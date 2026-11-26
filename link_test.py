"""Signal integrity tests of the links through the slipring.

USB MODE (default): USB -> Lindy extender -> Ethernet -> slipring -> Ethernet -> Lindy extender -> USB.
The far Lindy unit is a USB hub (Terminus FE1.1, VID 1A40 PID 0101): the PC sees it only while the
link through the ring works. Plug the near unit into the PC and the optical camera (UC50) into the far
unit, then run

    python link_test.py                 # watch the link + stream the camera at full size, preview window
    python link_test.py --size 800x600  # same at a low data rate (30 frames/s, far fewer bytes)
    python link_test.py --no-camera     # only watch which USB devices come and go

and turn the ring slowly. Every USB device arriving / leaving is printed with the time; once a second a
status line and a CSV row (linktest/<start>.csv): link (hub present), camera present, frames, frames/s,
read failures, cut-off frames (uniform bottom rows = JPEG ended early), repeated frames, longest gap
between frames. Linux also prints each device's USB speed (480 = high speed, 12 = full speed: a link that
fell back to full speed cannot carry the camera).

NETWORK MODE (Linux, root): the PC's Ethernet port -> slipring -> a USB-Ethernet adapter in the same PC,
so the strands are tested without the extenders. Frames with a sequence number and a known pattern go
both ways at once (raw Ethernet, EtherType 0x88B5: no IP addresses needed); each received frame is
checked. Once a second: link up / speed of both ports, frames sent / arrived / lost / changed per
direction, the ports' own CRC and other receive errors, link drops (linktest/<start>_net.csv).

    sudo python3 link_test.py --net                          # list the network ports
    sudo python3 link_test.py --net enp3s0 enx00e04c680001   # test between these two ports
    sudo python3 link_test.py --net A B --speed 100          # only 100 Mbit/s (pairs 1-2 and 3-6)
    sudo python3 link_test.py --net A B --cable-test         # first the ports' own cable test, if they can

Pairs (T568B): 1-2 orange, 3-6 green, 4-5 blue, 7-8 brown. 1000 Mbit/s sends on all four pairs in both
directions; 100 and 10 Mbit/s only on 1-2 and 3-6, one pair per direction. So: clean at 100 but errors or
no link at 1000 -> 4-5 or 7-8; errors in one direction only at 100 -> the pair carrying that direction
(which of 1-2 / 3-6 depends on the automatic crossover). --speed only advertises that speed (both ports
keep negotiating); the speeds advertised before are restored at the end. --cable-test (ethtool, many USB
adapters cannot) reports each pair as OK / open / short, with the distance to a fault; the link is down
while it runs. Use a port that is not the cable to the Windows PC, or that link is disturbed meanwhile.

In both modes: type a note into the console and press Enter (e.g. the ring angle, "90") to write it into
the CSV at that moment. q in the preview window or Ctrl+C ends with a summary.
"""
import argparse
import csv
import hashlib
import os
import random
import socket
import struct
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

WINDOWS = sys.platform == "win32"
FOLDER = Path(__file__).resolve().parent / "linktest"

KNOWN = {("1a40", "0101"): "Lindy extender hub", ("eba1", "7588"): "optical UC50",
         ("0403", "de37"): "thermal PI 640i"}
HUB = ("1a40", "0101")
CAMERA = ("eba1", "7588")
STALL = 1.0         # s without a frame counts as a stall in the summary


def usb_devices():
    """{device id: (vid, pid, speed Mbit/s or None)} of the USB devices present now (no interfaces)."""
    found = {}
    if WINDOWS:
        import ctypes
        cfg = ctypes.WinDLL("cfgmgr32")
        flags = 0x1 | 0x100                     # CM_GETIDLIST_FILTER_ENUMERATOR | _PRESENT
        size = ctypes.c_ulong()
        if cfg.CM_Get_Device_ID_List_SizeW(ctypes.byref(size), "USB", flags):
            return found
        buf = ctypes.create_unicode_buffer(size.value)
        if cfg.CM_Get_Device_ID_ListW("USB", buf, size.value, flags):
            return found
        for dev in buf[:size.value].split("\0"):
            up = dev.upper()
            if "VID_" in up and "&MI_" not in up:
                vid = up.split("VID_")[1][:4].lower()
                pid = up.split("PID_")[1][:4].lower() if "PID_" in up else "?"
                found[dev] = (vid, pid, None)
        return found
    for node in Path("/sys/bus/usb/devices").iterdir():
        try:
            vid, pid = (node / "idVendor").read_text().strip(), (node / "idProduct").read_text().strip()
            speed = (node / "speed").read_text().strip()
        except OSError:
            continue
        found[node.name] = (vid, pid, speed)
    return found


def describe(info):
    vid, pid, speed = info
    name = KNOWN.get((vid, pid), f"USB {vid}:{pid}")
    return name + (f" ({speed} Mbit/s)" if speed else "")


def stamp(t=None):
    return datetime.fromtimestamp(time.time() if t is None else t).strftime("%H:%M:%S.%f")[:-3]


class Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.frames = self.fails = self.cut = self.repeated = 0
        self.gap = 0.0                  # longest gap between frames in the current second
        self.last_frame = None
        self.notes = []
        self.link = self.camera = False
        self.devices = {}


def watch(stats, writer, out, stop):
    """Device arrivals / departures as they happen; a status line + CSV row every second."""
    summary = {"link_drops": 0, "link_down_s": 0.0, "camera_drops": 0, "stalls": 0, "longest_gap": 0.0,
               "frames": 0, "fails": 0, "cut": 0, "repeated": 0}
    next_row, down_since = time.time() + 1, None
    while not stop.is_set():
        now_devices = usb_devices()
        with stats.lock:
            old = stats.devices
            stats.devices = now_devices
        for dev in now_devices.keys() - old.keys():
            print(f"{stamp()}  + {describe(now_devices[dev])}   [{dev}]")
        for dev in old.keys() - now_devices.keys():
            print(f"{stamp()}  - {describe(old[dev])}   [{dev}]")
        link = any(v[:2] == HUB for v in now_devices.values())
        camera = any(v[:2] == CAMERA for v in now_devices.values())
        if stats.link and not link:
            summary["link_drops"] += 1
            down_since = time.time()
            print(f"{stamp()}  *** LINK LOST (extender hub gone)")
        if link and not stats.link and down_since is not None:
            summary["link_down_s"] += time.time() - down_since
            print(f"{stamp()}  *** link back after {time.time() - down_since:.1f} s")
            down_since = None
        if stats.camera and not camera:
            summary["camera_drops"] += 1
        stats.link, stats.camera = link, camera

        if time.time() >= next_row:
            next_row += 1
            with stats.lock:
                if stats.last_frame is not None:        # a gap still running counts too
                    stats.gap = max(stats.gap, time.monotonic() - stats.last_frame)
                row = [f"{time.time():.3f}", stamp(), int(link), int(camera), stats.frames, stats.fails,
                       stats.cut, stats.repeated, f"{stats.gap * 1000:.0f}", " | ".join(stats.notes)]
                for k in ("frames", "fails", "cut", "repeated"):
                    summary[k] += getattr(stats, k)
                if stats.gap >= STALL:
                    summary["stalls"] += 1
                summary["longest_gap"] = max(summary["longest_gap"], stats.gap)
                stats.frames = stats.fails = stats.cut = stats.repeated = 0
                stats.gap, stats.notes = 0.0, []
            writer.writerow(row)
            out.flush()
            flag = "" if link else "   LINK DOWN"
            flag += "   STALL" if float(row[8]) >= STALL * 1000 else ""
            flag += f"   {row[5]} read failures" if row[5] else ""
            flag += f"   {row[6]} cut-off" if row[6] else ""
            print(f"{row[1]}  link {'up  ' if link else 'DOWN'}  camera {'yes' if camera else 'no '}  "
                  f"{row[4]:>3} frames/s  longest gap {row[8]:>5} ms{flag}")
        time.sleep(0.1)
    if down_since is not None:
        summary["link_down_s"] += time.time() - down_since
    return summary


def notes(stats):
    for line in sys.stdin:
        if line.strip():
            with stats.lock:
                stats.notes.append(line.strip())
            print(f"{stamp()}  note: {line.strip()}")


def open_camera(size):
    import cv2                                       # only the USB mode needs OpenCV and the cameras
    from dual_camera import OPTICAL_NAME, video_devices
    index = next((i for i, n in video_devices() if OPTICAL_NAME in n.lower()), None)
    if index is None:
        return None
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if WINDOWS else cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    print(f"{stamp()}  camera opened: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    return cap


def stream(stats, size, show, stop):
    """Reads the camera as fast as it delivers; reopens it after it vanished. Runs in the main thread
    (DirectShow's device list wants COM in the thread that uses it; the preview window too)."""
    import cv2
    cap, last_hash, fails_in_row, retry = None, None, 0, 0.0
    while not stop.is_set():
        if cap is None:
            if time.monotonic() < retry:
                if show and cv2.waitKey(100) & 0xFF in (ord("q"), 27):
                    stop.set()
                elif not show:
                    time.sleep(0.1)
                continue
            cap = open_camera(size)
            retry = time.monotonic() + 1
            fails_in_row = 0
            continue
        ok, frame = cap.read()
        with stats.lock:
            if not ok:
                stats.fails += 1
            else:
                now = time.monotonic()
                if stats.last_frame is not None:
                    stats.gap = max(stats.gap, now - stats.last_frame)
                stats.last_frame = now
                stats.frames += 1
                bottom = frame[-8:]
                if bottom.min() == bottom.max():        # a JPEG that ended early leaves a flat bottom
                    stats.cut += 1
                h = hashlib.md5(frame[::8, ::8].tobytes()).digest()
                stats.repeated += h == last_hash        # sensor noise makes real frames differ
                last_hash = h
        fails_in_row = 0 if ok else fails_in_row + 1
        if fails_in_row >= 10:
            print(f"{stamp()}  camera delivers nothing; closing it, reopening when it is back")
            cap.release()
            cap, retry = None, time.monotonic() + 1
            with stats.lock:
                stats.last_frame = None
            continue
        if show and ok:
            cv2.imshow("link test (q quits)", cv2.resize(frame, (640, 640 * frame.shape[0] // frame.shape[1])))
        if show and cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            stop.set()
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


# ---- network mode (Linux) ----

SYS_NET = Path("/sys/class/net")
ETHERTYPE = 0x88B5                 # IEEE "local experimental": nothing else on the wire uses it
PAYLOAD = 1500                     # bytes per frame (full size): 2 direction + 8 sequence + pattern
PATTERNS = [bytes([b]) * 1490 for b in (0x00, 0xFF, 0x55, 0xAA)] + \
           [random.Random(n).randbytes(1490) for n in range(60)]   # fixed and random bit patterns
COUNTERS = ("rx_crc_errors", "rx_frame_errors", "rx_length_errors", "rx_missed_errors", "rx_errors", "tx_errors")
ADVERTISE = {10: 0x2, 100: 0x8, 1000: 0x20, 2500: 0x800000000000}      # ethtool mask of <speed>baseT/Full
MODES = {"10baseT/Half": 0x1, "10baseT/Full": 0x2, "100baseT/Half": 0x4, "100baseT/Full": 0x8,
         "1000baseT/Half": 0x10, "1000baseT/Full": 0x20, "2500baseT/Full": 0x800000000000,
         "5000baseT/Full": 0x1000000000000, "10000baseT/Full": 0x1000}


def sysfs(port, name):
    """A value of /sys/class/net/<port>/<name> as text, None if unreadable (e.g. speed while down)."""
    try:
        return (SYS_NET / port / name).read_text().strip()
    except OSError:
        return None


def ports():
    """The wired network ports: [(name, USB adapter?)]."""
    return [(p.name, "usb" in str((p / "device").resolve())) for p in sorted(SYS_NET.iterdir())
            if (p / "device").exists() and not (p / "wireless").exists() and not (p / "phy80211").exists()]


def link(port):
    """(link up, speed Mbit/s or None)."""
    up = sysfs(port, "carrier") == "1"
    speed = sysfs(port, "speed")
    return up, int(speed) if up and speed and int(speed) > 0 else None


def run(*command):
    """Output of a command, None if it is missing or fails."""
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout + done.stderr if done.returncode == 0 else None


def advertised(port):
    """ethtool mask of the link modes `port` advertises now, None if unknown."""
    text, mask, inside = run("ethtool", port) or "", 0, False
    for line in text.splitlines():
        if "Advertised link modes:" in line:
            inside, line = True, line.split(":", 1)[1]
        elif inside and ":" in line:
            break
        if inside:
            mask |= sum(MODES.get(mode, 0) for mode in line.split())
    return mask or None


class Direction:
    """Frames one way, A -> B or B -> A: counts for the current second and in total."""

    def __init__(self, tag, src, dst):
        self.tag, self.src, self.dst = tag.encode(), src, dst
        self.lock = threading.Lock()
        self.now = dict.fromkeys(("sent", "arrived", "lost", "changed", "late", "pc_dropped"), 0)
        self.total = dict(self.now)
        self.expected = None               # next sequence number expected

    def count(self, **add):
        with self.lock:
            for key, n in add.items():
                self.now[key] += n
                self.total[key] += n

    def second(self):
        """The counts of the second just ended (and starts the next)."""
        with self.lock:
            done, self.now = self.now, dict.fromkeys(self.now, 0)
        return done


def open_port(port):
    """Raw socket on `port` for our EtherType, with a large receive buffer (frames dropped because this
    PC read too slowly are counted apart: they are no fault of the cable)."""
    sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(ETHERTYPE))
    sock.bind((port, 0))
    try:
        sock.setsockopt(socket.SOL_SOCKET, 33, 16 << 20)        # SO_RCVBUFFORCE (root)
    except OSError:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 << 20)
    try:
        sock.setsockopt(263, 23, 1)           # SOL_PACKET, PACKET_IGNORE_OUTGOING: not our own frames back
    except OSError:
        pass                                  # older kernel: they are skipped in receive()
    sock.settimeout(0.2)
    return sock


def mac(port):
    return bytes.fromhex(sysfs(port, "address").replace(":", ""))


def send(sock, way, rate, stop):
    """Frames from way.src to way.dst at `rate` Mbit/s, paced in 10 ms steps."""
    head = mac(way.dst) + mac(way.src) + struct.pack("!H", ETHERTYPE) + way.tag
    per_step = max(1, round(rate * 1e6 * 0.01 / ((14 + PAYLOAD + 24) * 8)))   # + FCS, preamble, gap
    seq, due = 0, time.monotonic()
    while not stop.is_set():
        sent = 0
        for _ in range(per_step):
            try:
                sock.send(head + struct.pack("!Q", seq) + PATTERNS[seq % len(PATTERNS)])
            except OSError:                   # port down for a moment
                break
            seq, sent = seq + 1, sent + 1
        way.count(sent=sent)
        due += 0.01
        wait = due - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        elif wait < -0.1:                     # fell behind (e.g. the PC was busy): no catching up
            due = time.monotonic()


def receive(sock, way, stop):
    """Checks every frame of `way` arriving at its port: lost (gaps in the sequence), changed (pattern
    differs: a bit error the port's CRC check let through), late (out of order / repeated)."""
    while not stop.is_set():
        try:
            data, address = sock.recvfrom(4096)
        except socket.timeout:
            continue
        except OSError:
            time.sleep(0.1)
            continue
        if address[2] == socket.PACKET_OUTGOING or data[14:16] != way.tag or len(data) < 24:
            continue
        seq = struct.unpack("!Q", data[16:24])[0]
        changed = data[24:] != PATTERNS[seq % len(PATTERNS)]
        lost = late = 0
        if way.expected is not None and seq > way.expected:
            lost = seq - way.expected
        elif way.expected is not None and seq < way.expected:
            late = 1
        way.expected = seq + 1 if way.expected is None else max(way.expected, seq + 1)
        way.count(arrived=1, changed=int(changed), lost=lost, late=late)


def pc_dropped(sock):
    """Frames the kernel dropped for this socket since the last call (this PC too slow, not the cable)."""
    try:
        return struct.unpack("II", sock.getsockopt(263, 6, 8))[1]   # SOL_PACKET, PACKET_STATISTICS
    except OSError:
        return 0


def counters(port):
    return {name: int(sysfs(port, f"statistics/{name}") or 0) for name in COUNTERS}


def net_test(a, b, rate, speed, cable_test):
    names = [name for name, _ in ports()]
    if not a:
        print("Wired network ports (name, link, speed, USB adapter):")
        for name, usb in ports():
            up, mbit = link(name)
            print(f"    {name:18s} {'up  ' if up else 'down'} {f'{mbit} Mbit/s' if mbit else '':12s} {'USB' if usb else ''}")
        print("\nThen: sudo python3 link_test.py --net <port> <other port>")
        return
    if not b:
        sys.exit("Give two ports: the PC's own one and the USB adapter, e.g. --net enp3s0 enx00e04c680001")
    for port in (a, b):
        if port not in names:
            sys.exit(f"No wired network port {port!r}. These exist: {', '.join(names) or 'none'}")
    if os.geteuid() != 0:
        sys.exit("The network test sends raw Ethernet frames, which needs root: run it with sudo")
    for port in (a, b):
        if not int(sysfs(port, "flags") or "0", 16) & 1:      # IFF_UP
            print(f"{stamp()}  switching {port} on")
            run("ip", "link", "set", port, "up")
        if run("ip", "-4", "-o", "addr", "show", port):
            print(f"{stamp()}  note: {port} has an IP address, so it is also used for the network "
                  "(other traffic shares it; better not the cable to the Windows PC)")
    if cable_test:
        for port in (a, b):
            print(f"{stamp()}  cable test on {port} (link down meanwhile) ...")
            print(run("ethtool", "--cable-test", port) or "    not supported by this port (or ethtool missing)")
    before = {}
    if speed:
        for port in (a, b):
            before[port] = advertised(port)
            if run("ethtool", "-s", port, "advertise", hex(ADVERTISE[speed])) is None:
                print(f"{stamp()}  could not limit {port} to {speed} Mbit/s (ethtool missing or refused)")
    try:
        net_run(a, b, rate)
    finally:
        for port, mask in before.items():
            if mask:
                run("ethtool", "-s", port, "advertise", hex(mask))
                print(f"{stamp()}  {port}: the speeds advertised before restored")
            else:
                print(f"{stamp()}  {port}: speeds advertised before unknown; unplug it or run "
                      f"'sudo ethtool -s {port} advertise 0x3f' for 10/100/1000")


def net_run(a, b, rate):
    FOLDER.mkdir(exist_ok=True)
    path = FOLDER / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_net.csv"
    out = open(path, "w", newline="")
    writer = csv.writer(out)
    ways = [Direction("AB", a, b), Direction("BA", b, a)]
    writer.writerow(["time", "clock", "a_link", "a_mbit", "b_link", "b_mbit"]
                    + [f"{w.tag.decode()}_{k}" for w in ways for k in w.now]
                    + [f"{p}_{c}" for p in "ab" for c in COUNTERS] + ["note"])
    print(f"A = {a}, B = {b}; {rate:g} Mbit/s each way in {PAYLOAD}-byte frames; writing {path}")
    deadline = time.time() + 10
    while not (link(a)[0] and link(b)[0]) and time.time() < deadline:
        time.sleep(0.2)                       # the link takes a few seconds to come up after a change
    print(f"link: A {'up' if link(a)[0] else 'DOWN'} {link(a)[1] or ''}, B {'up' if link(b)[0] else 'DOWN'} "
          f"{link(b)[1] or ''} Mbit/s")
    print("Type a note (e.g. the ring angle) + Enter to mark the CSV. Ctrl+C ends.\n")
    sockets = {a: open_port(a), b: open_port(b)}
    stop, stats = threading.Event(), Stats()
    for way in ways:
        threading.Thread(target=send, args=(sockets[way.src], way, rate, stop), daemon=True).start()
        threading.Thread(target=receive, args=(sockets[way.dst], way, stop), daemon=True).start()
    threading.Thread(target=notes, args=(stats,), daemon=True).start()
    start, last = time.time(), {p: counters(p) for p in (a, b)}
    summary = {"drops": {a: 0, b: 0}, "speeds": set(), "errors": {a: 0, b: 0}}
    was_up = {a: link(a)[0], b: link(b)[0]}
    try:
        while True:
            time.sleep(1 - (time.time() - start) % 1)
            links = {p: link(p) for p in (a, b)}
            for p, (up, mbit) in links.items():
                if was_up[p] and not up:
                    summary["drops"][p] += 1
                    print(f"{stamp()}  *** {p}: LINK LOST")
                elif up and not was_up[p]:
                    print(f"{stamp()}  *** {p}: link back, {mbit} Mbit/s")
                was_up[p] = up
                if mbit:
                    summary["speeds"].add(mbit)
            for way in ways:
                way.count(pc_dropped=pc_dropped(sockets[way.dst]))
            done = [way.second() for way in ways]
            now = {p: counters(p) for p in (a, b)}
            delta = {p: {c: now[p][c] - last[p][c] for c in COUNTERS} for p in (a, b)}
            last = now
            for p in (a, b):
                summary["errors"][p] += max(delta[p]["rx_errors"], delta[p]["rx_crc_errors"])   # rx_errors includes CRC
            with stats.lock:
                note, stats.notes = " | ".join(stats.notes), []
            writer.writerow([f"{time.time():.3f}", stamp(), int(links[a][0]), links[a][1] or "", int(links[b][0]),
                             links[b][1] or ""] + [d[k] for d in done for k in d]
                            + [delta[p][c] for p in (a, b) for c in COUNTERS] + [note])
            out.flush()
            line = f"{stamp()}  A {links[a][1] or 'DOWN':>4} B {links[b][1] or 'DOWN':>4}"
            for way, d in zip(ways, done):
                line += f" | {way.tag.decode()[0]}>{way.tag.decode()[1]} {d['arrived']:>5}/{d['sent']:<5}"
                line += f" lost {d['lost']}" if d["lost"] else ""
                line += f" CHANGED {d['changed']}" if d["changed"] else ""
            bad = {p: max(delta[p]["rx_errors"], delta[p]["rx_crc_errors"]) for p in (a, b)}
            line += "".join(f"   {'A' if p == a else 'B'} CRC/rx errors {n}" for p, n in bad.items() if n)
            print(line)
    except KeyboardInterrupt:
        pass
    stop.set()
    time.sleep(0.5)
    out.close()
    print(f"\nSummary over {time.time() - start:.0f} s, speeds seen: "
          f"{', '.join(f'{s} Mbit/s' for s in sorted(summary['speeds'])) or 'no link'}")
    for way in ways:
        t = way.total
        name = f"{way.src} -> {way.dst}"
        share = t["lost"] / max(1, t["sent"])
        print(f"  {name}: {t['sent']} sent, {t['arrived']} arrived, {t['lost']} lost ({share:.4%}), "
              f"{t['changed']} changed, {t['late']} late; {t['pc_dropped']} of the lost dropped by this PC "
              "(too slow, not the cable)")
    for p in (a, b):
        print(f"  {p}: link lost {summary['drops'][p]} times, {summary['errors'][p]} CRC / receive errors")
    print(f"  per second: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--size", help="camera size WxH (default: OPTICAL_SIZE)")
    parser.add_argument("--no-camera", action="store_true", help="only watch the USB devices")
    parser.add_argument("--no-window", action="store_true", help="no preview window")
    parser.add_argument("--net", nargs="*", metavar="PORT", help="network mode: two ports (none: list them)")
    parser.add_argument("--rate", type=float, default=50, help="network mode: Mbit/s each way (default 50)")
    parser.add_argument("--speed", type=int, choices=sorted(ADVERTISE), help="network mode: only this link speed")
    parser.add_argument("--cable-test", action="store_true", help="network mode: the ports' own cable test first")
    args = parser.parse_args()
    if args.net is not None:
        if WINDOWS:
            sys.exit("The network mode works on Linux only")
        net_test(*(args.net + [None, None])[:2], args.rate, args.speed, args.cable_test)
        return
    if args.size:
        size = tuple(int(v) for v in args.size.lower().split("x"))
    else:
        from dual_camera import OPTICAL_SIZE
        size = OPTICAL_SIZE

    FOLDER.mkdir(exist_ok=True)
    path = FOLDER / f"{datetime.now():%Y-%m-%d_%H-%M-%S}.csv"
    out = open(path, "w", newline="")
    writer = csv.writer(out)
    writer.writerow(["time", "clock", "link", "camera", "frames", "read_failures", "cut_off", "repeated",
                     "longest_gap_ms", "note"])

    stats, stop, result = Stats(), threading.Event(), {}
    stats.devices = usb_devices()
    print("USB devices present:")
    for info in stats.devices.values():
        print(f"    {describe(info)}")
    stats.link = any(v[:2] == HUB for v in stats.devices.values())
    stats.camera = any(v[:2] == CAMERA for v in stats.devices.values())
    print(f"link {'up' if stats.link else 'DOWN (extender hub not seen)'}, "
          f"camera {'seen' if stats.camera else 'not seen'}; writing {path}")
    print("Type a note (e.g. the ring angle) + Enter to mark the CSV. q in the window or Ctrl+C ends.\n")

    watcher = threading.Thread(target=lambda: result.update(watch(stats, writer, out, stop)), daemon=True)
    watcher.start()
    threading.Thread(target=notes, args=(stats,), daemon=True).start()
    start = time.time()
    try:
        if args.no_camera:
            while True:
                time.sleep(0.5)
        else:
            stream(stats, size, not args.no_window, stop)
    except KeyboardInterrupt:
        pass
    stop.set()
    watcher.join(2)
    out.close()

    s = result
    if s:
        print(f"\nSummary over {time.time() - start:.0f} s:")
        print(f"  link lost {s['link_drops']} times, down {s['link_down_s']:.1f} s in total; "
              f"camera vanished {s['camera_drops']} times")
        if not args.no_camera:
            print(f"  {s['frames']} frames, {s['fails']} read failures, {s['cut']} cut-off, "
                  f"{s['repeated']} repeated; {s['stalls']} seconds with a gap >= {STALL:.0f} s, "
                  f"longest gap {s['longest_gap'] * 1000:.0f} ms")
    print(f"  per second: {path}")


if __name__ == "__main__":
    main()
