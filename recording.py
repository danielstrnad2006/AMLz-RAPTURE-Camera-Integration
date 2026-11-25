"""
recording.py - whole pictures recorded continuously with precise times, and put into the areas afterwards
with the plate angle from an AMCF log.

Recording (camera_service.Service.record_pictures: the app's "Record pictures" button, or the API):
recordings/<start time>/ gets every new picture pair (or one every n s), written by a thread of its own
so the cameras never wait for the disk (pairs that find PictureRecorder.QUEUE pairs waiting are dropped
and counted):
    index.csv               one row per pair: frame, timestamp (the grab, unix s), thermal_time /
                            optical_time (when each camera's frame arrived from the camera, unix s, empty if
                            none), rotation_deg (the plate rotation typed in / set through the API then),
                            thermal_file / optical_file (empty: that camera delivered nothing new, or the
                            thermal camera calibrated)
    <frame>_thermal.npy     deg C, float32 (as save_pair)
    <frame>_optical.png     BGR
Not compressed yet: about 1.2 MB (thermal) + 3-4 MB (optical) per pair.

Times: time.time() of the camera PC (UTC unix seconds, 0.1 us resolution). A frame was exposed a little
before it arrived: THERMAL_DELAY / OPTICAL_DELAY (to be measured, see the README). For the AMCF angles to
match, the clocks of the camera PC and the AMCF PC must agree (synchronised over the LAN, e.g. chrony).

Afterwards:  python recording.py recordings/<start> --angles amcf_log.csv [--areas] [--every N]
The AMCF log is a CSV with a header: a time column (unix seconds, or milliseconds, or an ISO date/time:
local time unless it names a zone) and an angle column (deg; --angle-scale 0.01 for 1/100 deg). Each
camera's frame gets the plate angle interpolated at its own time (ANGLE_SIGN, ANGLE_ZERO turn the AMCF
angle into our plate rotation), written to rotations.csv; --areas then puts the pictures into every area
(skipping the ones an area has already). Without --angles the rotation recorded with each pair is used.
"""
import argparse
import csv
import io
import math
import queue
import threading
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from area_history import load_areas, load_history, record
from dual_camera import FramePair, placeholder

RECORDING_DIR = Path(__file__).resolve().parent / "recordings"
INDEX = ("frame", "timestamp", "thermal_time", "optical_time", "rotation_deg", "thermal_file", "optical_file")
THERMAL_DELAY = 0.0      # s between the exposure and the arrival of a thermal frame (not measured yet)
OPTICAL_DELAY = 0.0      # s, the same for the optical camera (USB webcam: probably 0.1 to 0.4 s)
ANGLE_SIGN = 1.0         # our plate rotation = ANGLE_SIGN * (AMCF angle - ANGLE_ZERO); both still to be
ANGLE_ZERO = 0.0         # checked with a mark on the plate (the cameras may turn with the A axis: sign -1?)
MAX_GAP = 3.0            # s: a picture between two log entries further apart than this gets no angle


def number(text):
    return float(text) if text not in ("", None) else math.nan


class PictureRecorder:
    """Whole picture pairs into recordings/<start>/: every new pair (every=0) or one every `every` s."""

    QUEUE = 30           # pairs waiting for the disk at most (~330 MB of memory); further ones are dropped

    def __init__(self, every=0.0, folder=None):
        self.every = max(0.0, float(every or 0))
        self.folder = Path(folder or RECORDING_DIR / f"{datetime.now():%Y-%m-%d_%H-%M-%S}")
        self.folder.mkdir(parents=True, exist_ok=False)
        self.queue = queue.Queue(self.QUEUE)
        self.frames = self.saved = self.dropped = 0
        self.due = 0.0                                 # timestamp from which the next pair is taken
        self.last = (math.nan, math.nan)               # arrival times of the last pair taken
        self.started = datetime.now()
        self.index = (self.folder / "index.csv").open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.index, INDEX)
        self.writer.writeheader()
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()
        print(f"Recording pictures into recordings/{self.folder.name}/ "
              + (f"every {self.every:g} s" if self.every else "(every new picture)"))

    def add(self, pair):
        """From the camera thread, for every grab: queue the pair if it is due and a camera has a new frame."""
        times = (pair.thermal_time, pair.optical_time)
        if all(math.isnan(t) for t in times):          # no arrival times (placeholders only, a test rig)
            times = (pair.timestamp, pair.timestamp)
        if not any(not math.isnan(t) and t != last for t, last in zip(times, self.last)) or pair.timestamp < self.due:
            return
        if self.every:                                 # on a fixed rhythm; after a hold-up it starts anew
            late = not self.due or pair.timestamp - self.due >= self.every
            self.due = pair.timestamp + self.every if late else self.due + self.every
        self.last = times
        try:
            self.queue.put_nowait((self.frames, pair))
            self.frames += 1
        except queue.Full:
            self.dropped += 1

    def _write(self):
        while True:
            item = self.queue.get()
            if item is None:
                return
            frame, pair = item
            stem = f"{frame:06d}"
            row = dict.fromkeys(INDEX, "")
            row.update(frame=frame, timestamp=f"{pair.timestamp:.6f}", rotation_deg=f"{pair.rotation:g}")
            try:
                if pair.thermal_live and not np.isnan(pair.thermal_c).all():
                    np.save(self.folder / f"{stem}_thermal.npy", pair.thermal_c)
                    row.update(thermal_file=f"{stem}_thermal.npy", thermal_time=f"{pair.thermal_time:.6f}")
                if pair.optical_live:
                    cv2.imwrite(str(self.folder / f"{stem}_optical.png"), pair.optical)
                    row.update(optical_file=f"{stem}_optical.png", optical_time=f"{pair.optical_time:.6f}")
                self.writer.writerow(row)
                self.index.flush()
                self.saved += 1
            except OSError as exc:                     # disk full, ...
                self.dropped += 1
                print(f"Recording pictures: frame {frame} not saved ({exc})")

    def state(self):
        return {"folder": self.folder.name, "every_s": self.every, "saved": self.saved,
                "queued": self.queue.qsize(), "dropped": self.dropped}

    def stop(self):
        self.queue.put(None)                           # after the pairs still waiting
        self.thread.join()
        self.index.close()
        seconds = (datetime.now() - self.started).total_seconds()
        print(f"Recording pictures stopped: {self.saved} pairs in recordings/{self.folder.name}/ in {seconds:.0f} s"
              + (f", {self.dropped} dropped (the disk was too slow)" if self.dropped else ""))


# Afterwards -------------------------------------------------------------------------------------------------
def load_index(folder):
    """The rows of a recording's index.csv: times and rotation as float (NaN if empty)."""
    with (Path(folder) / "index.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in ("timestamp", "thermal_time", "optical_time", "rotation_deg"):
            row[key] = number(row[key])
    return rows


def load_frame(folder, row):
    """One recorded pair as a FramePair (the false-colour thermal picture made from the temperatures)."""
    folder = Path(folder)
    thermal_live, optical_live = bool(row["thermal_file"]), bool(row["optical_file"])
    if thermal_live:
        temps = np.load(folder / row["thermal_file"])
        low, high = np.nanmin(temps), np.nanmax(temps)
        grey = np.uint8(np.clip((temps - low) / max(high - low, 1e-6) * 255, 0, 255))
        thermal = cv2.applyColorMap(grey, cv2.COLORMAP_INFERNO)
    else:
        thermal = placeholder("Thermal: Optris PI 640i")
        temps = np.full(thermal.shape[:2], np.nan, np.float32)
    optical = cv2.imread(str(folder / row["optical_file"])) if optical_live else placeholder("Optical: MOKOSE UC50")
    return FramePair(row["timestamp"], thermal, temps, optical, thermal_live, optical_live,
                     0.0 if math.isnan(row["rotation_deg"]) else row["rotation_deg"],
                     row["thermal_time"], row["optical_time"])


def parse_time(text):
    """Unix seconds from a log's time cell: a number (s, or ms if > 1e11) or an ISO date/time."""
    try:
        value = float(text)
        return value / 1000 if value > 1e11 else value
    except ValueError:
        return datetime.fromisoformat(text.strip().replace("Z", "+00:00")).timestamp()


def load_angles(path, time_column=None, angle_column=None, scale=1.0):
    """An AMCF angle log: (times unix s, angles deg) sorted by time, the angle unwrapped (no jump at 360).
    Columns by name; without names the first whose name contains "time" / "angle"."""
    text = Path(path).read_text(encoding="utf-8-sig")
    try:
        dialect = csv.Sniffer().sniff(text[:4096], ",;\t")
    except csv.Error:
        dialect = csv.excel
    rows = list(csv.DictReader(io.StringIO(text), dialect=dialect))
    names = list(rows[0]) if rows else []

    def column(given, word):
        found = given or next((n for n in names if word in n.lower()), None)
        if found not in names:
            raise ValueError(f"{path}: no column {given or word!r} (columns: {names})")
        return found

    tc, ac = column(time_column, "time"), column(angle_column, "angle")
    data = np.array([(parse_time(r[tc]), float(r[ac]) * scale) for r in rows if r[tc] and r[ac]], float)
    data = data[np.argsort(data[:, 0])]
    return data[:, 0], np.degrees(np.unwrap(np.radians(data[:, 1])))


def angle_at(times, log_times, log_angles, max_gap=MAX_GAP):
    """The logged angle interpolated at `times`; NaN outside the log or where its entries are more than
    `max_gap` s apart (the plate may have stopped or sped up in between)."""
    times = np.asarray(times, float)
    angles = np.interp(times, log_times, log_angles, left=np.nan, right=np.nan)
    after = np.clip(np.searchsorted(log_times, times), 1, len(log_times) - 1)
    gap = log_times[after] - log_times[after - 1]
    return np.where(gap <= max_gap, angles, np.nan)


def rotations(rows, log, sign=None, zero=None, delays=None):
    """Per recorded pair the plate rotation when each camera's frame was taken: [(thermal, optical)],
    NaN where the camera has no frame or the log has no angle then. Defaults: ANGLE_SIGN, ANGLE_ZERO,
    (THERMAL_DELAY, OPTICAL_DELAY)."""
    sign = ANGLE_SIGN if sign is None else sign
    zero = ANGLE_ZERO if zero is None else zero
    delays = (THERMAL_DELAY, OPTICAL_DELAY) if delays is None else delays
    out = {}
    for camera, delay in zip(("thermal", "optical"), delays):
        exposed = np.array([row[f"{camera}_time"] for row in rows], float) - delay
        out[camera] = sign * (angle_at(exposed, *log) - zero)
    return list(zip(out["thermal"], out["optical"]))


def process(folder, log=None, areas=None, every=1, sign=None, zero=None):
    """Put a recording into `areas` (default: all in areas/): each pair with each camera's own rotation
    (from `log` = load_angles(...), else the rotation recorded with it); pairs an area has already are
    skipped. Writes rotations.csv when there is a log. Returns the number of pairs put in."""
    folder = Path(folder)
    rows = load_index(folder)[::max(1, every)]
    turned = rotations(rows, log, sign, zero) if log is not None else [(r["rotation_deg"],) * 2 for r in rows]
    if log is not None:
        with (folder / "rotations.csv").open("w", newline="", encoding="utf-8") as f:
            out = csv.writer(f)
            out.writerow(("frame", "thermal_rotation_deg", "optical_rotation_deg"))
            out.writerows((r["frame"], f"{t:.4f}", f"{o:.4f}") for r, (t, o) in zip(rows, turned))
    areas = load_areas() if areas is None else areas
    if not areas:
        print("No areas: make some first (Areas window, or defect_detection.py --create)")
        return 0
    known = {a["name"]: {r["picture"] for r in load_history(a)} for a in areas}
    done = 0
    for row, (t_rot, o_rot) in zip(rows, turned):
        picture = f"recordings/{folder.name}/{int(row['frame']):06d}"
        todo = [a for a in areas if picture not in known[a["name"]]]
        if not todo:
            continue
        pair = load_frame(folder, row)
        pair = replace(pair, thermal_live=pair.thermal_live and not math.isnan(t_rot),
                       optical_live=pair.optical_live and not math.isnan(o_rot))
        if not (pair.thermal_live or pair.optical_live):
            continue                                   # no angle for either camera
        rotation = t_rot if pair.thermal_live else o_rot
        pair = replace(pair, rotation=float(rotation), thermal_rotation=None if math.isnan(t_rot) else float(t_rot),
                       optical_rotation=None if math.isnan(o_rot) else float(o_rot))
        record(pair, todo, picture)
        done += 1
    print(f"recordings/{folder.name}: {done} of {len(rows)} pairs put into {len(areas)} areas")
    return done


def main():
    parser = argparse.ArgumentParser(description="Put a picture recording into the areas, with AMCF plate angles.")
    parser.add_argument("folder", help="recordings/<start>")
    parser.add_argument("--angles", help="AMCF log: CSV with a time and an angle column")
    parser.add_argument("--time-column")
    parser.add_argument("--angle-column")
    parser.add_argument("--angle-scale", type=float, default=1.0, help="angle unit in deg (0.01 for 1/100 deg)")
    parser.add_argument("--sign", type=float, help=f"default ANGLE_SIGN = {ANGLE_SIGN:g}")
    parser.add_argument("--zero", type=float, help=f"AMCF angle at our plate rotation 0, deg (default ANGLE_ZERO = {ANGLE_ZERO:g})")
    parser.add_argument("--every", type=int, default=1, help="use every n-th pair only")
    parser.add_argument("--areas", action="store_true", help="put the pictures into all areas")
    args = parser.parse_args()
    rows = load_index(args.folder)
    times = [t for r in rows for t in (r["thermal_time"], r["optical_time"]) if not math.isnan(t)]
    print(f"{args.folder}: {len(rows)} pairs, {sum(bool(r['thermal_file']) for r in rows)} thermal / "
          f"{sum(bool(r['optical_file']) for r in rows)} optical frames"
          + (f", {datetime.fromtimestamp(min(times)):%Y-%m-%d %H:%M:%S} to {datetime.fromtimestamp(max(times)):%H:%M:%S}"
             if times else ""))
    log = None
    if args.angles:
        log = load_angles(args.angles, args.time_column, args.angle_column, args.angle_scale)
        turned = np.array(rotations(rows, log, args.sign, args.zero), float)
        print(f"{args.angles}: {len(log[0])} angles, {datetime.fromtimestamp(log[0][0]):%H:%M:%S} to "
              f"{datetime.fromtimestamp(log[0][-1]):%H:%M:%S}; pairs with an angle: thermal "
              f"{np.isfinite(turned[:, 0]).sum()}, optical {np.isfinite(turned[:, 1]).sum()}")
    if args.areas:
        process(args.folder, log, every=args.every, sign=args.sign, zero=args.zero)
    elif log is not None:
        print("(--areas puts them into the areas)")
    print(f"Camera delays used: thermal {THERMAL_DELAY:g} s, optical {OPTICAL_DELAY:g} s (THERMAL_DELAY / OPTICAL_DELAY)")


if __name__ == "__main__":
    main()
