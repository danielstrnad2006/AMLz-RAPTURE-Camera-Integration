"""
camera_service.py - the cameras as a background service with an HTTP API (no window needed).

Run:  python camera_service.py [--host 0.0.0.0] [--port 8765]     (Ctrl+C stops it)
      app.py runs the same service inside itself and serves the same API while its windows are open
      (python app.py --host 0.0.0.0 to reach it from the network). Only one program can use the
      cameras: run either this or app.py.
Remote: python app.py --connect <camera PC>[:port] works with the service on another computer
      (RemoteService): live picture, state and log over the network; pictures taken by hand are
      fetched and stored on this computer; clips and picture recordings stay on the camera PC.

The Service owns the cameras (grabbing continuously in a thread of its own) and does everything that has to
go on without anyone watching: pictures, clips / timelapses, continuous recording of whole pictures
(recording.py), putting pictures into the areas, "record every n s", the plate rotation for the next
pictures (by hand now, later from the PLC).

HTTP API (JSON in and out; NaN = null). By default only reachable from this PC (127.0.0.1); --host 0.0.0.0
opens it to the network. There is no login: whoever reaches it can record and create areas.
    GET  /api/state                   cameras, recording, areas recording, rotation
    GET  /api/live.jpg                the live side-by-side picture
    GET  /api/live.mjpg               the same as a stream (open it in a browser)
    GET  /api/pair                    the latest picture pair, whole (NumPy .npz: thermal, thermal_c, optical,
                                      meta = JSON with timestamp, live flags, rotation, arrival times)
    GET  /api/log?after=n             what the service printed: {"next": n, "lines": [...]}
    GET  /api/temperature?x=&y=       deg C at build plate point x', y' mm, now
    GET  /api/areas                   the areas (corners in plate mm, pictures, last mean deg C)
    GET  /api/areas/<name>            one area and its history
    POST /api/picture                 take a picture into pictures/
    POST /api/record                  {"kind": "clip" | "timelapse"}: start recording
    POST /api/record/stop
    POST /api/pictures/record         {"every_s": n}: record whole pictures into recordings/ (0 = every new pair)
    POST /api/pictures/stop
    POST /api/areas                   {"name": ..., "corners": [[x', y'], ...]}: a new area (2 corners = a rectangle)
    POST /api/areas/record            the current moment into all areas
    POST /api/areas/every             {"seconds": n}: into all areas every n s; 0 or null = stop
    POST /api/rotation                {"deg": d}: the plate rotation for the pictures from now on
"""
import argparse
import collections
import io
import json
import math
import sys
import threading
import time
import traceback
from dataclasses import replace
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse
from urllib.request import Request, urlopen

import cv2
import numpy as np

from area_history import load_areas, load_history, new_area, record
from capture_dual_video import Recorder
from dual_camera import CameraRig, FramePair, load_pair, save_pair, side_by_side
from recording import PictureRecorder

HOST, PORT = "127.0.0.1", 8765
LOG = collections.deque(maxlen=2000)   # (number, line) of what was printed, for GET /api/log
_log = {"count": 0, "partial": "", "lock": threading.Lock()}


def remember(text):
    """Keep printed text, line by line, for GET /api/log (app.py's log and the headless service call it)."""
    with _log["lock"]:
        lines = (_log["partial"] + text).split("\n")
        _log["partial"] = lines.pop()
        for line in lines:
            _log["count"] += 1
            LOG.append((_log["count"], line))


class Tee:
    """stdout / stderr of the headless service: to the console and to remember()."""

    def __init__(self, console):
        self.console = console

    def write(self, text):
        self.console.write(text)
        remember(text)

    def flush(self):
        self.console.flush()


class Cameras(threading.Thread):
    """The CameraRig, grabbing continuously in its own thread: rig.grab() waits up to 0.4 s for the
    optical camera, nobody else must. Also feeds the video being recorded, so it gets every picture."""

    def __init__(self):
        super().__init__(daemon=True)
        self.lock = threading.Lock()       # held while the recorder is used
        self.latest = None                 # (pair, side_by_side(pair)) of the last grab
        self.count = 0                     # grabs so far
        self.calibrating = False           # the thermal camera of the last grab calibrates (no temperatures)
        self.rate = 0.0                    # grabs per second over the last 5 s
        self.recorder = None               # capture_dual_video.Recorder while recording
        self.pictures = None               # recording.PictureRecorder while recording whole pictures
        self.rotation = 0.0                # deg the plate is turned by: given to every pair grabbed
        self.jobs = []                     # functions f(rig) to run in this thread, e.g. the settings window
        self.stopping = False

    def run(self):
        times = []
        try:
            with CameraRig() as rig:
                while not self.stopping:
                    while self.jobs:
                        self.jobs.pop(0)(rig)
                    pair = replace(rig.grab(), rotation=self.rotation)
                    frame = side_by_side(pair)
                    with self.lock:
                        if self.recorder is not None:
                            self.recorder.add(pair, frame)
                        if self.pictures is not None:
                            self.pictures.add(pair)
                        self.calibrating = pair.thermal_live and bool(np.isnan(pair.thermal_c).all())
                        self.latest, self.count = (pair, frame), self.count + 1
                    now = time.monotonic()
                    times = [t for t in times if t > now - 5] + [now]
                    self.rate = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0.0
                    if not pair.optical_live:
                        time.sleep(0.03)       # no optical camera to wait for: about the thermal frame rate
        except Exception:
            traceback.print_exc()
            print("Camera thread stopped; restart the program")
        finally:
            with self.lock:
                if self.recorder is not None:
                    self.recorder.stop()
                    self.recorder = None
                if self.pictures is not None:
                    self.pictures.stop()
                    self.pictures = None


class Service:
    """The cameras and what is done with them, for app.py and the HTTP API alike (thread-safe)."""

    remote = False

    def __init__(self):
        self.cams = Cameras()
        self.keep = False                  # also keep the whole pictures of what goes into the areas
        self.every = None                  # s between pictures into the areas while "record every" is on
        self.areas_changed = 0             # counts changes of the areas / their histories (windows refresh)
        self._areas_lock = threading.Lock()    # one put() at a time
        self._wake = threading.Event()     # wakes the "record every" thread
        self._every_thread = None
        self._jpeg = (-1, b"")             # (cams.count, JPEG of that live picture)
        self._frame = (-1, None)           # (cams.count, live picture with the recording mark)
        self.closing = False

    @property
    def rotation(self):
        """deg the plate is turned by, for the pictures from now on."""
        return self.cams.rotation

    @rotation.setter
    def rotation(self, deg):
        self.cams.rotation = float(deg)

    def start(self):
        self.cams.start()
        return self

    def pair(self):
        """The latest FramePair, None before the first grab."""
        latest = self.cams.latest
        return latest[0] if latest else None

    def frame(self):
        """(number, the latest side-by-side picture with the recording mark), None before the first grab."""
        latest, count, recorder = self.cams.latest, self.cams.count, self.cams.recorder
        if latest is None:
            return None
        if self._frame[0] != count:
            frame = latest[1]
            if recorder is not None:
                frame = frame.copy()
                recorder.mark(frame)
            self._frame = (count, frame)
        return self._frame

    def state(self):
        """Cameras ("live", "calibrating", "not connected", "starting", "stopped"), recording, areas, rotation."""
        latest, recorder, pictures = self.cams.latest, self.cams.recorder, self.cams.pictures
        pair = latest[0] if latest else None
        stopped = self.cams.ident is not None and not self.cams.is_alive()

        def camera(live, calibrating=False):
            if stopped:
                return "stopped"
            if pair is None:
                return "starting"
            if not live:
                return "not connected"
            return "calibrating" if calibrating else "live"

        return {"thermal": camera(pair and pair.thermal_live, self.cams.calibrating),
                "optical": camera(pair and pair.optical_live),
                "pictures_per_s": round(self.cams.rate, 2),
                "picture_time": pair.timestamp if pair else None,
                "recording": None if recorder is None else "timelapse" if recorder.timelapse else "clip",
                "recording_file": recorder.path.name if recorder else None,
                "pictures": pictures.state() if pictures else None,
                "areas_every_s": self.every, "rotation_deg": self.rotation, "keep_whole_pictures": self.keep}

    def jpeg(self):
        """The live picture as JPEG (with the recording mark), encoded once per new picture."""
        shown = self.frame()
        if shown is None:
            return None
        if self._jpeg[0] != shown[0]:
            self._jpeg = (shown[0], cv2.imencode(".jpg", shown[1], [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes())
        return self._jpeg[1]

    def take_picture(self):
        """The current moment as pictures in pictures/ (see dual_camera.save_pair). Returns the path or None."""
        pair = self.pair()
        if pair is None:
            print("No picture from the cameras yet")
            return None
        path = save_pair(replace(pair, rotation=self.rotation))   # the rotation as set now (by hand / API)
        print(f"Picture saved: {path.parent.name}/{path.name} (side by side; next to it the original pictures "
              f"and the temperatures: {path.stem}_thermal.npy / _thermal.png / _optical.png)")
        return path

    def record(self, timelapse=None):
        """Start a clip (timelapse False) / timelapse (True); None = stop. Returns the recording kind or None."""
        cams = self.cams
        with cams.lock:
            if cams.recorder is not None and (timelapse is None or cams.recorder.timelapse != timelapse):
                cams.recorder.stop()
                cams.recorder = None
            if timelapse is not None and cams.recorder is None and cams.latest is not None:
                cams.recorder = Recorder(timelapse, cams.latest[1].shape[1::-1])
            return None if cams.recorder is None else "timelapse" if cams.recorder.timelapse else "clip"

    def record_pictures(self, every=None):
        """Record whole pictures into recordings/<start>/ (recording.PictureRecorder): every new pair
        (every 0) or one every `every` s; None = stop. Returns the recording's state or None."""
        cams = self.cams
        with cams.lock:
            old, cams.pictures = cams.pictures, None
        if old is not None:
            old.stop()                                 # outside the lock: writes what is still waiting
        if every is not None:
            recorder = PictureRecorder(every)
            with cams.lock:
                cams.pictures = recorder
        return cams.pictures.state() if cams.pictures else None

    def optical_settings(self):
        """The optical camera's own settings window (in the camera thread; Windows: live view pauses)."""
        if sys.platform == "win32":
            print("Optical camera settings window: the live view pauses until it is closed")
        self.cams.jobs.append(lambda rig: rig.optical_settings())

    # Areas -------------------------------------------------------------------------------------------------
    def new_area(self, name, corners):
        area = new_area(name, corners)
        self.areas_changed += 1
        print(f"Area '{area['name']}' created, corners {area['corners']} mm on the plate; its history goes to "
              f"areas/{area['folder'].name}/")
        return area

    def put(self, sources, rotation=None, quiet=False):
        """Pictures into all areas (blocking, one call at a time; None if not done). `sources` = FramePairs
        (turned by `rotation`, default self.rotation; whole pairs kept in pictures/ if self.keep) or paths
        of pairs stored by save_pair (left out of an area that has them already; their own rotation, or
        `rotation` if they have none stored). Returns [(pair, {area name: history row}), ...]."""
        rotation = self.rotation if rotation is None else rotation
        areas = load_areas()
        if not areas:
            print("No areas yet: make one first (Areas window: New area)")
            return None
        # quiet = record every: skips a turn while busy; else waits for the pictures being put in
        if not (self._areas_lock.acquire(blocking=False) if quiet else self._areas_lock.acquire(timeout=60)):
            if not quiet:
                print("Still putting the previous pictures into the areas")
            return None
        try:
            known = {a["name"]: {r["picture"] for r in load_history(a)} for a in areas}
            results = []
            for source in sources:
                if isinstance(source, Path):
                    pair, picture = load_pair(source), source.name
                    if not source.with_suffix(".json").exists():      # taken before rotations were stored
                        pair = replace(pair, rotation=rotation)
                    todo = [a for a in areas if picture not in known[a["name"]]]
                else:
                    pair = replace(source, rotation=rotation)
                    picture, todo = save_pair(pair).name if self.keep else "", areas
                done = record(pair, todo, picture)
                results.append((pair, done))
                parts = [f"{name} " + (f"{float(row['thermal_mean_C']):.1f} °C" if row["thermal_mean_C"] else "no °C")
                         + f" (seen: thermal {float(row['thermal_seen']):.0%}, optical {float(row['optical_seen']):.0%})"
                         for name, row in done.items()]
                parts += [f"{a['name']} not seen" for a in todo if a["name"] not in done]
                if len(todo) > 10:                     # e.g. the areas of a 3MF job: a summary
                    means = [float(row["thermal_mean_C"]) for row in done.values() if row["thermal_mean_C"]]
                    parts = [f"{len(done)} of {len(todo)} areas seen" + (
                        f", mean {min(means):.1f} to {max(means):.1f} °C in {len(means)}" if means else ", no °C")]
                print(f"{datetime.fromtimestamp(pair.timestamp):%H:%M:%S}" + (f", plate at {pair.rotation:g}°" if pair.rotation else "")
                      + ": " + ("; ".join(parts) if todo else f"{picture} is in all histories already")
                      + (f"  (whole pictures: pictures/{picture})" if self.keep and not isinstance(source, Path) else ""))
            return results
        finally:
            self.areas_changed += 1
            self._areas_lock.release()

    def put_now(self, quiet=False):
        """The current moment into all areas (see put)."""
        pair = self.pair()
        if pair is None:
            print("No picture from the cameras yet")
            return None
        return self.put([pair], quiet=quiet)

    def record_every(self, seconds):
        """The current moment into all areas every `seconds` s from now on (each new picture once);
        None or 0 = stop (a picture being put in then still goes in). Returns False (and says why) if it
        cannot start."""
        if not seconds:
            if self.every is not None:
                print("Recording into the areas stopped")
            self.every = None
        elif seconds < 0 or not load_areas():
            print("Record every: needs seconds > 0 and at least one area")
            return False
        else:
            self.every = max(0.5, float(seconds))
            print(f"Recording into the areas every {self.every:g} s")
            if self._every_thread is None:
                self._every_thread = threading.Thread(target=self._record_every, daemon=True)
                self._every_thread.start()
        self._wake.set()
        return True

    def _record_every(self):
        last = None                                    # timestamp of the last picture put into the areas
        while not self.closing:
            self._wake.clear()
            if self.every is None:
                self._wake.wait()
                continue
            pair = self.pair()
            if pair is not None and pair.timestamp != last and self.put([pair], quiet=True) is not None:
                last = pair.timestamp
            self._wake.wait(self.every)

    def close(self):
        self.closing = True
        self._wake.set()
        self.cams.stopping = True
        self.cams.join(timeout=3)
        with self.cams.lock:
            if self.cams.recorder is not None:     # camera thread stuck (e.g. settings window open)
                self.cams.recorder.stop()
                self.cams.recorder = None
            pictures, self.cams.pictures = self.cams.pictures, None
        if pictures is not None:
            pictures.stop()


# HTTP API ----------------------------------------------------------------------------------------------------
def plain(value):
    """`value` for JSON: NaN / inf -> None, numpy numbers and arrays -> Python, Paths -> text."""
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [plain(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Path):
        return value.name
    return value


PAIR_META = ("timestamp", "thermal_live", "optical_live", "rotation", "thermal_time", "optical_time")


def pair_bytes(pair):
    """A FramePair as an uncompressed NumPy .npz (GET /api/pair)."""
    out = io.BytesIO()
    np.savez(out, thermal=pair.thermal, thermal_c=pair.thermal_c, optical=pair.optical,
             meta=np.array(json.dumps(plain({k: getattr(pair, k) for k in PAIR_META}))))
    return out.getvalue()


def pair_from_bytes(data):
    with np.load(io.BytesIO(data)) as z:
        meta = json.loads(str(z["meta"]))
        return FramePair(meta["timestamp"], z["thermal"], z["thermal_c"], z["optical"], meta["thermal_live"],
                         meta["optical_live"], meta["rotation"],
                         *(math.nan if meta[k] is None else meta[k] for k in ("thermal_time", "optical_time")))


def area_summary(area):
    rows = load_history(area)
    last = next((r["thermal_mean_C"] for r in reversed(rows) if np.isfinite(r["thermal_mean_C"])), None)
    return {"name": area["name"], "corners": area["corners"], "pictures": len(rows), "last_mean_C": last,
            "created": area.get("created")}


class Handler(BaseHTTPRequestHandler):
    server_version = "RAPTUREcameras/1"

    def log_message(self, format, *args):              # no line per request
        pass

    def reply(self, status, data=None, body=None, kind="application/json"):
        body = json.dumps(plain(data), allow_nan=False).encode() if body is None else body
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        service, url = self.server.service, urlparse(self.path)
        path = url.path.rstrip("/")
        try:
            if path in ("", "/api"):
                self.reply(200, body=__doc__[__doc__.index("HTTP API (JSON"):].encode(), kind="text/plain; charset=utf-8")
            elif path == "/api/state":
                self.reply(200, service.state())
            elif path == "/api/live.jpg":
                jpeg = service.jpeg()
                self.reply(200, body=jpeg, kind="image/jpeg") if jpeg else self.reply(503, {"error": "no picture yet"})
            elif path == "/api/live.mjpg":
                self.stream(service)
            elif path == "/api/pair":
                pair = service.pair()
                self.reply(200, body=pair_bytes(pair), kind="application/octet-stream") if pair else self.reply(
                    503, {"error": "no picture yet"})
            elif path == "/api/log":
                after = int(parse_qs(url.query).get("after", ["0"])[0])
                lines = [(n, line) for n, line in list(LOG) if n > after]
                self.reply(200, {"next": lines[-1][0] if lines else _log["count"],
                                 "lines": [line for _, line in lines]})
            elif path == "/api/temperature":
                query, pair = parse_qs(url.query), service.pair()
                x, y = float(query["x"][0]), float(query["y"][0])
                self.reply(200, {"x": x, "y": y, "temperature_C": pair.temperature(x, y) if pair else None,
                                 "picture_time": pair.timestamp if pair else None})
            elif path == "/api/areas":
                self.reply(200, [area_summary(a) for a in load_areas()])
            elif path.startswith("/api/areas/"):
                name = unquote(path[len("/api/areas/"):])
                area = next((a for a in load_areas() if a["name"] == name), None)
                if area is None:
                    self.reply(404, {"error": f"no area '{name}'"})
                else:
                    self.reply(200, {**area_summary(area), "history": load_history(area)})
            else:
                self.reply(404, {"error": "unknown address; GET /api lists them"})
        except (KeyError, ValueError) as exc:
            self.reply(400, {"error": f"bad request: {exc}"})
        except ConnectionError:                        # the client went away
            pass
        except Exception as exc:                       # a bug: say so, and answer anyway
            traceback.print_exc()
            self.reply(500, {"error": f"{type(exc).__name__}: {exc}"})

    def do_POST(self):
        service, path = self.server.service, urlparse(self.path).path.rstrip("/")
        try:
            length = int(self.headers.get("Content-Length") or 0)
            data = json.loads(self.rfile.read(length) or b"{}") if length else {}
            if path == "/api/picture":
                picture = service.take_picture()
                self.reply(200, {"picture": f"pictures/{picture.name}"}) if picture else self.reply(503, {"error": "no picture yet"})
            elif path == "/api/record":
                kind = data.get("kind", "clip")
                if kind not in ("clip", "timelapse"):
                    raise ValueError('kind must be "clip" or "timelapse"')
                self.reply(200, {"recording": service.record(kind == "timelapse")})
            elif path == "/api/record/stop":
                self.reply(200, {"recording": service.record(None)})
            elif path == "/api/pictures/record":
                self.reply(200, {"pictures": service.record_pictures(float(data.get("every_s") or 0))})
            elif path == "/api/pictures/stop":
                self.reply(200, {"pictures": service.record_pictures(None)})
            elif path == "/api/areas":
                self.reply(200, area_summary(service.new_area(data["name"], np.asarray(data["corners"], float))))
            elif path == "/api/areas/record":
                results = service.put_now()
                if results is None:
                    self.reply(409, {"error": "no areas, no picture yet, or still busy with the previous one"})
                else:
                    self.reply(200, {"picture_time": results[-1][0].timestamp, "areas": results[-1][1]})
            elif path == "/api/areas/every":
                ok = service.record_every(data.get("seconds"))
                self.reply(200 if ok else 400, {"areas_every_s": service.every})
            elif path == "/api/rotation":
                service.rotation = float(data["deg"])
                self.reply(200, {"rotation_deg": service.rotation})
            else:
                self.reply(404, {"error": "unknown address; GET /api lists them"})
        except (KeyError, ValueError, TypeError, OSError) as exc:
            self.reply(400, {"error": f"bad request: {exc}"})
        except ConnectionError:                        # the client went away
            pass
        except Exception as exc:                       # a bug: say so, and answer anyway
            traceback.print_exc()
            self.reply(500, {"error": f"{type(exc).__name__}: {exc}"})

    def stream(self, service):
        """MJPEG: every new live picture, until the client goes away."""
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        shown = -1
        while not service.closing:
            latest = service.frame()
            if latest is None or latest[0] == shown:
                time.sleep(0.03)
                continue
            shown = latest[0]
            jpeg = service.jpeg()
            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpeg)
                             + jpeg + b"\r\n")


class Server(ThreadingHTTPServer):
    allow_reuse_address = sys.platform != "win32"    # Windows: else a second server opens the same port quietly
    daemon_threads = True                            # open live streams do not keep the program alive


def serve(service, host=HOST, port=PORT):
    """The HTTP API of `service` in a thread of its own; returns the server (shutdown() stops it) or None
    if the port is taken (then probably another camera service runs and holds the cameras)."""
    try:
        server = Server((host, port), Handler)
    except OSError as exc:
        print(f"Camera service API not started: port {port} not free ({exc.strerror}). Is another camera "
              "service (camera_service.py or app.py) running? It holds the cameras then.")
        return None
    server.service = service
    threading.Thread(target=server.serve_forever, daemon=True).start()
    shown = "this PC only" if host in ("127.0.0.1", "localhost") else "the network"
    print(f"Camera service API: http://{'localhost' if host == '0.0.0.0' else host}:{port}/api ({shown})")
    return server


class RemoteService:
    """A Service running on another computer, used through its HTTP API (app.py --connect): the same
    methods as Service for the app's main, calibration and alignment windows. The state and the camera
    PC's log are fetched twice a second, the live picture comes as the MJPEG stream; pictures taken by
    hand are fetched whole (GET /api/pair) and stored on this computer; clips and picture recordings
    are made and kept on the camera PC."""

    remote = True
    OFFLINE = {"thermal": "unreachable", "optical": "unreachable", "pictures_per_s": 0.0, "picture_time": None,
               "recording": None, "recording_file": None, "pictures": None, "areas_every_s": None,
               "rotation_deg": 0.0, "keep_whole_pictures": False}

    def __init__(self, address):
        url = urlparse(address if "://" in address else f"http://{address}")
        self.url = f"{url.scheme}://{url.hostname}:{url.port or PORT}"
        self.host = url.hostname
        self._state = dict(self.OFFLINE)
        self._frame = None                 # (number, side-by-side picture) from the stream
        self._log_next = None              # the camera PC's log: lines after this number are new
        self.reachable = None              # None until the first answer / failure
        self.closing = False

    def start(self):
        for target in (self._poll, self._stream):
            threading.Thread(target=target, daemon=True).start()
        return self

    def call(self, path, data=None, timeout=10):
        """GET (data None) or POST (data = JSON) to the camera PC; the answer's JSON (or bytes)."""
        request = Request(self.url + path, None if data is None else json.dumps(data).encode(),
                          {"Content-Type": "application/json"})
        with urlopen(request, timeout=timeout) as answer:
            body = answer.read()
            return json.loads(body) if answer.headers.get_content_type() == "application/json" else body

    def _poll(self):
        while not self.closing:
            try:
                self._state = self.call("/api/state", timeout=3)
                if self._log_next is None:             # older lines are not shown, only from now on
                    self._log_next = self.call("/api/log?after=1000000000", timeout=3)["next"]
                log = self.call(f"/api/log?after={self._log_next}", timeout=3)
                self._log_next = log["next"]
                for line in log["lines"]:
                    print(f"[{self.host}] {line}")
                if not self.reachable:
                    print(f"Connected to the camera service at {self.url}")
                self.reachable = True
            except (OSError, ValueError) as exc:
                if self.reachable is not False:
                    print(f"Camera service at {self.url} not reachable ({exc}); trying again")
                self.reachable, self._state = False, dict(self.OFFLINE)
            time.sleep(0.5)

    def _stream(self):
        count = 0
        while not self.closing:
            try:
                with urlopen(self.url + "/api/live.mjpg", timeout=10) as stream:
                    while not self.closing:
                        length = None
                        while True:                    # the part's header lines, up to the blank line
                            line = stream.readline()
                            if not line:
                                raise ConnectionError("stream ended")
                            if line.lower().startswith(b"content-length:"):
                                length = int(line.split(b":")[1])
                            elif not line.strip() and length is not None:    # the blank line
                                break
                        frame = cv2.imdecode(np.frombuffer(stream.read(length), np.uint8), cv2.IMREAD_COLOR)
                        if frame is not None:
                            count += 1
                            self._frame = (count, frame)
            except (OSError, ValueError):
                time.sleep(2)

    def state(self):
        return dict(self._state)

    def frame(self):
        return self._frame

    def pair(self):
        """The camera PC's latest picture pair, whole (about 11 MB over the network); None if not possible."""
        try:
            return pair_from_bytes(self.call("/api/pair", timeout=30))
        except (OSError, ValueError, KeyError) as exc:
            print(f"No picture from the camera service at {self.url} ({exc})")
            return None

    def take_picture(self):
        """The camera PC's latest pair, stored in pictures/ on this computer. Returns the path or None."""
        pair = self.pair()
        if pair is None:
            return None
        path = save_pair(pair)
        print(f"Picture fetched from {self.host} and saved here: {path.parent.name}/{path.name}")
        return path

    def record(self, timelapse=None):
        try:
            if timelapse is None:
                return self.call("/api/record/stop", {})["recording"]
            return self.call("/api/record", {"kind": "timelapse" if timelapse else "clip"})["recording"]
        except (OSError, ValueError) as exc:
            print(f"Recording on {self.host} not started / stopped ({exc})")

    def record_pictures(self, every=None):
        try:
            if every is None:
                return self.call("/api/pictures/stop", {})["pictures"]
            return self.call("/api/pictures/record", {"every_s": every})["pictures"]
        except (OSError, ValueError) as exc:
            print(f"Picture recording on {self.host} not started / stopped ({exc})")

    def optical_settings(self):
        print(f"The optical camera's settings window opens on the camera PC only ({self.host}: key s in its app, "
              "or qv4l2 / v4l2-ctl there)")

    def close(self):
        self.closing = True


def main():
    parser = argparse.ArgumentParser(description="The cameras as a background service with an HTTP API.")
    parser.add_argument("--host", default=HOST, help="address to listen on; 0.0.0.0 = the whole network "
                                                     "(default: this PC only)")
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()
    sys.stdout, sys.stderr = Tee(sys.stdout), Tee(sys.stderr)     # what is printed also goes to GET /api/log
    service = Service()
    server = serve(service, args.host, args.port)
    if server is None:
        return
    service.start()
    try:
        while service.cams.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    print("Stopping ...")
    server.shutdown()
    service.close()


if __name__ == "__main__":
    main()
