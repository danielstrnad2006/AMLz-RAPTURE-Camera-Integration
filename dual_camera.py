"""
dual_camera.py - Optris PI 640i (thermal) + MOKOSE UC50 (optical), both over USB, Windows or Linux (Ubuntu).

Install once:  Windows: python -m pip install opencv-python numpy matplotlib pygrabber
               Linux:   sudo apt install x11-xserver-utils qv4l2   (xrandr/xrdb for the window layout and
                        display scaling; qv4l2 = the optical camera's settings window)
                        The Optris SDK for Linux comes as .deb packages (libirimager, otcsdk) and puts
                        optris.otcsdk into /usr/lib/python3/dist-packages. It is built for NumPy 1.x; a
                        newer Ubuntu's system Python has NumPy 2 (import fails with "_ARRAY_API not
                        found"), so use a conda env (conda-forge: no Anaconda ToS needed) instead:
                        conda create -n rapture --override-channels -c conda-forge python=3.12 "numpy<2"
                                     "opencv<4.11" matplotlib tk
                        ln -s /usr/lib/python3/dist-packages/optris \
                              ~/miniconda3/envs/rapture/lib/python3.12/site-packages/optris
                        and run with  ~/miniconda3/envs/rapture/bin/python app.py
               plus the Optris thermal camera SDK (provides the Python module optris.otcsdk);
               Windows: restart the PC after installing it.
Run:           python app.py                  (the GUI: live view, recording, build plate, calibration)
               python dual_camera.py          (q or Esc to quit, s = optical camera settings,
                                               c = calibration picture: see calibrate_view)
               python dual_camera.py calibration/<time> [square mm]   (calibrate again from saved pictures)
The camera views in use (build plate mapping) are stored in views/thermal.json and views/optical.json.

From your own code (images are plain OpenCV / NumPy arrays):
    from dual_camera import CameraRig
    with CameraRig() as rig:
        pair = rig.grab()
        pair.thermal      # false-colour picture, BGR uint8 (for display)
        pair.thermal_c    # temperatures in deg C, float32, same height/width as pair.thermal;
                          # NaN while the camera calibrates after start-up, or if it is missing
        pair.optical      # BGR uint8
        pair.thermal_live, pair.optical_live    # False for placeholder pictures
        pair.timestamp                          # time.time() when the pair was taken
        pair.thermal_time, pair.optical_time    # time.time() when each camera's frame arrived (NaN: unknown)
        pair.rotation                           # deg the build plate had turned then (0 until the PLC tells)
        pair.temperature(x, y), pair.colour(x, y)   # at build plate point(s) x', y' in mm
    save_pair(pair) -> pictures/<time>.png,  load_pair(path) -> the FramePair again (no cameras needed)
"""
import json
import math
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

WINDOWS = sys.platform == "win32"   # else Linux: V4L2 instead of DirectShow for the optical camera
if WINDOWS:
    from pygrabber.dshow_graph import FilterGraph

try:
    import optris.otcsdk as otc
    _OTC_ERROR = None
except ImportError as exc:     # SDK not installed, Windows: the PC was not restarted after installing it,
    otc, _OTC_ERROR = None, exc    # Linux: the SDK was built for another Python / NumPy than this one

OPTICAL_NAME = "uc50"          # as Windows / Linux list it (case-insensitive; part of the name is enough)
OPTICAL_SIZE = (2048, 1536)    # requested resolution, None = camera default (actual size is printed)
                               # UC50: 2048x1536 gives 2.5 frames/s, 1600x1200 4, 1024x768 and 800x600 30
# Webcams have no ISO; gain is the equivalent. The camera only follows OPTICAL_GAIN with manual
# exposure, so set both. None = automatic. Exposure in s, e.g. 1/32 (longest for 30 frames/s) or 1/128
# (Windows rounds it to a power of 2). Press s in the live view to open the camera's own settings to try values.
OPTICAL_EXPOSURE = None
OPTICAL_GAIN = None            # UC50 accepted 0 to 30, rejected 100
ROTATE_180 = True              # turn both camera pictures upside down (cameras mounted upside down)
PANEL_HEIGHT = 480             # height of each picture in the preview window
FONT = cv2.FONT_HERSHEY_SIMPLEX



@dataclass(frozen=True)
class View:
    """How one camera sees the print plane (the plane the checkerboard lay in for calibrate_view).
    The camera looks straight down.

    Zoom: `size` = (width, height) mm of the print plane area the picture shows, as it would be without
    lens distortion. Lens distortion: OpenCV's `dist` = (k1, k2, p1, p2, k3) in plate mm from `center`
    divided by half the width (1 = the picture's left/right edge). Both come from calibrate_view().
    Mounting: the picture centre lies on the plate point `center` = (x', y') mm, the picture's rightward
    direction at `omega` deg from +x' (counter-clockwise seen from above); both measured by hand.
    """
    size: tuple
    center: tuple
    omega: float = 0.0
    dist: tuple = (0.0, 0.0, 0.0, 0.0, 0.0)

    def _camera_matrix(self, shape):
        h, w = shape[:2]
        return np.array([[w / 2, 0, (w - 1) / 2], [0, h / 2 * self.size[0] / self.size[1], (h - 1) / 2], [0, 0, 1]])

    def to_pixel(self, x, y, shape):
        """Where build plate point(s) (x', y') mm appear in a picture of shape `shape`.

        Returns (u, v) = (column, row) as float arrays; NaN for points the lens model cannot map.
        """
        (fx, _, cx), (_, fy, cy), _ = self._camera_matrix(shape)
        c, s = np.cos(np.radians(self.omega)), np.sin(np.radians(self.omega))
        dx, dy = np.asarray(x, float) - self.center[0], np.asarray(y, float) - self.center[1]
        half = self.size[0] / 2
        xn, yn = (dx * c + dy * s) / half, (dx * s - dy * c) / half             # without distortion
        k1, k2, p1, p2, k3 = self.dist
        r2 = xn**2 + yn**2
        radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
        xd = xn * radial + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn**2)
        yd = yn * radial + p1 * (r2 + 2 * yn**2) + 2 * p2 * xn * yn
        # Past the radius where the distortion polynomial stops growing it folds back and would put
        # far-away points inside the picture; those points are left out.
        r2_test = np.linspace(0, 4 * (1 + (self.size[1] / self.size[0]) ** 2), 1000)   # to 2x corner
        fold = 1 + 3 * k1 * r2_test + 5 * k2 * r2_test**2 + 7 * k3 * r2_test**3 <= 0
        ok = r2 < (r2_test[fold.argmax()] if fold.any() else r2_test[-1])
        return np.where(ok, fx * xd + cx, np.nan), np.where(ok, fy * yd + cy, np.nan)

    def to_plate(self, u, v, shape):
        """Build plate point(s) (x', y') mm seen at pixel (u, v) = (column, row) of a picture of shape `shape`."""
        u, v = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float))
        n = cv2.undistortPointsIter(np.stack([u.ravel(), v.ravel()], -1).reshape(-1, 1, 2),
                                    self._camera_matrix(shape), np.array(self.dist), None, None,
                                    (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-12))
        xn, yn = n[:, 0, 0].reshape(u.shape), n[:, 0, 1].reshape(u.shape)
        c, s = np.cos(np.radians(self.omega)), np.sin(np.radians(self.omega))
        half = self.size[0] / 2
        return (self.center[0] + half * (xn * c + yn * s),
                self.center[1] + half * (xn * s - yn * c))


def read_view(path):
    """(View, everything else stored with it, e.g. the calibration report) from a JSON file of save_view."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return View(tuple(d.pop("size")), tuple(d.pop("center")), d.pop("omega"), tuple(d.pop("dist"))), d


def load_view(camera, default):
    """The View in views/<camera>.json (camera = "thermal" or "optical"), or `default` if there is none."""
    path = VIEW_DIR / f"{camera}.json"
    return read_view(path)[0] if path.exists() else default


def save_view(camera, view, into=None, **info):
    """Write `view` as <into>/<camera>.json, with `info` (e.g. the calibration report) for the record.
    Without `into` it goes to views/, is the view in use from then on (THERMAL_VIEW / OPTICAL_VIEW)
    and is loaded again at the next start."""
    (into or VIEW_DIR).mkdir(parents=True, exist_ok=True)
    data = {**asdict(view), **info, "saved": f"{datetime.now():%Y-%m-%d %H:%M:%S}"}
    ((into or VIEW_DIR) / f"{camera}.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    if into is None:
        globals()[f"{camera.upper()}_VIEW"] = view


# Build plate frame x'-y': origin at the plate centre, mm, +x' right and +y' up seen from above; it turns
# with the plate (parts are fixed in it). The views map pixels to where x'-y' is at rotation 0 (the machine
# frame, fixed with the cameras): a plate point p is seen at turn(p, rotation).
PLATE_R_INNER, PLATE_R_OUTER = 15, 100     # mm


def turn(xy, deg):
    """Points xy (..., 2) mm turned by `deg` counter-clockwise (seen from above) about the plate centre."""
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    xy = np.asarray(xy, float)
    return np.stack([c * xy[..., 0] - s * xy[..., 1], s * xy[..., 0] + c * xy[..., 1]], -1)

BOARD_SQUARE = 8.00                      # mm, side of one square of the engraved calibration board
CALIBRATION_DIR = Path(__file__).resolve().parent / "calibration"   # checkerboard pictures, one folder per session
VIEW_DIR = Path(__file__).resolve().parent / "views"                # thermal.json, optical.json: the views in use
PICTURE_DIR = Path(__file__).resolve().parent / "pictures"          # pairs stored by save_pair (app.py: p)
# The views in use come from views/<camera>.json (written by app.py: Calibration). These defaults apply
# only if a file is missing: the views of 2026-11-25 (8 mm board, optical aligned to thermal, both then
# turned 90 deg counter-clockwise about the plate centre).
THERMAL_VIEW = load_view("thermal", View(size=(90.636, 67.441), center=(-57.0, -5.861), omega=190,
                                         dist=(-0.006422, 0.00061, -0.000135, -0.000114, 0.001731)))
OPTICAL_VIEW = load_view("optical", View(size=(84.245, 63.181), center=(-22.7442, -50.4373), omega=50.2837,
                                         dist=(-0.024221, -0.000281, 0.000245, -0.000783, 0.000462)))


def placeholder(title, size=(640, 480)):
    img = np.full((size[1], size[0], 3), 45, np.uint8)
    cv2.putText(img, title, (40, size[1] // 2), FONT, 0.9, (200, 200, 200), 2)
    cv2.putText(img, "not connected", (40, size[1] // 2 + 40), FONT, 0.8, (200, 200, 200), 2)
    return img


@dataclass
class FramePair:
    """Both cameras at one moment t_i."""
    timestamp: float
    thermal: np.ndarray      # false-colour picture, BGR uint8
    thermal_c: np.ndarray    # deg C per pixel, float32, same height/width as `thermal`
    optical: np.ndarray      # BGR uint8
    thermal_live: bool
    optical_live: bool
    rotation: float = 0.0    # deg the build plate had turned (counter-clockwise seen from above) when taken
    thermal_time: float = math.nan     # time.time() when each camera's frame arrived from the camera
    optical_time: float = math.nan     # (the frames of one pair are up to a frame interval apart)
    thermal_rotation: float | None = None   # the plate rotation when each camera's frame was taken, where
    optical_rotation: float | None = None   # known separately (recording.py, from an AMCF angle log); else rotation

    def turned(self, camera):
        """deg the plate had turned when `camera`'s frame was taken ("thermal" / "optical")."""
        own = self.thermal_rotation if camera == "thermal" else self.optical_rotation
        return self.rotation if own is None else own

    def seen_at(self, x, y, camera="thermal"):
        """Where `camera` saw build plate point(s) (x', y') mm: the points turned by its rotation."""
        there = turn(np.stack(np.broadcast_arrays(x, y), -1), self.turned(camera))
        return there[..., 0], there[..., 1]

    def temperature(self, x, y):
        """deg C at build plate point(s) (x', y') mm, interpolated from the original thermal picture.
        NaN where the thermal camera does not see the point (or while it calibrates)."""
        return _sample(self.thermal_c, *THERMAL_VIEW.to_pixel(*self.seen_at(x, y), self.thermal_c.shape))

    def colour(self, x, y):
        """Optical BGR (float32, last axis 3) at build plate point(s) (x', y') mm, interpolated from the
        original optical picture. NaN where the optical camera does not see the point."""
        bgr = _sample(self.optical, *OPTICAL_VIEW.to_pixel(*self.seen_at(x, y, "optical"), self.optical.shape))
        if not self.optical_live:
            bgr[...] = np.nan
        return bgr


def _sample(img, u, v):
    """Bilinear interpolation of a picture at pixel positions (u, v), as float32; NaN outside it."""
    h, w = img.shape[:2]
    inside = (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)          # also False for NaN
    u, v = np.where(inside, u, 0), np.where(inside, v, 0)
    u0, v0 = np.minimum(u.astype(int), w - 2), np.minimum(v.astype(int), h - 2)
    fu, fv = (u - u0).astype(np.float32), (v - v0).astype(np.float32)     # float32: half the time
    if img.ndim == 3:
        fu, fv, inside = fu[..., None], fv[..., None], inside[..., None]
    out = ((img[v0, u0] * (1 - fu) + img[v0, u0 + 1] * fu) * (1 - fv)
           + (img[v0 + 1, u0] * (1 - fu) + img[v0 + 1, u0 + 1] * fu) * fv)
    return np.where(inside, out, np.nan).astype(np.float32)


def temperature(pairs, x, y, t):
    """deg C at build plate point(s) (x', y') mm at time t (time.time() seconds), from the pair in
    `pairs` (e.g. a list of rig.grab() results) whose timestamp is nearest to t."""
    return min(pairs, key=lambda p: abs(p.timestamp - t)).temperature(x, y)


class Thermal(otc.IRImagerClient if otc else object):
    """Optris camera. The SDK pushes frames from its own thread; we keep only the latest one."""

    def __init__(self):
        super().__init__()
        self.connected = False
        self._lock = threading.Lock()
        self._latest = None                         # (false-colour picture, deg C, time.time() it arrived)
        self._imager = otc.IRImagerFactory.getInstance().create("native")
        self._imager.addClient(self)
        try:
            self._imager.connect(0)                 # 0 = first compatible camera
            self._builder = otc.ImageBuilder(colorFormat=otc.ColorFormat_BGR,
                                             widthAlignment=otc.WidthAlignment_OneByte)
            self._imager.runAsync()
        except Exception:
            self.close()
            raise
        self.connected = True
        print(f"Thermal camera: {otc.deviceTypeToString(self._imager.getDeviceType())} "
              f"(S/N {self._imager.getSerialNumber()})")

    def onFrame(self, evt):                         # called by the SDK (its thread) for every new frame
        # Converted at once, nothing of the SDK's is kept: the SDK reuses evt once this returns, and the
        # copies evt.clone() makes are never freed by the Python binding (not owned by Python: the memory
        # grew until the program crashed, seen on Linux 2026-11-26).
        arrived = time.time()
        frame = evt.thermalFrame
        if frame.isEmpty():
            return
        self._builder.setThermalFrame(thermalFrame=frame)
        self._builder.convertTemperatureToPaletteImage()
        h, w = self._builder.getHeight(), self._builder.getWidth()
        image = np.empty((h, w, 3), np.uint8)
        self._builder.copyImageDataTo(image)
        temps = np.empty(h * w, np.float32)
        frame.copyTemperaturesTo(temps)
        temps = temps.reshape(h, w)
        if "initializing" in otc.flagStateToString(evt.meta.getFlagState()).lower():
            temps[:] = np.nan                       # start-up calibration: temperatures unreliable
        with self._lock:
            self._latest = (image, temps, arrived)

    def onConnection(self, evt):                    # called by the SDK when the connection changes
        if evt.state == otc.ConnectionState_Lost or evt.state == otc.ConnectionState_Timeout:
            self.connected = False
            self._imager.disconnect()

    def read(self):
        """Latest frame as (false-colour BGR picture, temperatures in deg C, time.time() it arrived), or
        (None, None, NaN)."""
        with self._lock:
            latest = self._latest
        if not self.connected or latest is None:
            return None, None, math.nan
        image, temps, arrived = latest
        return image.copy(), temps.copy(), arrived  # the same frame may be read twice: each pair its own

    def close(self):
        self.connected = False
        if self._imager.isRunning():
            self._imager.stopRunning()
        self._imager.removeClient(self)


def video_devices():
    """(OpenCV index, name) of the video cameras. Windows: DirectShow's list, same order as OpenCV's
    indices. Linux: /dev/video<n> with its name from sysfs; a USB camera also has a metadata node, left
    out (its index within the camera is not 0)."""
    if WINDOWS:
        return list(enumerate(FilterGraph().get_input_devices()))
    found = []
    for node in Path("/sys/class/video4linux").glob("video*"):
        try:
            if int((node / "index").read_text()) == 0:
                found.append((int(node.name[5:]), (node / "name").read_text().strip()))
        except (OSError, ValueError):
            pass
    return sorted(found)


class CameraRig:
    """Works with either camera, both, or neither connected."""

    def __init__(self):
        self._placeholders = (placeholder("Thermal: Optris PI 640i"),
                              placeholder("Optical: MOKOSE UC50"))

        self.thermal = None
        if otc is None:
            print(f"Thermal camera: this Python ({sys.executable}, {sys.version.split()[0]}) cannot load the "
                  f"Optris SDK ({_OTC_ERROR}); showing placeholder")
        else:
            try:
                otc.Sdk.init(otc.Verbosity_Info, otc.Verbosity_Off, sys.argv[0])
                self.thermal = Thermal()
            except Exception as exc:                # no camera plugged in, or it is in use
                print(f"Thermal camera not available ({exc!r}; the SDK's own messages are in the terminal); "
                      "showing placeholder")

        self.optical = None
        devices = video_devices()
        index = next((i for i, n in devices if OPTICAL_NAME in n.lower()), None)
        if index is None:
            print(f"Optical camera '{OPTICAL_NAME}' not found. Cameras seen: {[n for _, n in devices]}")
            return
        self.optical_index = index
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if WINDOWS else cv2.CAP_V4L2)
        if OPTICAL_SIZE:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # needed for 3MP over USB 2.0
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, OPTICAL_SIZE[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, OPTICAL_SIZE[1])
        if OPTICAL_EXPOSURE is not None:
            if WINDOWS:     # DirectShow: 2**value s; setting it also switches auto exposure off
                cap.set(cv2.CAP_PROP_EXPOSURE, round(math.log2(OPTICAL_EXPOSURE)))
            else:           # V4L2: auto exposure off first (1 = manual, 3 = auto), then in units of 100 us
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                cap.set(cv2.CAP_PROP_EXPOSURE, round(OPTICAL_EXPOSURE * 1e4))
        if OPTICAL_GAIN is not None:
            cap.set(cv2.CAP_PROP_GAIN, OPTICAL_GAIN)
        good, end = 0, time.monotonic() + 3          # first frames can fail; skip a few so exposure settles
        while good < 5 and time.monotonic() < end:
            good += cap.read()[0]
        if good:
            self.optical = cap
            exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
            print(f"Optical camera: {'index ' if WINDOWS else '/dev/video'}{index}, "
                  f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}, "
                  f"exposure {(2 ** exposure if WINDOWS else exposure / 1e4) * 1000:.3g} ms, "
                  f"gain {cap.get(cv2.CAP_PROP_GAIN)}")
        else:
            cap.release()
            print("Optical camera found but delivers no frames (is another program using it, "
                  f"{'the Windows Camera app' if WINDOWS else 'e.g. cheese or guvcview'}?)")

    def grab(self) -> FramePair:
        """Latest image from each camera. A missing or failed camera gives a placeholder."""
        optical, optical_time = None, math.nan
        if self.optical is not None:
            ok, optical = self.optical.read()     # waits for the next frame
            optical_time = time.time()
            if not ok:        # unplugged while running
                optical = None
                self.optical.release()
                self.optical = None
        # Thermal after optical: the optical read waits for a new frame, so the thermal frame is the freshest.
        t_image, t_deg, thermal_time = self.thermal.read() if self.thermal is not None else (None, None, math.nan)
        thermal_live = t_image is not None
        if ROTATE_180:                       # live pictures only, placeholders stay readable
            optical = None if optical is None else cv2.rotate(optical, cv2.ROTATE_180)
            if thermal_live:
                t_image, t_deg = cv2.rotate(t_image, cv2.ROTATE_180), cv2.rotate(t_deg, cv2.ROTATE_180)
        if not thermal_live:
            t_image = self._placeholders[0].copy()
            t_deg = np.full(t_image.shape[:2], np.nan, np.float32)
        return FramePair(
            timestamp=time.time(),
            thermal=t_image,
            thermal_c=t_deg,
            optical=optical if optical is not None else self._placeholders[1].copy(),
            thermal_live=thermal_live,
            optical_live=optical is not None,
            thermal_time=thermal_time if thermal_live else math.nan,
            optical_time=optical_time if optical is not None else math.nan,
        )

    def optical_settings(self):
        """Open the optical camera's own settings window (exposure, gain, auto on/off, ...). Windows: the
        driver's dialog, returns when it is closed. Linux: qv4l2 (or guvcview's control panel) as a program
        of its own, returns at once: V4L2 lets another program change the settings while this one streams."""
        if self.optical is None:
            return
        if WINDOWS:
            self.optical.set(cv2.CAP_PROP_SETTINGS, 1)
            return
        device = f"/dev/video{self.optical_index}"
        for command in (["qv4l2", "-d", device], ["guvcview", "-d", device, "-z"]):
            if shutil.which(command[0]):
                subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
        print(f"No camera settings program: sudo apt install qv4l2 (or set them with v4l2-ctl -d {device})")

    def close(self):
        if self.optical is not None:
            self.optical.release()
            self.optical = None
        if self.thermal is not None:
            self.thermal.close()
            self.thermal = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def calibrate_view(view, pictures, square=BOARD_SQUARE, marked=None, report=None):
    """`view` with size and dist fitted to pictures of a flat checkerboard in the print plane.

    The board must cover the picture centre and at least 40% of the picture, better all of it (the
    distortion is only measured where there are corners); it may be larger than the picture, at any
    rotation, and moved between pictures. `square` = side of one square in mm. center and omega are
    kept: the board's place and rotation on the plate are not needed. Pictures may be BGR or 2D (e.g.
    pair.thermal_c; the thermal camera needs a board that shows up warm/cold). Prints how well the
    fitted view matches the corners. If `marked` is a list, one BGR picture per picture is appended
    showing what was found: yellow = corner candidates, green = checkerboard corners and grid,
    red = where the fitted view puts them, text = verdict. If `report` is a dict, the fit statistics
    are put in it (pictures used, corners, error rms/max, reach).
    """
    def corner_near(gray, strong, peaks, guess, step):   # checkerboard corner refined from `guess`, or None
        size = np.hypot(*step)                        # `step` = one square along a row or column, px
        win = max(2, int(0.3 * size))                 # window stays inside the squares
        near = peaks[np.hypot(*(peaks - guess).T).argmin()]
        if np.hypot(*(near - guess)) < 0.4 * size:    # a predicted corner can be off by a third of a square
            guess = near
        h, w = gray.shape
        if not (win <= guess[0] <= w - 1 - win and win <= guess[1] <= h - 1 - win):
            return None
        corner = cv2.cornerSubPix(gray, guess.astype(np.float32).reshape(1, 1, 2), (win, win), (-1, -1),
                                  (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 0.001))[0, 0]
        # In a flat area cornerSubPix stays put, so also check there is a corner (not flat, not an edge).
        if np.hypot(*(corner - guess)) >= 0.25 * size or not strong[round(corner[1]), round(corner[0])]:
            return None
        # And that it is a checkerboard corner (noise has corners too): of the four squares around it,
        # the diagonal ones look alike and the neighbouring ones clearly different.
        perp, k = np.array([-step[1], step[0]]), max(1, int(0.2 * size))
        q = [cv2.getRectSubPix(gray, (k, k), tuple(map(float, corner + 0.25 * (i * step + j * perp)))).mean()
             for i, j in ((1, 1), (-1, -1), (1, -1), (-1, 1))]
        if abs(q[0] + q[1] - q[2] - q[3]) > max(2 * (abs(q[0] - q[1]) + abs(q[2] - q[3])), 20):
            return corner.astype(float)
        return None

    def verdict(mark, number, text, ok=False):       # written onto the marked picture, printed if left out
        line = max(1, round(mark.shape[1] / 640))
        tw = cv2.getTextSize(text, FONT, 0.6 * line, line)[0][0]
        cv2.rectangle(mark, (0, 0), (tw + 16 * line, 34 * line), (0, 0, 0), -1)
        cv2.putText(mark, text, (8 * line, 24 * line), FONT, 0.6 * line, (0, 255, 0) if ok else (0, 0, 255), line)
        if not ok:
            print(f"Picture {number}: {text}")

    def without_lines(gray, width):                   # bright / dark lines up to `width` px wide wiped out
        k = 2 * round(width / 2) + 1                  # (the engraved outlines between the squares, scratches)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.morphologyEx(cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)

    def refine(img, corner, step):                    # where the board's edges through `corner` cross, or None
        # The four edges leaving a corner are measured from 0.2 to 0.8 squares away from it, leaving out the
        # crossing itself (the engraved outlines meet there in a blob or with a jog). Opposite edges are
        # fitted with one straight line, so a jog is averaged out; the corner is where the two lines cross.
        size = np.hypot(*step)
        t = np.r_[-np.linspace(0.8, 0.2, 7), np.linspace(0.2, 0.8, 7)] * size    # along the edges, px
        s = np.arange(-0.3 * size, 0.3 * size, 0.5)                                # across them
        lines = []
        for d in (step / size, np.array([-step[1], step[0]]) / size):
            n = np.array([-d[1], d[0]])
            xy = (corner + t[:, None, None] * d + s[None, :, None] * n).astype(np.float32)
            grad = np.abs(np.diff(cv2.remap(img, xy[..., 0], xy[..., 1], cv2.INTER_LINEAR), axis=1))
            i = grad.argmax(1).clip(1, grad.shape[1] - 2)
            a, b, c = (grad[np.arange(len(t)), i + k] for k in (-1, 0, 1))
            off = np.where(a - 2 * b + c < 0, 0.5 * (a - c) / np.minimum(a - 2 * b + c, -1e-9), 0)
            p = corner + t[:, None] * d + (s[i] + 0.25 + 0.5 * off)[:, None] * n     # the steepest point
            keep = inside = ((xy >= 0) & (xy <= np.array(img.shape[::-1]) - 1)).all((1, 2))   # profile in the picture
            for _ in range(3):                        # straight line, points off it (scratches, seams) dropped
                if (keep & (t < 0)).sum() < 3 or (keep & (t > 0)).sum() < 3:
                    return None
                mid = p[keep].mean(0)
                u = np.linalg.svd(p[keep] - mid)[2][0]
                off = np.abs((p - mid) @ np.array([-u[1], u[0]]))
                keep = inside & (off < max(0.02 * size, 3 * np.median(off[keep])))
            if (keep & (t < 0)).sum() < 3 or (keep & (t > 0)).sum() < 3:
                return None
            lines.append((mid, u))
        (m1, u1), (m2, u2) = lines
        k = np.linalg.solve(np.c_[u1, -u2], m2 - m1)[0]
        corner_new = m1 + k * u1
        return corner_new if np.hypot(*(corner_new - corner)) < 0.15 * size else None

    boards, used, warned = [], [], False           # per picture: corners in board mm, in pixels; its mark
    for number, pic in enumerate(pictures, 1):
        gray = pic if pic.ndim == 2 else cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
        if gray.dtype != np.uint8:                    # temperatures: 1%..99% -> 0..255, hot/cold spots clipped
            lo, hi = np.nanpercentile(gray, (1, 99)) if np.isfinite(gray).any() else (0, 1)
            gray = np.clip((np.nan_to_num(gray, nan=lo) - lo) * 255 / max(hi - lo, 1e-6), 0, 255).astype(np.uint8)
        h, w = gray.shape
        centre = np.array([(w - 1) / 2, (h - 1) / 2])
        mark = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if pic.ndim == 2 else pic.copy()
        line = max(1, round(w / 640))                 # line width in the marked picture
        if marked is not None:
            marked.append(mark)
        # Corners are searched in the picture scaled to <= 800 px wide: in a sharp high-resolution
        # picture paper texture and noise give many small corner features close together.
        zoom = min(1.0, 800 / w)
        small = cv2.resize(gray, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_AREA) if zoom < 1 else gray
        # Square size from the strongest repetition in the picture centre (noise has none): a
        # checkerboard's strongest Fourier component runs diagonally, with period sqrt(2) x square.
        # The outlines are wiped out first: their grid repeats every square, which could be taken for it.
        small = without_lines(small, 4)
        n = min(small.shape)
        y0, x0 = (small.shape[0] - n) // 2, (small.shape[1] - n) // 2
        crop = small[y0:y0 + n, x0:x0 + n].astype(np.float32)
        crop = (crop - cv2.GaussianBlur(crop, (0, 0), n / 8)) * np.outer(np.hanning(n), np.hanning(n))
        power = np.abs(np.fft.fft2(crop)) ** 2
        freq = np.hypot(*np.meshgrid(np.fft.fftfreq(n), np.fft.fftfreq(n)))   # cycles per px
        power[(freq < 4 / n) | (freq > 1 / 8)] = 0                            # periods 8 px .. n/4
        size = 1 / freq.flat[power.argmax()] / np.sqrt(2)
        expected = square * w / view.size[0]          # px, with the view in use
        if not warned and not 0.8 < size / zoom / expected < 1.25:
            warned = True
            print(f"Picture {number}: squares of ~{size / zoom:.0f} px, but with {square:g} mm squares the view in use "
                  f"expects ~{expected:.0f} px. Is the square size right?")
        small = without_lines(small, 0.08 * size)
        # Smoothed by a tenth of a square (removes pixel noise, keeps the corners) and the contrast evened
        # out over about a square (board warmer in one place than another, uneven light); the +2 grey
        # levels keep areas without a board from being amplified to full contrast.
        flat = cv2.GaussianBlur(small.astype(np.float32), (0, 0), 0.1 * size)
        flat -= cv2.GaussianBlur(flat, (0, 0), size)
        flat /= np.sqrt(cv2.GaussianBlur(flat * flat, (0, 0), size)) + 2
        small = np.clip(128 + 60 * flat, 0, 255).astype(np.uint8)
        # OpenCV's checkerboard detectors expect a border around the board and mis-number the corners
        # of a board larger than the picture. So: a 2x2 block of corners at the picture centre (from
        # the strongest corner features, one square apart), grown to the picture edges by
        # predicting the next corner along a row or column from the last two and refining it.
        # Candidates: saddle points of the smoothed picture. A checkerboard corner is one; the middle of a
        # (blurred, e.g. thermal) square and a straight edge are not, unlike for usual corner detectors.
        sm = cv2.GaussianBlur(small.astype(np.float32), (0, 0), 0.2 * size)
        saddle = cv2.Sobel(sm, -1, 1, 1) ** 2 - cv2.Sobel(sm, -1, 2, 0) * cv2.Sobel(sm, -1, 0, 2)
        sh, sw = small.shape
        strong = saddle > 0.05 * saddle[sh // 4:-sh // 4, sw // 4:-sw // 4].max()   # board covers the centre
        k = 2 * int(0.3 * size) + 1
        peaks = np.argwhere(strong & (saddle == cv2.dilate(saddle, np.ones((k, k), np.uint8))))[:, ::-1].astype(float)
        if len(peaks) < 9:
            verdict(mark, number, f"left out: hardly any corner features (square ~{size / zoom:.0f} px)")
            continue
        pts = peaks[np.argsort(np.hypot(*(peaks - centre * zoom).T))[:100]]
        for p in pts:
            cv2.circle(mark, tuple(np.round(p / zoom).astype(int)), 3 * line, (0, 255, 255), line)
        gaps = np.hypot(*(pts[:, None] - pts[None]).transpose(2, 0, 1))
        np.fill_diagonal(gaps, np.inf)
        found = None
        for i in range(min(20, len(pts))):
            a = pts[gaps[i].argmin()] - pts[i]            # one square along a row
            if not 0.75 * size < np.hypot(*a) < 1.33 * size:
                continue
            b = np.array([-a[1], a[0]])                   # one square along a column
            seed = {(r, c): corner_near(small, strong, peaks, pts[i] + c * a + r * b, a) for r in (0, 1) for c in (0, 1)}
            if all(p is not None for p in seed.values()):
                found = seed
                break
        if found is None:
            verdict(mark, number, f"left out: no 2x2 checkerboard corners near the centre "
                                  f"(square ~{size / zoom:.0f} px)")
            continue
        # Grown from there: every corner up to two squares away from the ones found is predicted from
        # the nearest found ones (a local affine fit, so the lens distortion hardly matters) and refined.
        # Two squares, so a corner that cannot be found (the hole, glare, an engraving seam) does not
        # stop the growth. A corner is tried again once more of its neighbours have been found.
        tried = {}                                    # cell -> neighbours found when it was last tried
        while True:
            cells, pts = np.array(list(found)), np.array(list(found.values()))
            todo = []
            for cell in {(r + dr, c + dc) for r, c in found for dr in range(-2, 3) for dc in range(-2, 3)} - set(found):
                nb = int((np.abs(cells - cell).max(1) == 1).sum())
                if tried.get(cell, -1) < nb:
                    todo.append((-nb, cell))
            added = False
            for nb, cell in sorted(todo):
                near = np.argsort(np.abs(cells - cell).max(1), kind="stable")[:8]
                a = np.c_[cells[near], np.ones(len(near))]
                if np.linalg.matrix_rank(a) < 3:
                    continue
                tried[cell] = -nb
                m = np.linalg.lstsq(a, pts[near], rcond=None)[0]   # rows: one row down, one column right, offset
                corner = corner_near(small, strong, peaks, np.array([*cell, 1.0]) @ m, m[1])
                if corner is not None:
                    found[cell], added = corner, True
            if not added:
                break
        # Refined in the full picture from the edges (one square along a row from an affine fit of all).
        cells, pts = np.array(list(found)), np.array(list(found.values()))
        step = np.linalg.lstsq(np.c_[cells, np.ones(len(cells))], pts, rcond=None)[0][1] / zoom
        img = cv2.GaussianBlur(without_lines(gray, 0.08 * size / zoom).astype(np.float32), (0, 0), 0.02 * size / zoom)
        full = {rc: refine(img, (p + 0.5) / zoom - 0.5, step) for rc, p in found.items()}
        found = {rc: p for rc, p in full.items() if p is not None}
        # Each corner must sit where its nearest neighbours put it (local affine fit, within 5% of a square):
        # noise or a scratch taken for a corner does not. The worst one is dropped until all agree.
        while len(found) > 4:
            cells, pts = np.array(list(found)), np.array(list(found.values()))
            err = np.full(len(cells), np.inf)
            for i, cell in enumerate(cells):
                gap = np.abs(cells - cell).max(1)
                gap[i] = 1 << 30
                near = np.argsort(gap, kind="stable")[:8]
                a = np.c_[cells[near], np.ones(len(near))]
                if np.linalg.matrix_rank(a) == 3:
                    err[i] = np.hypot(*(np.array([*cell, 1.0]) @ np.linalg.lstsq(a, pts[near], rcond=None)[0] - pts[i]))
            if err.max() < 0.05 * size / zoom:
                break
            del found[tuple(cells[err.argmax()])]
        if len(found) < 4:
            verdict(mark, number, f"left out: corners not confirmed by the edges (square ~{size / zoom:.0f} px)")
            continue
        for (r, c), p in found.items():
            for n in ((r, c + 1), (r + 1, c)):
                if n in found:
                    cv2.line(mark, tuple(np.round(p).astype(int)), tuple(np.round(found[n]).astype(int)),
                             (0, 255, 0), line)
            cv2.circle(mark, tuple(np.round(p).astype(int)), 4 * line, (0, 255, 0), -1)
        pix = np.array(list(found.values()))
        spread = cv2.contourArea(cv2.convexHull(pix.astype(np.float32))) / (w * h)
        if spread < 0.4:                              # the board fills the picture; this is something else
            verdict(mark, number, f"left out: {len(pix)} corners cover only {spread:.0%} of it "
                                  f"(square ~{size / zoom:.0f} px)")
            continue
        verdict(mark, number, f"{len(pix)} corners, cover {spread:.0%}, square ~{size / zoom:.0f} px", ok=True)
        # Board frame: x along the rows; the columns run 90 deg clockwise of that in the picture,
        # which is -y seen from above.
        rc = np.array(list(found), float)
        boards.append((np.stack([rc[:, 1], -rc[:, 0]], 1) * square, pix))
        used.append(mark)
    print(f"Checkerboard found in {len(boards)} of {len(pictures)} pictures")
    if not boards:
        raise RuntimeError("No checkerboard found, nothing calibrated")

    # Fit one lens to all pictures, each with its own board position and rotation (picture centre in
    # board mm, omega from the board's x), by Levenberg-Marquardt on the corner positions in pixels.
    start = [0.0] * 7                                 # size in mm (width, height), k1, k2, p1, p2, k3
    for xy, pix in boards:                            # first guess from a homography (no distortion)
        hom = cv2.findHomography(xy, pix)[0]
        c = cv2.perspectiveTransform(centre.reshape(1, 1, 2), np.linalg.inv(hom))[0, 0]
        d = np.diff(cv2.perspectiveTransform(np.array([[c, c + (1, 0)]]), hom)[0], axis=0)[0]   # 1 mm along x
        start[0], start[1] = w / np.hypot(*d), h / np.hypot(*d)
        start += [c[0], c[1], np.degrees(np.arctan2(d[1], d[0]))]

    def errors(p):                                    # fitted minus found corner positions, px, (n, 2)
        return np.concatenate([
            np.subtract(View(tuple(p[:2]), (cx, cy), om, tuple(p[2:7])).to_pixel(*xy.T, (h, w)), pix.T).T
            for (xy, pix), (cx, cy, om) in zip(boards, p[7:].reshape(-1, 3))])

    def fit(p):
        e, damping = errors(p).ravel(), 1e-3
        for _ in range(200):
            jac = np.empty((e.size, p.size))
            for k in range(p.size):
                dp = np.zeros_like(p)
                dp[k] = 1e-6 * max(1.0, abs(p[k]))
                jac[:, k] = (errors(p + dp).ravel() - e) / dp[k]
            a, g = jac.T @ jac, jac.T @ e
            while damping < 1e8:
                p_new = p - np.linalg.solve(a + damping * np.diag(np.diag(a)), g)
                e_new = errors(p_new).ravel()
                if e_new @ e_new < e @ e:             # False for NaN (lens model folded back)
                    break
                damping *= 10
            else:
                break
            done = e @ e - e_new @ e_new < 1e-10 * (e @ e)
            p, e, damping = p_new, e_new, damping / 10
            if done:
                break
        return p, np.hypot(*e.reshape(-1, 2).T)

    p, err = fit(np.array(start))
    keep = err < max(1.0, 3 * np.median(err))         # drop mis-found corners (glare, seams, dirt) and refit
    if not keep.all():
        keep = np.split(keep, np.cumsum([len(xy) for xy, _ in boards])[:-1])
        boards = [(xy[k], pix[k]) for (xy, pix), k in zip(boards, keep)]
        p, err = fit(p)
    fitted = errors(p) + np.vstack([pix for _, pix in boards])
    line = max(1, round(w / 640))
    for (_, pix), mark, at in zip(boards, used, np.split(fitted, np.cumsum([len(pix) for _, pix in boards])[:-1])):
        for u, v in at[np.isfinite(at).all(1)]:
            cv2.drawMarker(mark, (round(u), round(v)), (0, 0, 255), cv2.MARKER_CROSS, 10 * line, line)
    off = np.vstack([pix for _, pix in boards]) - centre
    quadrant = (off[:, 0] > 0) * 2 + (off[:, 1] > 0)
    reach = min(np.hypot(*off[quadrant == q].T).max(initial=0) for q in range(4)) / np.hypot(*centre)
    rms = np.sqrt(np.mean(err**2))
    print(f"{len(err)} corners, error rms {rms:.2f} px = {rms * p[0] / w:.3f} mm, max {err.max():.2f} px; "
          f"corners reach {reach:.0%} of the way to the picture corners")
    if report is not None:
        report.update(pictures_used=len(boards), pictures=len(pictures), corners=len(err),
                      rms_px=round(float(rms), 3), rms_mm=round(float(rms * p[0] / w), 4),
                      max_px=round(float(err.max()), 3), reach=round(float(reach), 3))
    return replace(view, size=tuple(round(float(s), 3) for s in p[:2]), dist=tuple(round(float(d), 6) for d in p[2:7]))


def side_by_side(pair):
    """Thermal and optical picture next to each other, PANEL_HEIGHT tall, each with a caption."""
    t = pair.thermal_c
    if not pair.thermal_live:
        thermal_text = "Thermal (no signal)"
    elif np.isnan(t).all():
        thermal_text = "Thermal (calibrating)"
    else:
        thermal_text = f"Thermal {np.nanmin(t):.1f} to {np.nanmax(t):.1f} C"
    h, w = pair.optical.shape[:2]
    optical_text = f"Optical {w}x{h}" if pair.optical_live else "Optical (no signal)"

    panels = []
    for text, img in ((thermal_text, pair.thermal), (optical_text, pair.optical)):
        h, w = img.shape[:2]
        panel = cv2.resize(img, (round(w * PANEL_HEIGHT / h), PANEL_HEIGHT))
        tw = cv2.getTextSize(text, FONT, 0.65, 1)[0][0]
        cv2.rectangle(panel, (0, 0), (tw + 16, 34), (0, 0, 0), -1)    # dark strip: readable on any image
        cv2.putText(panel, text, (8, 24), FONT, 0.65, (255, 255, 255), 1)
        panels.append(panel)
    return np.hstack(panels)


def save_board_picture(folder, number, pair, cameras=("thermal", "optical")):
    """Store `pair` as calibration picture `number` in `folder`: thermal_<n>.npy (= thermal_c) and
    thermal_<n>_colour.png, optical_<n>.png; only cameras in `cameras` that deliver. Returns those stored."""
    folder.mkdir(parents=True, exist_ok=True)
    stored = []
    if "thermal" in cameras and pair.thermal_live and not np.isnan(pair.thermal_c).all():   # not calibrating
        np.save(folder / f"thermal_{number}.npy", pair.thermal_c)
        cv2.imwrite(str(folder / f"thermal_{number}_colour.png"), pair.thermal)
        stored.append("thermal")
    if "optical" in cameras and pair.optical_live:
        cv2.imwrite(str(folder / f"optical_{number}.png"), pair.optical)
        stored.append("optical")
    return stored


def save_pair(pair, folder=None):
    """Store `pair` in pictures/ (or `folder`) as <time>.png (side by side, to look at) and the original
    pictures for processing: <time>_thermal.npy (= thermal_c, deg C), <time>_thermal.png (false colours),
    <time>_optical.png; only the cameras that deliver; <time>.json = the plate rotation and the times
    (timestamp, when each camera's frame arrived; unix s). Returns the path of <time>.png (see load_pair)."""
    folder = folder or PICTURE_DIR
    folder.mkdir(parents=True, exist_ok=True)
    stem = base = f"{datetime.fromtimestamp(pair.timestamp):%Y-%m-%d_%H-%M-%S}"
    while (folder / f"{stem}.png").exists():          # more than one in a second
        stem = f"{base}_{int(stem[len(base) + 1:] or 1) + 1}"
    cv2.imwrite(str(folder / f"{stem}.png"), side_by_side(pair))
    times = {k: getattr(pair, k) for k in ("timestamp", "thermal_time", "optical_time")}
    (folder / f"{stem}.json").write_text(json.dumps({"rotation_deg": pair.rotation, **{
        k: None if math.isnan(t) else round(t, 6) for k, t in times.items()}}), encoding="utf-8")
    if pair.thermal_live:
        np.save(folder / f"{stem}_thermal.npy", pair.thermal_c)
        cv2.imwrite(str(folder / f"{stem}_thermal.png"), pair.thermal)
    if pair.optical_live:
        cv2.imwrite(str(folder / f"{stem}_optical.png"), pair.optical)
    return folder / f"{stem}.png"


def load_pair(path):
    """The FramePair stored by save_pair; `path` = its <time>.png or any file stored with it. A camera
    not stored gives a placeholder, as in CameraRig.grab(). Rotation and times from <time>.json (older
    pairs: rotation 0 if there is none, timestamp from the file name, to the second)."""
    path = Path(path)
    stem = path.stem.removesuffix("_thermal").removesuffix("_optical").removesuffix("_points")
    files = {end: path.with_name(f"{stem}{end}") for end in ("_thermal.npy", "_thermal.png", "_optical.png", ".json")}
    thermal_live = files["_thermal.npy"].exists() and files["_thermal.png"].exists()
    optical_live = files["_optical.png"].exists()
    thermal = cv2.imread(str(files["_thermal.png"])) if thermal_live else placeholder("Thermal: Optris PI 640i")
    try:
        timestamp = datetime.strptime(stem[:19], "%Y-%m-%d_%H-%M-%S").timestamp()
    except ValueError:                                # renamed
        timestamp = path.stat().st_mtime
    info = json.loads(files[".json"].read_text(encoding="utf-8")) if files[".json"].exists() else {}
    times = {k: math.nan if info.get(k) is None else info[k] for k in ("thermal_time", "optical_time")}
    return FramePair(info.get("timestamp") or timestamp, thermal,
                     np.load(files["_thermal.npy"]) if thermal_live else np.full(thermal.shape[:2], np.nan, np.float32),
                     cv2.imread(str(files["_optical.png"])) if optical_live else placeholder("Optical: MOKOSE UC50"),
                     thermal_live, optical_live, info.get("rotation_deg", 0.0), **times)


def align_view(view, pixels, shape, plate, zoom=False):
    """`view` moved so that the pixels `pixels` [(u, v), ...] of its picture (of shape `shape`) land on
    the build plate points `plate` [(x', y') mm, ...], e.g. where the other camera sees the same features:
    new center and omega (with one point: center only), with `zoom` also size (both sides by the same
    factor; 2 points or more). Lens distortion is kept. Least squares (closed form).
    Returns (view, how far each point stays off, mm)."""
    pixels, plate = np.asarray(pixels, float), np.asarray(plate, float)
    own = np.stack(replace(view, center=(0, 0), omega=0).to_plate(*pixels.T, shape), 1)   # mm, picture's axes
    a, b = own - own.mean(0), plate - plate.mean(0)
    many = len(plate) > 1
    angle = np.arctan2(np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]), np.sum(a * b)) if many else np.radians(view.omega)
    turn = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    scale = np.sum(b * (a @ turn.T)) / np.sum(a * a) if zoom and many else 1.0
    center = plate.mean(0) - scale * turn @ own.mean(0)
    fitted = replace(view, center=tuple(round(float(c), 4) for c in center),
                     omega=round(float(np.degrees(angle)) % 360, 4),
                     size=view.size if scale == 1 else tuple(round(float(s * scale), 4) for s in view.size))
    return fitted, np.hypot(*(np.stack(fitted.to_plate(*pixels.T, shape), 1) - plate).T)


def calibrate_folder(folder, cameras=("thermal", "optical"), square=BOARD_SQUARE):
    """Calibrate the views of `cameras` from the pictures in `folder` (thermal_<n>.npy, optical_<n>.png)
    of a checkerboard with `square` mm squares, keeping center and omega of the views in use. Writes
    <camera>_<n>_corners.png next to the pictures (what was found) and, where it worked, <camera>.json
    (fitted view + report). Returns {camera: (view, report)} for the cameras where it worked; nothing
    is put in use."""
    results = {}
    for camera in cameras:
        load = "npy" if camera == "thermal" else "png"
        files = sorted((f for f in folder.glob(f"{camera}_*.{load}") if f.stem[len(camera) + 1:].isdigit()),
                       key=lambda f: int(f.stem[len(camera) + 1:]))     # not the *_corners.png ones
        if not files:
            continue
        pictures = [np.load(f) if load == "npy" else cv2.imread(str(f)) for f in files]
        print(f"\n{camera.upper()}_VIEW calibration ({', '.join(f.name for f in files)}; {square:g} mm squares):")
        marked, report = [], {"folder": folder.name, "square_mm": square}
        try:
            view = calibrate_view(THERMAL_VIEW if camera == "thermal" else OPTICAL_VIEW, pictures, square,
                                  marked=marked, report=report)
            save_view(camera, view, into=folder, **report)
            print(f"{camera.upper()}_VIEW = {view}   (saved as {folder.name}/{camera}.json)")
            results[camera] = view, report
        except RuntimeError as exc:
            print(exc)
        for f, mark in zip(files, marked):
            cv2.imwrite(str(f.with_name(f"{f.stem}_corners.png")), mark)
        print(f"Corners found: {folder / camera}_<n>_corners.png")
    return results


def main():
    if len(sys.argv) > 1:                             # python dual_camera.py calibration/<time> [square mm]
        calibrate_folder(Path(sys.argv[1]), square=float(sys.argv[2]) if len(sys.argv) > 2 else BOARD_SQUARE)
        print("\nTo use a result: app.py > Calibration > Save, or copy <folder>/<camera>.json to views/")
        return
    window = "Thermal | Optical"
    folder = CALIBRATION_DIR / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    taken = 0                                         # calibration pairs, taken with c
    with CameraRig() as rig:
        while True:
            pair = rig.grab()
            cv2.imshow(window, side_by_side(pair))
            key = cv2.waitKey(1 if pair.optical_live else 30) & 0xFF
            if key == ord("s"):
                rig.optical_settings()
            if key == ord("c"):
                taken += 1
                stored = save_board_picture(folder, taken, pair)
                print(f"Calibration picture {taken} ({' + '.join(stored) or 'no camera delivers'}) saved in {folder}")
            if key in (27, ord("q")) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    cv2.destroyAllWindows()
    if taken:
        calibrate_folder(folder)
        print("\nTo use a result: app.py > Calibration > Save, or copy <folder>/<camera>.json to views/")


if __name__ == "__main__":
    main()
