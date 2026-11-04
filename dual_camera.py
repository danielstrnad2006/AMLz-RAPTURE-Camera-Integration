"""
dual_camera.py - Optris PI 640i (thermal) + MOKOSE UC50 (optical), both over USB, Windows only.

Install once:  python -m pip install opencv-python numpy pygrabber
               plus the Optris thermal camera SDK (provides the Python module optris.otcsdk);
               restart the PC after installing it.
Run:           python dual_camera.py          (q or Esc to quit, s = optical camera settings,
                                               c = calibration picture: see calibrate_view)
               python dual_camera.py calibration/<time>   (calibrate again from saved pictures)

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
        pair.temperature(x, y), pair.colour(x, y)   # at build plate point(s) x', y' in mm
"""
import sys
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from pygrabber.dshow_graph import FilterGraph

try:
    import optris.otcsdk as otc
    _OTC_ERROR = None
except ImportError as exc:     # SDK not installed, or the PC was not restarted after installing it
    otc, _OTC_ERROR = None, exc

OPTICAL_NAME = "uc50"          # as Windows lists it (case-insensitive; part of the name is enough)
OPTICAL_SIZE = (2048, 1536)    # requested resolution, None = camera default (actual size is printed)
                               # UC50: 2048x1536 gives 2.5 frames/s, 1600x1200 4, 1024x768 and 800x600 30
# Webcams have no ISO; gain is the equivalent. The camera only follows OPTICAL_GAIN with manual
# exposure, so set both. None = automatic. Exposure is 2**value s: -5 = 1/32 s (longest for 30 frames/s),
# -7 = 1/128 s. Press s in the live view to open the camera's own settings window to try values.
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


# Build plate frame x'-y': origin at the plate centre, mm, +x' right and +y' up seen from above.
PLATE_R_INNER, PLATE_R_OUTER = 15, 100     # mm
# Placeholders until measured on the rig: center and omega by hand, size and dist from
# calibrate_view() (key c in the live view).
OPTICAL_VIEW = View(size=(88.209, 66.02), center=(-56.46213605, 27.8451025), omega=320, dist=(-0.030203, 0.014376, 0.000732, 9.4e-05, -0.006925))
THERMAL_VIEW = View(size=(115.5, 86.5), center=(-5.86124587, 57.00031636), omega=280)
BOARD_SQUARE = 5.00                      # mm, side of one calibration checkerboard square
CALIBRATION_DIR = Path(__file__).resolve().parent / "calibration"   # pictures taken with c


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

    def temperature(self, x, y):
        """deg C at build plate point(s) (x', y') mm, interpolated from the original thermal picture.
        NaN where the thermal camera does not see the point (or while it calibrates)."""
        return _sample(self.thermal_c, *THERMAL_VIEW.to_pixel(x, y, self.thermal_c.shape))

    def colour(self, x, y):
        """Optical BGR (float32, last axis 3) at build plate point(s) (x', y') mm, interpolated from the
        original optical picture. NaN where the optical camera does not see the point."""
        bgr = _sample(self.optical, *OPTICAL_VIEW.to_pixel(x, y, self.optical.shape))
        if not self.optical_live:
            bgr[...] = np.nan
        return bgr


def _sample(img, u, v):
    """Bilinear interpolation of a picture at pixel positions (u, v), as float32; NaN outside it."""
    h, w = img.shape[:2]
    inside = (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)          # also False for NaN
    u, v = np.where(inside, u, 0), np.where(inside, v, 0)
    u0, v0 = np.minimum(u.astype(int), w - 2), np.minimum(v.astype(int), h - 2)
    fu, fv = u - u0, v - v0
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
        self._event = None
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

    def onFrame(self, evt):                         # called by the SDK for every new frame
        with self._lock:
            self._event = evt.clone()               # the SDK reuses evt once this returns

    def onConnection(self, evt):                    # called by the SDK when the connection changes
        if evt.state == otc.ConnectionState_Lost or evt.state == otc.ConnectionState_Timeout:
            self.connected = False
            self._imager.disconnect()

    def read(self):
        """Latest frame as (false-colour BGR picture, temperatures in deg C), or (None, None)."""
        with self._lock:
            event = self._event
        if not self.connected or event is None or event.thermalFrame.isEmpty():
            return None, None
        self._builder.setThermalFrame(thermalFrame=event.thermalFrame)
        self._builder.convertTemperatureToPaletteImage()
        h, w = self._builder.getHeight(), self._builder.getWidth()
        image = np.empty((h, w, 3), np.uint8)
        self._builder.copyImageDataTo(image)
        temps = np.empty(h * w, np.float32)
        event.thermalFrame.copyTemperaturesTo(temps)
        temps = temps.reshape(h, w)
        if "initializing" in otc.flagStateToString(event.meta.getFlagState()).lower():
            temps[:] = np.nan                       # start-up calibration: temperatures unreliable
        return image, temps

    def close(self):
        self.connected = False
        if self._imager.isRunning():
            self._imager.stopRunning()
        self._imager.removeClient(self)


class CameraRig:
    """Works with either camera, both, or neither connected."""

    def __init__(self):
        self._placeholders = (placeholder("Thermal: Optris PI 640i"),
                              placeholder("Optical: MOKOSE UC50"))

        self.thermal = None
        if otc is None:
            print(f"Thermal camera: Optris SDK not available ({_OTC_ERROR}); showing placeholder")
        else:
            try:
                otc.Sdk.init(otc.Verbosity_Info, otc.Verbosity_Off, sys.argv[0])
                self.thermal = Thermal()
            except Exception as exc:                # no camera plugged in, or it is in use
                print(f"Thermal camera not available ({exc!r}); showing placeholder")

        self.optical = None
        names = FilterGraph().get_input_devices()    # same order as OpenCV's DirectShow indices
        index = next((i for i, n in enumerate(names) if OPTICAL_NAME in n.lower()), None)
        if index is None:
            print(f"Optical camera '{OPTICAL_NAME}' not found. Cameras seen: {names}")
            return
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if OPTICAL_SIZE:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # needed for 3MP over USB 2.0
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, OPTICAL_SIZE[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, OPTICAL_SIZE[1])
        if OPTICAL_EXPOSURE is not None:
            cap.set(cv2.CAP_PROP_EXPOSURE, OPTICAL_EXPOSURE)   # also switches auto exposure off
        if OPTICAL_GAIN is not None:
            cap.set(cv2.CAP_PROP_GAIN, OPTICAL_GAIN)
        good, end = 0, time.monotonic() + 3          # first frames can fail; skip a few so exposure settles
        while good < 5 and time.monotonic() < end:
            good += cap.read()[0]
        if good:
            self.optical = cap
            print(f"Optical camera: index {index}, "
                  f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}, "
                  f"exposure {cap.get(cv2.CAP_PROP_EXPOSURE)}, gain {cap.get(cv2.CAP_PROP_GAIN)}")
        else:
            cap.release()
            print("Optical camera found but delivers no frames (is the Windows Camera app still open?)")

    def grab(self) -> FramePair:
        """Latest image from each camera. A missing or failed camera gives a placeholder."""
        optical = None
        if self.optical is not None:
            ok, optical = self.optical.read()
            if not ok:        # unplugged while running
                optical = None
                self.optical.release()
                self.optical = None
        # Thermal after optical: the optical read waits for a new frame, so the thermal frame is the freshest.
        t_image, t_deg = self.thermal.read() if self.thermal is not None else (None, None)
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
        )

    def optical_settings(self):
        """Open the optical camera's own settings window (exposure, gain, auto on/off, ...)."""
        if self.optical is not None:
            self.optical.set(cv2.CAP_PROP_SETTINGS, 1)

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


def calibrate_view(view, pictures, square=BOARD_SQUARE, marked=None):
    """`view` with size and dist fitted to pictures of a flat checkerboard in the print plane.

    The board must cover the picture centre and at least 40% of the picture, better all of it (the
    distortion is only measured where there are corners); it may be larger than the picture, at any
    rotation, and moved between pictures. `square` = side of one square in mm. center and omega are
    kept: the board's place and rotation on the plate are not needed. Pictures may be BGR or 2D (e.g.
    pair.thermal_c; the thermal camera needs a board that shows up warm/cold). Prints how well the
    fitted view matches the corners. If `marked` is a list, one BGR picture per picture is appended
    showing what was found: yellow = corner candidates, green = checkerboard corners and grid,
    red = where the fitted view puts them, text = verdict.
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

    boards, used = [], []                          # per picture: corners in board mm, in pixels; its mark
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
        n = min(small.shape)
        y0, x0 = (small.shape[0] - n) // 2, (small.shape[1] - n) // 2
        crop = small[y0:y0 + n, x0:x0 + n].astype(np.float32)
        crop = (crop - cv2.GaussianBlur(crop, (0, 0), n / 8)) * np.outer(np.hanning(n), np.hanning(n))
        power = np.abs(np.fft.fft2(crop)) ** 2
        freq = np.hypot(*np.meshgrid(np.fft.fftfreq(n), np.fft.fftfreq(n)))   # cycles per px
        power[(freq < 4 / n) | (freq > 1 / 8)] = 0                            # periods 8 px .. n/4
        size = 1 / freq.flat[power.argmax()] / np.sqrt(2)
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
        todo = list(found)
        while todo:
            r, c = todo.pop()
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                new, back = (r + dr, c + dc), (r - dr, c - dc)
                if new in found or back not in found:
                    continue
                step = found[r, c] - found[back]
                corner = corner_near(small, strong, peaks, found[r, c] + step, step)
                if corner is not None:
                    found[new] = corner
                    todo.append(new)
        if zoom < 1:                                  # refine in the full picture
            win = max(2, int(0.25 * size / zoom))
            full = {rc: (p + 0.5) / zoom - 0.5 for rc, p in found.items()}
            pix = cv2.cornerSubPix(gray, np.array(list(full.values()), np.float32).reshape(-1, 1, 2), (win, win),
                                   (-1, -1), (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 0.001))
            found = {rc: p.astype(float) for (rc, guess), p in zip(full.items(), pix[:, 0])
                     if np.hypot(*(p - guess)) < 2 / zoom}
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
    keep = err < max(1.0, 5 * np.median(err))         # drop mis-found corners (glare, dirt) and refit
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


def calibrate_both(folder):
    """Calibrate both views from the pictures in `folder` (thermal_<n>.npy = thermal_c, optical_<n>.png)
    and print View lines to paste in. Writes <name>_<n>_corners.png next to them: what was found."""
    for name, view, kind, load in (("THERMAL_VIEW", THERMAL_VIEW, "thermal", "npy"),
                                   ("OPTICAL_VIEW", OPTICAL_VIEW, "optical", "png")):
        files = sorted((f for f in folder.glob(f"{kind}_*.{load}") if f.stem[len(kind) + 1:].isdigit()),
                       key=lambda f: int(f.stem[len(kind) + 1:]))     # not the *_corners.png ones
        if not files:
            continue
        pictures = [np.load(f) if load == "npy" else cv2.imread(str(f)) for f in files]
        print(f"\n{name} calibration ({', '.join(f.name for f in files)}):")
        marked = []
        try:
            print(f"{name} = {calibrate_view(view, pictures, marked=marked)}")
        except RuntimeError as exc:
            print(exc)
        for f, mark in zip(files, marked):
            cv2.imwrite(str(f.with_name(f"{f.stem}_corners.png")), mark)
        print(f"Corners found: {folder / kind}_<n>_corners.png")


def main():
    if len(sys.argv) > 1:                             # python dual_camera.py calibration/<time>
        calibrate_both(Path(sys.argv[1]))
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
                folder.mkdir(parents=True, exist_ok=True)
                if pair.thermal_live:
                    np.save(folder / f"thermal_{taken}.npy", pair.thermal_c)
                    cv2.imwrite(str(folder / f"thermal_{taken}_colour.png"), pair.thermal)
                if pair.optical_live:
                    cv2.imwrite(str(folder / f"optical_{taken}.png"), pair.optical)
                print(f"Calibration picture {taken} saved in {folder}")
            if key in (27, ord("q")) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    cv2.destroyAllWindows()
    if taken:
        calibrate_both(folder)


if __name__ == "__main__":
    main()
