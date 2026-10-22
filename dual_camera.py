"""
dual_camera.py - Optris PI 640i (thermal) + MOKOSE UC50 (optical), both over USB, Windows only.

Install once:  python -m pip install opencv-python numpy pygrabber
               plus the Optris thermal camera SDK (provides the Python module optris.otcsdk);
               restart the PC after installing it.
Run:           python dual_camera.py          (q or Esc to quit, s = optical camera settings)

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
import functools
import sys
import threading
import time
from dataclasses import dataclass

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
    """How one camera sees the build plate: lens model plus mounting. The camera looks straight down.

    Lens: pinhole with field of view `fov` = (horizontal, vertical) deg, plus OpenCV lens distortion
    `dist` = (k1, k2, p1, p2, k3); both come from calibrate_lens(). Principal point = picture centre.
    Mounting: lens `height` mm above the plate, straight above the plate point `center` = (x', y') mm,
    the picture's rightward direction at `omega` deg from +x' (counter-clockwise seen from above).
    """
    fov: tuple
    height: float
    center: tuple
    omega: float = 0.0
    dist: tuple = (0.0, 0.0, 0.0, 0.0, 0.0)

    def _intrinsics(self, shape):
        h, w = shape[:2]
        fx, fy = (n / 2 / np.tan(np.radians(a) / 2) for n, a in zip((w, h), self.fov))
        return fx, fy, (w - 1) / 2, (h - 1) / 2

    def to_pixel(self, x, y, shape):
        """Where build plate point(s) (x', y') mm appear in a picture of shape `shape`.

        Returns (u, v) = (column, row) as float arrays; NaN for points the lens model cannot map.
        """
        fx, fy, cx, cy = self._intrinsics(shape)
        c, s = np.cos(np.radians(self.omega)), np.sin(np.radians(self.omega))
        dx, dy = np.asarray(x, float) - self.center[0], np.asarray(y, float) - self.center[1]
        xn, yn = (dx * c + dy * s) / self.height, (dx * s - dy * c) / self.height   # ideal pinhole
        k1, k2, p1, p2, k3 = self.dist
        r2 = xn**2 + yn**2
        radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
        xd = xn * radial + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn**2)
        yd = yn * radial + p1 * (r2 + 2 * yn**2) + 2 * p2 * xn * yn
        # Past the radius where the distortion polynomial stops growing it folds back and would put
        # far-away points inside the picture; those points are left out.
        r2_test = np.linspace(0, 4 * np.sum(np.tan(np.radians(self.fov) / 2) ** 2), 1000)  # to 2x corner
        fold = 1 + 3 * k1 * r2_test + 5 * k2 * r2_test**2 + 7 * k3 * r2_test**3 <= 0
        ok = r2 < (r2_test[fold.argmax()] if fold.any() else r2_test[-1])
        return np.where(ok, fx * xd + cx, np.nan), np.where(ok, fy * yd + cy, np.nan)

    def to_plate(self, u, v, shape):
        """Build plate point(s) (x', y') mm seen at pixel (u, v) = (column, row) of a picture of shape `shape`."""
        fx, fy, cx, cy = self._intrinsics(shape)
        u, v = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float))
        n = cv2.undistortPointsIter(np.stack([u.ravel(), v.ravel()], -1).reshape(-1, 1, 2),
                                    np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]), np.array(self.dist),
                                    None, None, (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-12))
        xn, yn = n[:, 0, 0].reshape(u.shape), n[:, 0, 1].reshape(u.shape)
        c, s = np.cos(np.radians(self.omega)), np.sin(np.radians(self.omega))
        return (self.center[0] + self.height * (xn * c + yn * s),
                self.center[1] + self.height * (xn * s - yn * c))


# Build plate frame x'-y': origin at the plate centre, mm, +x' right and +y' up seen from above.
PLATE_R_INNER, PLATE_R_OUTER = 50, 200     # mm
PLOT_SCALE = 2                             # pixels per mm in plot_radial
# Placeholders until measured on the rig; fov and dist from calibrate_lens().
OPTICAL_VIEW = View(fov=(24, 18), height=100, center=(-100, -80), omega=70)
THERMAL_VIEW = View(fov=(50, 38), height=100, center=(-80, -100), omega=20)


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


def calibrate_lens(pictures, board=(9, 6)):
    """Lens field of view and distortion from pictures of a flat checkerboard.

    `board` = inner corners (columns, rows). Use 10-20 pictures with the board at different places,
    tilts and distances, together covering the whole picture, corners included. Pictures may be BGR
    or 2D (e.g. pair.thermal_c; the thermal camera needs a board that shows up warm/cold).
    Returns (fov, dist) for View(...).
    """
    corners_board = np.zeros((board[0] * board[1], 3), np.float32)
    corners_board[:, :2] = np.mgrid[:board[0], :board[1]].T.reshape(-1, 2)    # square size: irrelevant
    found_board, found_pic = [], []
    for pic in pictures:
        gray = pic if pic.ndim == 2 else cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
        if gray.dtype != np.uint8:
            gray = cv2.normalize(np.nan_to_num(gray), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        ok, corners = cv2.findChessboardCorners(gray, board)
        if ok:
            corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),
                                       (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 0.01))
            found_board.append(corners_board)
            found_pic.append(corners)
    print(f"Checkerboard found in {len(found_pic)} of {len(pictures)} pictures")
    if len(found_pic) < 5:
        raise RuntimeError("Too few checkerboard pictures for a calibration")
    h, w = gray.shape
    guess = np.array([[w, 0, (w - 1) / 2], [0, w, (h - 1) / 2], [0, 0, 1]], float)  # centre as in View
    err, k, dist, _, _ = cv2.calibrateCamera(found_board, found_pic, (w, h), guess, None,
                                             flags=cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT)
    fov = tuple(round(float(np.degrees(2 * np.arctan(n / 2 / f))), 3) for n, f in ((w, k[0, 0]), (h, k[1, 1])))
    dist = tuple(round(float(d), 6) for d in dist.ravel()[:5])
    print(f"fov={fov}, dist={dist}  (reprojection error {err:.2f} px)")
    return fov, dist


@functools.lru_cache(maxsize=8)
def _plate_view(view, shape, n, half, scale):
    """Which pixels of the plot_radial canvas a camera sees, and where they are in its picture."""
    rows, cols = np.mgrid[:n, :n]
    u, v = view.to_pixel(cols / scale - half, half - rows / scale, shape)
    h, w = shape[:2]
    seen = (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
    outline = cv2.findContours(seen.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    return seen, u[seen], v[seen], outline


def plot_radial(pair, scale=PLOT_SCALE):
    """Build plate from above with a radial grid and both camera views where they really are.

    Each canvas pixel is looked up in the original pictures (lens distortion and omega included).
    Returns a BGR picture, `scale` pixels per mm, plate centre in the middle, +x' right, +y' up.
    """
    half = PLATE_R_OUTER + 25                      # mm from the plate centre to the picture edge
    n = round(2 * half * scale)

    def px(x, y):                                  # build plate mm -> canvas pixel
        return round((x + half) * scale), round((half - y) * scale)

    canvas = np.full((n, n, 3), 255, np.uint8)
    cv2.circle(canvas, px(0, 0), round(PLATE_R_OUTER * scale), (225, 225, 225), -1, cv2.LINE_AA)
    cv2.circle(canvas, px(0, 0), round(PLATE_R_INNER * scale), (255, 255, 255), -1, cv2.LINE_AA)

    views = (("thermal", (0, 0, 255), _plate_view(THERMAL_VIEW, pair.thermal.shape, n, half, scale)),
             ("optical", (255, 0, 0), _plate_view(OPTICAL_VIEW, pair.optical.shape, n, half, scale)))
    for (_, _, (seen, u, v, _)), img in zip(views, (pair.thermal, pair.optical)):   # optical on top
        canvas[seen] = np.round(_sample(img, u, v)).astype(np.uint8)

    # Grid as an opacity mask, blended over everything in black or white, whichever shows up.
    grid = np.zeros((n, n), np.uint8)
    for step_r, step_deg, opacity in ((10, 10, 70), (50, 30, 230)):           # minor, then major
        for r in range(PLATE_R_INNER, PLATE_R_OUTER + 1, step_r):
            cv2.circle(grid, px(0, 0), round(r * scale), opacity, 1, cv2.LINE_AA)
        for deg in range(0, 360, step_deg):
            c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
            cv2.line(grid, px(PLATE_R_INNER * c, PLATE_R_INNER * s),
                     px(PLATE_R_OUTER * c, PLATE_R_OUTER * s), opacity, 1, cv2.LINE_AA)
    on = grid > 0
    opacity = grid[on, None] / 255
    contrast = np.where(canvas[on].mean(axis=1, keepdims=True) < 110, 255, 0)
    canvas[on] = canvas[on] * (1 - opacity) + contrast * opacity

    for r in range(PLATE_R_INNER, PLATE_R_OUTER + 1, 50):
        u, v = px(r, 0)
        tw = cv2.getTextSize(str(r), FONT, 0.4, 1)[0][0]
        cv2.putText(canvas, str(r), (u - tw - 3, v - 4), FONT, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    for deg in range(0, 360, 30):
        text = f"{deg}" + {0: " x'", 90: " y'"}.get(deg, "")
        tw, th = cv2.getTextSize(text, FONT, 0.4, 1)[0]
        r = PLATE_R_OUTER + 12
        u, v = px(r * np.cos(np.radians(deg)), r * np.sin(np.radians(deg)))
        cv2.putText(canvas, text, (u - tw // 2, v + th // 2), FONT, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    for name, colour, (_, _, _, outline) in views:
        if outline:
            cv2.polylines(canvas, outline, True, colour, 1, cv2.LINE_AA)
            points = np.vstack(outline)[:, 0]
            u, v = points[points[:, 1].argmin()]                                   # top of the view
            tw = cv2.getTextSize(name, FONT, 0.4, 1)[0][0]
            cv2.putText(canvas, name, (int(u) - tw // 2, int(v) - 6), FONT, 0.4, colour, 1, cv2.LINE_AA)
    return canvas


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


def main():
    window = "Thermal | Optical"
    with CameraRig() as rig:
        while True:
            pair = rig.grab()
            cv2.imshow(window, side_by_side(pair))
            cv2.imshow("Build plate", plot_radial(pair))
            key = cv2.waitKey(1 if pair.optical_live else 30) & 0xFF
            if key == ord("s"):
                rig.optical_settings()
            if key in (27, ord("q")) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
