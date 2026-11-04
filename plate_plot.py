"""
plate_plot.py - one moment of both cameras on the build plate, in a zoomable matplotlib window.

Install once:  python -m pip install matplotlib   (plus what dual_camera.py needs)
Run:           python plate_plot.py
               live preview; space = take this pair and show it on the build plate, s = optical camera
               settings, q or Esc = quit. In the plate window: zoom/pan with the toolbar, space or the
               button switches which camera lies on top; the status bar shows x', y', r, angle and the
               temperature under the mouse. Close the plate window to get back to the preview.
From your own code:  show_plate(pair)    (any FramePair, e.g. from rig.grab())
"""
from datetime import datetime

import cv2
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Polygon
from matplotlib.widgets import Button

from dual_camera import OPTICAL_VIEW, PLATE_R_INNER, PLATE_R_OUTER, THERMAL_VIEW, CameraRig, _sample, side_by_side

OVERSAMPLE = 1.5     # plate pixels per camera pixel (at the picture centre): more = sharper when zoomed, slower


def on_plate(view, img):
    """Picture `img` (taken through `view`) resampled onto a regular x'-y' grid.

    Returns (array, extent, outline): array rows run along +y' (imshow origin="lower"), NaN where the
    camera does not see the point; extent for imshow in mm; outline = the picture's border on the plate.
    """
    h, w = img.shape[:2]
    edge = np.linspace(0, 1, 50)
    u = np.concatenate([edge, np.ones(50), edge[::-1], np.zeros(50)]) * (w - 1)
    v = np.concatenate([np.zeros(50), edge, np.ones(50), edge[::-1]]) * (h - 1)
    outline = np.stack(view.to_plate(u, v, img.shape), 1)
    half = PLATE_R_OUTER + 25
    lo, hi = np.clip(outline.min(0), -half, half), np.clip(outline.max(0), -half, half)
    step = view.size[0] / w / OVERSAMPLE                                    # mm per plate pixel
    xs, ys = np.arange(lo[0], hi[0], step), np.arange(lo[1], hi[1], step)
    blocks = []
    for rows in np.array_split(ys, max(1, len(ys) // 200)):                  # in blocks: less memory
        x, y = np.meshgrid(xs, rows)
        blocks.append(_sample(img, *view.to_pixel(x, y, img.shape)))
    extent = (xs[0] - step / 2, xs[-1] + step / 2, ys[0] - step / 2, ys[-1] + step / 2)
    return np.concatenate(blocks), extent, outline


def rgba(bgr):
    """Resampled BGR picture (NaN where not seen) as RGBA uint8, transparent where not seen."""
    seen = ~np.isnan(bgr[..., 0])
    return np.dstack([np.nan_to_num(bgr[..., ::-1]).round(), seen * 255]).astype(np.uint8)


def show_plate(pair):
    """Build plate from above with a radial grid and both camera pictures where they really are.

    Each plate pixel is looked up in the original pictures (lens distortion and omega included), the
    thermal one as temperatures with a colour bar. Blocks until the window is closed.
    """
    fig, ax = plt.subplots(figsize=(10, 9))
    fig.canvas.manager.set_window_title("Build plate")
    ax.set_title(f"Build plate, {datetime.fromtimestamp(pair.timestamp):%Y-%m-%d %H:%M:%S}")
    ax.add_patch(Circle((0, 0), PLATE_R_OUTER, color="0.88", zorder=0))
    ax.add_patch(Circle((0, 0), PLATE_R_INNER, color="white", zorder=0))

    layers = {}                                    # camera name -> its image on the plate
    for name, live, view, colour in (("thermal", pair.thermal_live, THERMAL_VIEW, "red"),
                                     ("optical", pair.optical_live, OPTICAL_VIEW, "blue")):
        if not live:
            continue
        if name == "optical":
            picture, extent, outline = on_plate(view, pair.optical)
            layers[name] = ax.imshow(rgba(picture), extent=extent, origin="lower")
        else:
            temps, extent, outline = on_plate(view, pair.thermal_c)
            if np.isnan(temps).all():              # start-up calibration: no temperatures, false colours
                layers[name] = ax.imshow(rgba(on_plate(view, pair.thermal)[0]), extent=extent, origin="lower")
            else:
                layers[name] = ax.imshow(temps, extent=extent, origin="lower", cmap="inferno")
                fig.colorbar(layers[name], ax=ax, shrink=0.7, label="deg C")
        ax.add_patch(Polygon(outline, closed=True, fill=False, ec=colour, lw=1.2, zorder=4))
        top = outline[outline[:, 1].argmax()]
        ax.text(top[0], top[1] + 2, name, color=colour, ha="center", va="bottom", zorder=5)

    # Grid: dark lines with a light edge, so they show on dark and light pictures alike.
    t = np.radians(np.arange(361))
    for step_r, step_deg, width, alpha in ((10, 10, 0.4, 0.4), (50, 30, 0.9, 0.8)):   # minor, then major
        radii = [r for r in range(0, PLATE_R_OUTER + 1, step_r) if r >= PLATE_R_INNER]
        if step_r == 50:
            radii = sorted({PLATE_R_INNER, PLATE_R_OUTER, *radii})
        lines = [np.stack([r * np.cos(t), r * np.sin(t)], 1) for r in radii]
        lines += [[(r * np.cos(a), r * np.sin(a)) for r in (PLATE_R_INNER, PLATE_R_OUTER)]
                  for a in np.radians(range(0, 360, step_deg))]
        ax.add_collection(LineCollection(lines, colors="black", linewidths=width, alpha=alpha, zorder=3,
                                         path_effects=[pe.Stroke(linewidth=width + 1.2, foreground="white",
                                                                 alpha=0.5 * alpha), pe.Normal()]))
    for r in range(50, PLATE_R_OUTER + 1, 50):
        ax.text(r - 1, 1, str(r), fontsize=7, ha="right", va="bottom", zorder=5)
    for deg in range(0, 360, 30):
        r = PLATE_R_OUTER + 12
        ax.text(r * np.cos(np.radians(deg)), r * np.sin(np.radians(deg)),
                f"{deg}°" + {0: " x'", 90: " y'"}.get(deg, ""), fontsize=8, ha="center", va="center")

    half = PLATE_R_OUTER + 25
    ax.set(xlim=(-half, half), ylim=(-half, half), aspect="equal", xlabel="x' (mm)", ylabel="y' (mm)")

    def status(x, y):                              # status bar text for the mouse position
        text = f"x'={x:.1f} y'={y:.1f} mm   r={np.hypot(x, y):.1f} mm  {np.degrees(np.arctan2(y, x)) % 360:.1f}°"
        if pair.thermal_live:
            text += f"   {float(pair.temperature(x, y)):.1f} °C"
        return text
    ax.format_coord = status

    order = list(layers)                           # last one lies on top: optical to start with
    for z, name in enumerate(order, 1):
        layers[name].set_zorder(z)
    if len(layers) == 2:
        button = Button(fig.add_axes((0.02, 0.01, 0.3, 0.05)), f"{order[-1]} on top (space: switch)")

        def switch(_=None):
            order.reverse()
            for z, name in enumerate(order, 1):
                layers[name].set_zorder(z)
            button.label.set_text(f"{order[-1]} on top (space: switch)")
            fig.canvas.draw_idle()

        button.on_clicked(switch)
        fig.canvas.mpl_connect("key_press_event", lambda event: event.key == " " and switch())
    plt.show()


def main():
    window = "Space = show on build plate, q = quit"
    with CameraRig() as rig:
        while True:
            pair = rig.grab()
            cv2.imshow(window, side_by_side(pair))
            key = cv2.waitKey(1 if pair.optical_live else 30) & 0xFF
            if key == ord("s"):
                rig.optical_settings()
            if key == ord(" "):
                cv2.destroyWindow(window)          # it would stop responding while the plate window is open
                show_plate(pair)
            elif key in (27, ord("q")) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
