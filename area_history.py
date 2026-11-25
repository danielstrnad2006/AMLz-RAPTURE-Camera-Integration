"""
area_history.py - areas of the build plate and their history. Each picture taken for them is put into
every area in physical space: resampled onto a grid in plate mm through the views in use (lens
distortion taken out), so a cell is the same plate point in every picture. Thermal as deg C.

Run:  python area_history.py               (lists the areas)
      python area_history.py <area name>   (its history in a matplotlib window; no cameras needed)
The GUI: app.py > AREAS (draw areas on the build plate, take pictures into them, see the history).
From your own code:
    area = new_area("part 1", [(40, 0), (60, 0), (60, 20), (40, 20)])   # corners x', y' mm
    record(pair, load_areas())                    # pair from rig.grab() or load_pair(); into every area
    rows = load_history(area)                     # one dict per picture (= history.csv)
    temps, colours = entry(area, rows[-1])        # deg C array, BGRA array of that picture
    average, count = thermal_average(area, rows)  # deg C per cell averaged over those pictures

An area is a polygon fixed to the build plate, like the part on it: its corners are in the plate's own
frame x'-y', which turns with the plate. pair.rotation = how far the plate had turned when the pair was
taken (deg, counter-clockwise seen from above, like omega; later from the PLC): the cameras saw the area
turned by that much, and that is where it is looked up. The area's pictures stay in the plate's frame.
Stored in areas/<name>/ next to this script:
    area.json            name, corners, grid step per camera (mm), when it was created
    history.csv          one row per picture: time, rotation, thermal mean / min / max / std (deg C, over
                         the part of the area the camera sees; all cells are the same size on the plate,
                         so this is the mean over the area), share of the area each camera sees, the
                         whole picture in pictures/ if it was kept
    <time>_thermal.npy   deg C on the area's thermal grid (float32; NaN outside the area or not seen)
    <time>_optical.png   BGRA on the area's optical grid (transparent outside the area or not seen)
Grids: row 0 = largest y', column 0 = smallest x' (the plate seen from above); cell centres at
x' = x0 + (i + 0.5) step, y' = y1 - (j + 0.5) step, with x0, y1 = the smallest x', largest y' of the corners.
"""
import csv
import functools
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.dates as mdates
import numpy as np
from matplotlib.ticker import MaxNLocator

LINE, BAND, MUTED, INK = "#0F766E", "#D5EFEC", "#6B7280", "#111827"   # as the charts of the RAPTURE interface

import dual_camera
from dual_camera import _sample, turn

AREA_DIR = Path(__file__).resolve().parent / "areas"
AREA_STEP = {"thermal": 0.15, "optical": 0.05}   # mm per cell, about one camera pixel (90 mm / 640, 87 mm / 2048)
COLUMNS = ("time", "timestamp", "rotation_deg", "thermal_mean_C", "thermal_min_C", "thermal_max_C",
           "thermal_std_C", "thermal_seen", "optical_seen", "stem", "picture")
NUMBERS = COLUMNS[1:9]


def new_area(name, corners, **info):
    """A new area `name` with the polygon `corners` [(x', y') mm, ...] in the plate's frame (2 corners =
    the rectangle between them), stored as areas/<name>/area.json with `info` (e.g. the 3MF job and
    island it comes from, see defect_detection.py). Returns the area (dict)."""
    name = re.sub(r'[<>:"/\\|?*]', "_", name.strip()).strip(". ")     # a valid folder name on Windows
    corners = [(float(x), float(y)) for x, y in corners]
    if len(corners) == 2:
        (x0, y0), (x1, y1) = corners
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    if not name:
        raise ValueError("the area needs a name")
    if len(corners) < 3 or abs(cv2.contourArea(np.float32(corners))) < 0.01:
        raise ValueError("an area needs at least 3 corners around some surface (or 2 for a rectangle)")
    folder = AREA_DIR / name
    if folder.exists():
        raise ValueError(f"there is an area '{name}' already")
    area = {"name": name, "corners": [[round(x, 3), round(y, 3)] for x, y in corners], "steps": dict(AREA_STEP),
            "created": f"{datetime.now():%Y-%m-%d %H:%M:%S}", **info}
    folder.mkdir(parents=True)
    (folder / "area.json").write_text(json.dumps(area, indent=2), encoding="utf-8")
    return {**area, "folder": folder}


def load_areas():
    """All areas in areas/, oldest first (dicts as from new_area; the name is the folder's)."""
    areas = []
    for path in AREA_DIR.glob("*/area.json"):
        try:
            areas.append({**json.loads(path.read_text(encoding="utf-8")), "name": path.parent.name, "folder": path.parent})
        except ValueError:
            print(f"{path}: not readable, left out")
    return sorted(areas, key=lambda a: (a.get("created", ""), a["name"]))


def area_grid(area, camera):
    """The area's grid for `camera`: (xs, ys, inside) = cell centres x' (ascending) and y' (descending) in mm,
    and which cells lie inside the polygon (bool, rows along ys)."""
    step, corners = area["steps"][camera], np.array(area["corners"], float)
    (x0, y0), (x1, y1) = corners.min(0), corners.max(0)
    xs = x0 + (np.arange(max(1, int(np.ceil((x1 - x0) / step - 1e-9)))) + 0.5) * step
    ys = y1 - (np.arange(max(1, int(np.ceil((y1 - y0) / step - 1e-9)))) + 0.5) * step
    inside = np.zeros((len(ys), len(xs)), np.uint8)
    cells = np.stack([(corners[:, 0] - xs[0]) / step, (ys[0] - corners[:, 1]) / step], 1)   # in cells
    cv2.fillPoly(inside, [np.round(cells * 16).astype(np.int32)], 1, shift=4)
    return xs, ys, inside.astype(bool)


def extent(area, camera, shape):
    """imshow extent (left, right, bottom, top) mm of an area picture of `shape` (origin "upper")."""
    step, corners = area["steps"][camera], np.array(area["corners"], float)
    x0, y1 = corners[:, 0].min(), corners[:, 1].max()
    return x0, x0 + shape[1] * step, y1 - shape[0] * step, y1


@functools.lru_cache(maxsize=8)
def picture_outline(view, shape):
    """Where a picture of `shape` (height, width) lies through `view`, in the machine frame: points along
    its edge, mm."""
    h, w = shape
    side = np.linspace(0, 1, 25)
    u = np.r_[side * (w - 1), np.full(25, w - 1.0), (1 - side) * (w - 1), np.zeros(25)]
    v = np.r_[np.zeros(25), side * (h - 1), np.full(25, h - 1.0), (1 - side) * (h - 1)]
    return np.stack(view.to_plate(u, v, shape), -1)


def map_area(pair, area):
    """`pair` resampled into `area` (with the plate turned by pair.rotation) through the views in use:
    {"thermal": deg C array or None, "optical": BGRA array or None}; None where the camera delivers
    nothing, calibrates or sees none of the area."""
    out = {}
    for camera, picture, live in (("thermal", pair.thermal_c, pair.thermal_live),
                                  ("optical", pair.optical, pair.optical_live)):
        out[camera] = None
        if not live or (camera == "thermal" and np.isnan(picture).all()):
            continue
        view = dual_camera.THERMAL_VIEW if camera == "thermal" else dual_camera.OPTICAL_VIEW
        rotation = pair.turned(camera)                               # when this camera's frame was taken
        picture_at = turn(picture_outline(view, picture.shape[:2]), -rotation)   # in the plate's frame
        corners = np.array(area["corners"], float)
        if ((picture_at.min(0) > corners.max(0)) | (picture_at.max(0) < corners.min(0))).any():
            continue                                                 # bounding boxes apart: not in the picture
        xs, ys, inside = area_grid(area, camera)
        blocks = []
        for rows in np.array_split(ys, max(1, len(ys) // 200)):             # in blocks: less memory
            there = turn(np.stack(np.meshgrid(xs, rows), -1), rotation)    # where the camera saw it
            blocks.append(_sample(picture, *view.to_pixel(there[..., 0], there[..., 1], picture.shape)))
        values = np.concatenate(blocks)
        seen = inside & ~np.isnan(values if values.ndim == 2 else values[..., 0])
        if not seen.any():
            continue
        if camera == "thermal":
            out[camera] = np.where(seen, values, np.nan).astype(np.float32)
        else:
            out[camera] = np.dstack([np.where(seen[..., None], values, 0).round(), seen * 255]).astype(np.uint8)
    return out


def record(pair, areas, picture=""):
    """Put `pair` (taken with the plate turned by pair.rotation) into the history of each of `areas`: its
    files + a history.csv row, where at least one camera sees the area. `picture` = the name of the whole
    pair in pictures/, if it was kept. Returns {area name: row} for the areas that got one."""
    done = {}
    for area in areas:
        maps = map_area(pair, area)
        if maps["thermal"] is None and maps["optical"] is None:
            continue
        folder = area["folder"]
        stem = base = f"{datetime.fromtimestamp(pair.timestamp):%Y-%m-%d_%H-%M-%S}"
        while any((folder / f"{stem}{end}").exists() for end in ("_thermal.npy", "_optical.png")):
            stem = f"{base}_{int(stem[len(base) + 1:] or 1) + 1}"          # more than one in a second
        row = dict.fromkeys(COLUMNS, "")
        row.update(time=f"{datetime.fromtimestamp(pair.timestamp):%Y-%m-%d %H:%M:%S.%f}"[:-4],
                   timestamp=f"{pair.timestamp:.3f}", rotation_deg=f"{pair.rotation:g}", stem=stem, picture=picture)
        for camera in ("thermal", "optical"):
            inside = area_grid(area, camera)[2].sum()
            if maps[camera] is None:
                row[f"{camera}_seen"] = "0"
            elif camera == "thermal":
                temps = maps[camera][np.isfinite(maps[camera])]
                np.save(folder / f"{stem}_thermal.npy", maps[camera])
                row.update(thermal_mean_C=f"{temps.mean():.3f}", thermal_min_C=f"{temps.min():.3f}",
                           thermal_max_C=f"{temps.max():.3f}", thermal_std_C=f"{temps.std():.3f}",
                           thermal_seen=f"{min(1, temps.size / inside):.3f}")
            else:
                cv2.imwrite(str(folder / f"{stem}_optical.png"), maps[camera])
                row["optical_seen"] = f"{min(1, (maps[camera][..., 3] > 0).sum() / inside):.3f}"
        history = folder / "history.csv"
        new = not history.exists()
        with history.open("a", newline="", encoding="utf-8") as f:      # after the files: a row's files exist
            writer = csv.DictWriter(f, COLUMNS)
            if new:
                writer.writeheader()
            writer.writerow(row)
        done[area["name"]] = row
    return done


def load_history(area):
    """The rows of the area's history.csv, oldest first: dicts with COLUMNS; the NUMBERS as float (NaN if
    empty), time, stem and picture as text."""
    path = area["folder"] / "history.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in NUMBERS:
            row[key] = float(row[key]) if row.get(key) else np.nan
    return sorted(rows, key=lambda r: r["timestamp"])


def entry(area, row):
    """(deg C array or None, BGRA array or None) of the picture `row` of the area's history."""
    thermal, optical = (area["folder"] / f"{row['stem']}{end}" for end in ("_thermal.npy", "_optical.png"))
    return (np.load(thermal) if thermal.exists() else None,
            cv2.imread(str(optical), cv2.IMREAD_UNCHANGED) if optical.exists() else None)


def thermal_average(area, rows):
    """deg C per cell of the area's thermal grid, averaged over the pictures `rows` that saw the cell (NaN:
    none did), and in how many pictures each cell was seen. (None, None) without temperatures."""
    total = count = None
    for row in rows:
        path = area["folder"] / f"{row['stem']}_thermal.npy"
        if path.exists():
            temps = np.load(path)
            if total is None:
                total, count = np.zeros(temps.shape), np.zeros(temps.shape, int)
            seen = np.isfinite(temps)
            total[seen] += temps[seen]
            count += seen
    if total is None:
        return None, None
    return np.where(count > 0, total / np.maximum(count, 1), np.nan).astype(np.float32), count


def delete_entry(area, row):
    """Take the picture `row` out of the area's history: its files and its history.csv row."""
    for end in ("_thermal.npy", "_optical.png"):
        (area["folder"] / f"{row['stem']}{end}").unlink(missing_ok=True)
    path = area["folder"] / "history.csv"
    with path.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["stem"] != row["stem"]]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def draw_history(fig, area, rows, index=-1, average=None):
    """The history of `area` (`rows` from load_history) into the empty figure `fig`: picture `index` (thermal
    in deg C, optical), the thermal average over all `rows` (`average` from thermal_average; computed if
    None) and the area's temperature over time (mean, band from min to max). Returns the time plot's axes."""
    (ax_t, ax_o), (ax_a, ax_time) = fig.subplots(2, 2)
    outline = np.vstack([area["corners"], area["corners"][:1]])
    row = rows[index] if rows else None
    temps, colours = entry(area, row) if row else (None, None)
    if average is None and rows:
        average = thermal_average(area, rows)[0]
    when = f", {row['time'][:19]}" if row else ""
    for ax, camera, picture, title in ((ax_t, "thermal", temps, f"Thermal{when}"),
                                       (ax_o, "optical", colours, f"Optical{when}"),
                                       (ax_a, "thermal", average, f"Thermal average over {len(rows)} pictures")):
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_aspect("equal")
        ax.plot(*outline.T, color=MUTED, lw=1)
        ax.tick_params(labelsize=8)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        if picture is None or (picture.ndim == 2 and np.isnan(picture).all()):
            ax.text(0.5, 0.5, "no pictures yet" if not rows else f"the {camera} camera did not see the area",
                    ha="center", va="center", color=MUTED, transform=ax.transAxes)
            continue
        if picture.ndim == 2:                          # deg C
            shown = ax.imshow(picture, extent=extent(area, camera, picture.shape), cmap="inferno",
                              interpolation="nearest")
            fig.colorbar(shown, ax=ax, shrink=0.9).set_label("°C", rotation=0, va="center", labelpad=12)
        else:                                          # BGRA -> RGBA
            ax.imshow(picture[..., [2, 1, 0, 3]], extent=extent(area, camera, picture.shape))
    ax_a.set_xlabel("x' (mm)", fontsize=9)
    ax_a.set_ylabel("y' (mm)", fontsize=9)

    ax_time.set_title("Mean temperature, °C (band: min to max)", fontsize=10, loc="left")
    ax_time.tick_params(labelsize=8)
    ax_time.grid(True, color="#E5E7EB", lw=0.8)
    ax_time.set_axisbelow(True)
    for side in ("top", "right"):
        ax_time.spines[side].set_visible(False)
    mean = np.array([r["thermal_mean_C"] for r in rows])
    if not np.isfinite(mean).any():
        ax_time.text(0.5, 0.5, "no temperatures yet", ha="center", va="center", color=MUTED, transform=ax_time.transAxes)
        return ax_time
    times = [datetime.fromtimestamp(r["timestamp"]) for r in rows]
    ax_time.fill_between(times, [r["thermal_min_C"] for r in rows], [r["thermal_max_C"] for r in rows],
                         color=BAND, lw=0)
    ax_time.plot(times, mean, color=LINE, lw=2, marker="o", ms=4)
    partly = np.array([r["thermal_seen"] < 0.99 for r in rows])
    if partly.any():                                   # hollow: the area was partly outside the thermal picture
        ax_time.plot(np.array(times)[partly], mean[partly], ls="none", marker="o", ms=6, mfc="white", mec=LINE,
                     mew=1.5)
        ax_time.set_title("Mean temperature, °C (band: min to max; hollow: partly seen)", fontsize=10, loc="left")
    if np.isfinite(mean[index]):
        ax_time.plot(times[index], mean[index], marker="o", ms=11, mfc="none", mec=INK, mew=2)
    ax_time.axvline(times[index], color="#9CA3AF", lw=1, zorder=0)
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    ax_time.xaxis.set_major_locator(locator)
    ax_time.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    return ax_time


def main():
    areas = {a["name"]: a for a in load_areas()}
    if len(sys.argv) < 2 or sys.argv[1] not in areas:
        print("Areas:", ", ".join(areas) if areas else "none yet (draw one in app.py > AREAS)")
        print("Run:   python area_history.py <area name>")
        return
    import matplotlib.pyplot as plt
    area = areas[sys.argv[1]]
    rows = load_history(area)
    fig = plt.figure(figsize=(12, 9), layout="constrained")
    fig.canvas.manager.set_window_title(f"History of {area['name']}")
    draw_history(fig, area, rows)
    print(f"{area['name']}: {len(rows)} pictures, files in {area['folder']}")
    plt.show()


if __name__ == "__main__":
    main()
