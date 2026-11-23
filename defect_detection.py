"""
defect_detection.py - monitored areas from a sliced RAPTURE job (3MF with toolpaths), for defect detection.

Scope: defect detection only (no closed-loop control), for parts with a constant cross-section through
the height. RAPTURE prints on a turning plate: the job's toolpath layers are sectors (wedges) of
`sector_angle_deg`, one after another round and round (layer_height_per_rev_um per turn). With a
constant cross-section the first revolution, the sectors until their angles add up to 360 deg, holds
the whole cross-section; the later revolutions repeat it.

From the first revolution's scan vectors (hatches + contours) a melt mask of the plate is drawn
(`melt_mask`); its islands, kept `inset` mm inside the melted region (the alignment error and the melt
pool edge must not bring powder into an area), become the monitored areas, optionally cut into tiles
(`monitored_areas`). `create_areas` stores them as ordinary areas (area_history.new_area, with the job
and island in area.json), so app.py / camera_service.py record them like any other: every picture
(Take picture, Record every n s, the API, later the PLC trigger) goes into every area a camera sees.

Run:  python defect_detection.py <job.3mf> [--tile MM] [--inset MM] [--min-area MM2] [--create]
      shows the first layer (scan vectors, sectors, the monitored areas, where the cameras look) and
      prints the areas; --create stores them in areas/.
"""
import argparse
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import cv2
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.path import Path as Outline
from matplotlib.patches import Circle, Wedge

import dual_camera
from area_history import load_areas, new_area
from dual_camera import PLATE_R_INNER, PLATE_R_OUTER, turn
from plate_plot import COLOURS, PALETTE, edge

TOOLPATH = "{http://schemas.microsoft.com/3dmanufacturing/toolpath/2019/05}"
RAPTURE = "{https://rapture.ethz.ch/xml_namespace/20250203}"
STEP = 0.05         # mm per pixel of the melt mask
INSET = 0.3         # mm the areas stay inside the melted region
MIN_AREA = 2.0      # mm^2: smaller pieces (slivers at sector and plate edges) are not monitored
SHAPES = {"thermal": (480, 640), "optical": dual_camera.OPTICAL_SIZE[::-1]}    # picture (height, width)


def load_job(path):
    """The first revolution of a sliced job: {"name", "plate_diameter", "sector_deg", "sectors": [{"index"
    (layer number, from 1), "start", "end" (deg), "hatches" (n, 4) x1 y1 x2 y2 mm, "contours" [(m, 2) mm]}]}.
    Sectors are read until their angles add up to 360 deg."""
    path = Path(path)
    with zipfile.ZipFile(path) as z:
        model = ET.fromstring(z.read("3D/3dmodel.model"))
        resource = model.find(f".//{TOOLPATH}toolpathresource")
        unit = float(resource.get("unitfactor", "0.001"))            # toolpath units -> mm
        meta = resource.find(f".//{RAPTURE}metadata")
        parts = resource.find(f".//{RAPTURE}parts")
        job = {"name": path.stem, "path": str(path),
               "plate_diameter": float(parts.get("plate_diameter_mm", 2 * PLATE_R_OUTER)) if parts is not None else 2 * PLATE_R_OUTER,
               "sector_deg": float(meta.get("sector_angle_deg")) if meta is not None else None, "sectors": []}
        total = 0.0
        for index, layer in enumerate(resource.iter(f"{TOOLPATH}toolpathlayer"), 1):
            if total >= 360:
                break
            root = ET.fromstring(z.read(layer.get("path").lstrip("/")))
            sector = root.find(f".//{RAPTURE}sector_metadata")
            start, end = (int(sector.get(f"sector_{e}_angle_centideg")) / 100 for e in ("start", "end"))
            total += end - start
            hatches, contours = [], []
            for segment in root.iter(f"{TOOLPATH}segment"):
                if segment.get("type") == "hatch":
                    hatches += [[float(h.get(k)) for k in ("x1", "y1", "x2", "y2")] for h in segment.iter(f"{TOOLPATH}hatch")]
                else:                                                # polyline / loop: contours
                    contours.append(np.array([[float(p.get("x")), float(p.get("y"))]
                                              for p in segment.iter(f"{TOOLPATH}point")]) * unit)
            job["sectors"].append({"index": index, "start": start, "end": end, "contours": contours,
                                   "hatches": np.array(hatches, float).reshape(-1, 4) * unit})
    return job


def hatch_spacing(hatches):
    """Distance between neighbouring hatch lines, mm: the median step across the hatch direction from one
    hatch to the next in the file (they are written island by island, line after line)."""
    if len(hatches) < 2:
        return STEP
    d = hatches[:, 2:] - hatches[:, :2]
    d = d[np.argmax(np.hypot(*d.T))]
    across = ((hatches[:, :2] + hatches[:, 2:]) / 2) @ (np.array([-d[1], d[0]]) / np.hypot(*d))
    gaps = np.abs(np.diff(across))
    gaps = gaps[(gaps > 1e-3) & (gaps < 2)]                          # not a jump to another island
    return float(np.median(gaps)) if len(gaps) else STEP


def melt_mask(job, step=STEP):
    """What the first revolution melts, as a boolean raster of the plate: row 0 = largest y', column 0 =
    smallest x', pixel centres at ((col + 0.5) * step - r, r - (row + 0.5) * step) with r = plate radius."""
    r = job["plate_diameter"] / 2
    n = int(np.ceil(2 * r / step))
    mask = np.zeros((n, n), np.uint8)
    scale = 16 / step                                                # cv2 sub-pixel drawing (shift = 4)

    def px(xy):
        return np.round(np.c_[(xy[:, 0] + r) * scale, (r - xy[:, 1]) * scale]).astype(np.int32) - 8

    spacing = []
    for sector in job["sectors"]:
        h = sector["hatches"]
        if len(h):
            spacing.append(hatch_spacing(h))
            ends = px(h.reshape(-1, 2)).reshape(-1, 2, 2)
            cv2.polylines(mask, list(ends), False, 1, 1, cv2.LINE_8, 4)
        cv2.polylines(mask, [px(c) for c in sector["contours"] if len(c) > 1], False, 1, 1, cv2.LINE_8, 4)
    # Gaps up to twice the hatch spacing count as melted: the melt pool bridges them (the slicer leaves
    # such gaps where two sectors meet, ~0.1 mm in the 8 mm checkerboard job).
    gap = int(np.ceil(2 * max(spacing, default=step) / step)) + 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (gap, gap)))
    return mask.astype(bool)


def monitored_areas(job, tile=None, inset=INSET, min_area=MIN_AREA, step=STEP):
    """The islands of the melt mask, `inset` mm inside it (which also separates islands that only touch
    at a corner, as checkerboard squares do), cut into `tile` mm squares (None = one area per island).
    Returns [{"name", "corners" (m, 2) plate mm, "island", "tile", "area_mm2", "holes"}], in reading
    order on the plate (rows from the top, left to right)."""
    r = job["plate_diameter"] / 2
    mask = melt_mask(job, step).astype(np.uint8)
    k = int(round(inset / step))
    if k:
        mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    count, labels, stats, centres = cv2.connectedComponentsWithStats(mask, connectivity=4)
    order = sorted(range(1, count), key=lambda i: (round(centres[i][1] / (4 / step)), centres[i][0]))
    areas, island = [], 0
    for i in order:
        x0, y0, w, h = stats[i, :4]
        piece = (labels[y0:y0 + h, x0:x0 + w] == i).astype(np.uint8)
        if tile:                                                     # tiles aligned with the plate's axes
            cols = ((np.arange(x0, x0 + w) + 0.5) * step) // tile
            rows = ((np.arange(y0, y0 + h) + 0.5) * step) // tile
            cells = rows[:, None] * 10_000 + cols[None, :]
            pieces = [((cells == c) & (piece > 0)).astype(np.uint8) for c in np.unique(cells[piece > 0])]
        else:
            pieces = [piece]
        kept = [p for p in sorted(pieces, key=lambda p: -int(p.sum())) if int(p.sum()) * step ** 2 >= min_area]
        island += bool(kept)                                         # numbered without the dropped slivers
        for t, part in enumerate(kept, 1):
            outlines, hierarchy = cv2.findContours(part, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            outer = max(outlines, key=cv2.contourArea)
            outer = cv2.approxPolyDP(outer, 1.0, True)[:, 0] if len(outer) > 4 else outer[:, 0]
            corners = np.c_[(outer[:, 0] + x0 + 0.5) * step - r, r - (outer[:, 1] + y0 + 0.5) * step]
            name = f"{job['name']} {island:03d}" + (f"-{t:02d}" if tile and len(kept) > 1 else "")
            areas.append({"name": name, "corners": np.round(corners, 3), "island": island, "tile": t,
                          "area_mm2": round(int(part.sum()) * step ** 2, 2),
                          "holes": bool(hierarchy is not None and (hierarchy[0][:, 3] >= 0).any())})
    return areas


def footprint(camera):
    """Where the camera looks, in the machine frame (= the plate's frame at rotation 0): (n, 2) mm."""
    view, shape = getattr(dual_camera, f"{camera.upper()}_VIEW"), SHAPES[camera]
    outline = np.c_[view.to_plate(*edge(shape), shape)]
    return outline[np.r_[True, np.hypot(*np.diff(outline, axis=0).T) > 1e-9]]   # without repeated points


def visibility(areas, step_deg=1.0):
    """Per area and camera: whether the whole area is in the picture at plate rotation 0 ("now") and at
    some rotation ("turning"), and the share of rotations at which it is ("share")."""
    angles = np.arange(0, 360, step_deg)
    corners = np.vstack([a["corners"] for a in areas])
    owner = np.repeat(np.arange(len(areas)), [len(a["corners"]) for a in areas])
    seen = {a["name"]: {} for a in areas}
    for camera in SHAPES:
        points = turn(corners[None], angles[:, None])                # (angles, corners, 2)
        inside = Outline(footprint(camera)).contains_points(points.reshape(-1, 2)).reshape(len(angles), -1)
        whole = np.stack([inside[:, owner == i].all(1) for i in range(len(areas))], 1)   # (angles, areas)
        for i, area in enumerate(areas):
            seen[area["name"]][camera] = {"now": bool(whole[0, i]), "turning": bool(whole[:, i].any()),
                                          "share": float(whole[:, i].mean())}
    return seen


def draw_job(fig, job, areas, rotation=0.0):
    """The first layer on the plate (plate frame): scan vectors, the sectors, the monitored areas, and
    where the cameras look with the plate turned by `rotation` deg (solid) and while it turns (band)."""
    fig.clear()
    ax = fig.add_subplot()
    r = job["plate_diameter"] / 2
    ax.add_patch(Circle((0, 0), r, facecolor="#F3F4F6", edgecolor="#9CA3AF", lw=1))
    ax.add_patch(Circle((0, 0), PLATE_R_INNER, facecolor="white", edgecolor="#9CA3AF", lw=1))
    for camera, colour in COLOURS.items():                           # what each camera sees while the plate turns
        outline = footprint(camera)
        near = 0.0 if Outline(outline).contains_point((0, 0)) else np.hypot(*outline.T).min()   # dense outline
        far = np.hypot(*outline.T).max()
        ax.add_patch(Wedge((0, 0), min(far, r), 0, 360, width=min(far, r) - near, color=colour, alpha=0.06, lw=0))
        seen = turn(outline, -rotation)                              # the picture in the plate's frame
        ax.plot(*np.vstack([seen, seen[:1]]).T, color=colour, lw=1.8, zorder=6)
        ax.text(*seen[np.argmax(seen[:, 1])], f" {camera}", color=colour, fontsize=9, va="bottom", zorder=6)
    hatches = np.vstack([s["hatches"] for s in job["sectors"]])
    ax.add_collection(LineCollection(hatches.reshape(-1, 2, 2), colors="#6B7280", linewidths=0.25, zorder=2))
    ax.add_collection(LineCollection([c for s in job["sectors"] for c in s["contours"]], colors="#1F2933",
                                     linewidths=0.6, zorder=3))
    for s in job["sectors"]:                                          # sector boundaries and numbers
        a = np.radians(s["start"])
        ax.plot([PLATE_R_INNER * np.cos(a), r * np.cos(a)], [PLATE_R_INNER * np.sin(a), r * np.sin(a)],
                color="#9CA3AF", lw=0.7, ls=(0, (4, 4)), zorder=1)
        mid = np.radians((s["start"] + s["end"]) / 2)
        ax.text(1.05 * r * np.cos(mid), 1.05 * r * np.sin(mid), str(s["index"]), ha="center", va="center",
                fontsize=8, color="#1F2933" if len(s["hatches"]) else "#9CA3AF")
    colours = [PALETTE[a["island"] % len(PALETTE)] for a in areas]
    ax.add_collection(PolyCollection([a["corners"] for a in areas], facecolors=colours, alpha=0.25, zorder=4))
    ax.add_collection(PolyCollection([a["corners"] for a in areas], facecolors="none", edgecolors=colours,
                                     linewidths=1.2, linestyles=(0, (4, 2)), zorder=5))
    if len(areas) <= 200:
        for a, colour in zip(areas, colours):
            ax.text(*np.mean(a["corners"], 0), a["name"].rsplit(" ", 1)[-1], ha="center", va="center", fontsize=6,
                    color="white", zorder=7, path_effects=[pe.withStroke(linewidth=2, foreground=colour)])
    used = [s for s in job["sectors"] if len(s["hatches"]) or s["contours"]]
    ax.set_title(f"{job['name']}: first revolution = sectors 1 to {len(job['sectors'])} of {job['sector_deg']:g} deg "
                 f"({len(used)} with scan vectors); {len(areas)} monitored areas"
                 + (f"; cameras with the plate turned by {rotation:g} deg" if rotation else ""), fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlim(-1.12 * r, 1.12 * r)
    ax.set_ylim(-1.12 * r, 1.12 * r)
    ax.set_xlabel("x' mm")
    ax.set_ylabel("y' mm")
    return ax


def create_areas(job, areas):
    """The monitored areas as areas in areas/ (app.py and camera_service.py record them). Existing names
    are left as they are. Returns the number created."""
    have = {a["name"] for a in load_areas()}
    made = 0
    for a in areas:
        if a["name"] in have:
            continue
        new_area(a["name"], a["corners"], job=job["name"], island=a["island"], tile=a["tile"], area_mm2=a["area_mm2"])
        made += 1
    print(f"{made} areas created in areas/ ({len(areas) - made} were there already)")
    return made


def main():
    parser = argparse.ArgumentParser(description="Monitored areas from a sliced RAPTURE job (3MF).")
    parser.add_argument("job", help="the sliced job, .3mf")
    parser.add_argument("--tile", type=float, help="cut the islands into squares of this size, mm")
    parser.add_argument("--inset", type=float, default=INSET, help=f"mm inside the melted region (default {INSET})")
    parser.add_argument("--min-area", type=float, default=MIN_AREA, help=f"smallest area kept, mm^2 (default {MIN_AREA})")
    parser.add_argument("--create", action="store_true", help="store the areas in areas/ for recording")
    args = parser.parse_args()
    job = load_job(args.job)
    areas = monitored_areas(job, args.tile, args.inset, args.min_area)
    seen = visibility(areas)
    used = [s["index"] for s in job["sectors"] if len(s["hatches"]) or s["contours"]]
    print(f"{job['name']}: first revolution = {len(job['sectors'])} sectors of {job['sector_deg']:g} deg, scan vectors "
          f"in sectors {used}; {sum(len(s['hatches']) for s in job['sectors'])} hatches")
    print(f"{len(areas)} monitored areas ({args.inset:g} mm inside the melted region"
          + (f", tiles of {args.tile:g} mm" if args.tile else ", one per island") + f", >= {args.min_area:g} mm^2)")
    for camera in SHAPES:
        now = sum(seen[a["name"]][camera]["now"] for a in areas)
        turning = sum(seen[a["name"]][camera]["turning"] for a in areas)
        print(f"  {camera}: {now} wholly in the picture with the plate at 0 deg, {turning} at some rotation")
    holes = [a["name"] for a in areas if a["holes"]]
    if holes:
        print(f"  with holes (the area covers the hole too; use --tile): {holes}")
    if args.create:
        create_areas(job, areas)
    fig = plt.figure(figsize=(10, 10))
    draw_job(fig, job, areas)
    plt.show()


if __name__ == "__main__":
    main()
