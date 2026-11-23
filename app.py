"""
app.py - the GUI: live view of both cameras, pictures, video recording, build plate view, calibration
and alignment.

Run:  python app.py        (Windows or Linux; what to install: see dual_camera.py)
      Linux: with the conda env's Python, ~/miniconda3/envs/rapture/bin/python app.py (see dual_camera.py)
Main window: live thermal | optical view. Take a picture (p) into pictures/ (see dual_camera.save_pair),
record a real-time clip (r) or a timelapse (t) as MP4 in clips/ (see capture_dual_video.py), show the
current moment on the build plate (space, see plate_plot.py), open the optical camera's own settings
window (s), open the calibration and alignment windows. Everything the scripts print goes to the log at
the bottom (and to the console).
Calibration window: choose thermal, optical or both; take checkerboard pictures (c) into
calibration/<time>/ or open an earlier folder; Calibrate fits size and lens distortion (see
dual_camera.calibrate_view) and shows where the corners were found. center and omega (measured by hand)
are typed in. Save puts a view in use: it goes to views/<camera>.json, which everything loads at start-up.
Alignment window: a thermal + optical picture of one moment side by side and overlaid; click the same
feature in both (several points), Fit moves the optical view (center, omega, optionally zoom) so that
they land where the thermal camera sees them (the thermal view is the reference). Save puts it in use.
Areas window: draw areas on the build plate; pictures taken there (p, or every few seconds) go into each
area in plate mm, lens distortion taken out (see area_history.py); the selected area's history: thermal
(deg C) and optical picture of each moment, the thermal average, the mean temperature over time.
The cameras, recording and areas recording are camera_service.Service, which keeps running what needs no
window (record every n s goes on with the Areas window closed) and serves its HTTP API while the app runs.
Looks like the RAPTURE machine interface (the AMCF web client with its RAPTURE theme), see amcf_theme; every
window has the same tab bar (Live, Calibration, Alignment, Areas) and state chips for the cameras.
"""
import base64
import functools
import json
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
import traceback
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import matplotlib
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
import numpy as np
from matplotlib import font_manager
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import dual_camera
from area_history import delete_entry, draw_history, load_areas, load_history, thermal_average
from camera_service import Service, serve
from dual_camera import (BOARD_SQUARE, CALIBRATION_DIR, PICTURE_DIR, PLATE_R_INNER, PLATE_R_OUTER, VIEW_DIR,
                         _sample, align_view, calibrate_folder, load_pair, read_view, save_board_picture,
                         save_view)
from plate_plot import PALETTE, draw_plate, edge, plate_layers, rgba

CAMERAS = ("thermal", "optical")
PREVIEW_SIZE = (1280, 480)     # px, live view at start-up (shrinks with the window)
PICTURE_SHARE = 0.45           # calibration picture viewer at start-up: this share of the screen width

# The look of the RAPTURE machine interface: the AMCF web client with the RAPTURE theme (repo RAPTURE_AMCF,
# Plugins/UI/Theme/rapture.css, after docs/UI_REDESIGN.pptx): a navy control bar with the logo and state chips
# over a white tab bar, light grey page, white rounded cards with a thin line, outlined teal buttons with one
# filled main action per window, Segoe UI. Calm greys; colour only for state, with the IEC 60073 meanings
# (green ok / running, amber caution, red fault / stop / delete, blue operator action), never by colour alone.
TEAL, TEAL_SOFT, NAVY, AMBER, GREEN, BLUE = "#0F766E", "#D5EFEC", "#111827", "#ED8B00", "#2E7D32", "#1D4ED8"
PAGE, CARD, BORDER, TEXT, MUTED, FAINT = "#F4F5F7", "#FFFFFF", "#D5D8DD", "#1F2933", "#6B7280", "#9CA3AF"
ERROR, WARN, OFF = "#C62828", "#B45309", "#E5E7EB"       # WARN: amber text; OFF: disabled / empty
CHIPS = {"ok": GREEN, "info": BLUE, "warn": AMBER, "err": ERROR, "off": "#374151"}   # state chip colours
ASSETS = Path(__file__).resolve().parent       # rapture_logo.png, rapture_icon.png (from the AMCF resources)


def amcf_theme(root):
    """Style all windows of the app like the AMCF interface: ttk's "clam" theme recoloured (the native
    Windows theme ignores colours), rounded cards, buttons and chips from 9-slice images (`rounded`),
    Segoe UI, the RAPTURE rings as window icon."""
    families = set(tkfont.families(root))           # as rapture.css; Linux: the desktop's own sans
    family = next((f for f in ("Segoe UI", "Roboto", "Arial", "Ubuntu Sans", "Noto Sans", "DejaVu Sans")
                   if f in families), "Helvetica")

    def bold(size):                                # Tk elsewhere knows only normal and bold
        return ("Segoe UI Semibold", size) if "Segoe UI Semibold" in families else (family, size, "bold")

    unit = root.winfo_fpixels("1i") / 96           # screen px per px at 100% scaling

    def px(n):
        return max(1, round(n * unit))

    style = ttk.Style(root)
    style.theme_use("clam")
    root.configure(background=PAGE)
    root.rapture_images = []                       # Tk forgets images nobody keeps

    def element(name, radius, states, **options):
        """A rounded image element `name`: states = [(state, fill, line), ...], the first one the normal state."""
        images = [rounded(radius, fill, line, px(2) if line != fill else 0) for _, fill, line in states]
        root.rapture_images.extend(images)
        style.element_create(name, "image", images[0], *[(state, img) for (state, _, _), img in zip(states[1:], images[1:])],
                             border=radius, padding=0, sticky="nsew", width=2 * radius + 2, height=2 * radius + 2,
                             **options)                 # padding: else the content sits a whole radius in

    style.configure(".", background=CARD, foreground=TEXT, font=(family, 10), bordercolor=BORDER,
                    lightcolor=CARD, darkcolor=CARD, troughcolor=PAGE, focuscolor=CARD,
                    selectbackground=TEAL_SOFT, selectforeground=TEXT, insertcolor=TEXT)
    style.configure("Page.TFrame", background=PAGE)
    images = [rounded(px(12), CARD, BORDER, px(1))]
    root.rapture_images.extend(images)
    style.element_create("Card.bg", "image", images[0], border=px(12), padding=0, sticky="nsew",
                         width=px(24) + 2, height=px(24) + 2)
    style.layout("Card.TFrame", [("Card.bg", {"sticky": "nsew"})])
    style.configure("Card.TFrame", background=PAGE)          # the corners
    style.configure("Navy.TFrame", background=NAVY)
    style.configure("Navy.TLabel", background=NAVY, foreground="#E5E7EB", font=(family, 13))
    style.configure("Tabs.TFrame", background=CARD)
    style.configure("Tab.TLabel", background=CARD, foreground=MUTED, font=(family, 11))
    style.configure("TabHover.TLabel", background=CARD, foreground=TEXT, font=(family, 11))
    style.configure("TabActive.TLabel", background=CARD, foreground=TEAL, font=bold(11))
    style.configure("Title.TLabel", foreground=TEXT, font=bold(13))
    style.configure("Muted.TLabel", foreground=MUTED, font=(family, 9))
    style.configure("Warn.TLabel", foreground=WARN)
    # State chips in the control bar: pills, white text, colour by state (CHIPS)
    for kind, colour in CHIPS.items():
        element(f"Chip{kind}.bg", px(14), [("", colour, colour)])
        style.layout(f"Chip{kind}.TLabel", [(f"Chip{kind}.bg", {"sticky": "nsew", "children": [
            ("Label.padding", {"sticky": "nsew", "children": [("Label.label", {"sticky": "nsew"})]})]})])
        style.configure(f"Chip{kind}.TLabel", background=NAVY, foreground="white", font=bold(9),
                        padding=(px(14), px(5)), anchor="center")
    # Buttons: outlined teal; one filled main action per window; filled red to stop a recording; red outline to
    # delete; minor actions as teal text. Rounded 8 px; disabled grey.
    pressed, disabled = "#B7E2DD", ("disabled", OFF, OFF)
    for name, normal, hover, down, text in (
            ("TButton", (CARD, TEAL), (TEAL_SOFT, TEAL), (pressed, TEAL), TEAL),
            ("Main.TButton", (TEAL, TEAL), ("#0B5F58", "#0B5F58"), ("#094E48", "#094E48"), "white"),
            ("Danger.TButton", (ERROR, ERROR), ("#A61F1F", "#A61F1F"), ("#8E1A1A", "#8E1A1A"), "white"),
            ("Delete.TButton", (CARD, ERROR), ("#FDECEC", ERROR), ("#F8D0D0", ERROR), ERROR),
            ("Minor.TButton", (CARD, CARD), (TEAL_SOFT, TEAL_SOFT), (pressed, pressed), TEAL)):
        element(f"{name}.bg", px(8), [("", *normal), disabled, ("pressed", *down), ("active", *hover)])
        style.layout(name, [(f"{name}.bg", {"sticky": "nsew", "children": [
            ("Button.padding", {"sticky": "nsew", "children": [("Button.label", {"sticky": "nsew"})]})]})])
        style.configure(name, background=CARD, foreground=text, font=bold(10), padding=(px(16), px(9)),
                        anchor="center")
        style.map(name, foreground=[("disabled", FAINT)],              # background = the corners
                  background=[("disabled", CARD), ("pressed", CARD), ("active", CARD)])
    # Sub-tabs as pills (rapture.css .v-tab): grey, the active one light teal with a teal ring; on the page
    for name, states, text in (("Pill.TButton", [("", OFF, OFF), ("active", "#D9DCE1", "#D9DCE1")], MUTED),
                               ("PillActive.TButton", [("", TEAL_SOFT, TEAL)], TEAL)):
        element(f"{name}.bg", px(17), states)
        style.layout(name, [(f"{name}.bg", {"sticky": "nsew", "children": [
            ("Button.padding", {"sticky": "nsew", "children": [("Button.label", {"sticky": "nsew"})]})]})])
        style.configure(name, background=PAGE, foreground=text, font=bold(10), padding=(px(22), px(7)),
                        anchor="center")
        style.map(name, background=[("active", PAGE), ("pressed", PAGE)])
    style.configure("TEntry", fieldbackground="white", bordercolor=BORDER, lightcolor="white",
                    darkcolor="white", padding=(6, 4))
    style.map("TEntry", bordercolor=[("focus", TEAL)], lightcolor=[("focus", TEAL)])
    for name in ("TRadiobutton", "TCheckbutton"):
        style.configure(name, indicatorbackground="white", indicatorforeground=TEAL,
                        upperbordercolor=MUTED, lowerbordercolor=MUTED, padding=(0, 2, 12, 2),
                        indicatorsize=px(13), indicatormargin=(0, 0, px(6), 0))
        style.map(name, background=[("active", CARD)], indicatorbackground=[("pressed", TEAL_SOFT)])
    # Tables as in the parts table: navy heading, thin row lines, light teal selection
    style.configure("Treeview", background="white", fieldbackground="white", bordercolor=BORDER,
                    lightcolor="white", darkcolor="white", rowheight=px(26), font=(family, 10))
    style.map("Treeview", background=[("selected", TEAL_SOFT)], foreground=[("selected", TEXT)])
    style.configure("Treeview.Heading", background=NAVY, foreground="white", font=bold(9), relief="flat",
                    bordercolor=NAVY, lightcolor=NAVY, darkcolor=NAVY, padding=(px(6), px(4)))
    style.map("Treeview.Heading", background=[("active", "#1F2937")])
    style.configure("Horizontal.TScale", background=TEAL, troughcolor=OFF, bordercolor=OFF,
                    lightcolor=TEAL, darkcolor=TEAL)
    style.configure("Vertical.TScrollbar", background="#C8CCD2", troughcolor=CARD, bordercolor=CARD,
                    lightcolor="#C8CCD2", darkcolor="#C8CCD2", arrowcolor=MUTED, gripcount=0)
    style.map("Vertical.TScrollbar", background=[("active", FAINT)])
    style.configure("TSeparator", background=BORDER)
    try:                                           # plots in the same font, if matplotlib has it
        font_manager.findfont(family, fallback_to_default=False)
        matplotlib.rcParams["font.family"] = family
    except ValueError:
        pass
    matplotlib.rcParams.update({"text.color": TEXT, "axes.labelcolor": TEXT, "axes.edgecolor": FAINT,
                                "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": OFF})
    root.rapture_icon = image(ASSETS / "rapture_icon.png", 64)
    if root.rapture_icon:
        root.iconphoto(True, root.rapture_icon)    # for all windows of the app


def rounded(radius, fill, line, width=0):
    """A rounded rectangle as a Tk image for a 9-slice element: `fill` inside a `width` px `line`, the corners
    transparent; drawn 4x larger and scaled down (smooth edges). ttk tiles the middle, 256 px here: with a 2 px
    middle a full-window card took seconds to draw, with 256 px 57 ms against 32 for a plain frame. The
    element's smallest size is set separately (width/height in amcf_theme)."""
    size, scale = 2 * radius + 256, 4

    def shape(inset, r):                           # mask of a rounded rectangle, `inset` px in from the edge
        m = np.zeros((size * scale,) * 2, np.uint8)
        a, b, r = inset, size * scale - 1 - inset, max(0, r)
        cv2.rectangle(m, (a + r, a), (b - r, b), 255, -1)
        cv2.rectangle(m, (a, a + r), (b, b - r), 255, -1)
        for c in ((a + r, a + r), (b - r, a + r), (a + r, b - r), (b - r, b - r)):
            cv2.circle(m, c, r, 255, -1)
        return m

    def bgr(colour):
        return [int(colour[i:i + 2], 16) for i in (5, 3, 1)]

    picture = np.empty((size * scale, size * scale, 3), np.uint8)
    picture[:] = bgr(line)
    picture[shape(width * scale, (radius - width) * scale) > 0] = bgr(fill)
    picture = cv2.resize(np.dstack([picture, shape(0, radius * scale)]), (size, size), interpolation=cv2.INTER_AREA)
    return tk.PhotoImage(data=base64.b64encode(cv2.imencode(".png", picture)[1].tobytes()).decode(), format="png")


def image(path, height):
    """The PNG `path` (with transparency) `height` px tall as a Tk image; None if it is missing."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    img = cv2.resize(img, (round(img.shape[1] * height / img.shape[0]), height), interpolation=cv2.INTER_AREA)
    return tk.PhotoImage(data=base64.b64encode(cv2.imencode(".png", img)[1].tobytes()).decode(), format="png")


def toolbar(win, current=None, tabs=(), title=""):
    """The AMCF control bar at the top of `win`: navy, RAPTURE logo, `title`, room for state chips (bar.chips);
    under it the white tab bar, `tabs` = (caption, command), the tab `current` underlined in teal. Returns the bar."""
    unit = win.winfo_fpixels("1i") / 96
    bar = ttk.Frame(win, style="Navy.TFrame", padding=(round(14 * unit), round(8 * unit)))
    bar.pack(side="top", fill="x")
    bar.logo = image(ASSETS / "rapture_logo.png", round(38 * unit))     # a 54 px bar (AMCF: 64, more room here)
    ttk.Label(bar, image=bar.logo or "", text="" if bar.logo else "RAPTURE", style="Navy.TLabel").pack(side="left")
    if title:
        ttk.Label(bar, text=title, style="Navy.TLabel").pack(side="left", padx=(round(20 * unit), 0))
    bar.chips = ttk.Frame(bar, style="Navy.TFrame")
    bar.chips.pack(side="left", padx=(round(28 * unit), 0))
    if tabs:
        strip = ttk.Frame(win, style="Tabs.TFrame", padding=(round(14 * unit), 0))
        strip.pack(side="top", fill="x")
        for caption, command in tabs:
            tab = ttk.Frame(strip, style="Tabs.TFrame")
            tab.pack(side="left", padx=(0, round(16 * unit)))
            active = caption == current
            label = ttk.Label(tab, text=caption, style="TabActive.TLabel" if active else "Tab.TLabel")
            label.pack(padx=round(12 * unit), pady=(round(7 * unit), round(5 * unit)))
            tk.Frame(tab, height=round(3 * unit), background=TEAL if active else CARD).pack(fill="x")
            if not active:
                label.configure(cursor="hand2")
                label.bind("<Button-1>", lambda _, command=command: command())
                label.bind("<Enter>", lambda _, label=label: label.configure(style="TabHover.TLabel"))
                label.bind("<Leave>", lambda _, label=label: label.configure(style="Tab.TLabel"))
        tk.Frame(win, height=max(1, round(unit)), background=BORDER).pack(side="top", fill="x")
    return bar


def card(parent, title=None, **pack):
    """A white AMCF content box in `parent` (placed with `pack`), with an optional title. Returns its
    inside; with a title, inside.header is the title row, for things on its right."""
    unit = parent.winfo_fpixels("1i") / 96
    box = ttk.Frame(parent, style="Card.TFrame", padding=(round(16 * unit), round(12 * unit)))
    box.pack(**pack)
    inside = ttk.Frame(box)
    if title:
        inside.header = ttk.Frame(box)
        inside.header.pack(fill="x", pady=(0, 8))
        ttk.Label(inside.header, text=title, style="Title.TLabel").pack(side="left")
    inside.pack(fill="both", expand=True)
    return inside


def plot_tools(canvas, parent, background=PAGE):
    """matplotlib's toolbar (zoom, pan, coordinates under the mouse) for `canvas`, in `parent`, still to be placed."""
    tools = NavigationToolbar2Tk(canvas, parent, pack_toolbar=False)
    for widget in (tools, *tools.winfo_children()):
        try:
            widget.configure(background=background)
        except tk.TclError:                        # not every part has a background
            pass
    return tools


def photo(img, size):
    """BGR picture as a Tk image, scaled to fit `size` (width, height). PPM, so no Pillow needed."""
    h, w = img.shape[:2]
    scale = min(size[0] / w, size[1] / h)
    if abs(scale - 1) > 0.01:
        img = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))),
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return tk.PhotoImage(data=b"P6 %d %d 255\n" % rgb.shape[1::-1] + rgb.tobytes(), format="PPM")


def area_size(widget, default):
    """Size of `widget` on screen, or `default` before it is shown."""
    w, h = widget.winfo_width(), widget.winfo_height()
    return (w, h) if w > 20 and h > 20 else default


def x11(*command):
    """Output of an X11 tool (Linux), '' on Windows or when the tool is missing."""
    if sys.platform == "win32":
        return ""
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=3).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


@functools.cache
def monitor():
    """Linux: (x, y, width, height) of the primary monitor (else the first) from xrandr, None if unknown.
    Tk sees all monitors there as one wide screen (on Windows its screen is the main monitor)."""
    found = re.findall(r"^\s*\d+:\s+\+?(\*?)\S+\s+(\d+)/\d+x(\d+)/\d+\+(-?\d+)\+(-?\d+)",
                       x11("xrandr", "--listmonitors"), re.MULTILINE)
    if not found:
        return None
    _, w, h, x, y = max(found, key=lambda m: m[0])     # "*" = primary
    return int(x), int(y), int(w), int(h)


def screen(win):
    """(x, y, width, height) of the screen the windows go on."""
    return monitor() or (0, 0, win.winfo_screenwidth(), win.winfo_screenheight())


def fit_to_screen(win, size=None):
    """Fix the window's size (so it does not jump when a text changes): `size` or the size it asks for,
    at most 95% x 85% of the screen, near the top. Too big: the widget packed last (the picture) gives way."""
    win.update_idletasks()
    sx, sy, sw, sh = screen(win)
    w, h = size or (win.winfo_reqwidth(), win.winfo_reqheight())
    w, h = min(w, int(sw * 0.95)), min(h, int(sh * 0.85))
    win.geometry(f"{w}x{h}+{sx + (sw - w) // 2}+{sy + int(sh * 0.03)}")


def view_source(camera):
    """Where the view in use comes from, for display."""
    path = VIEW_DIR / f"{camera}.json"
    if not path.exists():
        return "defaults in dual_camera.py"
    return f"views/{camera}.json (saved {read_view(path)[1].get('saved', '?')})"


class Log:
    """Replaces sys.stdout and sys.stderr: what is printed, from any thread, goes to the console and to
    the log boxes of the open windows."""

    def __init__(self, console):
        self.console, self.queue, self.boxes = console, queue.Queue(), []

    def write(self, text):
        if self.console is not None:                   # None when started with pythonw
            try:
                self.console.write(text)
            except UnicodeEncodeError:                 # a console that lacks a character (e.g. output to a file)
                encoding = self.console.encoding or "ascii"
                self.console.write(text.encode(encoding, "replace").decode(encoding))
        self.queue.put(text)

    def flush(self):
        if self.console is not None:
            self.console.flush()

    def box(self, parent, height):
        """A new log box in `parent` (still to be placed), showing what is printed from now on."""
        frame = ttk.Frame(parent)
        bar = ttk.Scrollbar(frame, orient="vertical")
        box = tk.Text(frame, height=height, state="disabled", font=("Consolas", 9), wrap="word", relief="flat",
                      borderwidth=0, highlightthickness=0, background=CARD, foreground=TEXT, yscrollcommand=bar.set)
        bar.configure(command=box.yview)
        bar.pack(side="right", fill="y")
        box.pack(side="left", fill="both", expand=True)
        self.boxes.append(box)
        box.bind("<Destroy>", lambda _: box in self.boxes and self.boxes.remove(box))
        return frame

    def show(self):                                    # in the Tk thread: new text into the boxes
        text = ""
        while not self.queue.empty():
            text += self.queue.get()
        for box in self.boxes if text else ():
            box.configure(state="normal")
            box.insert("end", text)
            box.delete("1.0", "end-2000l")             # keep the last 2000 lines
            box.see("end")
            box.configure(state="disabled")


class App:
    """Main window: live view, pictures, recording, build plate view, optical settings; opens the calibration,
    alignment and areas windows."""

    def __init__(self, root):
        self.root = root
        self.log = Log(sys.stdout)
        sys.stdout = sys.stderr = self.log
        self.calls = queue.Queue()         # functions from worker threads, run in the Tk thread
        self.service = Service()           # the cameras, recording, areas (camera_service.py)
        self.cams = self.service.cams
        self.recording_shown = None        # what the record buttons show
        self.calibration = None            # the calibration window, while open
        self.alignment = None              # the alignment window, while open
        self.areas = None                  # the areas window, while open
        self.shown = -1                    # cams.count of the picture on screen
        self.rate = []                     # (time, cams.count), last few seconds: pictures per second
        self.chips = []                    # per window: its state chip labels (add_chips)
        self.chips_shown = None            # what they show

        root.title("RAPTURE cameras")
        self.add_chips(toolbar(root, "Live", self.tabs()))
        page = ttk.Frame(root, style="Page.TFrame", padding=12)
        page.pack(fill="both", expand=True)
        self.log.box(card(page, "Log", side="bottom", fill="x", pady=(12, 0)), 7).pack(fill="x")
        controls = card(page, side="bottom", fill="x", pady=(12, 0))
        buttons = ttk.Frame(controls)
        buttons.pack(fill="x")
        self.clip_button = ttk.Button(buttons, command=lambda: self.record(False))
        self.lapse_button = ttk.Button(buttons, command=lambda: self.record(True))
        self.plate_button = ttk.Button(buttons, text="Build plate view (space)", command=self.plate)
        for button in (ttk.Button(buttons, text="Take picture (P)", style="Main.TButton", command=self.take_picture),
                       self.clip_button, self.lapse_button, self.plate_button,
                       ttk.Button(buttons, text="Optical camera settings (S)", style="Minor.TButton",
                                  command=self.optical_settings)):
            button.pack(side="left", padx=(0, 8))
        self.views = ttk.Label(controls, style="Muted.TLabel")
        self.views.pack(fill="x", pady=(10, 0))
        live = card(page, "Live view", fill="both", expand=True)
        self.status = ttk.Label(live.header, text="Starting the cameras ...", style="Muted.TLabel")
        self.status.pack(side="right")
        self.preview_area = ttk.Frame(live, width=PREVIEW_SIZE[0], height=PREVIEW_SIZE[1])
        self.preview_area.pack(fill="both", expand=True)
        self.preview_area.pack_propagate(False)
        self.preview = ttk.Label(self.preview_area, anchor="center", text="Starting the cameras ...")
        self.preview.pack(fill="both", expand=True)

        for key, action in (("p", self.take_picture), ("r", lambda: self.record(False)),
                            ("t", lambda: self.record(True)), ("<space>", self.plate), ("s", self.optical_settings)):
            root.bind(key, lambda _, action=action: action())
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.update_buttons()
        self.views_changed()
        fit_to_screen(root)
        self.server = serve(self.service)  # the same HTTP API as camera_service.py, while the app runs
        self.service.start()
        self.tick()

    def tick(self):
        while not self.calls.empty():
            self.calls.get()()
        self.log.show()
        latest, count = self.cams.latest, self.cams.count
        if latest is not None and count != self.shown:
            self.shown = count
            frame, recorder = latest[1], self.cams.recorder
            if recorder is not None:
                frame = frame.copy()
                recorder.mark(frame)
            self.preview_photo = photo(frame, area_size(self.preview_area, PREVIEW_SIZE))
            self.preview.configure(image=self.preview_photo, text="")
            now = time.monotonic()
            self.rate = [(t, c) for t, c in self.rate if t > now - 5] + [(now, count)]
            if len(self.rate) > 1:
                (t0, c0), (t1, c1) = self.rate[0], self.rate[-1]
                self.status.configure(text=f"{(c1 - c0) / (t1 - t0):.1f} pictures/s")
        if not self.cams.is_alive():
            self.status.configure(text="Cameras stopped (see log)")
        state = self.service.state()
        self.update_chips(state)
        if state["recording"] != self.recording_shown:     # also started / stopped through the API
            self.update_buttons()
        if self.areas is not None:
            self.areas.poll(state)
        self.root.after(30, self.tick)

    def tabs(self):
        """The tab bar of every window: one tab per window, in the order of the work."""
        return [("Live", self.show_live), ("Calibration", self.open_calibration),
                ("Alignment", self.open_alignment), ("Areas", self.open_areas)]

    def show_live(self):
        self.root.deiconify()
        self.root.lift()

    def add_chips(self, bar):
        """State chips (thermal camera, optical camera, recording) in the control bar `bar` of a window."""
        chips = [ttk.Label(bar.chips, style="Navy.TLabel") for _ in range(3)]
        for chip in chips:
            chip.pack(side="left", padx=(0, round(8 * bar.winfo_fpixels("1i") / 96)))
        self.chips.append(chips)
        self.chips_shown = None                        # show them at the next tick

    def update_chips(self, state):
        """The state chips of all windows from service.state(): colour + symbol + text (never colour alone)."""
        look = {"live": ("ok", "✓"), "calibrating": ("info", "i"), "starting": ("off", "–"),
                "not connected": ("off", "–"), "stopped": ("err", "!")}
        states = tuple((look[state[name]][0], f"{look[state[name]][1]}  {name.upper()} {state[name].upper()}")
                       for name in CAMERAS)
        states += (("ok", f"●  REC {state['recording'].upper()}") if state["recording"] else None,)
        if states == self.chips_shown:
            return
        self.chips_shown = states
        self.chips = [chips for chips in self.chips if chips[0].winfo_exists()]
        for chips in self.chips:
            for chip, shown in zip(chips, states):
                chip.configure(style=f"Chip{shown[0]}.TLabel" if shown else "Navy.TLabel", text=shown[1] if shown else "")

    def work(self, job, then):
        """job() in a worker thread, then then(result) in the Tk thread; result None if job failed."""
        def run():
            try:
                result = job()
            except Exception:
                traceback.print_exc()
                result = None
            self.calls.put(lambda: then(result))
        threading.Thread(target=run, daemon=True).start()

    def take_picture(self):
        """The current moment as pictures in pictures/ (see Service.take_picture). Returns the path or None."""
        return self.service.take_picture()

    def record(self, timelapse):
        """Start a clip / timelapse, or stop the one being recorded."""
        self.service.record(None if self.cams.recorder is not None else timelapse)
        self.update_buttons()

    def update_buttons(self):
        recorder = self.cams.recorder
        self.recording_shown = None if recorder is None else "timelapse" if recorder.timelapse else "clip"
        for button, timelapse, name in ((self.clip_button, False, "clip (R)"), (self.lapse_button, True, "timelapse (T)")):
            mine = recorder is not None and recorder.timelapse == timelapse
            button.configure(text=f"■  Stop {name}" if mine else f"●  Record {name}",
                             style="Danger.TButton" if mine else "TButton")
            button.state(["disabled"] if recorder is not None and not mine else ["!disabled"])

    def plate(self, pair=None, optical_view=None):
        """`pair` (default: the current moment) on the build plate, in its own window, through the views in
        use or `optical_view` for the optical picture. The resampling runs in the background."""
        if pair is None and self.cams.latest is not None:
            pair = self.cams.latest[0]
        if pair is None or self.plate_button.instate(["disabled"]):
            return
        note = ("   (optical view of the alignment window, not saved)"
                if optical_view not in (None, dual_camera.OPTICAL_VIEW) else "")
        self.plate_button.state(["disabled"])
        print("Build plate view: putting the pictures on the plate (takes a few seconds) ...")
        self.work(lambda: plate_layers(pair, optical_view), lambda layers: self.plate_window(pair, layers, note))

    def plate_window(self, pair, layers, note=""):
        self.plate_button.state(["!disabled"])
        if layers is None:
            return
        win = tk.Toplevel(self.root)
        when = f"{datetime.fromtimestamp(pair.timestamp):%Y-%m-%d %H:%M:%S}"
        win.title(f"RAPTURE cameras: build plate {when}")
        toolbar(win, title=f"Build plate   {when}{note}")
        fig = Figure(figsize=(9, 8), facecolor=CARD)
        canvas = FigureCanvasTkAgg(fig, win)
        plot_tools(canvas, win).pack(side="bottom", fill="x")   # zoom / pan; x', y', r, angle, deg C under the mouse
        canvas.get_tk_widget().pack(fill="both", expand=True)
        draw_plate(fig, pair, layers)
        fit_to_screen(win, (int(screen(win)[3] * 0.8),) * 2)    # square, as the plate
        canvas.draw()
        canvas.get_tk_widget().focus_set()         # space switches which camera lies on top

    def optical_settings(self):
        if dual_camera.WINDOWS:                  # Linux: a program of its own, the live view keeps running
            print("Optical camera settings window: the live view pauses until it is closed")
        self.service.optical_settings()

    def open_calibration(self):
        if self.calibration is None:
            self.calibration = Calibration(self)
        else:
            self.calibration.win.deiconify()
            self.calibration.win.lift()

    def open_alignment(self):
        if self.alignment is None:
            self.alignment = Alignment(self)
        else:
            self.alignment.win.deiconify()
            self.alignment.win.lift()

    def open_areas(self):
        if self.areas is None:
            self.areas = Areas(self)
        else:
            self.areas.win.deiconify()
            self.areas.win.lift()

    def views_changed(self):
        """A view was put in use (Save in the calibration or alignment window)."""
        self.views.configure(text=f"Views in use:  thermal = {view_source('thermal')},  "
                                  f"optical = {view_source('optical')}")
        for window in (self.calibration, self.alignment, self.areas):
            if window is not None:
                window.views_changed()

    def close(self):
        if self.server is not None:
            self.server.shutdown()
        self.service.close()
        sys.stdout = sys.stderr = self.log.console
        self.root.destroy()


class Calibration:
    """Calibration window: checkerboard pictures, fit, where the corners were found, views to put in use."""

    def __init__(self, app):
        self.app = app
        self.win = win = tk.Toplevel(app.root)
        win.title("RAPTURE cameras: calibration")
        self.which = tk.StringVar(value="both")
        self.square = tk.StringVar(value=f"{BOARD_SQUARE:g}")    # mm, checkerboard square side
        self.folder = None                 # folder of the pictures; None = a new one with the next picture
        self.index = 0                     # which of self.pictures() is shown
        self.loaded = (None, None, None)   # (path, mtime, picture) last loaded, for redraws
        self.pending = {}                  # camera -> View: the one in use, or a calibration result not saved yet
        self.info = {}                     # camera -> what goes into the JSON with it (calibration report)
        self.fields = {}                   # camera -> StringVars "x", "y", "omega"
        self.base = {}                     # camera -> the view in use when its panel was last reset
        self.labels = {}                   # camera -> labels "source", "size", "dist", "fit", "state"

        app.add_chips(toolbar(win, "Calibration", app.tabs()))
        page = ttk.Frame(win, style="Page.TFrame", padding=12)
        page.pack(fill="both", expand=True)
        top = card(page, fill="x", pady=(0, 12))
        ttk.Label(top, text="Cameras").pack(side="left", padx=(0, 10))
        for value in ("both", *CAMERAS):
            ttk.Radiobutton(top, text=value.capitalize(), value=value, variable=self.which,
                            command=self.chosen).pack(side="left")
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=12)
        ttk.Label(top, text="Checkerboard square").pack(side="left")
        ttk.Entry(top, textvariable=self.square, width=6).pack(side="left", padx=6)
        ttk.Label(top, text="mm").pack(side="left")
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=12)
        self.buttons = [ttk.Button(top, text="Take picture (C)", command=self.take),
                        ttk.Button(top, text="Calibrate", style="Main.TButton", command=self.calibrate),
                        ttk.Button(top, text="New folder", style="Minor.TButton", command=self.new_folder),
                        ttk.Button(top, text="Open folder …", style="Minor.TButton", command=self.open_folder)]
        for button in self.buttons:
            button.pack(side="left", padx=(0, 8))
        self.folder_label = ttk.Label(top, style="Muted.TLabel")
        self.folder_label.pack(side="left", padx=8)

        self.log_box = app.log.box(card(page, "Log", side="bottom", fill="x", pady=(12, 0)), 6)
        self.log_box.pack(fill="x")
        side = ttk.Frame(page, style="Page.TFrame")         # the two view panels behind pill tabs (both are too tall)
        side.pack(side="right", fill="y", padx=(12, 0))
        pills = ttk.Frame(side, style="Page.TFrame")
        pills.pack(fill="x", pady=(0, 10))
        self.pills, self.panels = {}, {}
        for camera in CAMERAS:
            self.pills[camera] = ttk.Button(pills, text=f"{camera.capitalize()} view",
                                            command=lambda camera=camera: self.show_panel(camera))
            self.pills[camera].pack(side="left", padx=(0, 10))
            self.view_panel(side, camera)
        self.show_panel("thermal")
        viewer = card(page, "Pictures", fill="both", expand=True)
        self.picture_label = ttk.Label(viewer.header, style="Muted.TLabel")
        self.picture_label.pack(side="left", padx=12)
        nav = ttk.Frame(viewer)
        nav.pack(side="bottom", fill="x", pady=(8, 0))
        ttk.Button(nav, text="◀", width=3, style="Minor.TButton", command=lambda: self.step(-1)).pack(side="left")
        ttk.Button(nav, text="▶", width=3, style="Minor.TButton", command=lambda: self.step(1)).pack(side="left")
        ttk.Button(nav, text="Delete this picture", style="Delete.TButton", command=self.delete).pack(side="right")
        width = int(screen(win)[2] * PICTURE_SHARE)
        self.picture_size = (width, width * 3 // 4)       # 4:3 like both cameras
        self.picture_area = ttk.Frame(viewer, width=self.picture_size[0], height=self.picture_size[1])
        self.picture_area.pack(fill="both", expand=True)
        self.picture_area.pack_propagate(False)
        self.picture = ttk.Label(self.picture_area, anchor="center", justify="center")
        self.picture.pack(fill="both", expand=True)
        self.picture_area.bind("<Configure>", lambda _: self.show())

        for key, action in (("c", self.take), ("<Left>", lambda: self.step(-1)), ("<Right>", lambda: self.step(1))):
            win.bind(key, lambda event, action=action: isinstance(event.widget, tk.Entry) or action())
        win.protocol("WM_DELETE_WINDOW", self.close)
        fit_to_screen(win)
        print("Calibration: lay the checkerboard flat in the print plane, covering as much of the picture as "
              "possible; take pictures (c) with the board moved/turned a little in between, then Calibrate.")
        self.show()

    def show_panel(self, camera):
        """The view panel of `camera` (the other one waits behind its pill tab)."""
        for name, panel in self.panels.items():
            panel.pack_forget()
            self.pills[name].configure(style="PillActive.TButton" if name == camera else "Pill.TButton")
        self.panels[camera].pack(fill="x")

    def chosen(self):                                  # cameras chosen: their pictures, their panel
        if self.which.get() in CAMERAS:
            self.show_panel(self.which.get())
        self.show()

    def view_panel(self, parent, camera):
        box = card(parent, fill="x")
        self.panels[camera] = box.master
        labels = self.labels[camera] = {}
        for row, key in enumerate(("source", "size", "dist", "fit")):
            labels[key] = ttk.Label(box, wraplength="10c", justify="left",
                                    style="Muted.TLabel" if key in ("source", "fit") else "TLabel")
            labels[key].grid(row=row, column=0, columnspan=5, sticky="w", pady=1)
        fields = self.fields[camera] = {key: tk.StringVar() for key in ("x", "y", "omega")}
        for row, items in ((4, (("center x'", "x"), ("y'", "y"), "mm")), (5, (("omega", "omega"), "deg"))):
            column = 0
            for item in items:
                if isinstance(item, str):
                    ttk.Label(box, text=item).grid(row=row, column=column, sticky="w")
                    continue
                ttk.Label(box, text=item[0]).grid(row=row, column=column, sticky="w", padx=(0, 6), pady=(8, 0))
                ttk.Entry(box, textvariable=fields[item[1]], width=13).grid(row=row, column=column + 1,
                                                                             padx=(0, 10), pady=(8, 0))
                column += 2
        labels["state"] = ttk.Label(box, style="Warn.TLabel", wraplength="10c")
        labels["state"].grid(row=6, column=0, columnspan=5, sticky="w", pady=(8, 0))
        buttons = ttk.Frame(box)
        buttons.grid(row=7, column=0, columnspan=5, sticky="w", pady=(8, 0))
        ttk.Button(buttons, text="Save (put in use)", command=lambda: self.save(camera)).pack(side="left")
        ttk.Button(buttons, text="Revert", style="Minor.TButton", command=lambda: self.revert(camera)).pack(
            side="left", padx=8)
        for var in fields.values():
            var.trace_add("write", lambda *_: self.update_state(camera))
        self.revert(camera)

    def in_use(self, camera):
        return getattr(dual_camera, f"{camera.upper()}_VIEW")

    def revert(self, camera):
        """Back to the view in use: drops an unsaved calibration result and edits of center/omega."""
        view, path = self.in_use(camera), VIEW_DIR / f"{camera}.json"
        self.pending[camera] = self.base[camera] = view
        self.info[camera] = {k: v for k, v in read_view(path)[1].items() if k != "saved"} if path.exists() else {}
        fields = self.fields[camera]
        fields["x"].set(str(view.center[0]))
        fields["y"].set(str(view.center[1]))
        fields["omega"].set(str(view.omega))
        self.update_panel(camera)

    def update_panel(self, camera):
        view, info, labels = self.pending[camera], self.info[camera], self.labels[camera]
        labels["source"].configure(text=f"In use: {view_source(camera)}")
        labels["size"].configure(text=f"size {view.size[0]:.3f} x {view.size[1]:.3f} mm  (from calibration)")
        labels["dist"].configure(text="dist k1, k2, p1, p2, k3: " + ", ".join(f"{d:.6g}" for d in view.dist))
        if "rms_px" in info:
            square = f"{info['square_mm']:g}" if "square_mm" in info else "?"    # not stored before 2026-11-17
            labels["fit"].configure(text=f"fit: error rms {info['rms_px']} px = {info['rms_mm']} mm, max "
                                         f"{info['max_px']} px; {info['corners']} corners in {info['pictures_used']} "
                                         f"of {info['pictures']} pictures ({info.get('folder', '?')}, {square} mm "
                                         f"squares); corners reach {info['reach']:.0%} of the way to the picture corners")
        else:
            labels["fit"].configure(text="fit: not calibrated yet")
        self.update_state(camera)

    def edited(self, camera):
        """The pending view with center and omega from the fields, or None if they are not numbers."""
        fields = self.fields[camera]
        try:
            return replace(self.pending[camera], center=(float(fields["x"].get()), float(fields["y"].get())),
                           omega=float(fields["omega"].get()))
        except ValueError:
            return None

    def update_state(self, camera):
        view = self.edited(camera)
        if view is None:
            text = "center and omega must be numbers"
        elif view == self.in_use(camera):
            text = ""
        elif self.pending[camera] != self.in_use(camera):
            text = "New calibration, not saved yet: check the corners found, then Save"
        else:
            text = "center / omega changed, not saved yet"
        self.labels[camera]["state"].configure(text=text)
        if camera in self.pills:                       # also seen while the panel is behind its tab
            self.pills[camera].configure(text=f"{camera.capitalize()} view" + ("  · not saved" if text else ""))

    def views_changed(self):
        """Views were put in use (here or in the alignment window): panels without unsaved changes follow."""
        for camera in CAMERAS:
            if self.edited(camera) == self.base[camera]:
                self.revert(camera)
            else:
                self.update_state(camera)

    def save(self, camera):
        view = self.edited(camera)
        if view is None:
            messagebox.showerror("Calibration", f"{camera}: center and omega must be numbers", parent=self.win)
            return
        save_view(camera, view, **self.info[camera])
        print(f"{camera.upper()}_VIEW in use from now on (views/{camera}.json): {view}")
        self.revert(camera)
        self.app.views_changed()

    def cameras(self):
        return CAMERAS if self.which.get() == "both" else (self.which.get(),)

    def pictures(self, cameras=None):
        """(number, camera, file to show) of the pictures in the folder, of `cameras` (default: selected)."""
        found = []
        for camera in cameras or self.cameras():
            raw = "npy" if camera == "thermal" else "png"
            for f in self.folder.glob(f"{camera}_*.{raw}") if self.folder else ():
                n = f.stem[len(camera) + 1:]
                if n.isdigit():
                    corners = f.with_name(f"{camera}_{n}_corners.png")
                    plain = f.with_name(f"thermal_{n}_colour.png") if camera == "thermal" else f
                    found.append((int(n), camera, corners if corners.exists() else plain))
        return sorted(found)

    def show(self):
        """The picture at self.index (with the corners found, once calibrated) and the folder name."""
        self.folder_label.configure(text=f"Folder: {self.folder.name}" if self.folder else
                                    "Folder: a new one with the next picture")
        pictures = self.pictures()
        if not pictures:
            self.picture.configure(image="", text="No pictures yet.\n\nLay the checkerboard flat in the print plane,"
                                                  "\nthen Take picture (c).")
            self.picture_label.configure(text="")
            return
        self.index = max(0, min(self.index, len(pictures) - 1))
        n, camera, path = pictures[self.index]
        mtime = path.stat().st_mtime if path.exists() else None
        if self.loaded[:2] != (path, mtime):
            self.loaded = (path, mtime, cv2.imread(str(path)) if mtime else None)
        if self.loaded[2] is None:
            self.picture.configure(image="", text=f"Cannot show {path.name}")
        else:
            self.picture_photo = photo(self.loaded[2], area_size(self.picture_area, self.picture_size))
            self.picture.configure(image=self.picture_photo, text="")
        found = "corners found: green, fitted: red" if path.name.endswith("_corners.png") else "not calibrated yet"
        self.picture_label.configure(text=f"{self.index + 1} of {len(pictures)}:  {camera} picture {n}  ({found})")

    def step(self, by):
        count = len(self.pictures())
        if count:
            self.index = (self.index + by) % count
            self.show()

    def take(self):
        latest = self.app.cams.latest
        if latest is None:
            print("No picture from the cameras yet")
            return
        if self.folder is None:
            self.folder = CALIBRATION_DIR / f"{datetime.now():%Y-%m-%d_%H-%M-%S}"
        number = max((n for n, _, _ in self.pictures(CAMERAS)), default=0) + 1
        stored = save_board_picture(self.folder, number, latest[0], self.cameras())
        if not stored:
            print("No picture taken: the selected camera(s) deliver nothing, or the thermal camera still calibrates")
            return
        print(f"Picture {number} ({' + '.join(stored)}) saved in {self.folder}")
        self.index = next(i for i, (n, _, _) in enumerate(self.pictures()) if n == number)
        self.show()

    def delete(self):
        pictures = self.pictures()
        if not pictures:
            return
        n, camera, _ = pictures[self.index]
        if not messagebox.askyesno("Calibration", f"Delete {camera} picture {n} from {self.folder.name}?",
                                   parent=self.win):
            return
        names = ((f"thermal_{n}.npy", f"thermal_{n}_colour.png", f"thermal_{n}_corners.png") if camera == "thermal"
                 else (f"optical_{n}.png", f"optical_{n}_corners.png"))
        for name in names:
            (self.folder / name).unlink(missing_ok=True)
        print(f"Deleted {camera} picture {n} from {self.folder.name}")
        self.show()

    def new_folder(self):
        self.folder, self.index = None, 0
        self.show()

    def open_folder(self):
        chosen = filedialog.askdirectory(parent=self.win, title="Folder with calibration pictures",
                                         initialdir=CALIBRATION_DIR if CALIBRATION_DIR.exists() else None)
        if not chosen:
            return
        self.folder, self.index = Path(chosen), 0
        for camera in CAMERAS:                         # an earlier calibration result in it: ready to Save
            if (self.folder / f"{camera}.json").exists():
                view, info = read_view(self.folder / f"{camera}.json")
                self.pending[camera] = view
                self.info[camera] = {k: v for k, v in info.items() if k != "saved"}
                self.update_panel(camera)
                print(f"{camera}: calibration result from {self.folder.name} loaded (Save to put it in use)")
                if "square_mm" in info:                # the board these pictures were taken of
                    self.square.set(f"{info['square_mm']:g}")
        self.show()

    def calibrate(self):
        cameras = [c for c in self.cameras() if self.pictures((c,))]
        if not cameras:
            print("No pictures to calibrate in this folder")
            return
        try:
            square = float(self.square.get())
        except ValueError:
            square = 0
        if not square > 0:
            messagebox.showerror("Calibration", "Checkerboard square: the side of one square in mm, e.g. 5",
                                 parent=self.win)
            return
        for button in self.buttons:
            button.state(["disabled"])
        print(f"Calibrating {' and '.join(cameras)} from {self.folder.name} ({square:g} mm squares) ...")
        folder = self.folder
        self.app.work(lambda: calibrate_folder(folder, cameras, square), self.calibrated)

    def calibrated(self, results):
        if not self.win.winfo_exists():                # window closed meanwhile
            return
        for button in self.buttons:
            button.state(["!disabled"])
        for camera, (view, report) in (results or {}).items():
            self.pending[camera] = view                # center/omega: the fields' values are used on Save
            self.info[camera] = report
            self.update_panel(camera)
        if len(results or {}) == 1:
            self.show_panel(next(iter(results)))
        self.show()

    def close(self):
        self.app.calibration = None
        self.win.destroy()


class Alignment:
    """Alignment window: a thermal and an optical picture of one moment, points clicked where both show the
    same feature, and the optical view's center and omega (optionally its zoom) fitted so that those points
    land where the thermal camera puts them. The thermal view is the reference and stays as it is."""

    OVERLAY = 1.5                      # overlay pixels per thermal pixel

    def __init__(self, app):
        self.app = app
        self.win = win = tk.Toplevel(app.root)
        win.title("RAPTURE cameras: alignment")
        self.pair = self.path = None       # the pair shown, its <time>.png in pictures/
        self.points = []                   # per point [thermal (u, v), optical (u, v)] px, None until clicked
        self.selected = None               # index of the point selected in the list: clicks move it
        self.zoom = 1.0                    # optical size / size in use, from a fit with "fit the zoom too"
        self.base = None                   # the fields' texts when they were last reset to the view in use
        self.grid = None                   # plate points (x', y') of the overlay pixels (change with the thermal view)
        self.overlaid = None               # the optical view the overlay was made with
        self.layer = None                  # the optical picture in the overlay
        self.marks = []                    # points, outlines, plate edges drawn over the pictures
        self.job = None                    # redraw due after typing in the fields
        self.unit = win.winfo_fpixels("1i") / 96
        self.fields = {key: tk.StringVar() for key in ("x", "y", "omega")}
        self.fit_zoom = tk.BooleanVar(value=False)
        self.blend = tk.DoubleVar(value=0.5)

        app.add_chips(toolbar(win, "Alignment", app.tabs()))
        page = ttk.Frame(win, style="Page.TFrame", padding=12)
        page.pack(fill="both", expand=True)
        bottom = ttk.Frame(page, style="Page.TFrame")
        bottom.pack(side="bottom", fill="x", pady=(12, 0))
        self.view_panel(card(bottom, "Optical view", side="left", fill="y"))
        self.points_panel(card(bottom, "Points", side="left", fill="y", padx=(12, 0)))
        app.log.box(card(bottom, "Log", side="left", fill="both", expand=True, padx=(12, 0)), 8).pack(
            fill="both", expand=True)
        box = card(page, "Thermal and optical at one moment", fill="both", expand=True)
        self.picture_label = ttk.Label(box.header, style="Muted.TLabel")
        self.picture_label.pack(side="left", padx=12)
        for text, style, command in (("Build plate view (space)", "Minor.TButton", self.plate),
                                     ("Open picture …", "Minor.TButton", self.open_picture),
                                     ("Take picture (P)", "TButton", self.take)):
            ttk.Button(box.header, text=text, style=style, command=command).pack(side="right", padx=(8, 0))
        below = ttk.Frame(box)
        below.pack(side="bottom", fill="x")
        self.fig = Figure(facecolor=CARD)
        self.canvas = FigureCanvasTkAgg(self.fig, box)
        self.tools = plot_tools(self.canvas, below, CARD)
        self.tools.pack(side="left")
        ttk.Label(below, text="optical").pack(side="right")
        ttk.Scale(below, from_=0, to=1, variable=self.blend, length=round(160 * self.unit),
                  command=lambda _: self.blended()).pack(side="right", padx=8)
        ttk.Label(below, text="Overlay:  thermal").pack(side="right")
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("button_press_event", self.click)

        for key, action in (("p", self.take), ("f", self.fit), ("<space>", self.plate), ("<Delete>", self.delete),
                            ("<Escape>", lambda: self.table.selection_set(()))):
            win.bind(key, lambda event, action=action: isinstance(event.widget, tk.Entry) or action())
        win.protocol("WM_DELETE_WINDOW", self.close)
        fit_to_screen(win, screen(win)[2:])    # as large as it may be
        print("Alignment: take a picture (p) of things both cameras see where their pictures overlap (dashed "
              "outline), e.g. small warm objects with sharp edges. Click a feature in the thermal picture, then "
              "the same one in the optical picture; 3 or more points, spread out, then Fit (f). Zoom in with the "
              "toolbar's magnifier (click it again to get back to clicking). Right click: delete a point; select "
              "one in the list to move it (Esc: done). Save puts the optical view in use.")
        self.draw()
        self.revert()

    def view_panel(self, box):
        ttk.Label(box, text="Moved to fit the thermal view (the reference)", style="Muted.TLabel").grid(
            row=0, column=0, columnspan=5, sticky="w")
        for row, column, text, key in ((1, 0, "center x'", "x"), (1, 2, "y'", "y"), (2, 0, "omega", "omega")):
            ttk.Label(box, text=text).grid(row=row, column=column, sticky="w", padx=(0, 6), pady=(8, 0))
            ttk.Entry(box, textvariable=self.fields[key], width=11).grid(row=row, column=column + 1, sticky="w",
                                                                         padx=(0, 10), pady=(8, 0))
            self.fields[key].trace_add("write", lambda *_: self.later())
        ttk.Label(box, text="mm").grid(row=1, column=4, sticky="w", pady=(8, 0))
        ttk.Label(box, text="deg").grid(row=2, column=2, sticky="w", pady=(8, 0))
        self.size_label = ttk.Label(box)
        self.size_label.grid(row=3, column=0, columnspan=5, sticky="w", pady=(8, 0))
        ttk.Checkbutton(box, text="Fit the zoom (size) too", variable=self.fit_zoom).grid(
            row=4, column=0, columnspan=5, sticky="w", pady=(4, 0))
        self.state = ttk.Label(box, style="Warn.TLabel", wraplength="9c")
        self.state.grid(row=5, column=0, columnspan=5, sticky="w", pady=(8, 0))
        buttons = ttk.Frame(box)
        buttons.grid(row=6, column=0, columnspan=5, sticky="w", pady=(8, 0))
        ttk.Button(buttons, text="Fit (F)", style="Main.TButton", command=self.fit).pack(side="left")
        ttk.Button(buttons, text="Save (put in use)", command=self.save).pack(side="left", padx=8)
        ttk.Button(buttons, text="Revert", style="Minor.TButton", command=self.revert).pack(side="left")

    def points_panel(self, box):
        frame = ttk.Frame(box)
        frame.pack(fill="both", expand=True)
        self.table = ttk.Treeview(frame, columns=("thermal", "optical", "apart"), height=6, selectmode="browse")
        for column, text, width in (("#0", "#", 45), ("thermal", "thermal u, v px", 125),
                                    ("optical", "optical u, v px", 135), ("apart", "apart mm", 75)):
            self.table.heading(column, text=text, anchor="w")
            self.table.column(column, width=round(width * self.unit), stretch=False, anchor="w")
        bar = ttk.Scrollbar(frame, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        self.table.pack(side="left", fill="both", expand=True)
        self.table.bind("<<TreeviewSelect>>", lambda _: self.select())
        self.summary = ttk.Label(box, style="Muted.TLabel", wraplength="10c", justify="left")
        self.summary.pack(fill="x", pady=(6, 0))
        buttons = ttk.Frame(box)
        buttons.pack(fill="x", pady=(6, 0))
        ttk.Button(buttons, text="Delete point (Del)", style="Delete.TButton", command=self.delete).pack(side="left")
        ttk.Button(buttons, text="Delete all", style="Delete.TButton", command=self.delete_all).pack(side="left", padx=8)

    # The pair --------------------------------------------------------------------------------------------
    def take(self):
        path = self.app.take_picture()
        if path is not None:
            self.open(path)

    def open_picture(self):
        chosen = filedialog.askopenfilename(parent=self.win, title="Picture (any of its files)",
                                            initialdir=PICTURE_DIR if PICTURE_DIR.exists() else None,
                                            filetypes=[("Pictures", "*.png *.npy *.json"), ("All files", "*.*")])
        if chosen:
            self.open(Path(chosen))

    def open(self, path):
        stem = path.stem.removesuffix("_thermal").removesuffix("_optical").removesuffix("_points")
        self.path = path.with_name(f"{stem}.png")
        self.pair = load_pair(self.path)
        missing = [c for c, live in zip(CAMERAS, (self.pair.thermal_live, self.pair.optical_live)) if not live]
        if missing:
            print(f"{self.path.name}: no {' and no '.join(missing)} picture; aligning needs both")
        points = self.points_file()
        self.points = ([[tuple(p) if p else None for p in point] for point in json.loads(points.read_text(encoding="utf-8"))]
                       if points.exists() else [])
        self.selected = None
        self.picture_label.configure(text=f"{self.path.parent.name}/{self.path.name}")
        print(f"Opened {self.path.name}" + (f" with the {len(self.points)} points clicked earlier" if self.points else ""))
        self.draw()

    def points_file(self):
        return self.path.with_name(f"{self.path.stem}_points.json")

    def draw(self):
        """The pictures of self.pair, from scratch (after opening one)."""
        self.fig.clear()
        self.marks, self.grid, self.overlaid, self.layer = [], None, None, None
        axes = self.fig.subplots(1, 3)
        self.axes = dict(zip(("thermal", "optical", "overlay"), axes))
        self.fig.subplots_adjust(left=0.005, right=0.995, bottom=0.01, top=0.9, wspace=0.03)
        for ax in axes:
            ax.set_axis_off()
        self.tools.update()                            # forget the zoom history
        if self.pair is None:
            axes[1].text(0.5, 0.5, "Take a picture (p) of something both cameras see,\nor open one taken earlier.",
                         ha="center", va="center", color=MUTED, transform=axes[1].transAxes)
            self.canvas.draw_idle()
            return
        pair, (h, w) = self.pair, self.pair.thermal.shape[:2]
        if np.isfinite(pair.thermal_c).any():
            lo, hi = np.nanpercentile(pair.thermal_c, (1, 99))
            thermal, title = dict(X=pair.thermal_c, cmap="inferno", vmin=lo, vmax=hi), f"{lo:.1f} to {hi:.1f} °C"
        else:                                          # false colours: calibrating then, or missing
            thermal = dict(X=pair.thermal[..., ::-1])
            title = "calibrating, no temperatures" if pair.thermal_live else "no picture"
        axes[0].imshow(**thermal)
        axes[1].imshow(pair.optical[..., ::-1])
        axes[2].imshow(**thermal)
        self.layer = axes[2].imshow(np.zeros((1, 1, 4), np.uint8), extent=(-0.5, w - 0.5, h - 0.5, -0.5),
                                    alpha=self.blend.get())
        for ax, text, camera in zip(axes, (f"Thermal (reference), {title}", "Optical",
                                           "Optical over thermal (blend: slider below)"), ("thermal", "optical", "thermal")):
            ax.set_title(text, fontsize=10, loc="left")
            ax.set_autoscale_on(False)                 # lines past the picture edge must not widen the view
            ax.format_coord = lambda u, v, camera=camera: self.where(camera, u, v)
        self.update()
        # The overlay zoomed onto the part both cameras see, in a box shaped like the picture (keeps the panel size).
        seen = np.argwhere(np.asarray(self.layer.get_array())[..., 3] > 0) if pair.optical_live else []
        if len(seen):
            (v0, u0), (v1, u1) = (seen.min(0) + 0.5) / self.OVERLAY - 0.5, (seen.max(0) + 0.5) / self.OVERLAY - 0.5
            du, dv = 1.1 * (u1 - u0), 1.1 * (v1 - v0)
            du, dv = max(du, dv * w / h, 40), max(dv, du * h / w)
            if du < 0.9 * w:
                cu = np.clip((u0 + u1) / 2, du / 2 - 0.5, w - 0.5 - du / 2)
                cv = np.clip((v0 + v1) / 2, dv / 2 - 0.5, h - 0.5 - dv / 2)
                axes[2].set(xlim=(cu - du / 2, cu + du / 2), ylim=(cv + dv / 2, cv - dv / 2))
                self.canvas.draw_idle()

    def where(self, camera, u, v):
        """Status bar text for the mouse at pixel (u, v) of the `camera` picture."""
        view = dual_camera.THERMAL_VIEW if camera == "thermal" else self.edited() or dual_camera.OPTICAL_VIEW
        x, y = (float(c) for c in view.to_plate(u, v, getattr(self.pair, camera).shape))
        temp = float(_sample(self.pair.thermal_c, *dual_camera.THERMAL_VIEW.to_pixel(x, y, self.pair.thermal_c.shape)))
        return f"u={u:.1f} v={v:.1f} px   x'={x:.2f} y'={y:.2f} mm" + (f"   {temp:.1f} °C" if np.isfinite(temp) else "")

    # The view --------------------------------------------------------------------------------------------
    def edited(self):
        """The optical view in use with center and omega from the fields and the fitted zoom; None if the
        fields are not numbers."""
        use = dual_camera.OPTICAL_VIEW
        try:
            center = float(self.fields["x"].get()), float(self.fields["y"].get())
            omega = float(self.fields["omega"].get())
        except ValueError:
            return None
        return replace(use, center=center, omega=omega,
                       size=use.size if self.zoom == 1 else tuple(round(s * self.zoom, 4) for s in use.size))

    def apart(self, view):
        """Per point: mm between where the thermal camera and (through `view`) the optical camera see it on
        the plate; None if it is not clicked in both pictures yet."""
        return [float(np.hypot(*np.subtract(dual_camera.THERMAL_VIEW.to_plate(*t, self.pair.thermal.shape),
                                             view.to_plate(*o, self.pair.optical.shape))))
                if t and o else None for t, o in self.points]

    def fit(self):
        both = [(t, o) for t, o in self.points if t and o]
        if not both:
            print("Fit: first click at least one point in both pictures")
            return
        t, o = np.array(both).transpose(1, 0, 2)
        use, zoom = dual_camera.OPTICAL_VIEW, self.fit_zoom.get() and len(both) > 1
        plate = np.stack(dual_camera.THERMAL_VIEW.to_plate(*t.T, self.pair.thermal.shape), 1)
        view, off = align_view(use, o, self.pair.optical.shape, plate, zoom)
        before = np.array([a for a in self.apart(use) if a is not None])
        self.zoom = view.size[0] / use.size[0]
        for key, value in zip(("x", "y", "omega"), (*view.center, view.omega)):
            self.fields[key].set(str(value))
        print(f"Fit to {len(both)} point{'s' * (len(both) > 1)}: center ({view.center[0]:.3f}, {view.center[1]:.3f}) mm, "
              f"omega {view.omega:.3f} deg" + (f", zoom {self.zoom:.4f}" if zoom else "") +
              f"; points apart rms {np.sqrt(np.mean(off ** 2)):.2f} mm, max {off.max():.2f} mm (view in use: rms "
              f"{np.sqrt(np.mean(before ** 2)):.2f} mm)" +
              (". With 1 point only the center moves; 2 or more also turn it." if len(both) == 1 else ""))
        self.update()

    def later(self):                                   # fields typed in: redraw once typing pauses
        if self.job is not None:
            self.win.after_cancel(self.job)
        self.job = self.win.after(400, self.update)

    def revert(self):
        """Back to the optical view in use."""
        use, self.zoom = dual_camera.OPTICAL_VIEW, 1.0
        for key, value in zip(("x", "y", "omega"), (*use.center, use.omega)):
            self.fields[key].set(str(value))
        self.base = tuple(var.get() for var in self.fields.values())
        self.update()

    def save(self):
        view = self.edited()
        if view is None:
            messagebox.showerror("Alignment", "center and omega must be numbers", parent=self.win)
            return
        path = VIEW_DIR / "optical.json"
        info = {k: v for k, v in read_view(path)[1].items() if k not in ("saved", "alignment")} if path.exists() else {}
        apart = np.array([a for a in (self.apart(view) if self.pair else []) if a is not None])
        if len(apart):                                 # for the record: what it was fitted to
            info["alignment"] = {"picture": self.path.name, "points": len(apart),
                                 "rms_mm": round(float(np.sqrt(np.mean(apart ** 2))), 4),
                                 "max_mm": round(float(apart.max()), 4), "zoom": round(self.zoom, 5)}
        save_view("optical", view, **info)
        print(f"OPTICAL_VIEW in use from now on (views/optical.json): {view}")
        self.revert()
        self.app.views_changed()

    def views_changed(self):
        """Views were put in use (here or in the calibration window)."""
        if self.zoom == 1 and tuple(var.get() for var in self.fields.values()) == self.base:
            self.revert()                              # nothing changed here: follow the new view
        self.grid = self.overlaid = None               # the thermal view may have changed
        self.update()

    def plate(self):
        if self.pair is None:
            print("Take or open a picture first")
            return
        self.app.plate(self.pair, self.edited() or dual_camera.OPTICAL_VIEW)

    # The points ------------------------------------------------------------------------------------------
    def click(self, event):
        side = {self.axes["thermal"]: 0, self.axes["optical"]: 1}.get(event.inaxes)
        if self.pair is None or self.tools.mode or side is None or event.xdata is None:
            return                                     # zooming / panning, or not in a picture
        if event.button == 3:                          # right click: delete the point nearest to it
            near = [(np.hypot(*(event.inaxes.transData.transform(p[side]) - (event.x, event.y))), i)
                    for i, p in enumerate(self.points) if p[side]]
            if near and min(near)[0] < 20 * self.unit:
                self.delete(min(near)[1])
            return
        if event.button != 1:
            return
        target = self.selected                         # moved; else the last point if it lacks a side, or a new one
        if target is None:
            if not self.points or None not in self.points[-1]:
                self.points.append([None, None])
            target = len(self.points) - 1
        self.points[target][side] = (round(float(event.xdata), 2), round(float(event.ydata), 2))
        self.changed()

    def select(self):
        chosen = self.table.selection()
        new = int(chosen[0]) if chosen else None
        if new != self.selected:
            self.selected = new
            self.update()

    def delete(self, index=None):
        """Delete point `index`, else the selected one, else the last one."""
        index = index if index is not None else self.selected if self.selected is not None else len(self.points) - 1
        if 0 <= index < len(self.points):
            del self.points[index]
            self.selected = None
            self.changed()

    def delete_all(self):
        if self.points and messagebox.askyesno("Alignment", f"Delete all {len(self.points)} points?", parent=self.win):
            self.points, self.selected = [], None
            self.changed()

    def changed(self):                                 # points changed: stored next to the picture
        self.points_file().write_text(json.dumps(self.points), encoding="utf-8")
        self.update()

    # Drawing ---------------------------------------------------------------------------------------------
    def update(self):
        """Points list, state, overlay and marks with the optical view in the fields."""
        if self.job is not None:
            self.win.after_cancel(self.job)
            self.job = None
        use, view = dual_camera.OPTICAL_VIEW, self.edited()
        size = (view or use).size
        self.size_label.configure(text=f"size {size[0]:.3f} x {size[1]:.3f} mm  " +
                                  (f"(zoom {self.zoom:.4f} from the fit)" if self.zoom != 1 else "(from calibration)"))
        self.state.configure(text="center and omega must be numbers" if view is None else "" if view == use else
                             "Not saved yet: check the overlay and the build plate view, then Save")
        view = view or use
        apart = self.apart(view) if self.pair else []
        self.table.delete(*self.table.get_children())
        for i, ((t, o), a) in enumerate(zip(self.points, apart)):
            self.table.insert("", "end", iid=str(i), text=str(i + 1), values=(
                f"{t[0]:.1f}, {t[1]:.1f}" if t else "", f"{o[0]:.1f}, {o[1]:.1f}" if o else "",
                f"{a:.2f}" if a is not None else "click both"))
        if self.selected is not None:
            self.table.selection_set(str(self.selected))
            self.table.see(str(self.selected))
        done = np.array([a for a in apart if a is not None])
        now = np.array([a for a in self.apart(use) if a is not None]) if self.pair else done
        text = (f"{len(done)} points in both pictures, apart rms {np.sqrt(np.mean(done ** 2)):.2f} mm, max "
                f"{done.max():.2f} mm" + (f" (view in use: rms {np.sqrt(np.mean(now ** 2)):.2f} mm)" if view != use else "")
                if len(done) else "No points yet: click a feature in the thermal picture, then in the optical one.")
        if self.selected is not None:
            text = f"Point {self.selected + 1} selected: clicks move it (Esc: done).  " + text
        self.summary.configure(text=text)
        if self.pair is None:
            return
        if self.pair.optical_live and view != self.overlaid:
            self.layer.set_data(self.warped(view))
            self.overlaid = view
        self.mark(view)
        self.canvas.draw_idle()

    def warped(self, view):
        """The optical picture as the thermal camera would see it (through both views): RGBA at OVERLAY x the
        thermal resolution, transparent where the optical camera does not see."""
        h, w = self.pair.thermal.shape[:2]
        if self.grid is None:
            f = self.OVERLAY
            u, v = np.meshgrid((np.arange(round(f * w)) + 0.5) / f - 0.5, (np.arange(round(f * h)) + 0.5) / f - 0.5)
            self.grid = dual_camera.THERMAL_VIEW.to_plate(u, v, (h, w))
        return rgba(_sample(self.pair.optical, *view.to_pixel(*self.grid, self.pair.optical.shape)))

    def blended(self):
        if self.layer is not None:
            self.layer.set_alpha(self.blend.get())
            self.canvas.draw_idle()

    def mark(self, view):
        """Points (with where the views predict a point not clicked yet), the other camera's picture border
        (dashed) and the plate's edges (dotted, where each view puts them) over the pictures."""
        for artist in self.marks:
            artist.remove()
        self.marks = []
        thermal, t_shape, o_shape = dual_camera.THERMAL_VIEW, self.pair.thermal.shape, self.pair.optical.shape
        ax_t, ax_o, ax_b = self.axes.values()

        def to_t(x, y):                                # plate -> thermal picture
            return thermal.to_pixel(x, y, t_shape)

        def to_o(x, y):                                # plate -> optical picture
            return view.to_pixel(x, y, o_shape)

        def plot(ax, uv, **style):
            self.marks.extend(ax.plot(*uv, **style))

        turn = np.radians(np.arange(0, 360.5, 0.5))
        for r in (PLATE_R_INNER, PLATE_R_OUTER):
            for ax, to in ((ax_t, to_t), (ax_o, to_o), (ax_b, to_t)):
                plot(ax, to(r * np.cos(turn), r * np.sin(turn)), color="#2DD4BF", lw=1.5, ls=":")
        for ax in (ax_t, ax_b):
            plot(ax, to_t(*view.to_plate(*edge(o_shape), o_shape)), color="white", lw=1, ls="--")
        plot(ax_o, to_o(*thermal.to_plate(*edge(t_shape), t_shape)), color="white", lw=1, ls="--")
        halo = [pe.withStroke(linewidth=3, foreground="white")]
        for i, (t, o) in enumerate(self.points):
            colour, chosen = PALETTE[i % len(PALETTE)], i == self.selected
            ring = dict(marker="o", ms=14 if chosen else 9, mew=2.5 if chosen else 1.5, mfc="none", mec=colour, ls="none")
            seen_t = t or to_t(*view.to_plate(*o, o_shape))     # not clicked yet: where the other picture's point predicts it
            seen_o = o or to_o(*thermal.to_plate(*t, t_shape))
            for ax, p, clicked in ((ax_t, seen_t, t is not None), (ax_o, seen_o, o is not None)):
                if np.isfinite(p).all():
                    plot(ax, p, alpha=1 if clicked else 0.5, **ring)
                    self.marks.append(ax.annotate(f"{i + 1}" if clicked else f"{i + 1}?", p, xytext=(7, 7),
                                                  textcoords="offset points", color=colour, fontsize=10,
                                                  fontweight="bold", path_effects=halo, alpha=1 if clicked else 0.6))
            if t and o:                                # overlay: thermal point (ring), optical point (cross), gap
                there = to_t(*view.to_plate(*o, o_shape))
                plot(ax_b, ([t[0], there[0]], [t[1], there[1]]), color=colour, lw=1.5)
                plot(ax_b, t, **ring)
                plot(ax_b, there, marker="x", ms=9, mew=2, color=colour, ls="none")

    def close(self):
        self.app.alignment = None
        self.win.destroy()


class Areas:
    """Areas window: areas drawn on the build plate, pictures put into each of them in physical space (plate
    mm, lens distortion taken out; see area_history.py), and the history of the selected one."""

    DPI = 75                           # figures smaller than matplotlib's usual 100: smaller text, more picture

    def __init__(self, app):
        self.app = app
        self.win = win = tk.Toplevel(app.root)
        win.title("RAPTURE cameras: areas")
        self.areas = []                    # area_history.load_areas()
        self.selected = None               # name of the area whose history is shown
        self.rows = []                     # its history (load_history)
        self.average = None                # its thermal average over all pictures
        self.sums = (None, set(), None, None)  # (area, pictures, sum, count) behind it: new pictures are added
        self.index = -1                    # the picture of the history shown; -1 = the newest
        self.drawing = False               # clicks on the plate add corners of a new area
        self.pair = None                   # the moment shown on the plate
        self.plate_next = None             # the moment to show on it next, while it is busy with another
        self.plate_ax = self.time_ax = None
        self.marks = []                    # areas drawn over the plate
        self.busy = self.plate_busy = False    # putting pictures into the areas / onto the plate (worker thread)
        self.changes = app.service.areas_changed   # areas / histories as shown (poll refreshes on a change)
        self.shown = None                  # (rotation, record every) of the service as shown
        self.closed = False
        self.name, self.corners = tk.StringVar(), tk.StringVar()
        self.rotation = tk.StringVar(value=f"{app.service.rotation:g}")
        self.every = tk.StringVar(value=f"{app.service.every or 10:g}")
        self.keep = tk.BooleanVar(value=app.service.keep)
        self.rotation.trace_add("write", lambda *_: self.rotation_typed())
        self.keep.trace_add("write", lambda *_: setattr(app.service, "keep", self.keep.get()))

        app.add_chips(toolbar(win, "Areas", app.tabs()))
        page = ttk.Frame(win, style="Page.TFrame", padding=12)
        page.pack(fill="both", expand=True)
        bottom = ttk.Frame(page, style="Page.TFrame")
        bottom.pack(side="bottom", fill="x", pady=(12, 0))
        self.areas_panel(card(bottom, "Areas", side="left", fill="y"))
        self.record_panel(card(bottom, "Pictures into all areas", side="left", fill="y", padx=(12, 0)))
        app.log.box(card(bottom, "Log", side="left", fill="both", expand=True, padx=(12, 0)), 8).pack(
            fill="both", expand=True)
        top = ttk.Frame(page, style="Page.TFrame")
        top.pack(fill="both", expand=True)
        top.rowconfigure(0, weight=1)
        for column, weight in ((0, 2), (1, 3)):        # plate : history
            top.columnconfigure(column, weight=weight, uniform="top")
        left, right = ttk.Frame(top, style="Page.TFrame"), ttk.Frame(top, style="Page.TFrame")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        right.grid(row=0, column=1, sticky="nsew")

        plate = card(left, "Build plate", fill="both", expand=True)
        self.plate_label = ttk.Label(plate.header, style="Muted.TLabel")
        self.plate_label.pack(side="left", padx=12)
        below = ttk.Frame(plate)
        below.pack(side="bottom", fill="x")
        self.plate_fig = Figure(facecolor=CARD, dpi=self.DPI)
        self.plate_canvas = FigureCanvasTkAgg(self.plate_fig, plate)
        self.plate_tools = plot_tools(self.plate_canvas, below, CARD)
        self.plate_tools.pack(side="left")
        self.plate_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.plate_canvas.mpl_connect("button_press_event", self.click)

        history = card(right, "History", fill="both", expand=True)
        self.history_label = ttk.Label(history.header, style="Muted.TLabel")
        self.history_label.pack(side="left", padx=12)
        ttk.Button(history.header, text="Delete this picture", style="Delete.TButton",
                   command=self.delete_picture).pack(side="right")
        for text, by in (("▶", 1), ("◀", -1)):
            ttk.Button(history.header, text=text, width=3, style="Minor.TButton",
                       command=lambda by=by: self.step(by)).pack(side="right")
        below = ttk.Frame(history)
        below.pack(side="bottom", fill="x")
        self.fig = Figure(facecolor=CARD, dpi=self.DPI, layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.fig, history)
        self.tools = plot_tools(self.canvas, below, CARD)
        self.tools.pack(side="left")
        self.entry_label = ttk.Label(below, style="Muted.TLabel")
        self.entry_label.pack(side="right")
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("button_press_event", self.pick)

        for key, action in (("p", self.take), ("n", self.start_area), ("<Left>", lambda: self.step(-1)),
                            ("<Right>", lambda: self.step(1))):
            win.bind(key, lambda event, action=action: isinstance(event.widget, tk.Entry) or action())
        win.bind("<Return>", lambda _: self.finish())
        win.bind("<Escape>", lambda _: self.cancel())
        win.protocol("WM_DELETE_WINDOW", self.close)
        fit_to_screen(win, screen(win)[2:])    # as large as it may be
        print("Areas: New area (n), click the corners of the area on the build plate (or type them; 2 corners = a "
              "rectangle), Enter. Take picture (p) or Record every puts the moment into every area, resampled into "
              "plate mm (lens distortion taken out), thermal as deg C. Click an area on the plate to see its "
              "history; click the time plot or use the arrow buttons / keys for one picture of it.")
        self.refresh()
        self.redraw_plate()
        if app.cams.latest is not None:
            self.show_plate(app.cams.latest[0])

    def areas_panel(self, box):
        frame = ttk.Frame(box)
        frame.pack(fill="both", expand=True)
        self.table = ttk.Treeview(frame, columns=("size", "pictures", "last"), height=3, selectmode="browse")
        for column, text, width in (("#0", "name", 120), ("size", "size mm", 105), ("pictures", "pictures", 70),
                                    ("last", "last mean °C", 100)):
            self.table.heading(column, text=text, anchor="w")
            self.table.column(column, width=round(width * self.win.winfo_fpixels("1i") / 96), stretch=False, anchor="w")
        bar = ttk.Scrollbar(frame, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        self.table.pack(side="left", fill="both", expand=True)
        self.table.bind("<<TreeviewSelect>>", lambda _: self.select())
        new = ttk.Frame(box)
        new.pack(fill="x", pady=(8, 0))
        ttk.Label(new, text="New area").grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Entry(new, textvariable=self.name, width=16).grid(row=0, column=1, sticky="w")
        ttk.Label(new, text="corners x' y' mm").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(6, 0))
        ttk.Entry(new, textvariable=self.corners, width=38).grid(row=1, column=1, sticky="we", pady=(6, 0))
        self.corners.trace_add("write", lambda *_: self.typed())
        buttons = ttk.Frame(box)
        buttons.pack(fill="x", pady=(8, 0))
        self.new_button = ttk.Button(buttons, command=self.start_area)
        self.new_button.pack(side="left")
        self.cancel_button = ttk.Button(buttons, text="Cancel (Esc)", style="Minor.TButton", command=self.cancel)
        self.cancel_button.pack(side="left", padx=8)
        ttk.Button(buttons, text="Delete area", style="Delete.TButton", command=self.delete_area).pack(side="right")
        self.area_state = ttk.Label(box, style="Muted.TLabel", wraplength="11c", justify="left")
        self.area_state.pack(fill="x", pady=(6, 0))
        self.update_buttons()

    def record_panel(self, box):
        ttk.Button(box, text="Take picture (P)", style="Main.TButton", command=self.take).pack(anchor="w")
        row = ttk.Frame(box)
        row.pack(fill="x", pady=(8, 0))
        self.every_button = ttk.Button(row, command=self.toggle_every)
        self.every_button.pack(side="left")
        ttk.Entry(row, textvariable=self.every, width=5).pack(side="left", padx=(8, 4))
        ttk.Label(row, text="s").pack(side="left")
        row = ttk.Frame(box)
        row.pack(fill="x", pady=(10, 0))
        ttk.Label(row, text="Plate turned by").pack(side="left")
        ttk.Entry(row, textvariable=self.rotation, width=7).pack(side="left", padx=6)
        ttk.Label(row, text="deg  (for the next pictures; later from the PLC)").pack(side="left")
        ttk.Checkbutton(box, text="Also keep the whole pictures (pictures/)", variable=self.keep).pack(
            anchor="w", pady=(8, 0))
        ttk.Button(box, text="Add saved pictures …", style="Minor.TButton", command=self.add_saved).pack(
            anchor="w", pady=(4, 0))
        self.record_state = ttk.Label(box, style="Muted.TLabel", wraplength="8c", justify="left")
        self.record_state.pack(fill="x", pady=(4, 0))
        self.update_every()

    # The areas -------------------------------------------------------------------------------------------
    def area(self):
        """The selected area, or None."""
        return next((a for a in self.areas if a["name"] == self.selected), None)

    def refresh(self):
        """Areas from areas/ into the list, and the history of the selected one."""
        self.areas = load_areas()
        if self.area() is None:                        # deleted, or none chosen yet: the newest area
            self.selected, self.index = (self.areas[-1]["name"] if self.areas else None), -1
        self.table.delete(*self.table.get_children())
        for i, area in enumerate(self.areas):
            rows = load_history(area)
            (x0, y0), (x1, y1) = np.min(area["corners"], 0), np.max(area["corners"], 0)
            last = next((r["thermal_mean_C"] for r in reversed(rows) if np.isfinite(r["thermal_mean_C"])), np.nan)
            self.table.insert("", "end", iid=str(i), text=area["name"], values=(
                f"{x1 - x0:.1f} x {y1 - y0:.1f}", len(rows), f"{last:.1f}" if np.isfinite(last) else ""))
            if area["name"] == self.selected:
                self.table.selection_set(str(i))
                self.table.see(str(i))
        self.show_history()
        self.mark()

    def select(self):
        chosen = self.table.selection()
        name = self.areas[int(chosen[0])]["name"] if chosen else None
        if name is not None and name != self.selected:
            self.selected, self.index = name, -1
            self.show_history()
            self.mark()

    def new_corners(self):
        """The corners typed / clicked for a new area as an (n, 2) array (plate frame), None if not pairs of numbers."""
        numbers = re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)", self.corners.get())
        return None if len(numbers) % 2 else np.array(numbers, float).reshape(-1, 2)

    def typed(self):                                   # corners changed (typed or clicked)
        if self.corners.get().strip() and not self.drawing:
            self.start_area()
        self.mark()

    def start_area(self):
        """New area (n): clicks on the plate add corners from now on. While drawing: Finish."""
        if self.drawing:
            self.finish()
            return
        self.drawing = True
        names = {a["name"] for a in self.areas}
        if not self.name.get().strip() or self.name.get().strip() in names:
            n = 1
            while f"area {n}" in names:
                n += 1
            self.name.set(f"area {n}")
        self.area_state.configure(text="Click the corners on the build plate (right click: take the last one back) "
                                       "or type them as x' y' pairs; 2 corners = a rectangle. Enter: done, Esc: cancel.")
        self.update_buttons()
        self.mark()

    def finish(self):
        if not self.drawing:
            return
        corners = self.new_corners()
        try:
            if corners is None:
                raise ValueError("the corners must be pairs of numbers x' y'")
            area = self.app.service.new_area(self.name.get(), corners)
        except (ValueError, OSError) as exc:
            self.area_state.configure(text=f"Not created: {exc}")
            print(f"Area not created: {exc}")
            return
        self.drawing = False
        self.corners.set("")
        self.name.set("")
        self.selected, self.index = area["name"], -1
        self.area_state.configure(text="")
        self.update_buttons()
        self.refresh()

    def cancel(self):
        if self.drawing:
            self.drawing = False
            self.corners.set("")
            self.area_state.configure(text="")
            self.update_buttons()
            self.mark()

    def update_buttons(self):
        self.new_button.configure(text="Finish (Enter)" if self.drawing else "New area (N)")
        self.cancel_button.state(["!disabled"] if self.drawing else ["disabled"])

    def delete_area(self):
        area = self.area()
        if area is not None and messagebox.askyesno(
                "Areas", f"Delete the area '{area['name']}' and its history ({len(self.rows)} pictures)?", parent=self.win):
            shutil.rmtree(area["folder"])
            self.app.service.areas_changed += 1
            print(f"Area '{area['name']}' deleted")
            self.refresh()

    def rotation_deg(self):
        """The plate rotation typed in, deg; None if it is not a number."""
        try:
            return float(self.rotation.get())
        except ValueError:
            return None

    def rotation_typed(self):                          # the service takes it for the pictures from now on
        rotation = self.rotation_deg()
        if rotation is not None:
            self.app.service.rotation = rotation

    def poll(self, state):
        """Each tick of the app: refresh what changed in the service (also through its API, or recording
        every n s in the background)."""
        if self.app.service.areas_changed != self.changes:
            self.changes = self.app.service.areas_changed
            self.refresh()
        shown = (state["rotation_deg"], state["areas_every_s"])
        if shown != self.shown:
            if self.shown is not None and shown[0] != self.shown[0] and shown[0] != self.rotation_deg():
                self.rotation.set(f"{shown[0]:g}")         # set through the API
            self.shown = shown
            self.update_every()

    # Pictures into the areas -----------------------------------------------------------------------------
    def take(self):
        latest = self.app.cams.latest
        if latest is None:
            print("No picture from the cameras yet")
        else:
            self.put([latest[0]], plate=True)

    def add_saved(self):
        chosen = filedialog.askopenfilenames(parent=self.win, title="Pictures to put into all areas",
                                             initialdir=PICTURE_DIR if PICTURE_DIR.exists() else None,
                                             filetypes=[("Pictures", "*.png *.npy"), ("All files", "*.*")])
        stems = {Path(c).stem.removesuffix("_thermal").removesuffix("_optical").removesuffix("_points"): Path(c)
                 for c in chosen}
        if stems:
            self.put(sorted(path.with_name(f"{stem}.png") for stem, path in stems.items()))

    def put(self, sources, plate=False):
        """Pictures into all areas (Service.put, in a worker thread), with the plate rotation typed in.
        `plate`: then show the last on the plate."""
        rotation = self.rotation_deg()
        if rotation is None:
            print("Plate turned by: must be a number (deg)")
            return
        if not self.areas:
            print("No areas yet: New area (n), then click its corners on the build plate")
            return
        if self.busy:
            print("Still putting the previous pictures into the areas")
            return
        self.busy = True
        self.record_state.configure(text="Putting the pictures into the areas ...")
        self.app.work(lambda: self.app.service.put(sources, rotation),
                      lambda results: self.put_done(results[-1][0] if results else None, plate))

    def put_done(self, pair, plate):
        self.busy = False
        if self.closed:
            return
        self.record_state.configure(text="")
        if plate and pair is not None:
            self.show_plate(pair)

    def toggle_every(self):
        """Record every n s, in the service: goes on with this window closed, until stopped."""
        if self.app.service.every is not None:
            self.app.service.record_every(None)
        else:
            try:
                seconds = float(self.every.get())
            except ValueError:
                seconds = 0
            self.app.service.record_every(seconds)
        self.update_every()

    def update_every(self):
        on = self.app.service.every is not None
        self.every_button.configure(text="■  Stop recording" if on else "●  Record every",
                                    style="Danger.TButton" if on else "TButton")

    # The build plate -------------------------------------------------------------------------------------
    def show_plate(self, pair):
        """`pair` on the plate drawing (resampled in the background; after the one being resampled)."""
        if self.plate_busy:
            self.plate_next = pair
            return
        self.plate_busy = True
        self.plate_label.configure(text="putting the picture on the plate (a few seconds) ...")
        self.app.work(lambda: plate_layers(pair), lambda layers: self.plate_done(pair, layers))

    def plate_done(self, pair, layers):
        self.plate_busy = False
        if not self.closed and layers is not None:
            self.pair = pair
            self.redraw_plate(layers)
        if not self.closed and self.plate_next is not None:
            pair, self.plate_next = self.plate_next, None
            self.show_plate(pair)

    def redraw_plate(self, layers=()):
        self.plate_fig.clear()
        draw_plate(self.plate_fig, self.pair, layers)
        self.plate_ax = self.plate_fig.axes[0]
        self.plate_ax.set_autoscale_on(False)
        self.plate_tools.update()
        self.marks = []
        self.plate_label.configure(text=f"the moment of {datetime.fromtimestamp(self.pair.timestamp):%H:%M:%S}, "
                                        f"plate at {self.pair.rotation:g}° (Take picture updates it)"
                                   if self.pair else "no picture yet")
        self.mark()

    def mark(self):
        """The areas (the selected one filled) and the corners of a new one over the plate (all in the plate's
        frame, like the picture: the camera pictures are the ones turned)."""
        if self.plate_ax is None:
            return
        for artist in self.marks:
            artist.remove()
        self.marks = []
        ax = self.plate_ax
        halo = [pe.withStroke(linewidth=3.5, foreground="black", alpha=0.55)]
        many = len(self.areas) > 30                    # e.g. from a 3MF job: names only for the selected one
        for i, area in enumerate(self.areas):         # as the parts in the AMCF plate view: dashed, the selected solid teal
            xy = np.vstack([area["corners"], area["corners"][:1]])
            chosen, colour = area["name"] == self.selected, PALETTE[i % len(PALETTE)]
            self.marks += ax.fill(*xy.T, color=TEAL if chosen else colour, alpha=0.3 if chosen else 0.1, lw=0, zorder=6)
            self.marks += ax.plot(*xy.T, color=TEAL if chosen else colour, lw=3 if chosen else 1 if many else 1.5,
                                  ls="-" if chosen else (0, (6, 4)), zorder=7,
                                  path_effects=[pe.withStroke(linewidth=5, foreground="white")] if chosen else halo)
            if many and not chosen:
                continue
            top = xy[xy[:, 1].argmax()]
            self.marks.append(ax.text(top[0], top[1] + 1, area["name"], color="white" if chosen else colour, fontsize=9,
                                      ha="center", va="bottom", fontweight="bold" if chosen else "normal", zorder=8,
                                      path_effects=[pe.withStroke(linewidth=3.5, foreground=TEAL if chosen else "black",
                                                                  alpha=1 if chosen else 0.55)]))
        corners = self.new_corners() if self.drawing else None
        if corners is not None and len(corners):
            if len(corners) == 2:                          # a rectangle
                (x0, y0), (x1, y1) = corners
                corners = np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
            xy = np.vstack([corners, corners[:1]]) if len(corners) > 2 else corners
            self.marks += ax.plot(*xy.T, color="white", lw=1.5, ls="--", marker="o", ms=5, zorder=9, path_effects=halo)
        self.plate_canvas.draw_idle()

    def click(self, event):
        if event.inaxes is not self.plate_ax or self.plate_tools.mode or event.xdata is None:
            return                                     # zooming / panning, or not on the plate
        x, y = event.xdata, event.ydata
        if self.drawing:
            corners = self.new_corners()
            corners = [] if corners is None else corners.tolist()
            if event.button == 1:
                corners.append((x, y))
            elif event.button == 3 and corners:
                corners.pop()
            self.corners.set("; ".join(f"{cx:.1f} {cy:.1f}" for cx, cy in corners))
        elif event.button == 1:                        # select the (smallest) area clicked into
            hits = [(abs(cv2.contourArea(np.float32(a["corners"]))), i) for i, a in enumerate(self.areas)
                    if cv2.pointPolygonTest(np.float32(a["corners"]), (float(x), float(y)), False) >= 0]
            if hits:
                self.table.selection_set(str(min(hits)[1]))

    def views_changed(self):
        """Views were put in use: the plate picture again through them. The histories keep what was put in."""
        if self.pair is not None:
            self.show_plate(self.pair)

    # The history -----------------------------------------------------------------------------------------
    def show_history(self, reload=True):
        """The selected area's history; `reload`: read it from its files again."""
        area = self.area()
        if reload:
            self.rows = load_history(area) if area else []
            self.average = self.averaged(area, self.rows) if self.rows else None
        self.fig.clear()
        self.tools.update()
        self.time_ax = None
        if area is None:
            ax = self.fig.add_subplot()
            ax.set_axis_off()
            ax.text(0.5, 0.5, "No areas yet: New area (n), then click its corners on the build plate.",
                    ha="center", va="center", color=MUTED, transform=ax.transAxes)
            self.history_label.configure(text="")
            self.entry_label.configure(text="")
        else:
            self.time_ax = draw_history(self.fig, area, self.rows, self.index, self.average)
            n = len(self.rows)
            self.history_label.configure(text=f"{area['name']}: {n} picture{'s' * (n != 1)}, in areas/{area['folder'].name}/")
            text = "No pictures yet: Take picture (p) or Record every"
            if n:
                row = self.rows[self.index]
                text = f"picture {self.index % n + 1} of {n}: {row['time'][:19]}, plate at {row['rotation_deg']:g}°"
                if np.isfinite(row["thermal_mean_C"]):
                    text += f", mean {row['thermal_mean_C']:.2f} °C over {row['thermal_seen']:.0%} of the area"
            self.entry_label.configure(text=text)
        self.canvas.draw_idle()

    def averaged(self, area, rows):
        """The thermal average of `rows` of `area`, reading only the pictures new since the last time (reading
        all of them after each new one would slow down recording)."""
        name, stems, total, count = self.sums
        now = {r["stem"] for r in rows}
        if name != area["name"] or not stems <= now:   # another area, or pictures taken out: from scratch
            stems, total, count = set(), None, None
        mean, seen = thermal_average(area, [r for r in rows if r["stem"] not in stems])
        if mean is not None:
            if total is None:
                total, count = np.zeros(mean.shape), np.zeros(mean.shape, int)
            total, count = total + np.nan_to_num(mean) * seen, count + seen
        self.sums = area["name"], now, total, count
        return None if total is None else np.where(count > 0, total / np.maximum(count, 1), np.nan)

    def step(self, by):
        n = len(self.rows)
        if n:
            k = min(max(self.index % n + by, 0), n - 1)
            self.index = -1 if k == n - 1 else k       # at the newest: stay with the newest as pictures come in
            self.show_history(reload=False)

    def pick(self, event):                             # a click into the time plot: the picture nearest to it
        if event.inaxes is not self.time_ax or self.tools.mode or event.xdata is None or not self.rows:
            return
        times = mdates.date2num([datetime.fromtimestamp(r["timestamp"]) for r in self.rows])
        k = int(np.argmin(np.abs(times - event.xdata)))
        self.index = -1 if k == len(self.rows) - 1 else k
        self.show_history(reload=False)

    def delete_picture(self):
        area = self.area()
        if area is None or not self.rows:
            return
        row = self.rows[self.index]
        if messagebox.askyesno("Areas", f"Take the picture of {row['time'][:19]} out of the history of "
                                        f"'{area['name']}'?", parent=self.win):
            delete_entry(area, row)
            if self.index >= len(self.rows) - 2:       # was the newest or the one before: the newest now
                self.index = -1
            print(f"{area['name']}: picture of {row['time'][:19]} taken out of the history")
            self.refresh()

    def close(self):
        if self.app.service.every is not None:
            print(f"Recording into the areas every {self.app.service.every:g} s goes on (stop: Areas window, "
                  "Stop recording)")
        self.closed = True
        self.app.areas = None
        self.win.destroy()


def main():
    try:                                   # sharp text and pictures on scaled Windows displays
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass
    root = tk.Tk()
    dpi = re.search(r"^Xft\.dpi:\s*([\d.]+)", x11("xrdb", "-query"), re.MULTILINE)
    if dpi:          # Linux: Tk ignores the desktop's display scaling; GNOME and KDE give it as Xft.dpi (96 = 100%)
        root.tk.call("tk", "scaling", float(dpi.group(1)) / 72)    # fonts and amcf_theme's `unit` follow it
    amcf_theme(root)
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
