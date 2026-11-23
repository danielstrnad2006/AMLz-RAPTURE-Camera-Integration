# AMLz RAPTURE Camera Integration

Thermal and optical cameras for the RAPTURE build plate. The program shows both cameras live, maps their
pictures onto the build plate in millimetres and records the history of chosen areas of the plate.

- **Thermal:** Optris PI 640i (USB), through the Optris Thermal Camera SDK
- **Optical:** MOKOSE UC50 webcam (USB)

The program also runs with only one camera connected, or none. A missing camera shows a placeholder.

## Install

### Windows

1. Install the [Optris Thermal Camera SDK](https://optris.com/software/otc-sdk/) and restart the PC.
2. Install the Python packages:

   ```
   python -m pip install opencv-python numpy matplotlib pygrabber
   ```

3. Close the Windows Camera app and PIX Connect. Only one program can use a camera at a time.

### Linux (tested on Ubuntu 26.04, x86_64)

The Optris SDK's Python part is built for NumPy 1.x, but Ubuntu's own Python has NumPy 2. So the program
runs in a conda environment.

1. Install the Optris SDK `.deb` packages (libirimager, otcsdk).
2. Install the tools for the window layout and the camera settings window:

   ```
   sudo apt install x11-xserver-utils qv4l2
   ```

3. Create the environment (packages from conda-forge) and link the SDK's Python module into it:

   ```
   conda create -n rapture --override-channels -c conda-forge python=3.12 "numpy<2" "opencv<4.11" matplotlib tk
   ln -s /usr/lib/python3/dist-packages/optris ~/miniconda3/envs/rapture/lib/python3.12/site-packages/optris
   ```

4. Start the program with the environment's Python, as shown under Run.

## Run

```
python app.py                                  # Windows
~/miniconda3/envs/rapture/bin/python app.py    # Linux
```

Every window has the same tabs: **Live**, **Calibration**, **Alignment** and **Areas**.

| Window | What it does |
|---|---|
| Live | Both cameras side by side. Take a picture (P, into `pictures/`), record a real-time clip (R) or a timelapse (T) as MP4 in `clips/`, show the moment on the build plate (Space), open the optical camera's settings (S). |
| Calibration | Fits each camera's zoom and lens distortion from pictures of a checkerboard lying flat in the print plane (5 mm squares by default). Save puts the result in use. |
| Alignment | Makes the optical camera agree with the thermal one: click the same feature in both pictures, then Fit and Save. The thermal camera is the reference. |
| Areas | Draw areas on the build plate (in the plate's own millimetres) and record their history: thermal (°C) and optical pictures, the average temperature and the mean temperature over time. Each picture goes into every area, with lens distortion removed. |

### Background service

To run the cameras without any window, for example on the PC at the machine:

```
python camera_service.py                  # reachable from this PC only
python camera_service.py --host 0.0.0.0   # reachable from the network too
```

It does everything that has to go on without anyone watching:
- taking pictures and recording clips or timelapses;
- putting pictures into the areas, including "record every n s";
- keeping the plate rotation to use for the next pictures.

Ctrl+C stops it. There is no login: anyone who can reach the port can record and create areas.

`app.py` runs the same service inside itself, so while the app is open the same API is available at
`http://localhost:8765/api`. Only one program can use the cameras at a time, so run either the app or the
background service.

Use the API with any HTTP client. Opening `http://localhost:8765/api` in a browser lists all the addresses.

| Request | What it does |
|---|---|
| `GET /api/state` | Camera states, recording, area recording, plate rotation |
| `GET /api/live.jpg`, `GET /api/live.mjpg` | The live picture, or a live stream (open it in a browser) |
| `GET /api/temperature?x=..&y=..` | °C at a build plate point, now |
| `GET /api/areas`, `GET /api/areas/<name>` | The areas, or one area with its history |
| `POST /api/picture` | Take a picture into `pictures/` |
| `POST /api/record` `{"kind": "clip"}` or `{"kind": "timelapse"}`, `POST /api/record/stop` | Start or stop a recording |
| `POST /api/areas` `{"name": "part 1", "corners": [[x, y], ...]}` | Create an area (2 corners make a rectangle) |
| `POST /api/areas/record` | Put the current moment into all areas |
| `POST /api/areas/every` `{"seconds": 10}` | Put a picture into all areas every 10 s (0 stops it) |
| `POST /api/rotation` `{"deg": 12.5}` | Plate rotation for the pictures from now on |

For example:
```
curl -X POST localhost:8765/api/areas/every -H "Content-Type: application/json" -d "{\"seconds\": 10}"
```

### Monitored areas from a sliced job (defect detection)

```
python defect_detection.py job.3mf            # show the first layer and the monitored areas
python defect_detection.py job.3mf --create   # also store the areas in areas/
```

This covers parts whose cross-section stays the same through the whole height.

- **First layer:** the job's sectors are read until they add up to 360°, which is one revolution. With a
  constant cross-section, that revolution contains the whole cross-section.
- **Melted islands:** its scan vectors are drawn into a map of what gets melted. Each separate melted
  island becomes one monitored area.
- **Margin:** each area is kept 0.3 mm inside the melted region (`--inset`), so powder doesn't end up in
  an area.
- **Tiles:** `--tile 4` cuts the islands into 4 mm squares.
- **Plot:** the window shows the scan vectors, the sectors, the areas and where both cameras look.

With `--create`, the areas are stored like any other area, so the app and the background service
record them: Take picture, Record every, or the API. Each picture goes into every area a camera sees at
that moment.

### Other entry points


- `python dual_camera.py`: plain live view (q quits).
- `python dual_camera.py calibration/<time> [square mm]`: calibrate again from saved pictures, without cameras.
- `python plate_plot.py`: live view; Space shows the moment on the build plate.
- `python area_history.py [area name]`: list the areas, or show one area's history, without cameras.

## Files

| Path | Contents |
|---|---|
| `app.py` | The GUI, styled like the RAPTURE machine interface (AMCF) |
| `camera_service.py` | The cameras as a background service with an HTTP API (also used by `app.py`) |
| `dual_camera.py` | Camera access, build plate mapping, calibration, saving and loading pictures |
| `plate_plot.py` | Drawing the cameras' pictures onto the build plate |
| `area_history.py` | Areas and their history |
| `defect_detection.py` | Monitored areas from a sliced 3MF job |
| `capture_dual_video.py` | Clip and timelapse recording |
| `views/` | The camera views in use (`thermal.json`, `optical.json`) |
| `pictures/`, `calibration/`, `areas/`, `clips/` | What the program saves (not in git) |

## Build plate coordinates

- **Origin and axes:** x' and y' are in millimetres, with the origin at the plate centre, +x' to the
  right and +y' up, seen from above.
- **Plate size:** the plate is a ring with radii 15 to 100 mm.
- **Rotation:** areas stay fixed to the plate while it turns. A picture taken with the plate turned is
  mapped back into the plate's own frame.
- **Plate angle:** for now it is typed in by hand in the Areas window; later it will come from the
  machine's PLC.

## Using the cameras from your own code

```python
from dual_camera import CameraRig

with CameraRig() as rig:
    pair = rig.grab()
    pair.thermal_c                  # temperature in °C per pixel (NaN while the camera calibrates)
    pair.optical                    # BGR picture
    pair.temperature(x, y)          # °C at build plate point(s) x', y' in mm
```
