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
| Live | Both cameras side by side. Take a picture (P, into `pictures/`), record a real-time clip (R) or a timelapse (T) as MP4 in `clips/`, record whole pictures continuously into `recordings/` (Record pictures, every n s), show the moment on the build plate (Space), open the optical camera's settings (S). |
| Calibration | Fits each camera's zoom and lens distortion from pictures of a checkerboard lying flat in the print plane (the engraved board with 8 mm squares by default, for both cameras). Save puts the result in use. |
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
| `POST /api/pictures/record` `{"every_s": 1}`, `POST /api/pictures/stop` | Start or stop recording whole pictures (0 = every new picture) |
| `GET /api/pair` | The latest picture pair, whole (NumPy `.npz`) |
| `GET /api/log?after=n` | What the service printed |

For example:
```
curl -X POST localhost:8765/api/areas/every -H "Content-Type: application/json" -d "{\"seconds\": 10}"
```

### Recording whole pictures continuously

**Record pictures** in the main window (or `POST /api/pictures/record`) stores the original pictures with
their exact times into `recordings/<start time>/`. It takes one pair every n seconds, or every new pair
with 0. Nothing is put into areas yet, so the pictures can be processed later, after everything is tested.

- **`index.csv`:** one row per pair. It holds the time of the grab and, for each camera, the time its
  frame arrived from the camera (unix seconds, from the camera PC's clock). It also holds the plate
  rotation set then and the file names.
- **`000123_thermal.npy`, `000123_optical.png`:** °C as float32, and the optical picture.
- **Writing:** a separate thread writes to disk, so the cameras never wait for it. If the disk falls
  behind by more than 30 pairs, further pairs are dropped. The window and the log show how many.
- **Disk space:** not compressed yet. That's about 5 MB per pair: 18 GB per hour at one pair per second,
  and about 35 GB per hour at full rate.

### Plate angle from AMCF, afterwards

AMCF writes a log of the plate angle with times. A CSV with a header is enough:

```
time,angle_deg
1795606788.95,123.40
1795606789.95,124.71
```

- **Time:** unix seconds, milliseconds, or an ISO date and time. ISO times count as local time unless
  they name a zone.
- **Angle:** in degrees; use `--angle-scale 0.01` for 1/100°.
- **How often:** once a second is enough while the plate turns at a steady speed; 10 per second is safer.
  Where two log entries are more than 3 s apart, the pictures in between get no angle.

```
python recording.py recordings/<start> --angles amcf_log.csv            # check: how many pictures get an angle
python recording.py recordings/<start> --angles amcf_log.csv --areas    # put them into all areas
```

Each camera's frame gets the angle interpolated at its own time, minus that camera's delay. The two
frames of one pair are up to 0.4 s apart, so this matters. The angles are written to `rotations.csv`. A
pair is never put into an area twice, so the command can be run again after adding areas.

Two things still need setting, once, at the top of `recording.py`:
- **`ANGLE_SIGN`, `ANGLE_ZERO`:** how the AMCF angle turns into our plate rotation. Check them with a mark
  on the plate at a known angle. The cameras may turn with the A axis, in which case the sign is −1.
- **`THERMAL_DELAY`, `OPTICAL_DELAY`:** how long before its arrival each camera's frame was exposed. Record
  the turning plate with a mark on it and process with delay 0. The mark then appears shifted by
  speed × delay, so the delay is the shift divided by the speed.

### Cameras on another PC (Linux camera PC, controlled from Windows)

On the camera PC (Linux):
```
~/miniconda3/envs/rapture/bin/python camera_service.py --host 0.0.0.0
```
(or `app.py --host 0.0.0.0` to have its windows there too). On the Windows PC:
```
python app.py --connect 192.168.10.2        # the camera PC's address; :port if not 8765
```

**What works over the network:**
- the live picture, the camera states and the camera PC's log (lines marked with its address);
- clips, timelapses and picture recordings, which are made and kept on the camera PC;
- pictures taken by hand, calibration pictures, alignment pictures and the build plate view: the whole
  picture pair is fetched (about 11 MB) and stored on the Windows PC.

**What doesn't:**
- **The Areas window:** it works where the cameras are, so open it on the camera PC, or process a
  recording afterwards.
- **The optical camera's settings window:** it opens on the camera PC only.
- **Views:** calibrations and alignments saved on Windows go into its own `views/`. Copy
  `views/*.json` to the camera PC if areas are recorded there.

**Network:**
- **Connection:** a direct LAN cable between the two PCs, or both on the machine network, with fixed
  addresses, e.g. 192.168.10.1 (Windows) and 192.168.10.2 (Linux), mask 255.255.255.0.
- **Firewall:** if Ubuntu's firewall is on, allow the port with
  `sudo ufw allow from 192.168.10.0/24 to any port 8765`.
- **No login:** anyone who reaches the port can control the cameras, so keep it on that cable or network.

### Clocks

The plate angles from AMCF only fit the pictures if the AMCF PC and the camera PC agree on the time. Let
one be the time source and the other follow it, over the same cable. On Ubuntu that's `chrony`:

- **Time source:** in `/etc/chrony/chrony.conf`, add `allow 192.168.10.0/24` and `local stratum 8`.
- **Follower:** add `server <source address> iburst`.
- **Check:** `chronyc tracking` shows the remaining offset, typically well under 1 ms on a direct cable.

On Windows, `w32tm /config /manualpeerlist:<source address> /syncfromflags:manual /update` reaches a few
milliseconds. At about 1.3°/s, 10 ms is 0.01°, far below what the optical camera's 0.4 s between frames
already limits.

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
| `recording.py` | Continuous recording of whole pictures; putting them into the areas with AMCF angles |
| `link_test.py` | Signal tests through the slipring: the USB extender link, or Ethernet strands between two network ports (Linux); results in `linktest/` |
| `capture_dual_video.py` | Clip and timelapse recording |
| `views/` | The camera views in use (`thermal.json`, `optical.json`) |
| `pictures/`, `calibration/`, `areas/`, `clips/`, `recordings/` | What the program saves (not in git) |

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
