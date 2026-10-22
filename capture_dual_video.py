"""
capture_dual_video.py - record videos of the thermal and optical view side by side.

Run:  python capture_dual_video.py
      r = start / stop a real-time clip
      t = start / stop a timelapse (one picture every TIMELAPSE_INTERVAL s, played back at CLIP_FPS)
      s = optical camera settings window (exposure, gain)
      q or Esc = quit
Videos are saved as MP4 in the "clips" folder next to this script.
"""
from datetime import datetime
from pathlib import Path

import cv2

from dual_camera import FONT, CameraRig, side_by_side

CLIP_FPS = 25                  # frame rate of the saved videos
TIMELAPSE_INTERVAL = 1.0       # s of real time per timelapse picture (1.0 at 25 fps = 25x faster)
CLIP_DIR = Path(__file__).resolve().parent / "clips"


def clock(seconds):
    s = int(seconds)
    return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}"


def main():
    window = "Capture (r = clip, t = timelapse, q = quit)"
    writer, path, size, timelapse = None, None, None, False   # the video being recorded
    start, elapsed, last_slot, last_frame, written, grabbed = None, 0.0, -1, None, 0, 0

    def stop():
        nonlocal writer
        writer.release()
        writer = None
        video = written / CLIP_FPS
        if timelapse:
            print(f"Saved {path}: {clock(elapsed)} of real time in {video:.1f} s of video "
                  f"({elapsed / video:.0f}x faster)")
        else:
            print(f"Saved {path}: {video:.1f} s, cameras delivered {(grabbed - 1) / max(elapsed, 1e-9):.1f} frames/s")

    try:
        with CameraRig() as rig:
            while True:
                pair = rig.grab()
                frame = side_by_side(pair)
                preview = frame
                if writer is not None:
                    if frame.shape[1::-1] != size:     # a camera dropped out: placeholder has another size
                        frame = cv2.resize(frame, size)
                    if start is None:
                        start = pair.timestamp
                    elapsed = pair.timestamp - start
                    grabbed += 1
                    # Real time: one video frame per 1/CLIP_FPS s, repeating the previous picture while
                    # the cameras are slower than that. Timelapse: one picture per TIMELAPSE_INTERVAL s.
                    slot = int(elapsed / (TIMELAPSE_INTERVAL if timelapse else 1 / CLIP_FPS))
                    if slot > last_slot:
                        if not timelapse:
                            for _ in range(slot - last_slot - 1):
                                writer.write(last_frame)
                        writer.write(frame)
                        written += 1 if timelapse else slot - last_slot
                        last_slot, last_frame = slot, frame
                    preview = frame.copy()             # the REC marker goes on the preview only
                    text = f"{'TIMELAPSE' if timelapse else 'REC'} {clock(elapsed)}"
                    tw = cv2.getTextSize(text, FONT, 0.65, 2)[0][0]
                    cv2.circle(preview, (size[0] - tw - 30, 22), 8, (0, 0, 255), -1)
                    cv2.putText(preview, text, (size[0] - tw - 14, 29), FONT, 0.65, (0, 0, 255), 2)
                cv2.imshow(window, preview)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("s"):
                    rig.optical_settings()
                if key in (ord("r"), ord("t")) and writer is None:
                    timelapse = key == ord("t")
                    CLIP_DIR.mkdir(exist_ok=True)
                    path = CLIP_DIR / f"{'timelapse' if timelapse else 'clip'}_{datetime.now():%Y%m%d_%H%M%S}.mp4"
                    size = frame.shape[1::-1]
                    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), CLIP_FPS, size)
                    start, elapsed, last_slot, written, grabbed = None, 0.0, -1, 0, 0
                    print(f"Recording {path.name} ...")
                elif writer is not None and key in (ord("r"), ord("t"), 27, ord("q")):
                    stop()
                if key in (27, ord("q")) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        if writer is not None:                         # window closed or error while recording
            stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
