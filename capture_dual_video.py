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


class Recorder:
    """One video being recorded from side_by_side pictures, placed by their timestamps: real time = one
    video frame per 1/CLIP_FPS s, the previous picture repeated while the cameras are slower; timelapse
    = one picture per TIMELAPSE_INTERVAL s."""

    def __init__(self, timelapse, size):
        self.timelapse, self.size = timelapse, size          # size = (width, height) of the video
        CLIP_DIR.mkdir(exist_ok=True)
        self.path = CLIP_DIR / f"{'timelapse' if timelapse else 'clip'}_{datetime.now():%Y%m%d_%H%M%S}.mp4"
        self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), CLIP_FPS, size)
        self.start, self.elapsed, self.last_slot, self.last_frame, self.written, self.grabbed = None, 0.0, -1, None, 0, 0
        print(f"Recording {self.path.name} ...")

    def add(self, pair, frame):
        """Add `frame` = side_by_side(pair)."""
        if frame.shape[1::-1] != self.size:                  # a camera dropped out: placeholder has another size
            frame = cv2.resize(frame, self.size)
        if self.start is None:
            self.start = pair.timestamp
        self.elapsed = pair.timestamp - self.start
        self.grabbed += 1
        slot = int(self.elapsed / (TIMELAPSE_INTERVAL if self.timelapse else 1 / CLIP_FPS))
        if slot > self.last_slot:
            if not self.timelapse:
                for _ in range(slot - self.last_slot - 1):
                    self.writer.write(self.last_frame)
            self.writer.write(frame)
            self.written += 1 if self.timelapse else slot - self.last_slot
            self.last_slot, self.last_frame = slot, frame

    def mark(self, preview):
        """Red REC / TIMELAPSE marker with the elapsed time, top right in `preview` (not in the video)."""
        text = f"{'TIMELAPSE' if self.timelapse else 'REC'} {clock(self.elapsed)}"
        w = preview.shape[1]
        tw = cv2.getTextSize(text, FONT, 0.65, 2)[0][0]
        cv2.circle(preview, (w - tw - 30, 22), 8, (0, 0, 255), -1)
        cv2.putText(preview, text, (w - tw - 14, 29), FONT, 0.65, (0, 0, 255), 2)

    def stop(self):
        self.writer.release()
        video = self.written / CLIP_FPS
        if self.timelapse:
            print(f"Saved {self.path}: {clock(self.elapsed)} of real time in {video:.1f} s of video "
                  f"({self.elapsed / max(video, 1e-9):.0f}x faster)")
        else:
            print(f"Saved {self.path}: {video:.1f} s, cameras delivered "
                  f"{(self.grabbed - 1) / max(self.elapsed, 1e-9):.1f} frames/s")


def main():
    window = "Capture (r = clip, t = timelapse, q = quit)"
    recorder = None
    try:
        with CameraRig() as rig:
            while True:
                pair = rig.grab()
                preview = frame = side_by_side(pair)
                if recorder is not None:
                    recorder.add(pair, frame)
                    preview = cv2.resize(frame, recorder.size)    # the REC marker goes on the preview only
                    recorder.mark(preview)
                cv2.imshow(window, preview)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("s"):
                    rig.optical_settings()
                if key in (ord("r"), ord("t")) and recorder is None:
                    recorder = Recorder(key == ord("t"), frame.shape[1::-1])
                elif recorder is not None and key in (ord("r"), ord("t"), 27, ord("q")):
                    recorder.stop()
                    recorder = None
                if key in (27, ord("q")) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        if recorder is not None:                         # window closed or error while recording
            recorder.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
