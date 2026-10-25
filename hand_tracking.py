"""MediaPipe hand tracking + gesture classification.

`classify_gesture` and the helper classes are pure Python so they can be tested without
a webcam or MediaPipe. MediaPipe is imported lazily inside `HandTracker`.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

Point = Tuple[float, float]

POINT, PEACE, OPEN, FIST, NONE = "point", "peace", "open", "fist", "none"
WRIST = 0
FINGERS = {"index": (8, 6), "middle": (12, 10), "ring": (16, 14), "pinky": (20, 18)}  # (tip, pip)


@dataclass
class HandState:
    tip: Optional[Point] = None      # normalized (0-1), already mirrored
    gesture: str = NONE


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def extended_fingers(lm: Sequence[Point]) -> dict:
    """A finger counts as extended if its tip is clearly farther from the wrist than its PIP joint.
    This works at any hand rotation (unlike comparing y coordinates)."""
    wrist = lm[WRIST]
    return {n: _dist(lm[t], wrist) > _dist(lm[p], wrist) * 1.15 for n, (t, p) in FINGERS.items()}


def classify_gesture(lm: Optional[Sequence[Point]]) -> str:
    """POINT: index only (draw). PEACE: index+middle (pen up). OPEN: four fingers (clear). Else FIST."""
    if lm is None or len(lm) < 21:
        return NONE
    e = extended_fingers(lm)
    others = e["middle"] or e["ring"] or e["pinky"]
    if e["index"] and not others:
        return POINT
    if e["index"] and e["middle"] and not e["ring"] and not e["pinky"]:
        return PEACE
    if all(e.values()):
        return OPEN
    return FIST


class GestureStabilizer:
    """Only switch gesture after it has been seen `min_frames` times in a row (anti-flicker)."""

    def __init__(self, min_frames: int = 3):
        self.min_frames = min_frames
        self.current = NONE
        self._cand, self._count = NONE, 0

    def update(self, g: str) -> str:
        if g == self.current:
            self._cand, self._count = g, 0
            return self.current
        if g == self._cand:
            self._count += 1
        else:
            self._cand, self._count = g, 1
        if self._count >= self.min_frames or g == NONE:
            self.current, self._cand, self._count = g, g, 0
        return self.current


class HoldDetector:
    """Fires once when `target` gesture has been held continuously for `seconds`."""

    def __init__(self, target: str = OPEN, seconds: float = 1.0):
        self.target, self.seconds = target, seconds
        self._since: Optional[float] = None
        self._fired = False

    def progress(self, now: float) -> float:
        return 0.0 if self._since is None else min((now - self._since) / self.seconds, 1.0)

    def update(self, gesture: str, now: float) -> bool:
        if gesture != self.target:
            self._since, self._fired = None, False
            return False
        if self._since is None:
            self._since = now
        if not self._fired and now - self._since >= self.seconds:
            self._fired = True
            return True
        return False


class HandTracker:
    """Thin wrapper around mediapipe.solutions.hands (mediapipe==0.10.14 recommended)."""

    def __init__(self, mirror: bool = True):
        try:
            import mediapipe as mp
            self._hands = mp.solutions.hands.Hands(
                max_num_hands=1, model_complexity=0,
                min_detection_confidence=0.6, min_tracking_confidence=0.5)
        except Exception as exc:  # ImportError or AttributeError on newer mediapipe
            raise RuntimeError(
                "MediaPipe Hands is unavailable (install mediapipe==0.10.14 on Python 3.9-3.12): "
                f"{exc}") from exc
        self.mirror = mirror
        self._stab = GestureStabilizer()

    def process(self, frame_rgb) -> HandState:
        res = self._hands.process(frame_rgb)
        if not res.multi_hand_landmarks:
            self._stab.update(NONE)
            return HandState()
        lm = [(p.x, p.y) for p in res.multi_hand_landmarks[0].landmark]
        gesture = self._stab.update(classify_gesture(lm))
        x, y = lm[8]
        return HandState(tip=(1.0 - x if self.mirror else x, y), gesture=gesture)

    def close(self) -> None:
        self._hands.close()


class HandTrackingWorker(threading.Thread):
    """Runs landmark detection on the newest camera frame, independently of rendering."""

    def __init__(self, capture):
        super().__init__(daemon=True)
        self.capture = capture
        self.error: Optional[str] = None
        self._state = HandState()
        self._lock = threading.Lock()
        self._halt = threading.Event()

    def run(self) -> None:
        import cv2
        try:
            tracker = HandTracker(mirror=False)  # camera frames are already mirrored
        except RuntimeError as exc:
            self.error = str(exc)
            return
        last_id = -1
        while not self._halt.is_set():
            frame, fid = self.capture.latest()
            if frame is None or fid == last_id:
                time.sleep(0.005)
                continue
            last_id = fid
            state = tracker.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            with self._lock:
                self._state = state
        tracker.close()

    def state(self) -> HandState:
        with self._lock:
            return self._state

    def stop(self) -> None:
        self._halt.set()
