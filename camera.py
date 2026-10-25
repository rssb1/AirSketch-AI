"""Threaded webcam capture so the UI never blocks on camera reads."""
from __future__ import annotations

import sys
import threading
import time
from typing import Optional, Tuple

import numpy as np


class CameraCapture(threading.Thread):
    def __init__(self, index: int = 0, width: int = 640, height: int = 480, mirror: bool = True):
        super().__init__(daemon=True)
        self.index, self.width, self.height, self.mirror = index, width, height, mirror
        self.error: Optional[str] = None
        self.opened = threading.Event()
        self._frame: Optional[np.ndarray] = None
        self._id = 0
        self._lock = threading.Lock()
        self._halt = threading.Event()

    def run(self) -> None:
        import cv2
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.index, backend)
        if not cap.isOpened():
            self.error = ("Could not open the webcam. Check that no other app is using it and "
                          "that camera access is allowed for desktop apps in Windows privacy settings.")
            self.opened.set()
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.opened.set()
        failures = 0
        while not self._halt.is_set():
            ok, frame = cap.read()
            if not ok:
                failures += 1
                if failures > 60:
                    self.error = "Webcam stopped delivering frames."
                    break
                time.sleep(0.02)
                continue
            failures = 0
            if self.mirror:
                frame = cv2.flip(frame, 1)
            with self._lock:
                self._frame, self._id = frame, self._id + 1
        cap.release()

    def latest(self) -> Tuple[Optional[np.ndarray], int]:
        with self._lock:
            return self._frame, self._id

    def stop(self) -> None:
        self._halt.set()
