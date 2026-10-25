"""Virtual canvas: stroke rendering, smoothing, and export. No camera or UI dependencies."""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

CANVAS_W, CANVAS_H = 640, 480

# name -> RGB
COLORS = {
    "Ink": (30, 30, 30),
    "Red": (220, 50, 47),
    "Orange": (245, 130, 20),
    "Green": (40, 160, 70),
    "Blue": (40, 100, 220),
    "Purple": (140, 70, 200),
}
BRUSH_SIZES = (4, 8, 14)


class PointSmoother:
    """Exponential moving average to reduce fingertip jitter."""

    def __init__(self, alpha: float = 0.45):
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self._p: Optional[Tuple[float, float]] = None

    def update(self, x: float, y: float) -> Tuple[float, float]:
        if self._p is None:
            self._p = (x, y)
        else:
            px, py = self._p
            self._p = (px + self.alpha * (x - px), py + self.alpha * (y - py))
        return self._p

    def reset(self) -> None:
        self._p = None


class DrawingCanvas:
    def __init__(self, width: int = CANVAS_W, height: int = CANVAS_H):
        self.width, self.height = width, height
        self.color = COLORS["Ink"]
        self.brush = BRUSH_SIZES[1]
        self.smoother = PointSmoother()
        self.frozen = False  # True once time runs out / drawing submitted
        self._last: Optional[Tuple[int, int]] = None
        self._img = np.full((height, width, 3), 255, np.uint8)  # RGB
        self._ink = np.zeros((height, width), bool)

    # ------------------------------------------------------------ drawing
    def add_point(self, x: float, y: float) -> bool:
        """Pen-down point. Connects to the previous point of the same stroke."""
        if self.frozen:
            return False
        sx, sy = self.smoother.update(x, y)
        pt = (int(round(min(max(sx, 0), self.width - 1))),
              int(round(min(max(sy, 0), self.height - 1))))
        if self._last is not None and np.hypot(pt[0] - self._last[0], pt[1] - self._last[1]) < 1.5:
            return True  # dead zone: ignore sub-pixel jitter
        start = self._last or pt
        cv2.line(self._img, start, pt, self.color, self.brush, cv2.LINE_AA)
        cv2.line(self._ink.view(np.uint8), start, pt, 1, self.brush)
        if self._last is None:  # single tap -> dot
            cv2.circle(self._img, pt, max(self.brush // 2, 1), self.color, -1, cv2.LINE_AA)
        self._last = pt
        return True

    def lift(self) -> None:
        """End the current stroke (pen up, hand lost, gesture changed)."""
        self._last = None
        self.smoother.reset()

    def clear(self) -> None:
        self._img[:] = 255
        self._ink[:] = False
        self.lift()

    # ------------------------------------------------------------ settings
    def set_color(self, rgb: Tuple[int, int, int]) -> None:
        self.color = tuple(int(c) for c in rgb)

    def set_brush(self, size: int) -> None:
        self.brush = int(max(1, size))

    # ------------------------------------------------------------ export
    def is_empty(self, min_pixels: int = 20) -> bool:
        return int(self._ink.sum()) < min_pixels

    @property
    def array(self) -> np.ndarray:
        return self._img

    def to_pil(self) -> Image.Image:
        return Image.fromarray(self._img.copy(), "RGB")


def normalized_to_canvas(nx: float, ny: float, width: int = CANVAS_W, height: int = CANVAS_H,
                         margin: float = 0.12) -> Tuple[float, float]:
    """Map a normalized camera coordinate to canvas pixels.

    Only the central (1 - 2*margin) region of the camera frame is used so the
    player can reach the canvas edges without leaving the camera's view.
    """
    span = 1.0 - 2 * margin
    u = min(max((nx - margin) / span, 0.0), 1.0)
    v = min(max((ny - margin) / span, 0.0), 1.0)
    return u * (width - 1), v * (height - 1)
