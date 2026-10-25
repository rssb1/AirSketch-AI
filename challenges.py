"""Reference challenges, drawn locally with Pillow (no external images needed)."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from PIL import Image, ImageDraw

SIZE = 400
LINE = 8
DIFFICULTIES = ("easy", "medium", "hard")
ASSET_DIR = Path(__file__).resolve().parent / "assets" / "challenges"


# ---------------------------------------------------------------- helpers
def _pts(points):
    return [tuple(p) for p in points]


def _poly(d: ImageDraw.ImageDraw, points) -> None:
    pts = _pts(points)
    d.line(pts + [pts[0]], fill=0, width=LINE, joint="curve")


def _line(d, *points) -> None:
    d.line(_pts(points), fill=0, width=LINE, joint="curve")


def _circle(d, cx, cy, r, fill=False) -> None:
    d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=0, fill=0 if fill else None, width=LINE)


def _ellipse(d, box) -> None:
    d.ellipse(box, outline=0, width=LINE)


def _rect(d, box) -> None:
    x0, y0, x1, y1 = box
    _poly(d, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _battlements(d, x0, x1, y, h=24, n=3) -> None:
    step = (x1 - x0) / (2 * n - 1)
    pts = [(x0, y)]
    for i in range(2 * n - 1):
        x = x0 + step * i
        if i % 2 == 0:
            pts += [(x, y - h), (x + step, y - h)]
        else:
            pts += [(x, y), (x + step, y)]
    pts.append((x1, y))
    d.line(_pts(pts), fill=0, width=LINE, joint="curve")


# ---------------------------------------------------------------- easy
def _sun(d):
    _circle(d, 200, 200, 60)
    for k in range(8):
        a = k * math.pi / 4
        _line(d, (200 + 90 * math.cos(a), 200 + 90 * math.sin(a)),
              (200 + 140 * math.cos(a), 200 + 140 * math.sin(a)))


def _house(d):
    _rect(d, (110, 190, 290, 340))
    _poly(d, [(85, 190), (200, 85), (315, 190)])
    _rect(d, (175, 360 - 100, 225, 340))
    _rect(d, (130, 215, 160, 245))


def _tree(d):
    _rect(d, (175, 250, 225, 355))
    _ellipse(d, (95, 60, 305, 270))


def _flower(d):
    _circle(d, 200, 140, 24)
    for k in range(6):
        a = k * math.pi / 3
        _circle(d, 200 + 62 * math.cos(a), 140 + 62 * math.sin(a), 30)
    _line(d, (200, 230), (200, 365))
    _ellipse(d, (200, 280, 275, 320))


def _fish(d):
    _ellipse(d, (80, 140, 290, 260))
    _poly(d, [(285, 200), (355, 140), (355, 260)])
    _circle(d, 140, 185, 7, fill=True)
    _line(d, (110, 200), (125, 215), (145, 215))


def _star(d):
    pts = []
    for k in range(10):
        r = 150 if k % 2 == 0 else 62
        a = -math.pi / 2 + k * math.pi / 5
        pts.append((200 + r * math.cos(a), 205 + r * math.sin(a)))
    _poly(d, pts)


# ---------------------------------------------------------------- medium
def _cat(d):
    _ellipse(d, (105, 150, 295, 330))
    _poly(d, [(112, 215), (122, 85), (200, 155)])
    _poly(d, [(288, 215), (278, 85), (200, 155)])
    _circle(d, 160, 220, 9, fill=True)
    _circle(d, 240, 220, 9, fill=True)
    _poly(d, [(188, 252), (212, 252), (200, 268)])
    _line(d, (60, 250), (130, 265))
    _line(d, (60, 295), (130, 280))
    _line(d, (340, 250), (270, 265))
    _line(d, (340, 295), (270, 280))


def _rocket(d):
    _poly(d, [(200, 45), (262, 140), (262, 300), (138, 300), (138, 140)])
    _circle(d, 200, 175, 30)
    _poly(d, [(138, 240), (85, 325), (138, 300)])
    _poly(d, [(262, 240), (315, 325), (262, 300)])
    _poly(d, [(168, 300), (200, 365), (232, 300)])


def _umbrella(d):
    d.arc((55, 90, 345, 350), 180, 360, fill=0, width=LINE)
    _line(d, (55, 220), (345, 220))
    _line(d, (200, 220), (200, 325))
    d.arc((160, 305, 200, 345), 0, 180, fill=0, width=LINE)


def _sailboat(d):
    _poly(d, [(75, 290), (325, 290), (270, 355), (130, 355)])
    _line(d, (200, 290), (200, 55))
    _poly(d, [(212, 70), (212, 270), (315, 270)])
    _poly(d, [(188, 110), (188, 270), (105, 270)])


def _car(d):
    _rect(d, (55, 190, 345, 285))
    _poly(d, [(115, 190), (155, 120), (250, 120), (290, 190)])
    _circle(d, 130, 290, 36)
    _circle(d, 270, 290, 36)


def _mushroom(d):
    d.chord((65, 70, 335, 310), 180, 360, outline=0, width=LINE)
    _rect(d, (160, 190, 240, 335))
    _circle(d, 135, 150, 16)
    _circle(d, 210, 115, 14)
    _circle(d, 270, 160, 16)


# ---------------------------------------------------------------- hard
def _bicycle(d):
    _circle(d, 100, 280, 62)
    _circle(d, 300, 280, 62)
    a, b, c, e = (100, 280), (190, 280), (160, 175), (265, 170)
    _line(d, a, b, c, a)
    _line(d, c, e, b)
    _line(d, e, (300, 280))
    _line(d, (138, 165), (182, 165))
    _line(d, e, (275, 130), (310, 130))


def _castle(d):
    _rect(d, (70, 125, 140, 340))
    _rect(d, (260, 125, 330, 340))
    _rect(d, (140, 200, 260, 340))
    _battlements(d, 70, 140, 125)
    _battlements(d, 260, 330, 125)
    _battlements(d, 140, 260, 200, h=20, n=4)
    _rect(d, (178, 265, 222, 340))
    d.arc((178, 243, 222, 287), 180, 360, fill=0, width=LINE)


def _robot(d):
    _rect(d, (130, 70, 270, 165))
    _circle(d, 170, 115, 13)
    _circle(d, 230, 115, 13)
    _line(d, (165, 145), (235, 145))
    _line(d, (200, 70), (200, 38))
    _circle(d, 200, 28, 9)
    _rect(d, (110, 185, 290, 320))
    _line(d, (110, 205), (60, 275))
    _line(d, (290, 205), (340, 275))
    _rect(d, (140, 320, 180, 372))
    _rect(d, (220, 320, 260, 372))
    _circle(d, 200, 250, 18)


def _windmill(d):
    _poly(d, [(155, 355), (245, 355), (226, 175), (174, 175)])
    _circle(d, 200, 165, 11)
    for k in range(4):
        a = math.pi / 4 + k * math.pi / 2
        dx, dy = math.cos(a), math.sin(a)
        tip = (200 + 135 * dx, 165 + 135 * dy)
        side = (tip[0] - 32 * dy, tip[1] + 32 * dx)
        _poly(d, [(200, 165), tip, side])
    _rect(d, (183, 300, 217, 355))


def _lighthouse(d):
    _poly(d, [(150, 355), (250, 355), (232, 135), (168, 135)])
    _rect(d, (172, 80, 228, 135))
    _poly(d, [(160, 80), (200, 40), (240, 80)])
    for y in (210, 280):
        off = (y - 135) * 18 / 220
        _line(d, (168 - off, y), (232 + off, y))
    _line(d, (228, 105), (345, 70))
    _line(d, (228, 105), (345, 140))
    _line(d, (105, 355), (295, 355))


def _airplane(d):
    _ellipse(d, (55, 170, 345, 235))
    _poly(d, [(70, 195), (35, 120), (105, 180)])
    _poly(d, [(190, 205), (245, 335), (285, 335), (260, 205)])
    _poly(d, [(190, 200), (235, 75), (275, 75), (260, 200)])
    for x in (250, 285, 320):
        _circle(d, x, 200, 8)


@dataclass(frozen=True)
class Challenge:
    key: str
    name: str
    difficulty: str
    renderer: Callable[[ImageDraw.ImageDraw], None]

    @property
    def filename(self) -> str:
        return f"{self.difficulty}_{self.key}.png"

    def render(self) -> Image.Image:
        img = Image.new("RGB", (SIZE, SIZE), "white")
        self.renderer(ImageDraw.Draw(img))
        return img


CHALLENGES: List[Challenge] = [
    Challenge("sun", "sun", "easy", _sun),
    Challenge("house", "house", "easy", _house),
    Challenge("tree", "tree", "easy", _tree),
    Challenge("flower", "flower", "easy", _flower),
    Challenge("fish", "fish", "easy", _fish),
    Challenge("star", "star", "easy", _star),
    Challenge("cat", "cat", "medium", _cat),
    Challenge("rocket", "rocket", "medium", _rocket),
    Challenge("umbrella", "umbrella", "medium", _umbrella),
    Challenge("sailboat", "sailboat", "medium", _sailboat),
    Challenge("car", "car", "medium", _car),
    Challenge("mushroom", "mushroom", "medium", _mushroom),
    Challenge("bicycle", "bicycle", "hard", _bicycle),
    Challenge("castle", "castle", "hard", _castle),
    Challenge("robot", "robot", "hard", _robot),
    Challenge("windmill", "windmill", "hard", _windmill),
    Challenge("lighthouse", "lighthouse", "hard", _lighthouse),
    Challenge("airplane", "airplane", "hard", _airplane),
]


def by_difficulty(difficulty: str) -> List[Challenge]:
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"Unknown difficulty: {difficulty!r}")
    return [c for c in CHALLENGES if c.difficulty == difficulty]


def pick_challenge(difficulty: str, exclude: Optional[str] = None,
                   rng: Optional[random.Random] = None) -> Challenge:
    """Random challenge for a difficulty, avoiding `exclude` (a key) when possible."""
    rng = rng or random
    pool = by_difficulty(difficulty)
    if exclude and len(pool) > 1:
        pool = [c for c in pool if c.key != exclude]
    return rng.choice(pool)


def ensure_assets(directory: Path = ASSET_DIR) -> None:
    """Write any missing reference PNGs to disk (also useful for the README screenshots)."""
    directory.mkdir(parents=True, exist_ok=True)
    for c in CHALLENGES:
        path = directory / c.filename
        if not path.exists():
            c.render().save(path)


def load_reference(challenge: Challenge, directory: Path = ASSET_DIR) -> Image.Image:
    """Load the PNG for a challenge; re-render it if the file is missing or unreadable."""
    path = directory / challenge.filename
    try:
        with Image.open(path) as im:
            return im.convert("RGB")
    except (FileNotFoundError, OSError):
        return challenge.render()


def all_references() -> Dict[str, Image.Image]:
    return {c.name: c.render() for c in CHALLENGES}
