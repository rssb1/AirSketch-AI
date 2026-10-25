"""Drawing evaluation.

Pipeline (see README for details):
  1. Local shape comparison (always available): both images are reduced to normalized ink masks and
     compared with a tolerance-based precision/recall F-score. The drawing is also matched against every
     reference challenge to report a "closest match".
  2. Optional vision-language model (OpenAI, key from the OPENAI_API_KEY env var / .env) that
     identifies the drawing blind (without being told the target) and rates semantic similarity.
  The final score blends both when the API is available; otherwise the local score is returned and
  labelled approximate. Neither is an objective measure of artistic skill.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol

import cv2
import numpy as np
from PIL import Image

MASK_SIZE = 128
TOLERANCE_PX = 3          # how far (in 128px space) ink may be from the reference and still count
F_LOW, F_HIGH = 0.30, 0.80  # F-scores mapped linearly onto 0..100 (below/above are clipped)
SEMANTIC_WEIGHT, VISUAL_WEIGHT = 0.6, 0.4
MIN_INK_PIXELS = 20
DENSE_INK_RATIO = 2.0     # drawings with more than 2x the reference's ink are penalized

SCORE_EXPLANATION = (
    "The score blends how recognizable your drawing is (AI judgment) with how closely its overall shape and "
    "layout match the reference. It rewards resemblance to this particular reference, not artistic skill."
)
LOCAL_EXPLANATION = (
    "Approximate score: computed offline by comparing normalized outlines (no AI model). "
    "It measures shape overlap with the reference, not artistic skill."
)


class EvaluationUnavailable(Exception):
    """Raised by an AI backend that cannot run (no key, no package, network error, bad reply)."""


@dataclass
class EvaluationResult:
    score: float
    recognized: str = ""
    feedback: str = ""
    suggestions: List[str] = field(default_factory=list)
    method: str = "local"          # "ai+local" or "local"
    approximate: bool = False
    visual_score: Optional[float] = None
    semantic_score: Optional[float] = None
    explanation: str = ""
    note: str = ""                 # e.g. why the AI backend was skipped
    error: str = ""                # non-empty => the drawing could not be evaluated

    @property
    def ok(self) -> bool:
        return not self.error


# ------------------------------------------------------------------ image processing
def ink_mask(img: Image.Image) -> np.ndarray:
    """Boolean mask of non-white pixels (works for any ink color, including light yellow)."""
    arr = np.asarray(img.convert("RGB"), dtype=np.int16)
    return (255 - arr.min(axis=2)) > 40


def normalize_mask(mask: np.ndarray, size: int = MASK_SIZE, pad: int = 8) -> Optional[np.ndarray]:
    """Crop to the ink bounding box, scale (keeping aspect ratio) to fit `size`, center. None if empty."""
    ys, xs = np.nonzero(mask)
    if len(xs) < MIN_INK_PIXELS:
        return None
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8) * 255
    h, w = crop.shape
    scale = (size - 2 * pad) / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    small = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    small = (small > 40).astype(np.uint8)
    out = np.zeros((size, size), np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = small
    return out.astype(bool)


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(mask.astype(np.uint8), k).astype(bool)


def shape_f_score(ref: np.ndarray, drawing: np.ndarray, tol: int = TOLERANCE_PX):
    """Return (f, precision, recall) using ink-within-tolerance matching on thinned-then-compared masks."""
    ref_t = _dilate(ref, 1)
    dr_t = _dilate(drawing, 1)
    precision = float((dr_t & _dilate(ref, tol)).sum() / max(dr_t.sum(), 1))
    recall = float((ref_t & _dilate(drawing, tol)).sum() / max(ref_t.sum(), 1))
    f = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return f, precision, recall


def f_to_score(f: float) -> float:
    return float(np.clip((f - F_LOW) / (F_HIGH - F_LOW), 0.0, 1.0) * 100.0)


def visual_similarity(reference: Image.Image, drawing: Image.Image) -> Optional[Dict[str, float]]:
    """Shape similarity 0-100 plus diagnostics, or None if either image has no ink."""
    r = normalize_mask(ink_mask(reference))
    d = normalize_mask(ink_mask(drawing))
    if r is None or d is None:
        return None
    f, p, rec = shape_f_score(r, d)
    ink_ratio = float(d.sum() / max(r.sum(), 1))
    score = f_to_score(f)
    if ink_ratio > DENSE_INK_RATIO:  # discourage "scribble everywhere" strategies
        score *= DENSE_INK_RATIO / ink_ratio
    return {"score": score, "precision": p, "recall": rec, "ink_ratio": ink_ratio,
            "aspect_ref": _aspect(ink_mask(reference)), "aspect_drawing": _aspect(ink_mask(drawing))}


def _aspect(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 1.0
    return float((xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1))


def closest_match(drawing: Image.Image, candidates: Dict[str, Image.Image]):
    """Nearest reference by shape similarity: [(name, score), ...] sorted best-first."""
    d = normalize_mask(ink_mask(drawing))
    if d is None:
        return []
    out = []
    for name, ref in candidates.items():
        r = normalize_mask(ink_mask(ref))
        if r is not None:
            out.append((name, f_to_score(shape_f_score(r, d)[0])))
    return sorted(out, key=lambda t: t[1], reverse=True)


def local_feedback(diag: Dict[str, float], recognized_ok: Optional[bool]) -> (str, List[str]):
    """Feedback derived strictly from the measured diagnostics."""
    tips: List[str] = []
    if diag["recall"] < 0.5:
        tips.append("Parts of the reference are missing - add more of its main features and details.")
    if diag["precision"] < 0.5:
        tips.append("Some of your lines fall away from the reference shape - try fewer stray strokes.")
    if diag["ink_ratio"] < 0.4:
        tips.append("Your drawing is quite sparse - draw bolder, more complete outlines.")
    elif diag["ink_ratio"] > 2.5:
        tips.append("Your drawing has a lot more lines than the reference - keep it simpler.")
    ar, ad = diag["aspect_ref"], diag["aspect_drawing"]
    if ad > ar * 1.4:
        tips.append("Your drawing is wider than the reference - make it taller.")
    elif ad < ar / 1.4:
        tips.append("Your drawing is taller than the reference - make it wider.")
    if recognized_ok is False:
        tips.append("Another reference looked closer to your drawing than the target one.")
    if not tips:
        tips.append("Good overall match - refine the proportions for an even closer fit.")
    head = ("Shape overlap with the reference is "
            f"{diag['score']:.0f}/100 (precision {diag['precision']:.0%}, coverage {diag['recall']:.0%}).")
    return head, tips[:3]


# ------------------------------------------------------------------ backends
class AIBackend(Protocol):
    def evaluate(self, reference: Image.Image, drawing: Image.Image) -> Dict: ...


def _png_b64(img: Image.Image, max_side: int = 768) -> str:
    img = img.convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


PROMPT = (
    "You are judging a Pictionary-style game. Image 1 is the REFERENCE drawing the player was asked to "
    "recreate. Image 2 is the player's hand-drawn attempt (drawn in the air with a fingertip, so lines are "
    "wobbly - judge shapes, not neatness).\n"
    "First decide, WITHOUT looking at Image 1, what Image 2 depicts. Then compare to Image 1.\n"
    "Reply with ONLY a JSON object, no markdown:\n"
    '{"recognized": "<short noun phrase for what Image 2 looks like>", '
    '"same_object": <true|false>, "semantic_score": <0-100 integer: does Image 2 represent the same object '
    'as Image 1>, "feedback": "<one or two friendly sentences>", '
    '"suggestions": ["<specific improvement>", "<specific improvement>"]}'
)


class OpenAIBackend:
    """Vision evaluation through the OpenAI API. The key is read from the environment only."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, timeout: float = 30.0):
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        self.model = model or os.environ.get("AIRSKETCH_MODEL", "gpt-4o-mini")
        self.timeout = timeout

    def available(self) -> bool:
        return bool(self.api_key.strip())

    def evaluate(self, reference: Image.Image, drawing: Image.Image) -> Dict:
        if not self.available():
            raise EvaluationUnavailable("No OPENAI_API_KEY configured.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise EvaluationUnavailable("The 'openai' package is not installed.") from exc

        def url(img):
            return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + _png_b64(img)}}

        try:
            client = OpenAI(api_key=self.api_key, timeout=self.timeout, max_retries=1)
            resp = client.chat.completions.create(
                model=self.model, max_tokens=500,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": PROMPT}, url(reference), url(drawing)]}])
            text = resp.choices[0].message.content or ""
        except Exception as exc:  # network, auth, rate limit...
            raise EvaluationUnavailable(f"AI request failed: {type(exc).__name__}") from exc
        return parse_ai_reply(text)


def parse_ai_reply(text: str) -> Dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise EvaluationUnavailable("AI reply contained no JSON.")
    try:
        data = json.loads(m.group(0))
        score = float(data["semantic_score"])
    except (ValueError, KeyError, TypeError) as exc:
        raise EvaluationUnavailable("AI reply was malformed.") from exc
    sugg = data.get("suggestions") or []
    return {"recognized": str(data.get("recognized", "")).strip(),
            "semantic_score": float(np.clip(score, 0, 100)),
            "feedback": str(data.get("feedback", "")).strip(),
            "suggestions": [str(s) for s in sugg][:3] if isinstance(sugg, list) else []}


# ------------------------------------------------------------------ facade
class ImageEvaluator:
    def __init__(self, ai_backend: Optional[AIBackend] = None):
        self.backend = ai_backend if ai_backend is not None else OpenAIBackend()

    def evaluate(self, reference: Optional[Image.Image], drawing: Optional[Image.Image],
                 target_name: str = "", candidates: Optional[Dict[str, Image.Image]] = None
                 ) -> EvaluationResult:
        if reference is None:
            return EvaluationResult(0.0, error="The reference image is missing, so the drawing cannot be judged.")
        if drawing is None or normalize_mask(ink_mask(drawing)) is None:
            return EvaluationResult(0.0, error="The canvas is empty - draw something first!")

        diag = visual_similarity(reference, drawing)
        if diag is None:
            return EvaluationResult(0.0, error="The reference image has no visible lines.")
        visual = diag["score"]

        ranking = closest_match(drawing, candidates) if candidates else []
        recognized_ok = None
        local_label = ""
        if ranking:
            local_label = f"closest match: {ranking[0][0]}"
            recognized_ok = (not target_name) or ranking[0][0] == target_name

        note = ""
        try:
            ai = self.backend.evaluate(reference, drawing) if self.backend else None
            if ai is None:
                raise EvaluationUnavailable("AI backend disabled.")
            sem = ai["semantic_score"]
            score = SEMANTIC_WEIGHT * sem + VISUAL_WEIGHT * visual
            return EvaluationResult(
                score=float(np.clip(score, 0, 100)), recognized=ai["recognized"] or local_label,
                feedback=ai["feedback"], suggestions=ai["suggestions"] or local_feedback(diag, recognized_ok)[1],
                method="ai+local", approximate=False, visual_score=visual, semantic_score=sem,
                explanation=SCORE_EXPLANATION)
        except EvaluationUnavailable as exc:
            note = str(exc)

        head, tips = local_feedback(diag, recognized_ok)
        score = visual
        if recognized_ok is False:
            score *= 0.7  # a different reference matched better: penalize, but stay transparent in the label
        return EvaluationResult(
            score=float(np.clip(score, 0, 100)), recognized=local_label or "unknown (offline mode)",
            feedback=head, suggestions=tips, method="local", approximate=True, visual_score=visual,
            explanation=LOCAL_EXPLANATION, note=f"AI evaluation unavailable ({note}) - using offline comparison.")
