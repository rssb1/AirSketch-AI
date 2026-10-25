import unittest

import numpy as np
from PIL import Image, ImageDraw

import challenges as ch
import image_evaluator as ev

REFS = ch.all_references()


class RaisingBackend:
    def evaluate(self, reference, drawing):
        raise ev.EvaluationUnavailable("no network")


class FixedBackend:
    def __init__(self, semantic=80.0, recognized="a cat"):
        self.reply = {"recognized": recognized, "semantic_score": semantic,
                      "feedback": "Nice ears!", "suggestions": ["Add whiskers"]}
        self.calls = 0

    def evaluate(self, reference, drawing):
        self.calls += 1
        return dict(self.reply)


def blank(size=(640, 480)):
    return Image.new("RGB", size, "white")


def scaled_copy(name, size=(640, 480)):
    """Same picture, different canvas size and ink color - like a faithful air drawing."""
    img = REFS[name].convert("L").point(lambda v: 255 if v > 128 else 30)
    side = min(size)
    out = Image.new("RGB", size, "white")
    out.paste(Image.merge("RGB", (img, img, img)).resize((side, side)), ((size[0] - side) // 2, 0))
    return out


class PreprocessingTests(unittest.TestCase):
    def test_blank_canvas_has_no_ink(self):
        self.assertIsNone(ev.normalize_mask(ev.ink_mask(blank())))

    def test_tiny_speck_is_not_a_drawing(self):
        img = blank()
        img.putpixel((10, 10), (0, 0, 0))
        self.assertIsNone(ev.normalize_mask(ev.ink_mask(img)))

    def test_light_colors_still_count_as_ink(self):
        img = blank()
        ImageDraw.Draw(img).line([(50, 50), (300, 300)], fill=(255, 220, 0), width=8)
        self.assertIsNotNone(ev.normalize_mask(ev.ink_mask(img)))

    def test_normalization_is_position_and_scale_invariant(self):
        a, b = blank(), blank()
        ImageDraw.Draw(a).rectangle((100, 100, 200, 180), outline="black", width=6)
        ImageDraw.Draw(b).rectangle((300, 200, 600, 440), outline="black", width=6)
        ma, mb = ev.normalize_mask(ev.ink_mask(a)), ev.normalize_mask(ev.ink_mask(b))
        f, _, _ = ev.shape_f_score(ma, mb)
        self.assertGreater(f, 0.5)


class ScoringTests(unittest.TestCase):
    def test_f_to_score_boundaries(self):
        self.assertEqual(ev.f_to_score(0.0), 0.0)
        self.assertEqual(ev.f_to_score(ev.F_LOW), 0.0)
        self.assertEqual(ev.f_to_score(ev.F_HIGH), 100.0)
        self.assertEqual(ev.f_to_score(1.0), 100.0)
        mid = ev.f_to_score((ev.F_LOW + ev.F_HIGH) / 2)
        self.assertAlmostEqual(mid, 50.0)

    def test_identical_images_score_near_100(self):
        diag = ev.visual_similarity(REFS["cat"], REFS["cat"])
        self.assertGreaterEqual(diag["score"], 99)

    def test_faithful_copy_beats_wrong_object(self):
        good = ev.visual_similarity(REFS["house"], scaled_copy("house"))["score"]
        bad = ev.visual_similarity(REFS["house"], scaled_copy("star"))["score"]
        self.assertGreater(good, 90)
        self.assertGreater(good, bad + 25)

    def test_closest_match_identifies_object(self):
        ranking = ev.closest_match(scaled_copy("rocket"), REFS)
        self.assertEqual(ranking[0][0], "rocket")

    def test_closest_match_empty_drawing(self):
        self.assertEqual(ev.closest_match(blank(), REFS), [])


class EvaluatorTests(unittest.TestCase):
    def test_empty_drawing_is_an_error_not_a_score(self):
        res = ev.ImageEvaluator(FixedBackend()).evaluate(REFS["cat"], blank(), "cat", REFS)
        self.assertFalse(res.ok)
        self.assertEqual(res.score, 0)
        self.assertIn("empty", res.error.lower())

    def test_empty_drawing_never_calls_the_api(self):
        backend = FixedBackend()
        ev.ImageEvaluator(backend).evaluate(REFS["cat"], blank(), "cat", REFS)
        self.assertEqual(backend.calls, 0)

    def test_missing_reference(self):
        res = ev.ImageEvaluator(RaisingBackend()).evaluate(None, scaled_copy("cat"), "cat", REFS)
        self.assertFalse(res.ok)
        self.assertIn("reference", res.error.lower())

    def test_missing_drawing(self):
        res = ev.ImageEvaluator(RaisingBackend()).evaluate(REFS["cat"], None)
        self.assertFalse(res.ok)

    def test_fallback_when_api_unavailable(self):
        res = ev.ImageEvaluator(RaisingBackend()).evaluate(
            REFS["cat"], scaled_copy("cat"), "cat", REFS)
        self.assertTrue(res.ok)
        self.assertEqual(res.method, "local")
        self.assertTrue(res.approximate)
        self.assertGreater(res.score, 85)
        self.assertIn("approximate", res.explanation.lower())
        self.assertIn("unavailable", res.note.lower())
        self.assertTrue(res.suggestions)

    def test_fallback_penalizes_wrong_object(self):
        evl = ev.ImageEvaluator(RaisingBackend())
        right = evl.evaluate(REFS["cat"], scaled_copy("cat"), "cat", REFS)
        wrong = evl.evaluate(REFS["cat"], scaled_copy("windmill"), "cat", REFS)
        self.assertGreater(right.score, wrong.score + 40)
        self.assertEqual(wrong.recognized, "closest match: windmill")

    def test_no_api_key_means_fallback(self):
        evl = ev.ImageEvaluator(ev.OpenAIBackend(api_key=""))
        res = evl.evaluate(REFS["sun"], scaled_copy("sun"), "sun", REFS)
        self.assertEqual(res.method, "local")
        self.assertTrue(res.approximate)

    def test_ai_score_blends_semantic_and_visual(self):
        res = ev.ImageEvaluator(FixedBackend(semantic=80)).evaluate(
            REFS["cat"], scaled_copy("cat"), "cat", REFS)
        expected = ev.SEMANTIC_WEIGHT * 80 + ev.VISUAL_WEIGHT * res.visual_score
        self.assertAlmostEqual(res.score, expected)
        self.assertEqual(res.method, "ai+local")
        self.assertFalse(res.approximate)
        self.assertEqual(res.recognized, "a cat")

    def test_score_always_within_bounds(self):
        for sem in (-50, 0, 100, 400):
            res = ev.ImageEvaluator(FixedBackend(semantic=sem)).evaluate(
                REFS["cat"], scaled_copy("cat"), "cat", REFS)
            self.assertTrue(0 <= res.score <= 100)


class ParseReplyTests(unittest.TestCase):
    def test_parses_json_wrapped_in_text(self):
        out = ev.parse_ai_reply('Sure!\n{"recognized":"cat","semantic_score":"88","feedback":"ok",'
                                '"suggestions":["a","b","c","d"]}')
        self.assertEqual(out["semantic_score"], 88.0)
        self.assertEqual(len(out["suggestions"]), 3)

    def test_clamps_semantic_score(self):
        out = ev.parse_ai_reply('{"semantic_score": 900}')
        self.assertEqual(out["semantic_score"], 100.0)

    def test_malformed_replies_raise_unavailable(self):
        for bad in ("no json here", '{"recognized": "x"}', '{"semantic_score": "abc"}', "{not json}"):
            with self.assertRaises(ev.EvaluationUnavailable):
                ev.parse_ai_reply(bad)


if __name__ == "__main__":
    unittest.main()
