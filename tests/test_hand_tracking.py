import math
import unittest

import hand_tracking as ht


def make_hand(index=False, middle=False, ring=False, pinky=False):
    """Synthetic 21-point hand. Extended fingers have their tip far from the wrist."""
    lm = [(0.5, 0.9)] * 21
    lm = list(lm)
    angles = {"index": -100, "middle": -90, "ring": -80, "pinky": -70}
    flags = {"index": index, "middle": middle, "ring": ring, "pinky": pinky}
    for name, (tip, pip) in ht.FINGERS.items():
        a = math.radians(angles[name])
        tip_r, pip_r = (0.5, 0.25) if flags[name] else (0.18, 0.25)
        lm[pip] = (0.5 + pip_r * math.cos(a), 0.9 + pip_r * math.sin(a))
        lm[tip] = (0.5 + tip_r * math.cos(a), 0.9 + tip_r * math.sin(a))
    return lm


class GestureTests(unittest.TestCase):
    def test_point(self):
        self.assertEqual(ht.classify_gesture(make_hand(index=True)), ht.POINT)

    def test_peace(self):
        self.assertEqual(ht.classify_gesture(make_hand(index=True, middle=True)), ht.PEACE)

    def test_open_palm(self):
        self.assertEqual(ht.classify_gesture(make_hand(True, True, True, True)), ht.OPEN)

    def test_fist(self):
        self.assertEqual(ht.classify_gesture(make_hand()), ht.FIST)

    def test_no_hand(self):
        self.assertEqual(ht.classify_gesture(None), ht.NONE)
        self.assertEqual(ht.classify_gesture([(0, 0)] * 5), ht.NONE)

    def test_rotation_invariant(self):
        lm = make_hand(index=True)
        rotated = [(0.5 + (y - 0.9), 0.9 - (x - 0.5)) for x, y in lm]  # 90 degrees
        self.assertEqual(ht.classify_gesture(rotated), ht.POINT)


class StabilizerTests(unittest.TestCase):
    def test_requires_consecutive_frames(self):
        s = ht.GestureStabilizer(min_frames=3)
        self.assertEqual(s.update(ht.POINT), ht.NONE)
        self.assertEqual(s.update(ht.POINT), ht.NONE)
        self.assertEqual(s.update(ht.POINT), ht.POINT)

    def test_single_glitch_frame_ignored(self):
        s = ht.GestureStabilizer(min_frames=3)
        for _ in range(4):
            s.update(ht.POINT)
        self.assertEqual(s.update(ht.FIST), ht.POINT)
        self.assertEqual(s.update(ht.POINT), ht.POINT)

    def test_hand_loss_is_immediate(self):
        s = ht.GestureStabilizer(min_frames=3)
        for _ in range(4):
            s.update(ht.POINT)
        self.assertEqual(s.update(ht.NONE), ht.NONE)


class HoldDetectorTests(unittest.TestCase):
    def test_fires_once_after_hold(self):
        h = ht.HoldDetector(ht.OPEN, 1.0)
        self.assertFalse(h.update(ht.OPEN, 0.0))
        self.assertFalse(h.update(ht.OPEN, 0.9))
        self.assertTrue(h.update(ht.OPEN, 1.0))
        self.assertFalse(h.update(ht.OPEN, 1.5))   # does not re-fire while still held

    def test_interruption_resets(self):
        h = ht.HoldDetector(ht.OPEN, 1.0)
        h.update(ht.OPEN, 0.0)
        h.update(ht.POINT, 0.8)
        self.assertFalse(h.update(ht.OPEN, 1.0))
        self.assertTrue(h.update(ht.OPEN, 2.0))

    def test_progress(self):
        h = ht.HoldDetector(ht.OPEN, 2.0)
        h.update(ht.OPEN, 10.0)
        self.assertAlmostEqual(h.progress(11.0), 0.5)
        self.assertEqual(h.progress(20.0), 1.0)


if __name__ == "__main__":
    unittest.main()
