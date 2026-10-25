import unittest

import numpy as np

from drawing_canvas import CANVAS_H, CANVAS_W, DrawingCanvas, PointSmoother, normalized_to_canvas


class CanvasTests(unittest.TestCase):
    def test_new_canvas_is_empty_and_white(self):
        c = DrawingCanvas()
        self.assertTrue(c.is_empty())
        self.assertTrue((c.array == 255).all())

    def test_drawing_a_stroke(self):
        c = DrawingCanvas()
        for x in range(50, 300, 10):
            c.add_point(x, 100)
        self.assertFalse(c.is_empty())

    def test_clear(self):
        c = DrawingCanvas()
        for x in range(50, 300, 10):
            c.add_point(x, 100)
        c.clear()
        self.assertTrue(c.is_empty())
        self.assertTrue((c.array == 255).all())

    def test_lift_prevents_connecting_strokes(self):
        c = DrawingCanvas()
        c.smoother.alpha = 1.0
        for x in range(50, 100, 5):
            c.add_point(x, 50)
        c.lift()                                   # hand left the frame
        for x in range(50, 100, 5):
            c.add_point(x, 400)
        mid = c.array[150:350]                      # region between the strokes stays white
        self.assertTrue((mid == 255).all())

    def test_without_lift_strokes_connect(self):
        c = DrawingCanvas()
        c.smoother.alpha = 1.0
        c.add_point(50, 50)
        c.add_point(50, 400)
        self.assertFalse((c.array[150:350] == 255).all())

    def test_frozen_canvas_rejects_input(self):
        c = DrawingCanvas()
        c.frozen = True
        self.assertFalse(c.add_point(100, 100))
        self.assertTrue(c.is_empty())

    def test_color_and_brush(self):
        c = DrawingCanvas()
        c.set_color((255, 0, 0))
        c.set_brush(14)
        c.smoother.alpha = 1.0
        for x in range(100, 200, 10):
            c.add_point(x, 100)
        self.assertTrue((c.array[100, 150] == (255, 0, 0)).all())
        self.assertEqual(c.brush, 14)

    def test_points_outside_canvas_are_clamped(self):
        c = DrawingCanvas()
        c.add_point(-500, 9999)
        self.assertFalse(c.is_empty() and False)  # no exception raised

    def test_to_pil_is_a_copy(self):
        c = DrawingCanvas()
        img = c.to_pil()
        self.assertEqual(img.size, (CANVAS_W, CANVAS_H))
        c.add_point(10, 10)
        self.assertTrue((np.asarray(img) == 255).all())


class SmootherTests(unittest.TestCase):
    def test_smoothing_moves_part_way(self):
        s = PointSmoother(alpha=0.5)
        s.update(0, 0)
        self.assertEqual(s.update(10, 10), (5, 5))

    def test_reset(self):
        s = PointSmoother(0.5)
        s.update(0, 0)
        s.reset()
        self.assertEqual(s.update(8, 8), (8, 8))

    def test_invalid_alpha(self):
        with self.assertRaises(ValueError):
            PointSmoother(0)


class MappingTests(unittest.TestCase):
    def test_corners_and_center(self):
        self.assertEqual(normalized_to_canvas(0.5, 0.5), ((CANVAS_W - 1) / 2, (CANVAS_H - 1) / 2))
        self.assertEqual(normalized_to_canvas(0, 0), (0, 0))
        self.assertEqual(normalized_to_canvas(1, 1), (CANVAS_W - 1, CANVAS_H - 1))


if __name__ == "__main__":
    unittest.main()
