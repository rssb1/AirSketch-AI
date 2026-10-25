import random
import unittest

import challenges as ch
from game_logic import Game, GameState, clamp_score, grade_for_score


class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


def make_game(duration=60):
    clock = FakeClock()
    return Game(duration=duration, clock=clock, rng=random.Random(0)), clock


class ScoreBoundaryTests(unittest.TestCase):
    def test_clamp(self):
        self.assertEqual(clamp_score(-5), 0)
        self.assertEqual(clamp_score(0), 0)
        self.assertEqual(clamp_score(100), 100)
        self.assertEqual(clamp_score(250), 100)

    def test_grade_thresholds(self):
        self.assertEqual(grade_for_score(100), "Masterpiece!")
        self.assertEqual(grade_for_score(85), "Masterpiece!")
        self.assertEqual(grade_for_score(84.99), "Great job!")
        self.assertEqual(grade_for_score(65), "Great job!")
        self.assertEqual(grade_for_score(40), "Getting there")
        self.assertEqual(grade_for_score(15), "Needs work")
        self.assertEqual(grade_for_score(0), "Keep practicing")
        self.assertEqual(grade_for_score(-10), "Keep practicing")


class StateTransitionTests(unittest.TestCase):
    def test_initial_state(self):
        g, _ = make_game()
        self.assertEqual(g.state, GameState.MENU)
        self.assertEqual((g.best_score, g.current_score, g.rounds_completed), (0, 0, 0))
        self.assertFalse(g.can_draw)

    def test_start_picks_challenge_of_difficulty(self):
        for d in ch.DIFFICULTIES:
            g, _ = make_game()
            c = g.start(d)
            self.assertEqual(c.difficulty, d)
            self.assertEqual(g.state, GameState.PLAYING)
            self.assertTrue(g.can_draw)

    def test_unknown_difficulty(self):
        g, _ = make_game()
        with self.assertRaises(ValueError):
            g.start("impossible")

    def test_timer_counts_down_and_expires_once(self):
        g, clock = make_game(60)
        g.start("easy")
        clock.advance(45)
        self.assertAlmostEqual(g.time_left, 15)
        self.assertFalse(g.tick())
        clock.advance(15)
        self.assertEqual(g.time_left, 0)
        self.assertTrue(g.tick())          # fires once...
        self.assertFalse(g.tick())         # ...never twice
        self.assertEqual(g.state, GameState.EVALUATING)
        self.assertTrue(g.timed_out)
        self.assertFalse(g.can_draw)

    def test_no_drawing_after_time_runs_out_even_before_tick(self):
        g, clock = make_game(10)
        g.start("easy")
        clock.advance(11)
        self.assertFalse(g.can_draw)

    def test_duplicate_submit_is_rejected(self):
        g, clock = make_game()
        g.start("easy")
        clock.advance(5)
        self.assertTrue(g.submit())
        self.assertFalse(g.submit())
        self.assertEqual(g.state, GameState.EVALUATING)

    def test_submit_after_timeout_is_rejected(self):
        g, clock = make_game(5)
        g.start("easy")
        clock.advance(6)
        g.tick()
        self.assertFalse(g.submit())

    def test_record_result_updates_history(self):
        g, clock = make_game()
        g.start("easy")
        clock.advance(20)
        g.submit()
        rec = g.record_result(72.5, "local")
        self.assertEqual(g.state, GameState.RESULTS)
        self.assertEqual(rec.score, 72.5)
        self.assertAlmostEqual(rec.seconds_used, 20)
        self.assertEqual((g.current_score, g.best_score, g.rounds_completed), (72.5, 72.5, 1))

    def test_record_result_clamps_and_requires_pending_evaluation(self):
        g, _ = make_game()
        self.assertIsNone(g.record_result(50))
        g.start("easy")
        self.assertIsNone(g.record_result(50))           # still playing
        g.submit()
        self.assertEqual(g.record_result(500).score, 100)
        self.assertIsNone(g.record_result(10))           # duplicate result ignored
        self.assertEqual(g.rounds_completed, 1)

    def test_best_score_tracks_maximum(self):
        g, _ = make_game()
        for s in (40, 90, 60):
            g.start("easy")
            g.submit()
            g.record_result(s)
        self.assertEqual(g.best_score, 90)
        self.assertEqual(g.current_score, 60)
        self.assertEqual(g.rounds_completed, 3)

    def test_skip_gives_new_challenge_without_scoring(self):
        g, _ = make_game()
        first = g.start("medium")
        second = g.skip()
        self.assertNotEqual(first.key, second.key)
        self.assertEqual(g.skipped, 1)
        self.assertEqual(g.rounds_completed, 0)
        self.assertEqual(g.state, GameState.PLAYING)

    def test_skip_only_while_playing(self):
        g, _ = make_game()
        self.assertIsNone(g.skip())

    def test_play_again_keeps_challenge_and_resets_timer(self):
        g, clock = make_game()
        c = g.start("hard")
        clock.advance(30)
        g.submit()
        g.record_result(50)
        again = g.play_again()
        self.assertEqual(again.key, c.key)
        self.assertAlmostEqual(g.time_left, 60)
        self.assertFalse(g.timed_out)

    def test_new_challenge_differs_and_keeps_difficulty(self):
        g, _ = make_game()
        c = g.start("easy")
        g.submit()
        g.record_result(10)
        n = g.new_challenge()
        self.assertNotEqual(n.key, c.key)
        self.assertEqual(n.difficulty, "easy")

    def test_abort_evaluation_allows_retry(self):
        g, _ = make_game()
        g.start("easy")
        g.submit()
        g.abort_evaluation()
        self.assertEqual(g.state, GameState.RESULTS)
        self.assertEqual(g.rounds_completed, 0)
        self.assertIsNotNone(g.play_again())

    def test_to_menu_keeps_session_stats(self):
        g, _ = make_game()
        g.start("easy")
        g.submit()
        g.record_result(55)
        g.to_menu()
        self.assertEqual(g.state, GameState.MENU)
        self.assertEqual(g.best_score, 55)


if __name__ == "__main__":
    unittest.main()
