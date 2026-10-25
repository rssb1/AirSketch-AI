import shutil
import tempfile
import unittest
from pathlib import Path

import challenges as ch


class ChallengeTests(unittest.TestCase):
    def test_at_least_15_challenges_in_every_difficulty(self):
        self.assertGreaterEqual(len(ch.CHALLENGES), 15)
        for d in ch.DIFFICULTIES:
            self.assertGreaterEqual(len(ch.by_difficulty(d)), 3)

    def test_unique_keys(self):
        keys = [c.key for c in ch.CHALLENGES]
        self.assertEqual(len(keys), len(set(keys)))

    def test_all_render_with_ink(self):
        for c in ch.CHALLENGES:
            img = c.render()
            self.assertEqual(img.size, (ch.SIZE, ch.SIZE))
            self.assertLess(img.convert("L").getextrema()[0], 50, c.key)

    def test_missing_asset_is_regenerated_in_memory(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            img = ch.load_reference(ch.CHALLENGES[0], tmp)
            self.assertEqual(img.size, (ch.SIZE, ch.SIZE))
        finally:
            shutil.rmtree(tmp)

    def test_ensure_assets_writes_files(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            ch.ensure_assets(tmp)
            self.assertEqual(len(list(tmp.glob("*.png"))), len(ch.CHALLENGES))
        finally:
            shutil.rmtree(tmp)

    def test_pick_excludes_previous(self):
        for _ in range(30):
            self.assertNotEqual(ch.pick_challenge("easy", exclude="sun").key, "sun")

    def test_bad_difficulty(self):
        with self.assertRaises(ValueError):
            ch.by_difficulty("nope")


if __name__ == "__main__":
    unittest.main()
