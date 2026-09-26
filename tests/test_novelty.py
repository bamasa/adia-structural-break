"""The novelty module streams twenty finite channels per point and survives
degenerate input.  Run with:  PYTHONPATH=src python -m unittest discover tests
"""

import unittest

import numpy as np

from structural_break.novelty import NOVELTY_CHANNELS, Novelty


class NoveltyStream(unittest.TestCase):
    def test_width_and_warmup(self):
        rng = np.random.default_rng(0)
        nv = Novelty(rng.standard_normal(1000))
        rows = np.array([nv.update(float(v)) for v in rng.standard_normal(300)])
        self.assertEqual(rows.shape, (300, NOVELTY_CHANNELS))
        self.assertTrue(np.all(rows[:4] == 0.0))
        self.assertTrue(np.isfinite(rows).all())
        self.assertTrue(np.any(rows[5:] != 0.0))

    def test_a_shift_reads_as_novel(self):
        rng = np.random.default_rng(1)
        nv = Novelty(rng.standard_normal(2000))
        quiet = np.array([nv.update(float(v)) for v in rng.standard_normal(300)])
        loud = np.array([nv.update(float(v)) for v in 4.0 * rng.standard_normal(300)])
        # Window 100, log nearest-neighbour distance: channel 10.
        self.assertGreater(loud[150:, 10].mean(), quiet[150:, 10].mean() + 1.0)

    def test_constant_history_does_not_crash(self):
        nv = Novelty(np.ones(1000))
        rows = np.array([nv.update(float(v)) for v in np.arange(50)])
        self.assertEqual(rows.shape, (50, NOVELTY_CHANNELS))
        self.assertTrue(np.isfinite(rows).all())

    def test_short_history_falls_back_to_zeros(self):
        nv = Novelty(np.random.default_rng(2).standard_normal(60))
        rows = np.array([nv.update(float(v)) for v in np.random.default_rng(3).standard_normal(40)])
        self.assertTrue(np.all(rows[:, 10:] == 0.0))        # windows 100 and 250 have no history windows
        self.assertTrue(np.isfinite(rows).all())


if __name__ == "__main__":
    unittest.main()
