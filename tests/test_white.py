"""The whitened-stream module: ninety finite channels per point; a scale
break and a dependence break at constant variance are both seen on a
heteroskedastic AR history.  Run with:  PYTHONPATH=src python -m unittest discover tests
"""

import unittest

import numpy as np

from structural_break.white import WHITE_CHANNELS, WHITE_ODDS_CHANNELS, WhiteMonitor


def _ar1(rng, phi, n, sd=1.0):
    x = np.zeros(n)
    e = rng.normal(0.0, sd, n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


class WhitenedStream(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.hist = _ar1(self.rng, 0.6, 3000) * (1 + 0.5 * np.sin(np.arange(3000) / 300))

    def test_width_and_finiteness(self):
        wm = WhiteMonitor(self.hist)
        rows = np.array([wm.update(float(v)) for v in _ar1(self.rng, 0.6, 200)])
        self.assertEqual(rows.shape, (200, WHITE_CHANNELS))
        self.assertTrue(np.isfinite(rows).all())
        self.assertTrue(np.all(rows[:, -1] == np.arange(200)))     # the step index

    def test_scale_break_is_seen(self):
        wm = WhiteMonitor(self.hist)
        quiet = np.array([wm.update(float(v)) for v in _ar1(self.rng, 0.6, 300)])
        loud = np.array([wm.update(float(v)) for v in _ar1(self.rng, 0.6, 300, sd=1.5)])
        self.assertGreater(loud[200:, 4].mean(), quiet[200:, 4].mean() + 5.0)   # scale CUSUM

    def test_dependence_break_at_constant_variance_is_seen(self):
        wm = WhiteMonitor(self.hist)
        quiet = np.array([wm.update(float(v)) for v in _ar1(self.rng, 0.6, 300)])
        sd = np.sqrt((1 - 0.9 ** 2) / (1 - 0.6 ** 2))
        shifted = np.array([wm.update(float(v)) for v in _ar1(self.rng, 0.9, 300, sd=sd)])
        self.assertGreater(shifted[200:, 54].mean(), quiet[200:, 54].mean() + 3.0)  # dependence GLR max

    def test_odds_ride_along(self):
        wm = WhiteMonitor(self.hist, odds=True)
        quiet = np.array([wm.update(float(v)) for v in _ar1(self.rng, 0.6, 300)])
        loud = np.array([wm.update(float(v)) for v in _ar1(self.rng, 0.6, 300, sd=1.6)])
        self.assertEqual(quiet.shape[1], WHITE_ODDS_CHANNELS)
        self.assertTrue(np.isfinite(loud).all())
        # The variance-up mixture (fourth from the end) rises after a scale break.
        self.assertGreater(loud[200:, -4].mean(), quiet[200:, -4].mean() + 5.0)

    def test_constant_history_does_not_crash(self):
        wm = WhiteMonitor(np.ones(1000))
        rows = np.array([wm.update(float(v)) for v in np.arange(20)])
        self.assertEqual(rows.shape, (20, WHITE_CHANNELS))
        self.assertTrue(np.isfinite(rows).all())


if __name__ == "__main__":
    unittest.main()
