"""The dependence-CUSUM module: twenty-three finite channels per point, and a
change in dependence at constant variance is seen.  Run with:
PYTHONPATH=src python -m unittest discover tests
"""

import unittest

import numpy as np

from structural_break.depcusum import DEP_CHANNELS, DepCusum


def _ar1(rng, phi, n, marginal_sd=1.0):
    x = np.zeros(n)
    e = rng.normal(0.0, marginal_sd * np.sqrt(1.0 - phi ** 2), n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


class DependenceCusum(unittest.TestCase):
    def test_width_and_finiteness(self):
        rng = np.random.default_rng(0)
        dc = DepCusum(_ar1(rng, 0.3, 2000))
        rows = np.array([dc.update(float(v)) for v in _ar1(rng, 0.3, 300)])
        self.assertEqual(rows.shape, (300, DEP_CHANNELS))
        self.assertTrue(np.isfinite(rows).all())

    def test_dependence_change_at_constant_variance_is_seen(self):
        rng = np.random.default_rng(1)
        dc = DepCusum(_ar1(rng, 0.2, 3000))
        quiet = np.array([dc.update(float(v)) for v in _ar1(rng, 0.2, 300)])
        shifted = np.array([dc.update(float(v)) for v in _ar1(rng, 0.7, 300)])
        # Lag-1 CUSUM peak (channel 0) and portmanteau (channel 15) rise.
        self.assertGreater(shifted[200:, 0].mean(), 3 * quiet[200:, 0].mean() + 1.0)
        self.assertGreater(shifted[200:, 15].mean(), quiet[200:, 15].mean())

    def test_constant_history_does_not_crash(self):
        dc = DepCusum(np.ones(1000))
        rows = np.array([dc.update(float(v)) for v in np.arange(30)])
        self.assertEqual(rows.shape, (30, DEP_CHANNELS))
        self.assertTrue(np.isfinite(rows).all())


if __name__ == "__main__":
    unittest.main()
