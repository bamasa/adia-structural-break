"""The whitened stream must reproduce its batch builder row by row.

``structural_break.white_batch.white_channels`` is the vectorised code that made
the WHITE90, SR22 and HISTCTX8 training matrices; ``WhiteMonitor`` streams one
point at a time and is what ships. Any drift between the two would put the
models on channels they were never trained on. Three synthetic series -- an
AR(1) history, a GARCH-like heteroskedastic one, and a short history whose
online part is shorter than the smallest test window -- with the odds and the
context on (119 columns), compared in float64 to 1e-5. Run with:
PYTHONPATH=src python -m unittest discover -s tests
"""

import unittest

import numpy as np

from structural_break.white import CONTEXT_CHANNELS, WHITE_CHANNELS, WHITE_ODDS_CHANNELS, WhiteMonitor
from structural_break.white_batch import white_channels

TOLERANCE = 1e-5
FULL_WIDTH = WHITE_ODDS_CHANNELS + CONTEXT_CHANNELS


def _ar1(rng, phi, n, sd=1.0):
    x = np.zeros(n)
    e = rng.normal(0.0, sd, n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def _garch_ar(rng, n, phi=0.3, omega=0.05, alpha=0.1, beta=0.85):
    """AR(1) with GARCH(1,1) innovations: a heteroskedastic history."""
    x = np.zeros(n)
    s2 = omega / (1 - alpha - beta)
    e = 0.0
    for i in range(1, n):
        s2 = omega + alpha * e * e + beta * s2
        e = np.sqrt(s2) * rng.standard_normal()
        x[i] = phi * x[i - 1] + e
    return x


def _stream(hist, online, **kwargs):
    wm = WhiteMonitor(hist, **kwargs)
    return np.array([wm.update(float(v)) for v in online], dtype="float64")


class BatchEqualsStream(unittest.TestCase):
    def assertRowsEqual(self, hist, online):
        batch = white_channels(hist, online, odds=True, context=True)
        stream = _stream(hist, online, odds=True, context=True)
        self.assertEqual(batch.shape, (len(online), FULL_WIDTH))
        self.assertEqual(stream.shape, batch.shape)
        self.assertEqual(batch.dtype, np.float64)
        self.assertEqual(stream.dtype, np.float64)
        self.assertTrue(np.isfinite(batch).all())
        gap = np.abs(batch - stream).max(axis=0)
        worst = int(gap.argmax())
        self.assertLessEqual(gap[worst], TOLERANCE, f"column {worst} differs by {gap[worst]:.3g}")
        self.assertTrue(np.array_equal(batch[:, WHITE_CHANNELS - 1], np.arange(len(online))))   # the step index
        return batch, stream

    def test_ar1_history_with_a_scale_break(self):
        rng = np.random.default_rng(1)
        hist = _ar1(rng, 0.6, 3000)
        online = np.concatenate([_ar1(rng, 0.6, 300), _ar1(rng, 0.6, 300, sd=2.5)])
        self.assertGreaterEqual(WhiteMonitor(hist).p, 1)          # the whitening is not trivial
        self.assertRowsEqual(hist, online)

    def test_heteroskedastic_history_with_a_mean_shift(self):
        rng = np.random.default_rng(2)
        hist = _garch_ar(rng, 2500)
        online = np.concatenate([_garch_ar(rng, 250), _garch_ar(rng, 250) + 0.8])
        self.assertLess(WhiteMonitor(hist).lam, 1.0)               # the conditional scale is in play
        self.assertRowsEqual(hist, online)

    def test_short_history_and_online_shorter_than_the_smallest_window(self):
        rng = np.random.default_rng(3)
        hist = _ar1(rng, -0.4, 1000)
        online = _ar1(rng, -0.4, 50)
        self.assertLess(len(online), 64)
        batch, stream = self.assertRowsEqual(hist, online)
        # The narrower layouts are prefixes of the full one, on both sides.
        self.assertTrue(np.array_equal(white_channels(hist, online), batch[:, :WHITE_CHANNELS]))
        self.assertTrue(np.array_equal(white_channels(hist, online, odds=True), batch[:, :WHITE_ODDS_CHANNELS]))
        self.assertTrue(np.array_equal(_stream(hist, online), stream[:, :WHITE_CHANNELS]))
        self.assertTrue(np.array_equal(_stream(hist, online, odds=True), stream[:, :WHITE_ODDS_CHANNELS]))


if __name__ == "__main__":
    unittest.main()
