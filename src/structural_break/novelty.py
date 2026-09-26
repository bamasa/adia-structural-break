"""Window novelty against the history's own windows.

Every other member compares the online window with the history's mean and
variance -- one number per statistic. A history that is itself heterogeneous
(regimes, changing volatility) makes that a poor null: a window ordinary for
the history looks anomalous against its average, and a break into a quiet
regime looks like nothing. Here a window is described by seven summaries and
compared with every history window of the same length: the distance to the
nearest, and the rank of that distance among the history-against-history
nearest distances -- an empirical p-value against the history's own variety
-- plus the same ranks for scale and first-lag dependence alone.

Four windows, five channels each (experiment 140). The history windows and
their reference distributions are computed once per series; each online step
costs one nearest-neighbour pass over at most a few hundred history windows.
Reproduces the batch builder that made the training matrix to 1e-6 --
including the builder's third channel, the median-distance rank, which its
NaN-poisoned reference left constant at -0.5 in the matrix. The classifier
trained on that matrix never split on it, so it is held constant here too
until the matrix is rebuilt (141).
"""

from __future__ import annotations

import numpy as np

NOVELTY_WINDOWS = (25, 50, 100, 250)
NOVELTY_CHANNELS = len(NOVELTY_WINDOWS) * 5
_MIN_POINTS = 5


def _novelty_summary(w: np.ndarray) -> np.ndarray:
    """Mean, log scale, skewness, excess kurtosis, lag-1 autocorrelation and
    the 5th and 95th percentiles of one window."""
    m = w.mean()
    sd = w.std() + 1e-9
    z = (w - m) / sd
    with np.errstate(all="ignore"):
        ac = np.corrcoef(w[:-1], w[1:])[0, 1] if len(w) > 4 else 0.0
    return np.array([m, np.log(sd), (z ** 3).mean(), (z ** 4).mean() - 3.0, ac, np.quantile(w, 0.05), np.quantile(w, 0.95)])


class _WindowNovelty:
    """One window length: the history's windows, their spread, and the
    reference distributions the online window is ranked against."""

    def __init__(self, zh: np.ndarray, W: int) -> None:
        self.W = W
        step = max(W // 2, 1)
        starts = range(0, len(zh) - W + 1, step)
        H = np.array([_novelty_summary(zh[j: j + W]) for j in starts]) if len(zh) >= W else np.zeros((0, 7))
        self.ok = len(H) >= 2
        if not self.ok:
            return
        self.scale = H.std(0) + 1e-6
        self.Hn = H / self.scale
        D = np.sqrt(((self.Hn[:, None, :] - self.Hn[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(D, np.inf)
        self.nn_ref = np.sort(D.min(1))
        self.nn_med = np.median(self.nn_ref) + 1e-9
        self.sd_med, self.ac_med = np.median(self.Hn[:, 1]), np.median(self.Hn[:, 4])
        self.sd_ref = np.sort(np.abs(self.Hn[:, 1] - self.sd_med))
        self.ac_ref = np.sort(np.abs(self.Hn[:, 4] - self.ac_med))

    def channels(self, w: np.ndarray) -> list[float]:
        if not self.ok or len(w) < _MIN_POINTS:
            return [0.0] * 5
        v = _novelty_summary(w) / self.scale
        d = np.sqrt(((self.Hn - v) ** 2).sum(1))
        dnn = d.min()
        p_nn = np.searchsorted(self.nn_ref, dnn) / len(self.nn_ref)
        p_sd = np.searchsorted(self.sd_ref, abs(v[1] - self.sd_med)) / len(self.sd_ref)
        p_ac = np.searchsorted(self.ac_ref, abs(v[4] - self.ac_med)) / len(self.ac_ref)
        # The median-distance rank: constant in the training matrix (see the
        # module docstring), so constant here.
        return [float(np.log(dnn / self.nn_med + 1e-9)), p_nn - 0.5, -0.5, p_sd - 0.5, p_ac - 0.5]


class Novelty:
    """Streaming window novelty: one update per online point, twenty channels out.

    The windows are the last W online points only -- never the history -- so
    early channels compare a short prefix with full-length history windows
    exactly as the batch builder did.
    """

    def __init__(self, history: np.ndarray) -> None:
        h = np.asarray(history, dtype="float64")
        self.mu, self.sd = float(h.mean()), float(h.std()) + 1e-12
        zh = (h - self.mu) / self.sd
        self.parts = [_WindowNovelty(zh, W) for W in NOVELTY_WINDOWS]
        self.buf: list[float] = []
        self._wmax = max(NOVELTY_WINDOWS)

    def update(self, x: float) -> list[float]:
        self.buf.append((float(x) - self.mu) / self.sd)
        if len(self.buf) > 4 * self._wmax:
            del self.buf[: -self._wmax]
        if len(self.buf) < _MIN_POINTS:
            return [0.0] * NOVELTY_CHANNELS
        arr = np.asarray(self.buf)
        out: list[float] = []
        for part in self.parts:
            out.extend(float(np.clip(np.nan_to_num(v), -20, 20)) for v in part.channels(arr[-part.W:]))
        return out
