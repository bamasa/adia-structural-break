"""A mass battery of simple statistics, built independently of the channels.

Four families of change-point detector in a row measured something real and
added nothing when appended to the two hundred engineered channels (096, 097,
110, 112). Experiment 114 found the reason: what the ensemble lacks is not a
better detector but a member that *disagrees* with it. Ninety plain statistics,
poured into the same model, correlate 0.84 with the ensemble and give nothing;
trained as their own classifier they correlate 0.68 and add 0.004.

Six representations of the standardised series -- the value, its magnitude, its
square, its increment, the deviation of the running sum, and its sign -- over
six exponential windows from ten to five hundred points. Each is compared with
the history by a standardised mean gap and a log variance ratio, and three
exceedance rates of the history's upper quantiles complete the set.

Everything is a rolling update: the cost is one small matrix operation per
step, about 0.03 ms, and nothing is recomputed from the past.
"""

from __future__ import annotations

import numpy as np

#: Exponential window lengths, in points.
MASS_WINDOWS = (10, 25, 50, 100, 250, 500)
#: Representations of the series, in the order the channels are emitted.
MASS_REPRESENTATIONS = 6
#: Total channels: six representations x six windows x two comparisons, plus
#: three exceedance rates per window.
MASS_CHANNELS = MASS_REPRESENTATIONS * len(MASS_WINDOWS) * 2 + len(MASS_WINDOWS) * 3


class MassBattery:
    """Streaming mass battery: one update per online point, ninety channels out."""

    def __init__(self, history: np.ndarray) -> None:
        h = np.asarray(history, dtype="float64")
        self.mu = float(h.mean())
        self.sd = float(h.std()) + 1e-12
        zh = (h - self.mu) / self.sd
        reps = [zh, np.abs(zh), zh * zh, np.diff(zh, prepend=zh[0]), np.zeros_like(zh), np.sign(zh)]
        self.hist_mean = np.array([r.mean() for r in reps])
        self.hist_var = np.array([r.var() + 1e-9 for r in reps])
        self.quantiles = np.quantile(np.abs(zh), [0.75, 0.95, 0.99])
        self.windows = np.array(MASS_WINDOWS, dtype="float64")
        self.alpha = 1.0 / self.windows
        self.s1 = np.zeros((MASS_REPRESENTATIONS, len(MASS_WINDOWS)))
        self.s2 = np.zeros((MASS_REPRESENTATIONS, len(MASS_WINDOWS)))
        self.exceed = np.zeros((3, len(MASS_WINDOWS)))
        self.se = np.sqrt(self.hist_var[:, None] / self.windows[None, :])
        self.expected_exceed = np.array([0.25, 0.05, 0.01])[:, None]
        self._prev = float(zh[-1])
        self._cum = 0.0
        self._step = 0

    def update(self, x: float) -> list[float]:
        z = (float(x) - self.mu) / self.sd
        self._cum += z
        vals = np.array([z, abs(z), z * z, z - self._prev,
                         self._cum / np.sqrt(self._step + 1), np.sign(z)])
        a = self.alpha
        self.s1 = (1 - a) * self.s1 + a * vals[:, None]
        self.s2 = (1 - a) * self.s2 + a * (vals * vals)[:, None]
        var = np.maximum(self.s2 - self.s1 * self.s1, 1e-9)
        gap = (self.s1 - self.hist_mean[:, None]) / self.se
        ratio = np.log(var / self.hist_var[:, None])
        exc = (abs(z) > self.quantiles[:, None]).astype("float64")
        self.exceed = (1 - a[None, :]) * self.exceed + a[None, :] * exc
        out = np.concatenate([gap.ravel(), ratio.ravel(),
                              (self.exceed - self.expected_exceed).ravel()])
        self._prev = z
        self._step += 1
        return np.clip(np.nan_to_num(out, nan=0.0, posinf=20.0, neginf=-20.0), -20, 20).tolist()
