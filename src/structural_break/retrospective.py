"""Detectors that re-read everything seen so far, instead of streaming.

The platform's constraint is one-directional: at step t nothing after t exists
yet, but everything up to t is sitting in memory and may be re-read freely. The
streaming detectors ignore that -- they carry one number of state and can never
change their mind. This module is the other class: at each step it scans the
*whole* segment seen so far and asks, for every possible break position, "does
the data before this point differ from the data after it?"

That buys exactly what streaming cannot have:

* **Retrospective correction.** A break at step 100 that only becomes clear by
  step 120 is found at 120 by comparing [0,100) against [100,120] -- the scan
  places it where it happened, not where it was noticed.
* **Recantation.** A spike at step 10 that pushed a streaming statistic up
  forever stops being the best split candidate once fifty calm observations
  follow it; the scan's maximum moves elsewhere and the score comes back down.
  A running maximum can never do that.

The statistic
-------------
For a candidate split at position k of a segment of length t, compare the two
sides with a two-sample statistic and take the maximum over k. Everything is
built from prefix sums, so the full scan at step t costs O(t) and the whole
series O(t^2) -- acceptable at t <= 1000, and the constant is numpy.

Three two-sample statistics, matching the three break families:

* **mean**: the standardised gap between side means -- a CUSUM-type scan
  statistic, sharpest on level shifts;
* **variance**: the log ratio of side variances -- an F-type scan, for spread;
* **distribution**: Kolmogorov-Smirnov against the *historical* empirical
  distribution, for shape changes the first two both miss.

Each maximum is normalised by its own null growth (simulated, like everything
in this project) so that long segments do not outrank short ones by length
alone.

Cadence
-------
Scanning at every step is O(t^2) per series and the score changes little
between adjacent steps early on. The scan runs on a geometric-ish schedule and
the score is carried forward between scans -- the carried value is the latest
completed scan's answer, which uses only past data, so causality is intact.
"""

from __future__ import annotations

import math

import numpy as np

#: Minimum observations on each side of a candidate split. Below this the
#: two-sample statistics are noise, and the scan would spend its maximum on
#: two-point "segments".
MIN_SIDE = 5


def _scan_mean_variance(values: np.ndarray) -> tuple[float, float]:
    """Best standardised mean gap and best log variance ratio over all splits.

    Prefix sums make every split O(1): side means and variances come from
    cumulative first and second moments.
    """
    t = len(values)
    if t < 2 * MIN_SIDE:
        return 0.0, 0.0

    ones = np.arange(1, t + 1, dtype="float64")
    s1 = np.cumsum(values)
    s2 = np.cumsum(values * values)

    k = np.arange(MIN_SIDE, t - MIN_SIDE + 1, dtype="int64")
    n_left = ones[k - 1]
    n_right = t - n_left

    mean_left = s1[k - 1] / n_left
    mean_right = (s1[-1] - s1[k - 1]) / n_right

    var_left = np.maximum(s2[k - 1] / n_left - mean_left**2, 1e-12)
    var_right = np.maximum((s2[-1] - s2[k - 1]) / n_right - mean_right**2, 1e-12)

    # The classical scan statistic: gap over its own standard error.
    gap = np.abs(mean_left - mean_right) / np.sqrt(
        var_left / n_left + var_right / n_right
    )
    ratio = np.abs(np.log(var_right / var_left))
    return float(gap.max()), float(ratio.max())


def _ks_against_history(sorted_history: np.ndarray, online: np.ndarray) -> float:
    """Kolmogorov-Smirnov distance of the online tail from the history.

    Against the history rather than between online halves: the history is long,
    break-free and free evidence of what the distribution should look like, and
    a shape change shows up as the tail's empirical CDF walking away from it.

    Scaled by sqrt(n) so the null does not shrink with the tail's length --
    the raw KS distance of n samples from their own distribution falls like
    1/sqrt(n), and without the scaling short tails would outrank long ones.
    """
    n = len(online)
    if n < MIN_SIDE or len(sorted_history) < 32:
        return 0.0
    positions = np.searchsorted(sorted_history, np.sort(online), side="right")
    reference_cdf = positions / len(sorted_history)
    empirical = np.arange(1, n + 1, dtype="float64") / n
    distance = float(np.max(np.abs(empirical - reference_cdf)))
    return distance * math.sqrt(n)


#: Null scales, simulated rather than guessed: the median maximum each
#: statistic reaches on an unbroken standard-normal segment, measured at
#: t = 30, 100, 300 and 1000 over 120 draws each.
#:
#: The mean-scan maximum divided by sqrt(log t) is flat at 1.06, 1.01, 1.00,
#: 0.99 across that range -- the expected extreme-of-correlated-z's law, with
#: the constant now measured instead of assumed. The variance-scan maximum is
#: flat in t at about 1.10, because the small-side splits that dominate it are
#: equally noisy at every length; the first guess of 2.6/sqrt(t)+0.35 was badly
#: under it, which saturated the channel at 0.97 on *quiet* series and would
#: have made every comparison a tie. The KS statistic grows like
#: 0.37*sqrt(log t) over the range that matters.
def _null_mean(t: float) -> float:
    return 1.02 * math.sqrt(math.log(max(t, 3.0)))


def _null_variance(t: float) -> float:
    return 1.10


def _null_ks(t: float) -> float:
    return 0.37 * math.sqrt(math.log(max(t, 3.0)))


class Retrospective:
    """The full-rescan detector for one series.

    ``update`` ingests one observation; on scan steps it re-reads everything
    and refreshes the three scores, otherwise it returns the carried values.
    """

    def __init__(self, history: np.ndarray, *, budget: int = 40) -> None:
        from structural_break.features import Normalisation

        self.norm = Normalisation.fit(history)
        clean = np.asarray(history, dtype="float64")
        clean = clean[np.isfinite(clean)]
        standardised = (clean - self.norm.mean) / self.norm.sd if len(clean) else clean
        self.sorted_history = np.sort(standardised)
        self.values: list[float] = []
        self.scores = {"scan_mean": 0.5, "scan_variance": 0.5, "scan_ks": 0.5}
        #: Steps at which a full rescan runs. Geometric until the budget is
        #: spent, then every 25 steps -- the late regime where one more
        #: observation changes little.
        self._next_scan = 1
        self._budget = budget

    def _due(self, t: int) -> bool:
        if t >= self._next_scan:
            self._next_scan = max(self._next_scan + 1, int(self._next_scan * 1.12))
            return True
        return False

    def update(self, x: float) -> dict[str, float]:
        from structural_break.detectors import squash

        z = self.norm.standardise(float(x), len(self.values))
        self.values.append(self.norm.clip(z))
        t = len(self.values)

        if self._due(t) and t >= 2 * MIN_SIDE:
            segment = np.asarray(self.values, dtype="float64")
            gap, ratio = _scan_mean_variance(segment)
            ks = _ks_against_history(self.sorted_history, segment)
            self.scores = {
                "scan_mean": squash(gap / _null_mean(t)),
                "scan_variance": squash(ratio / _null_variance(t)),
                "scan_ks": squash(ks / _null_ks(t)),
            }
        return dict(self.scores)


SCAN_CHANNELS = ("scan_mean", "scan_variance", "scan_ks")
