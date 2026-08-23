"""The rank view: online observations as quantiles of the history.

The z-score view calibrates the detectors against a unit Gaussian, and a
series whose noise is genuinely heavy-tailed spends its whole life in that
view's tails -- a clean-but-jittery series produces spikes, the spikes freeze
into peaks, and the series outranks real breaks forever. That failure is
cross-sectional, and the inspection notebook names it as where the score is
actually lost.

The rank view removes the distribution from the problem. Each online value is
placed within the *empirical* distribution of the standardised history -- its
rank -- and the rank is mapped back to a Gaussian through the inverse normal
CDF. Under the null the result is N(0,1) by construction, whatever the shape
of the series' noise: heavy tails, skew, bimodality all become exactly the
distribution the detectors are calibrated for. What survives the mapping is
the only thing that should: values falling where history says they should not.

The price is paid where it ought to be cheap. Ranks see any monotone
transformation of the data identically, so a break that only stretches the
extreme tail is muted; the z-view keeps that case. The two views disagree
precisely on the series where one of them is being fooled, which is what a
combiner is for.

A mid-history plateau: many identical values in the history give the empirical
CDF a step, and midranking places repeated online values at the step's centre
rather than randomly within it -- deterministic, as the platform requires.
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np

from structural_break.detectors import Cusum, PageHinkley, VarianceRatio
from structural_break.features import Normalisation

_NORMAL = NormalDist()

#: Families, in the order their columns appear (peak score, then current).
RANK_CHANNELS = tuple(
    f"rank_{family}{suffix}"
    for suffix in ("", "_now")
    for family in ("cusum", "page_hinkley", "variance_ratio")
)


class RankView:
    """Detectors running on the probit of each observation's historical rank."""

    def __init__(self, norm: Normalisation, history: np.ndarray) -> None:
        self.norm = norm
        # The history standardised at its own (negative) steps, so the fitted
        # trend is removed from it the same way it is removed online.
        n = len(history)
        z_hist = np.asarray(
            [norm.standardise(float(v), i - n) for i, v in enumerate(history)]
        )
        self.sorted_z = np.sort(z_hist)
        self.detectors = (Cusum(), PageHinkley(), VarianceRatio())
        self._step = 0

    def gauss_rank(self, x: float) -> float:
        """Probit of the observation's midrank within the history."""
        z = self.norm.standardise(float(x), self._step)
        lo = float(np.searchsorted(self.sorted_z, z, side="left"))
        hi = float(np.searchsorted(self.sorted_z, z, side="right"))
        u = (0.5 * (lo + hi) + 0.5) / (len(self.sorted_z) + 1.0)
        return float(_NORMAL.inv_cdf(min(max(u, 1e-9), 1.0 - 1e-9)))

    def update(self, x: float) -> dict[str, float]:
        g = self.gauss_rank(x)
        self._step += 1
        # The mean detectors get the same dependence inflation as the raw
        # view: ranks preserve serial correlation. The variance detector does
        # not -- the marginal variance of the probit rank is one under the
        # null regardless of dependence.
        adjusted = g / self.norm.inflation
        out: dict[str, float] = {}
        for detector, name in zip(self.detectors, ("cusum", "page_hinkley", "variance_ratio")):
            value = g if name == "variance_ratio" else adjusted
            out[f"rank_{name}"] = detector.update(value)
            out[f"rank_{name}_now"] = detector.now
        return out
