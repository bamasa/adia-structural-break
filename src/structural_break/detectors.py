"""Three textbook online change detectors, and why these three.

The task is to watch a stream and, after each observation, say how confident we
are that the process has already changed. Three families answer that, and they
fail in different places — which is the reason to run all three rather than
pick one:

**CUSUM** accumulates standardised deviations and fires when the running sum
drifts. It is the sharpest of the three on a *sustained shift in level*, and it
is deliberately blind to a single large observation: one outlier moves the sum
once, a changed mean moves it every step. That is the property the task wants,
since a spike that reverts is not a structural break.

**Page-Hinkley** is CUSUM's cousin with a different bookkeeping: it tracks the
gap between the running mean and its own minimum. It reacts to smaller shifts
than CUSUM at the same false-alarm rate, and pays for it with a longer delay.

**Variance ratio** ignores the level entirely and compares recent dispersion to
the historical dispersion. A break that doubles the noise without moving the
mean is invisible to both detectors above and obvious to this one.

All three are one-sided in the sense that matters here: they answer "has
something changed", not "which way". The score they emit is monotone in
confidence and bounded, because the metric is an AUC across series at each step
— what matters is the *ordering* between series, not the calibration of any one
of them.

Streaming, not windowed
-----------------------
Every update below is O(1) in the number of observations seen. That is not an
optimisation: the platform reveals one point at a time and re-deriving a
statistic from the whole history at each step would be quadratic, and on
thousand-point series with ten thousand of them, too slow to finish.

The historical segment is used only to fix the reference — mean, standard
deviation — and is never re-read after the stream starts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


#: Standard errors beyond which a single observation stops counting more.
#: Without it one huge value moves a cumulative sum as far as a genuine shift
#: does over many steps -- and the whole point of these detectors is that a
#: spike which reverts is not a structural break. Winsorising is the standard
#: robustification and it is what separates the two cases.
WINSOR = 4.0


def _z(x: float, reference: Reference) -> float:
    """Standardised, dependence-adjusted, and clipped."""
    z = (x - reference.mean) / (reference.sd * reference.inflation)
    return float(np.clip(z, -WINSOR, WINSOR))


def _null_scale(steps: float, growth: float, law: str) -> float:
    """How large the statistic gets on an unbroken series by this step.

    A cumulative statistic grows with the length of the stream even when
    nothing happens, so a raw value confounds "something changed" with "this
    series is long". The metric compares series against each other at each
    step, and series have different lengths, so that confound would be scored
    directly. Dividing by the null's own growth removes it.

    The two laws are not interchangeable, and simulation says which is which.
    A CUSUM with a drift term is reflected at zero, so its running maximum
    grows like the *logarithm* of the stream — measured here at 0.84·log(n),
    stable across n = 50, 200 and 800. Page-Hinkley with a small delta is
    essentially the range of a random walk, which grows like its *square root*
    — measured at 1.0·sqrt(n), equally stable. Using one law for both leaves a
    length effect in the score, and length is not evidence of a break.
    """
    n = max(steps, 2.0)
    base = math.log(n) if law == "log" else math.sqrt(n)
    return growth * base + 1e-9


def _squash(ratio: float, steepness: float = 2.0) -> float:
    """Map a statistic, already divided by its null scale, onto (0, 1).

    Logistic and centred on one, so a series behaving exactly like the null
    lands at 0.5 and the two tails have room. A ``tanh`` of the raw value would
    have put the null at 0.76 and crushed every genuine break into the same
    0.98-1.00 band -- fine for a threshold, useless for a metric that only ever
    compares series against each other.

    Never clipped, for the same reason: a clip makes every value above the cap
    a tie, and ties are exactly the comparisons an AUC is made of. This project
    has already lost one experiment to a clip that flattened a ranking.
    """
    if not math.isfinite(ratio):
        return 0.5
    return float(1.0 / (1.0 + math.exp(-steepness * (ratio - 1.0))))


@dataclass
class Reference:
    """What the quiet historical segment establishes."""

    mean: float
    sd: float
    #: Lag-1 autocorrelation of the history, kept because a series with strong
    #: dependence has a wider natural spread of any running statistic, and a
    #: detector tuned on independent data would fire on it constantly.
    rho: float = 0.0

    @classmethod
    def fit(cls, history: np.ndarray) -> Reference:
        x = np.asarray(history, dtype="float64")
        x = x[np.isfinite(x)]
        if len(x) < 8:
            return cls(mean=0.0, sd=1.0)
        mean = float(x.mean())
        sd = float(x.std(ddof=1))
        rho = 0.0
        if len(x) > 32 and sd > 0:
            centred = x - mean
            denominator = float(np.dot(centred, centred))
            if denominator > 0:
                rho = float(np.dot(centred[:-1], centred[1:]) / denominator)
        return cls(mean=mean, sd=max(sd, 1e-9), rho=float(np.clip(rho, -0.95, 0.95)))

    @property
    def inflation(self) -> float:
        """How much dependence widens the spread of a running mean.

        For an AR(1) process the variance of a mean of n observations is larger
        than the independent case by roughly (1+rho)/(1-rho). Ignoring it is why
        naive detectors alarm on autocorrelated series that never changed.
        """
        return math.sqrt(max((1.0 + self.rho) / (1.0 - self.rho), 1e-6))


@dataclass
class Cusum:
    """Two-sided CUSUM on standardised observations.

    ``drift`` is the slack subtracted at each step: the shift size, in standard
    errors, the detector is tuned to ignore. Without it the sum wanders upward
    on noise alone and every long series eventually alarms.
    """

    reference: Reference
    drift: float = 0.5
    #: Null growth per log(step), measured by simulation.
    growth: float = 0.84
    up: float = 0.0
    down: float = 0.0
    peak: float = 0.0
    steps: float = 0.0

    def update(self, x: float) -> float:
        if math.isfinite(x):
            z = _z(x, self.reference)
            self.up = max(0.0, self.up + z - self.drift)
            self.down = max(0.0, self.down - z - self.drift)
            # The running maximum, not the current value: a break that happened
            # and then partly reverted is still a break, and the score is asked
            # whether one has *already* occurred.
            self.peak = max(self.peak, self.up, self.down)
            self.steps += 1.0
        return _squash(self.peak / _null_scale(self.steps, self.growth, "log"))


@dataclass
class PageHinkley:
    """Page-Hinkley on the running mean, two-sided.

    Tracks how far the cumulative deviation has travelled from its own extreme.
    More sensitive than CUSUM to small persistent shifts, slower to react to
    large ones — which is exactly why both are here.
    """

    reference: Reference
    delta: float = 0.05
    #: Null growth per sqrt(step), measured by simulation.
    growth: float = 1.00
    total_up: float = 0.0
    total_down: float = 0.0
    #: Running minima of each cumulative sum: the extreme the statistic is
    #: measured back from.
    floor_up: float = 0.0
    floor_down: float = 0.0
    peak: float = 0.0
    steps: float = 0.0

    def update(self, x: float) -> float:
        if math.isfinite(x):
            z = _z(x, self.reference)
            self.total_up += z - self.delta
            self.total_down += -z - self.delta
            self.floor_up = min(self.floor_up, self.total_up)
            self.floor_down = min(self.floor_down, self.total_down)
            self.peak = max(
                self.peak,
                self.total_up - self.floor_up,
                self.total_down - self.floor_down,
            )
            self.steps += 1.0
        return _squash(self.peak / _null_scale(self.steps, self.growth, "sqrt"))


@dataclass
class VarianceRatio:
    """Recent dispersion against the historical dispersion.

    Blind to the level by construction, which is the point: a break that widens
    the noise without moving the mean is invisible to the two detectors above.

    The recent estimate is exponentially weighted so it is O(1), and it is
    compared on a log scale so that halving and doubling are treated as equally
    large departures rather than one being bounded by zero.
    """

    reference: Reference
    alpha: float = 0.02
    scale: float = 0.55
    #: Effective observations behind the running estimate. Until it is large
    #: enough the ratio is mostly noise, and reporting it would put short series
    #: at the top of the ranking for no reason.
    warmup: float = 20.0
    ewma_mean: float = field(default=0.0)
    ewma_var: float = field(default=0.0)
    n_eff: float = 0.0
    peak: float = 0.0

    def __post_init__(self) -> None:
        self.ewma_mean = self.reference.mean
        self.ewma_var = self.reference.sd**2

    def update(self, x: float) -> float:
        if not math.isfinite(x):
            return _squash(self.peak / self.scale)
        # Winsorised like the other two, and for the same reason turned around:
        # a single huge value inflates a variance estimate far more than it
        # shifts a mean, so without this the dispersion detector fires on
        # exactly the outliers CUSUM was built to ignore. Measured: it took the
        # spike case from an AUC of 1.000 -- perfectly detecting a non-break --
        # down to chance.
        delta = self.reference.sd * _z(x, self.reference) + self.reference.mean - self.ewma_mean
        delta = float(np.clip(delta, -WINSOR * self.reference.sd, WINSOR * self.reference.sd))
        self.ewma_mean += self.alpha * delta
        self.ewma_var = (1.0 - self.alpha) * (self.ewma_var + self.alpha * delta * delta)
        self.n_eff += 1.0
        if self.n_eff < self.warmup:
            return _squash(self.peak / self.scale)
        ratio = max(self.ewma_var, 1e-12) / max(self.reference.sd**2, 1e-12)
        self.peak = max(self.peak, abs(math.log(ratio)))
        # Divided by the dispersion a log-variance-ratio reaches on an unbroken
        # series, so the same centring applies as to the other two.
        return _squash(self.peak / self.scale)


#: The three, in the order they are combined.
DETECTORS = ("cusum", "page_hinkley", "variance_ratio")


def build(history: np.ndarray) -> dict[str, object]:
    """One of each, sharing a reference fitted on the quiet segment."""
    reference = Reference.fit(history)
    return {
        "cusum": Cusum(reference),
        "page_hinkley": PageHinkley(reference),
        "variance_ratio": VarianceRatio(reference),
    }


def combine(scores: dict[str, float]) -> float:
    """One number from three, by the maximum.

    A mean would let two quiet detectors bury the one that spotted the change,
    and the three are deliberately sensitive to *different* breaks — so the
    question "did any of them see something" is the right one. The cost is a
    higher false-alarm rate, which an AUC across series tolerates far better
    than a missed detection.
    """
    return max(scores.values()) if scores else 0.0
