"""Three online change detectors, on an already-standardised stream.

Normalisation lives in :mod:`structural_break.features` and is done before
anything here sees a value. That separation is deliberate: these detectors then
assume a stream of roughly unit-variance, roughly independent, roughly
zero-mean numbers, and every threshold below is calibrated against exactly that
assumption. A detector that also owned its own normalisation would hide which
of the two was responsible when it misfired.

Why three
---------
They fail in different places, and the failures are complementary rather than
overlapping:

* **CUSUM** accumulates deviations past a slack term and reports its running
  maximum. Sharpest on a sustained shift in level, and deliberately unmoved by a
  lone spike — one outlier enters the sum once, a changed mean enters it every
  step. That distinction is the whole task: a spike that reverts is not a
  structural break.
* **Page-Hinkley** tracks the same cumulative sum measured back from its own
  running minimum. More sensitive than CUSUM to small persistent shifts, slower
  on large ones.
* **Variance ratio** compares recent dispersion to the reference, on a log
  scale so that halving and doubling count equally. Blind to the level by
  construction, which is the point: a break that widens the noise without
  moving the mean is invisible to the two above.

Measured on synthetic cases: on a +0.5 mean shift the first two reach 0.95–1.00
AUC while the dispersion detector sits at 0.42; on a threefold variance change
it reaches 1.00 while they fall to 0.22–0.28.

The null normalisation, and why the growth law differs
------------------------------------------------------
A cumulative statistic grows with stream length even when nothing happens, so a
raw value confounds "something changed" with "this series is long". The metric
compares series against each other at each step and series have different
lengths, so that confound would be scored directly.

Simulation gives the two laws, both stable across n = 50, 200 and 800: a
reflected CUSUM's running maximum grows like 0.84·log(n); Page-Hinkley with a
small delta is essentially the range of a random walk and grows like 1.0·√n.
Using one law for both leaves a length effect in the score, and length is not
evidence of a break.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def null_scale(steps: float, growth: float, law: str) -> float:
    """How large a statistic gets on an unbroken stream by this step."""
    n = max(steps, 2.0)
    return growth * (math.log(n) if law == "log" else math.sqrt(n)) + 1e-9


def squash(ratio: float, steepness: float = 2.0) -> float:
    """Map a null-normalised statistic onto (0, 1), monotonically.

    Logistic and centred on one, so a stream behaving exactly like the null
    lands at 0.5 with room in both tails. Never a clip: a clip makes every value
    above its cap a tie, and ties are precisely the comparisons an AUC is built
    from.
    """
    if not math.isfinite(ratio):
        return 0.5
    return float(1.0 / (1.0 + math.exp(-steepness * (ratio - 1.0))))


@dataclass
class Cusum:
    """Two-sided CUSUM on standardised observations.

    ``drift`` is the slack subtracted each step: the shift, in standard errors,
    the detector is tuned to ignore. Without it the sum wanders upward on noise
    alone and every long stream eventually alarms.
    """

    drift: float = 0.5
    growth: float = 0.84
    up: float = 0.0
    down: float = 0.0
    peak: float = 0.0
    now: float = 0.5
    steps: float = 0.0

    def update(self, z: float) -> float:
        if math.isfinite(z):
            self.up = max(0.0, self.up + z - self.drift)
            self.down = max(0.0, self.down - z - self.drift)
            # The running maximum, not the current value: a break that happened
            # and then partly reverted is still a break, and the question asked
            # is whether one has *already* occurred.
            self.peak = max(self.peak, self.up, self.down)
            self.steps += 1.0
        scale = null_scale(self.steps, self.growth, "log")
        # The current value alongside the peak: the peak can never recant a
        # false alarm, the current value drains back to zero once the stream
        # behaves again. A combiner holding both can learn the difference
        # between "was loud once" and "is loud still".
        self.now = squash(max(self.up, self.down) / scale)
        return squash(self.peak / scale)


@dataclass
class PageHinkley:
    """Page-Hinkley, two-sided, measured back from each sum's own minimum."""

    delta: float = 0.05
    growth: float = 1.00
    total_up: float = 0.0
    total_down: float = 0.0
    floor_up: float = 0.0
    floor_down: float = 0.0
    peak: float = 0.0
    now: float = 0.5
    steps: float = 0.0

    def update(self, z: float) -> float:
        if math.isfinite(z):
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
        scale = null_scale(self.steps, self.growth, "sqrt")
        # Page-Hinkley's excursion above its floor never shrinks, but the
        # growing null does the reverting for it -- slowly. Reported anyway,
        # for symmetry with CUSUM; the trees decide whether it earns its keep.
        self.now = squash(
            max(self.total_up - self.floor_up, self.total_down - self.floor_down)
            / scale
        )
        return squash(self.peak / scale)


@dataclass
class VarianceRatio:
    """Recent dispersion against the reference, on a log scale.

    The recent estimate is exponentially weighted so it stays O(1). ``warmup``
    withholds the score until enough observations are behind it: below that the
    ratio is mostly noise, and reporting it would put short streams at the top
    of the ranking for no reason.
    """

    alpha: float = 0.02
    scale: float = 0.55
    warmup: float = 20.0
    ewma_mean: float = 0.0
    ewma_var: float = 1.0
    n_eff: float = 0.0
    peak: float = 0.0
    now: float = 0.5

    def update(self, z: float) -> float:
        if not math.isfinite(z):
            return squash(self.peak / self.scale)
        delta = z - self.ewma_mean
        self.ewma_mean += self.alpha * delta
        self.ewma_var = (1.0 - self.alpha) * (self.ewma_var + self.alpha * delta * delta)
        self.n_eff += 1.0
        if self.n_eff < self.warmup:
            return squash(self.peak / self.scale)
        # Against 1.0, since the input is standardised: the reference variance
        # is unity by construction.
        current = abs(math.log(max(self.ewma_var, 1e-12)))
        self.peak = max(self.peak, current)
        # The EWMA forgets on its own, so this one genuinely comes home after
        # a false alarm -- the cleanest reverting channel of the three.
        self.now = squash(current / self.scale)
        return squash(self.peak / self.scale)
