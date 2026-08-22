"""Three classical online change detectors, combined.

Submission for the ADIA Lab Structural Break Challenge: Real-Time Edition.

The task hands over a long quiet history and then reveals a short online segment
one observation at a time, asking after each for the confidence that the process
has already changed. Three detector families answer that, and they fail in
different places -- which is the reason to run all three rather than choose:

* **CUSUM** on standardised observations: sharpest on a sustained shift in
  level, deliberately blind to a lone spike, since one outlier moves the sum
  once while a changed mean moves it every step.
* **Page-Hinkley**: the same idea with different bookkeeping, more sensitive to
  small persistent shifts and slower on large ones.
* **Variance ratio**: ignores the level entirely, so a break that widens the
  noise without moving the mean -- invisible to the first two -- is obvious.

Measured on synthetic cases, the split is clean: on a +0.5 mean shift the first
two reach 0.95-1.00 AUC while the dispersion detector sits at 0.42; on a
threefold variance change it reaches 1.00 while they fall to 0.22-0.28. The
maximum of the three covers both.

Three details that decide whether this works at all
---------------------------------------------------
**Winsorising.** Every observation is clipped to four standard errors before it
enters any statistic. A single huge value otherwise moves a cumulative sum as
far as a genuine shift does over many steps, and inflates a variance estimate
much further. Without it the dispersion detector scored 1.00 AUC on a *spike
that reverts* -- perfectly detecting a non-break. With it, 0.60.

**Null normalisation, with the right growth law.** A cumulative statistic grows
with stream length even when nothing happens, so a raw value confounds "changed"
with "long". Since the metric compares series against each other at each step
and series have different lengths, that confound would be scored directly.
Simulation gives the laws: a reflected CUSUM's running maximum grows like
0.84*log(n), Page-Hinkley's like 1.0*sqrt(n) -- both stable across n = 50, 200
and 800. Using one law for both leaves a length effect in the score.

**Logistic squashing, never a clip.** The score is centred so a series behaving
exactly like its own history lands at 0.5. A clip would make every value above
its cap a tie, and ties are precisely the comparisons an AUC is built from.

**Dependence.** The historical lag-1 autocorrelation widens the natural spread
of any running mean by roughly sqrt((1+rho)/(1-rho)), and detectors tuned on
independent data alarm constantly on dependent series that never changed. The
reference carries that inflation.

Everything is O(1) per observation: the platform reveals points one at a time,
and re-deriving a statistic from the whole history at each step would be
quadratic -- too slow at ten thousand series of up to a thousand points.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

import joblib
import numpy as np

#: Standard errors beyond which one observation stops counting more.
WINSOR = 4.0


@dataclass
class Reference:
    """What the quiet historical segment establishes."""

    mean: float
    sd: float
    rho: float = 0.0

    @classmethod
    def fit(cls, history: np.ndarray) -> "Reference":
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
        """How much serial dependence widens the spread of a running mean."""
        return math.sqrt(max((1.0 + self.rho) / (1.0 - self.rho), 1e-6))


def _z(x: float, reference: Reference) -> float:
    z = (x - reference.mean) / (reference.sd * reference.inflation)
    return float(np.clip(z, -WINSOR, WINSOR))


def _null_scale(steps: float, growth: float, law: str) -> float:
    n = max(steps, 2.0)
    return growth * (math.log(n) if law == "log" else math.sqrt(n)) + 1e-9


def _squash(ratio: float, steepness: float = 2.0) -> float:
    """Logistic, centred on one: null behaviour lands at 0.5."""
    if not math.isfinite(ratio):
        return 0.5
    return float(1.0 / (1.0 + math.exp(-steepness * (ratio - 1.0))))


@dataclass
class Cusum:
    reference: Reference
    drift: float = 0.5
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
            # The running maximum: a break that happened and then partly
            # reverted is still a break, and the question is whether one has
            # *already* occurred.
            self.peak = max(self.peak, self.up, self.down)
            self.steps += 1.0
        return _squash(self.peak / _null_scale(self.steps, self.growth, "log"))


@dataclass
class PageHinkley:
    reference: Reference
    delta: float = 0.05
    growth: float = 1.00
    total_up: float = 0.0
    total_down: float = 0.0
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
    reference: Reference
    alpha: float = 0.02
    scale: float = 0.55
    #: Effective observations before the ratio is reported. Below this it is
    #: mostly noise, and reporting it would put short series at the top of the
    #: ranking for no reason.
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
        delta = self.reference.sd * _z(x, self.reference) + self.reference.mean - self.ewma_mean
        delta = float(np.clip(delta, -WINSOR * self.reference.sd, WINSOR * self.reference.sd))
        self.ewma_mean += self.alpha * delta
        self.ewma_var = (1.0 - self.alpha) * (self.ewma_var + self.alpha * delta * delta)
        self.n_eff += 1.0
        if self.n_eff < self.warmup:
            return _squash(self.peak / self.scale)
        ratio = max(self.ewma_var, 1e-12) / max(self.reference.sd**2, 1e-12)
        self.peak = max(self.peak, abs(math.log(ratio)))
        return _squash(self.peak / self.scale)


DETECTORS = ("cusum", "page_hinkley", "variance_ratio")


def train(
    datasets: List[Tuple[int, List[float], List[float], Optional[int]]],
    model_directory_path: str,
) -> None:
    """Nothing is fitted.

    The detectors are calibrated against a simulated null rather than against
    the training labels, so this submission has no learned parameters. Recorded
    explicitly instead of left as an empty function, so the next version can be
    compared against a stated starting point.
    """
    joblib.dump(
        {
            "note": "three classical detectors, calibrated on a simulated null",
            "detectors": list(DETECTORS),
            "winsor": WINSOR,
            "growth": {"cusum": 0.84, "page_hinkley": 1.00},
        },
        os.path.join(model_directory_path, "model.joblib"),
    )


def infer(
    datasets: Iterable[Tuple[List[float], Iterable[float]]],
    model_directory_path: str,
):
    """Stream per-step scores: the maximum of the three detectors.

    The maximum rather than the mean. The three are deliberately sensitive to
    *different* breaks, so two quiet ones would bury the one that saw the
    change. The cost is a higher false-alarm rate, which an AUC across series
    tolerates far better than a missed detection.
    """
    joblib.load(os.path.join(model_directory_path, "model.joblib"))

    yield  # Signal readiness to the runner.

    for x_historical, x_online in datasets:
        reference = Reference.fit(np.asarray(x_historical, dtype="float64"))
        state = (
            Cusum(reference),
            PageHinkley(reference),
            VarianceRatio(reference),
        )
        for point in x_online:
            x = float(point)
            yield max(detector.update(x) for detector in state)
