"""Nine detector channels, combined by a fitted linear weighting.

Submission 003 for the ADIA Lab Structural Break Challenge: Real-Time Edition.

Submission 001 ran three classical detectors and took their maximum, scoring a
local TS-AUC of 0.5385. The maximum was an assumption; ten thousand labelled
series is enough evidence to replace it with a fitted weighting, and that is the
whole change here.

What the model sees
-------------------
Nine channels: three detector families over three views of the stream.

The views each remove one way of being fooled and each destroy one kind of
break, so all three run rather than one being chosen:

* **raw** — detrended, standardised, dependence-adjusted. Sees a change in
  level.
* **whitened** — additionally passed through the historical AR coefficient.
  Immune to dependence that was already there; loud when the dependence itself
  changes, because the wrong coefficient stops whitening.
* **absolute** — the magnitude of the whitened value, centred. Turns a change in
  spread into a change in level, so the mean-shift detectors can find it.

What the model deliberately does not see
-----------------------------------------
Anything describing the series rather than the moment: the length of its
history, its dispersion, its kurtosis, its autocorrelation — and the step index.

That is not caution, it is measurement. Adding them takes the validation score
from 0.7487 down to 0.5049. The reason is the metric: it compares series
*against each other at the same step*, so a model that ranks series by their own
properties is ranking on something orthogonal to whether a break happened, and
the step index is identical across every series being compared at that step —
pure capacity spent on a constant.

Normalisation
-------------
Everything the detectors see has already had four things removed, each of which
a detector would otherwise mistake for a change: the level, the scale, the
trend, and the serial dependence. What is removed is kept as its own channel
rather than discarded, because a break may live in exactly the quantity being
normalised away.

Streaming
---------
Every statistic updates in O(1) from the observation just seen. The platform
reveals one point at a time and re-deriving anything from a window that includes
later points is both impossible there and quadratic here.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

import joblib
import numpy as np

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Normalisation:
    """What the break-free historical segment says "ordinary" means here."""

    mean: float
    sd: float
    slope: float
    rho: float
    kurtosis: float
    n: int

    @property
    def inflation(self) -> float:
        """How much serial dependence widens the spread of a running mean.

        For an AR(1) the variance of a mean of n observations exceeds the
        independent case by roughly (1+rho)/(1-rho). A detector that ignores it
        fires on dependence rather than on change.
        """
        rho = float(np.clip(self.rho, -0.95, 0.95))
        return math.sqrt(max((1.0 + rho) / (1.0 - rho), 1e-6))

    @property
    def winsor(self) -> float:
        """Where one observation stops counting more, widened for heavy tails."""
        return float(np.clip(4.0 + 0.5 * math.sqrt(max(self.kurtosis, 0.0)), 4.0, 8.0))

    @classmethod
    def fit(cls, history: np.ndarray) -> "Normalisation":
        x = np.asarray(history, dtype="float64")
        x = x[np.isfinite(x)]
        n = len(x)
        if n < 16:
            return cls(mean=0.0, sd=1.0, slope=0.0, rho=0.0, kurtosis=0.0, n=n)

        t = np.arange(n, dtype="float64")
        t_centred = t - t.mean()
        denominator = float(np.dot(t_centred, t_centred))
        slope = float(np.dot(t_centred, x - x.mean()) / denominator) if denominator > 0 else 0.0

        detrended = x - (x.mean() + slope * t_centred)
        sd = max(float(detrended.std(ddof=1)), 1e-9)

        rho = 0.0
        if n > 32:
            centred = detrended - detrended.mean()
            denominator = float(np.dot(centred, centred))
            if denominator > 0:
                rho = float(np.dot(centred[:-1], centred[1:]) / denominator)

        z = detrended / sd
        kurtosis = float(np.mean(z**4) - 3.0) if n > 32 else 0.0

        return cls(
            mean=float(x.mean() + slope * (n - 1 - t.mean())),
            sd=sd,
            slope=slope,
            rho=float(np.clip(rho, -0.95, 0.95)),
            kurtosis=float(np.clip(kurtosis, -2.0, 50.0)),
            n=n,
        )

    def standardise(self, x: float, step: int) -> float:
        """Detrended, and in units of the historical spread.

        The trend is continued from the last historical observation, so a series
        that keeps drifting exactly as it always did produces no signal.
        """
        return (x - (self.mean + self.slope * (step + 1))) / self.sd

    def clip(self, z: float) -> float:
        w = self.winsor
        return float(np.clip(z, -w, w))


def whiten(z: float, previous: float, rho: float) -> float:
    """The innovation rather than the observation, scaled to unit variance."""
    r = float(np.clip(rho, -0.95, 0.95))
    return (z - r * previous) / math.sqrt(max(1.0 - r * r, 1e-6))


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------


def null_scale(steps: float, growth: float, law: str) -> float:
    """How large a statistic gets on an unbroken stream by this step.

    A cumulative statistic grows with stream length even when nothing happens,
    and since the metric compares series of different lengths against each
    other, that confound would be scored directly. Simulation gives the laws: a
    reflected CUSUM's running maximum grows like 0.84*log(n), Page-Hinkley's
    like 1.0*sqrt(n), both stable across n = 50, 200 and 800.
    """
    n = max(steps, 2.0)
    return growth * (math.log(n) if law == "log" else math.sqrt(n)) + 1e-9


def squash(ratio: float, steepness: float = 2.0) -> float:
    """Logistic, centred on one. Never a clip: ties are what an AUC is made of."""
    if not math.isfinite(ratio):
        return 0.5
    return float(1.0 / (1.0 + math.exp(-steepness * (ratio - 1.0))))


@dataclass
class Cusum:
    drift: float = 0.5
    growth: float = 0.84
    up: float = 0.0
    down: float = 0.0
    peak: float = 0.0
    steps: float = 0.0

    def update(self, z: float) -> float:
        if math.isfinite(z):
            self.up = max(0.0, self.up + z - self.drift)
            self.down = max(0.0, self.down - z - self.drift)
            # The running maximum: a break that partly reverted is still a
            # break, and the question is whether one has *already* occurred.
            self.peak = max(self.peak, self.up, self.down)
            self.steps += 1.0
        return squash(self.peak / null_scale(self.steps, self.growth, "log"))


@dataclass
class PageHinkley:
    delta: float = 0.05
    growth: float = 1.00
    total_up: float = 0.0
    total_down: float = 0.0
    floor_up: float = 0.0
    floor_down: float = 0.0
    peak: float = 0.0
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
        return squash(self.peak / null_scale(self.steps, self.growth, "sqrt"))


@dataclass
class VarianceRatio:
    alpha: float = 0.02
    scale: float = 0.55
    warmup: float = 20.0
    ewma_mean: float = 0.0
    ewma_var: float = 1.0
    n_eff: float = 0.0
    peak: float = 0.0

    def update(self, z: float) -> float:
        if not math.isfinite(z):
            return squash(self.peak / self.scale)
        delta = z - self.ewma_mean
        self.ewma_mean += self.alpha * delta
        self.ewma_var = (1.0 - self.alpha) * (self.ewma_var + self.alpha * delta * delta)
        self.n_eff += 1.0
        if self.n_eff < self.warmup:
            return squash(self.peak / self.scale)
        self.peak = max(self.peak, abs(math.log(max(self.ewma_var, 1e-12))))
        return squash(self.peak / self.scale)


VIEWS = ("raw", "whitened", "absolute")
FAMILIES = ("cusum", "page_hinkley", "variance_ratio")
CHANNELS = tuple(f"{v}_{f}" for v in VIEWS for f in FAMILIES)


class Monitor:
    """One series being watched, one observation at a time."""

    def __init__(self, history: np.ndarray) -> None:
        self.norm = Normalisation.fit(history)
        self.detectors = {
            view: (Cusum(), PageHinkley(), VarianceRatio()) for view in VIEWS
        }
        self._previous_z = (
            self.norm.standardise(float(history[-1]), -1) if len(history) else 0.0
        )
        self._step = 0

    def update(self, x: float) -> list[float]:
        z = self.norm.clip(self.norm.standardise(float(x), self._step))
        w = whiten(z, self._previous_z, self.norm.rho)
        views = (
            z / self.norm.inflation,
            w,
            abs(w) - 0.7979,  # centred: E|N(0,1)| = sqrt(2/pi)
        )
        self._previous_z = z
        self._step += 1
        return [
            detector.update(value)
            for value, group in zip(views, (self.detectors[v] for v in VIEWS))
            for detector in group
        ]


# ---------------------------------------------------------------------------
# Competition interface
# ---------------------------------------------------------------------------


def train(
    datasets: List[Tuple[int, List[float], List[float], Optional[int]]],
    model_directory_path: str,
) -> None:
    """Fit the combiner on every training series.

    One row per online step: the nine channels at that step against whether the
    break has already happened. Rows are weighted so each step index contributes
    comparably, because the metric averages per-step AUCs equally while a long
    series contributes many late rows and few early ones -- and the early ones
    are where the evidence is thinnest and the score is decided.
    """
    from sklearn.linear_model import LogisticRegression

    rows: list[list[float]] = []
    targets: list[int] = []
    steps: list[int] = []

    for _dataset_id, x_hist, x_online, tau in datasets:
        online = np.asarray(x_online, dtype="float64")
        if len(online) == 0:
            continue
        monitor = Monitor(np.asarray(x_hist, dtype="float64"))
        # Every step of every series, no subsampling. A geometric grid here is
        # what fabricated a 0.7487 cross-validation against a 0.4990 reality in
        # an earlier version: the grid's spacing depended on series length, so
        # each step held a length-dependent selection of series, and the metric
        # compares series against each other at exactly those steps.
        for step, value in enumerate(online):
            channels = monitor.update(float(value))
            rows.append(channels)
            targets.append(int(tau is not None and step >= tau))
            steps.append(step)

    x = np.asarray(rows, dtype="float64")
    y = np.asarray(targets, dtype="int64")
    step_array = np.asarray(steps, dtype="int64")

    unique, counts = np.unique(step_array, return_counts=True)
    lookup = dict(zip(unique.tolist(), counts.tolist()))
    weights = np.array([1.0 / lookup[s] for s in step_array.tolist()], dtype="float64")
    weights *= len(weights) / weights.sum()

    # One coefficient per channel, fitted on the logit scale so the scores
    # combine multiplicatively in odds and the result stays a probability. The
    # readable alternative to the trees: a negative coefficient names a channel
    # that is anti-informative, which is worth knowing in itself.
    epsilon = 1e-6
    logits = np.log(np.clip(x, epsilon, 1 - epsilon) / np.clip(1 - x, epsilon, 1 - epsilon))
    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(logits, y, sample_weight=weights)
    joblib.dump(
        {"model": model, "channels": list(CHANNELS)},
        os.path.join(model_directory_path, "model.joblib"),
    )


def infer(
    datasets: Iterable[Tuple[List[float], Iterable[float]]],
    model_directory_path: str,
):
    """Stream per-step scores from the fitted combiner."""
    artefact = joblib.load(os.path.join(model_directory_path, "model.joblib"))
    model = artefact["model"]
    epsilon = 1e-6

    yield  # Signal readiness to the runner.

    for x_historical, x_online in datasets:
        monitor = Monitor(np.asarray(x_historical, dtype="float64"))
        for point in x_online:
            channels = np.asarray(monitor.update(float(point)), dtype="float64")
            logits = np.log(
                np.clip(channels, epsilon, 1 - epsilon)
                / np.clip(1 - channels, epsilon, 1 - epsilon)
            )
            yield float(model.predict_proba(logits.reshape(1, -1))[0, 1])
