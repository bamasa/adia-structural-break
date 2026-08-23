"""Forty detector channels, combined by tuned boosted trees.

Assembled from the library by scripts/assemble_submission.py — edits belong in
src/structural_break/, never here.

Submission 005. Three families of channels feed one fitted combiner:

* nine streaming detectors — CUSUM, Page-Hinkley and a variance ratio over
  raw, AR-whitened and absolute views of the standardised stream;
* twenty-four multi-scale discrepancies — exponential windows at six receptive
  fields (5 to 200 observations), each reporting mean- and spread-departure
  from the break-free history, current value and running peak;
* seven retrospective verdicts that re-scan everything seen so far on a
  geometric cadence — best-split statistics at several trailing depths, the
  *stability* of where the best split lands (a real break pins it; noise
  wanders it), and the calmness of the tail after the split, which lets a
  spike-induced alarm be cancelled once ordinary data follows.

Grouped 5-fold CV: 0.5684 against 0.5568 for the nine streaming channels
alone. Hyperparameters from a 12-configuration sweep selected by mean minus
fold spread.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

import joblib
import json
import numpy as np

#: One worker per pair of cores. Left unset, the platform runs a single worker
#: on a sixteen-core machine, and quota is billed in wall-clock hours.
INFER_PARALLELISM = 8



# --- features.py -------------------------------------------------


@dataclass(frozen=True)
class Normalisation:
    """What the historical segment says "ordinary" means for one series."""

    mean: float
    sd: float
    #: Slope per observation, estimated on the history and extended forward.
    slope: float
    #: Lag-1 autocorrelation of the detrended history.
    rho: float
    #: Excess kurtosis; 0 for a Gaussian, large for heavy tails.
    kurtosis: float
    #: Length of the historical segment, which sets how much any of the above
    #: can be trusted.
    n: int

    @property
    def inflation(self) -> float:
        """How much serial dependence widens the spread of a running mean."""
        rho = float(np.clip(self.rho, -0.95, 0.95))
        return math.sqrt(max((1.0 + rho) / (1.0 - rho), 1e-6))

    @property
    def winsor(self) -> float:
        """Where to clip a single observation, widened for heavy tails.

        Four standard errors on Gaussian data. A series whose history is
        genuinely heavy-tailed produces observations beyond that routinely, and
        clipping them all to the same value would throw away the ordering the
        metric is built from — so the threshold grows with the measured excess
        kurtosis, and stops growing at eight, beyond which the clip is doing
        nothing anyway.
        """
        return float(np.clip(4.0 + 0.5 * math.sqrt(max(self.kurtosis, 0.0)), 4.0, 8.0))

    @classmethod
    def fit(cls, history: np.ndarray) -> "Normalisation":
        x = np.asarray(history, dtype="float64")
        x = x[np.isfinite(x)]
        n = len(x)
        if n < 16:
            return cls(mean=0.0, sd=1.0, slope=0.0, rho=0.0, kurtosis=0.0, n=n)

        # Slope by least squares on the index. Estimated on the whole history,
        # since the history is guaranteed break-free and there is nothing to
        # protect against within it.
        t = np.arange(n, dtype="float64")
        t_centred = t - t.mean()
        denominator = float(np.dot(t_centred, t_centred))
        slope = float(np.dot(t_centred, x - x.mean()) / denominator) if denominator > 0 else 0.0

        detrended = x - (x.mean() + slope * t_centred)
        sd = float(detrended.std(ddof=1))
        sd = max(sd, 1e-9)

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

    def expected(self, step: int) -> float:
        """Where the level should be at online step ``step`` if nothing changed.

        ``mean`` is anchored at the last historical observation, so the trend is
        continued from there rather than from the middle of the history.
        """
        return self.mean + self.slope * (step + 1)

    def standardise(self, x: float, step: int) -> float:
        """One observation, detrended and in units of historical spread."""
        return (x - self.expected(step)) / self.sd

    def clip(self, z: float) -> float:
        w = self.winsor
        return float(np.clip(z, -w, w))


def whiten(z: float, previous: float, rho: float) -> float:
    """Remove one-step dependence: the innovation rather than the observation.

    Under an AR(1) the innovation is independent by construction, which is what
    every detector below assumes. The residual carries a factor 1/sqrt(1-rho^2)
    to keep unit variance, so thresholds calibrated on independent data still
    apply.

    Kept as a *separate channel* rather than applied to everything: if the break
    is a change in the dependence itself, whitening through the pre-break
    coefficient is precisely what makes it visible — and precisely what would
    hide it if the detector only ever saw whitened values.
    """
    r = float(np.clip(rho, -0.95, 0.95))
    scale = math.sqrt(max(1.0 - r * r, 1e-6))
    return (z - r * previous) / scale


# --- detectors.py ------------------------------------------------


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
        return squash(self.peak / null_scale(self.steps, self.growth, "log"))


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
        self.peak = max(self.peak, abs(math.log(max(self.ewma_var, 1e-12))))
        return squash(self.peak / self.scale)


# --- retrospective.py --------------------------------------------


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


# --- retro2.py ---------------------------------------------------


#: Trailing depths for the confirm ladder.
DEPTHS = (10, 20, 50, 100)

RETRO2_CHANNELS = (
    "scan_best",
    *(f"recent_{d}" for d in DEPTHS),
    "split_stability",
    "tail_calm",
)


def _recent_vs_before(values: np.ndarray, depth: int) -> float:
    """Standardised mean gap of the last ``depth`` against everything before."""
    t = len(values)
    if t < depth + MIN_SIDE:
        return 0.0
    head, tail = values[: t - depth], values[t - depth :]
    gap = abs(tail.mean() - head.mean())
    spread = math.sqrt(
        max(head.var(ddof=1), 1e-9) / len(head) + max(tail.var(ddof=1), 1e-9) / depth
    )
    return gap / max(spread, 1e-9)


class Retro2:
    """Confirm-and-cancel review of everything seen so far."""

    def __init__(self, history: np.ndarray) -> None:
        self.norm = Normalisation.fit(history)
        self.values: list[float] = []
        self.scores = dict.fromkeys(RETRO2_CHANNELS, 0.5)
        self._argmax_trail: list[int] = []
        self._next_scan = 1

    def _due(self, t: int) -> bool:
        if t >= self._next_scan:
            self._next_scan = max(self._next_scan + 1, int(self._next_scan * 1.12))
            return True
        return False

    def update(self, x: float) -> dict[str, float]:
        z = self.norm.standardise(float(x), len(self.values))
        self.values.append(self.norm.clip(z))
        t = len(self.values)

        if self._due(t) and t >= 2 * MIN_SIDE:
            segment = np.asarray(self.values, dtype="float64")

            gap, _ratio = _scan_mean_variance(segment)
            self.scores["scan_best"] = squash(gap / _null_mean(t))

            for depth in DEPTHS:
                self.scores[f"recent_{depth}"] = squash(
                    _recent_vs_before(segment, depth) / 2.6
                )

            # Where the best split lands, and whether it keeps landing there.
            k = np.arange(MIN_SIDE, t - MIN_SIDE + 1)
            if len(k):
                s1 = np.cumsum(segment)
                n_left = k.astype("float64")
                n_right = t - n_left
                mean_left = s1[k - 1] / n_left
                mean_right = (s1[-1] - s1[k - 1]) / n_right
                s2 = np.cumsum(segment**2)
                var_left = np.maximum(s2[k - 1] / n_left - mean_left**2, 1e-12)
                var_right = np.maximum(
                    (s2[-1] - s2[k - 1]) / n_right - mean_right**2, 1e-12
                )
                stat = np.abs(mean_left - mean_right) / np.sqrt(
                    var_left / n_left + var_right / n_right
                )
                best = int(k[int(stat.argmax())])
                self._argmax_trail.append(best)
                trail = self._argmax_trail[-6:]
                if len(trail) >= 3:
                    # Positions are compared as fractions of the current length,
                    # since the same break drifts forward in absolute index as
                    # the segment grows behind it stays fixed -- it is the
                    # *fraction of history before it* that stabilises.
                    fractions = np.asarray(trail, dtype="float64") / t
                    wander = float(fractions.std())
                    self.scores["split_stability"] = squash(
                        (0.08 - wander) / 0.04 + 1.0
                    )

                # The tail after the best split, against the history: calm tail
                # means the "break" did not persist, and persistence is the
                # definition of the label.
                tail = segment[best:]
                if len(tail) >= MIN_SIDE:
                    drift = abs(tail.mean()) + abs(
                        math.log(max(tail.var(ddof=1) if len(tail) > 1 else 1.0, 1e-9))
                    )
                    self.scores["tail_calm"] = squash(drift / 0.9)
        return dict(self.scores)


# --- multiscale.py -----------------------------------------------


#: Effective window lengths of the exponential kernels.
SCALES = (5, 10, 20, 50, 100, 200)


def _alpha(length: int) -> float:
    return 2.0 / (length + 1.0)


#: Null standard error of an EWMA mean of unit-variance noise, per scale.
_SE_MEAN = tuple(math.sqrt(_alpha(s) / (2.0 - _alpha(s))) for s in SCALES)

#: Null standard deviation of log(EWMA of z^2), simulated: for chi-square
#: inputs the EWMA of z^2 has relative spread sqrt(2 * alpha / (2 - alpha)),
#: and the log is that to first order.
_SE_VAR = tuple(math.sqrt(2.0 * _alpha(s) / (2.0 - _alpha(s))) for s in SCALES)


@dataclass
class MultiScale:
    """Twelve current discrepancies and their running peaks, O(1) per step."""

    means: list[float] = field(default_factory=lambda: [0.0] * len(SCALES))
    variances: list[float] = field(default_factory=lambda: [1.0] * len(SCALES))
    #: Effective observations absorbed, per scale, for the warm-up correction:
    #: an EWMA five observations old has variance far above its asymptote, and
    #: without the correction every series starts with a false alarm.
    weight: list[float] = field(default_factory=lambda: [0.0] * len(SCALES))
    peak_mean: list[float] = field(default_factory=lambda: [0.0] * len(SCALES))
    peak_var: list[float] = field(default_factory=lambda: [0.0] * len(SCALES))

    def update(self, z: float) -> list[float]:
        """Ingest one standardised observation; return the 24 channel values.

        Order: for each scale, the current |mean| discrepancy and current
        |spread| discrepancy; then for each scale, their running peaks.
        """
        current: list[float] = []
        peaks: list[float] = []
        z2 = z * z
        for i, scale in enumerate(SCALES):
            a = _alpha(scale)
            self.means[i] = (1 - a) * self.means[i] + a * z
            self.variances[i] = (1 - a) * self.variances[i] + a * z2
            self.weight[i] = (1 - a) * self.weight[i] + 1.0

            # Warm-up: the variance of a young EWMA exceeds its asymptote by
            # roughly (full weight / current weight); shrink the discrepancy
            # accordingly so the first steps do not alarm by construction.
            maturity = min(self.weight[i] * a * (2.0 - a), 1.0)
            mean_z = abs(self.means[i]) / _SE_MEAN[i] * math.sqrt(maturity)
            var_z = (
                abs(math.log(max(self.variances[i], 1e-9)))
                / _SE_VAR[i]
                * math.sqrt(maturity)
            )
            self.peak_mean[i] = max(self.peak_mean[i], mean_z)
            self.peak_var[i] = max(self.peak_var[i], var_z)
            current.append(mean_z)
            current.append(var_z)
        for i in range(len(SCALES)):
            peaks.append(self.peak_mean[i])
            peaks.append(self.peak_var[i])
        return current + peaks


MULTISCALE_CHANNELS = tuple(
    f"{kind}_{stat}_{scale}"
    for kind in ("now", "peak")
    for scale in SCALES
    for stat in ("mean", "var")
)


# --- competition interface ---------------------------------------------------

VIEWS = ("raw", "whitened", "absolute")


class Monitor:
    """One series watched one observation at a time: all forty channels."""

    def __init__(self, history: np.ndarray) -> None:
        self.norm = Normalisation.fit(history)
        self.detectors = {view: (Cusum(), PageHinkley(), VarianceRatio()) for view in VIEWS}
        self.multiscale = MultiScale()
        self.retro = Retro2.__new__(Retro2)
        # Retro2 refits the same normalisation; share it instead.
        self.retro.norm = self.norm
        self.retro.values = []
        self.retro.scores = dict.fromkeys(RETRO2_CHANNELS, 0.5)
        self.retro._argmax_trail = []
        self.retro._next_scan = 1
        self._previous_z = self.norm.standardise(float(history[-1]), -1) if len(history) else 0.0
        self._step = 0

    def update(self, x: float) -> list[float]:
        z = self.norm.clip(self.norm.standardise(float(x), self._step))
        w = whiten(z, self._previous_z, self.norm.rho)
        views = (z / self.norm.inflation, w, abs(w) - 0.7979)
        self._previous_z = z
        self._step += 1
        stream = [
            detector.update(value)
            for value, group in zip(views, (self.detectors[v] for v in VIEWS))
            for detector in group
        ]
        scales = self.multiscale.update(z)
        retro = self.retro.update(float(x))
        return stream + scales + [retro[c] for c in RETRO2_CHANNELS]


N_CHANNELS = 9 + 24 + len(RETRO2_CHANNELS)


def train(
    datasets: List[Tuple[int, List[float], List[float], Optional[int]]],
    model_directory_path: str,
) -> None:
    """Rebuild every channel for every training series and fit the combiner.

    Hyperparameters are the winners of a 12-configuration sweep on grouped
    3-fold CV (selected by mean minus spread), confirmed on 5 folds. Rows are
    weighted so each step index contributes comparably: the metric averages
    per-step AUCs equally, while a long series floods the late steps.
    """
    import lightgbm as lgb

    rows: list[list[float]] = []
    targets: list[int] = []
    steps: list[int] = []
    for _sid, x_hist, x_online, tau in datasets:
        online = np.asarray(x_online, dtype="float64")
        if len(online) == 0:
            continue
        monitor = Monitor(np.asarray(x_hist, dtype="float64"))
        for step, value in enumerate(online):
            rows.append(monitor.update(float(value)))
            targets.append(int(tau is not None and step >= tau))
            steps.append(step)

    x = np.asarray(rows, dtype="float32")
    y = np.asarray(targets, dtype="int64")
    step_array = np.asarray(steps, dtype="int64")
    unique, counts = np.unique(step_array, return_counts=True)
    lookup = dict(zip(unique.tolist(), counts.tolist()))
    weights = np.array([1.0 / lookup[s] for s in step_array.tolist()], dtype="float64")
    weights *= len(weights) / weights.sum()

    booster = lgb.LGBMClassifier(
        objective="binary",
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=500,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=10.0,
        n_estimators=300,
        random_state=0,
        n_jobs=4,
        deterministic=True,
        force_row_wise=True,
        verbose=-1,
    )
    booster.fit(x, y, sample_weight=weights)
    joblib.dump({"booster": booster}, os.path.join(model_directory_path, "model.joblib"))


def infer(
    datasets: Iterable[Tuple[List[float], Iterable[float]]],
    model_directory_path: str,
):
    model = joblib.load(os.path.join(model_directory_path, "model.joblib"))
    booster = model["booster"]

    yield  # Signal readiness to the runner.

    for x_historical, x_online in datasets:
        monitor = Monitor(np.asarray(x_historical, dtype="float64"))
        for point in x_online:
            channels = np.asarray(monitor.update(float(point)), dtype="float64")
            yield float(booster.predict_proba(channels.reshape(1, -1))[0, 1])
