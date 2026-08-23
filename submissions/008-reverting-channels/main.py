"""Fifty channels: the forty-one of submission 006 plus nine reverting ones.

Assembled from the library by scripts/assemble_submission.py — edits belong in
src/structural_break/, never here.

Each detector's running peak answers "has it ever looked broken" and can never
take a false alarm back; its current statistic answers "does it look broken
still" and drains once the stream behaves again. Both are now channels, and the
combiner learns when a peak deserves to be discounted. Grouped 5-fold CV:
0.5719 against 0.5662 for the same folds without them, better on all five.
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

#: The dilated-convolution channel's fitted weights, embedded so the submission
#: has no torch dependency and no external file: 721 parameters, trained once
#: locally with a fixed seed, forward pass reimplemented in numpy below and
#: asserted against torch at export time.
CNN_WEIGHTS = json.loads(r"""{"weights": [[[[-0.07266596704721451, 0.36535537242889404, -0.38401609659194946, -0.28864219784736633, -0.1740281879901886]], [[0.04633694887161255, -0.2653433382511139, 0.3749937117099762, -0.08925638347864151, 0.11243904381990433]], [[-0.18263278901576996, 0.327459454536438, -0.4635157287120819, -0.31172114610671997, -0.09763425588607788]], [[-0.1484927535057068, 0.319388210773468, 0.3934409022331238, -0.19953441619873047, -0.3052457869052887]], [[0.1963605135679245, 0.4150046706199646, 0.16825689375400543, 0.3154437243938446, 0.016427600756287575]], [[0.008882584981620312, 0.39959582686424255, -0.5141045451164246, -0.24813681840896606, -0.13653503358364105]], [[-0.03034358099102974, 0.2227049022912979, 0.02118147537112236, -0.05396464839577675, -0.09483098983764648]], [[-0.3941996395587921, -0.019504593685269356, 0.3313857614994049, 0.2779979407787323, 0.1430901139974594]]], [[[-0.059138134121894836, 0.04617363214492798, 0.15371888875961304, 0.17044954001903534, 0.0798509269952774], [-0.04190114885568619, -0.016669372096657753, -0.2779388129711151, -0.11069845408201218, -0.23617416620254517], [-0.10453129559755325, -0.039935559034347534, 0.10067954659461975, 0.07706207036972046, -0.08804690837860107], [-0.1130729615688324, -0.11141818016767502, -0.22384576499462128, -0.12037637084722519, -0.05234265327453613], [0.08196742832660675, 0.0859467014670372, -0.13212136924266815, -0.17143306136131287, 0.02716292254626751], [0.12277842313051224, 0.13620881736278534, 0.16621318459510803, -0.02597498893737793, -0.1505914032459259], [0.06804097443819046, -0.03308410197496414, -0.06715670228004456, 0.19195275008678436, 0.1749388873577118], [-0.09446234256029129, 0.11104535311460495, 0.0010715776588767767, 0.06990113109350204, -0.010398758575320244]], [[0.07344873249530792, -0.008373993448913097, 0.1024591401219368, 0.12097416818141937, 0.1271982342004776], [-0.045978568494319916, -0.09013072401285172, -0.357136994600296, -0.1608702391386032, -0.13989733159542084], [-0.09329688549041748, -0.029072577133774757, 0.16961684823036194, 0.14916692674160004, -0.0660950243473053], [-0.2386324554681778, -0.3692983388900757, -0.27624648809432983, -0.2268332988023758, -0.2869633138179779], [-0.028568744659423828, -0.021838944405317307, -0.06358036398887634, -0.010861807502806187, -0.02288881689310074], [0.10633326321840286, 0.1452413946390152, -0.1189064234495163, -0.09047595411539078, -0.05856044217944145], [0.08933336287736893, 0.13315947353839874, 0.16477148234844208, 0.08561810106039047, 0.02134598046541214], [0.16016885638237, 0.010045374743640423, 0.015190277248620987, 0.05346348136663437, 0.02399212121963501]], [[0.07522007077932358, -0.06560633331537247, 0.006296883337199688, 0.07098760455846786, -0.06372906267642975], [-0.3095565140247345, 0.04410812258720398, -0.2428840547800064, 0.3187541663646698, 0.11791616678237915], [0.15452903509140015, 0.14923174679279327, 0.15936680138111115, 0.1255631446838379, 0.16919894516468048], [-0.12204264849424362, -0.15758198499679565, 0.05179924890398979, 0.08420871943235397, 0.07260526716709137], [0.004545318894088268, -0.12207717448472977, 0.1755431592464447, -0.01996743679046631, 0.09010405838489532], [0.26369088888168335, 0.11848132312297821, -0.13378682732582092, 0.10232344269752502, -0.008707409724593163], [0.058634497225284576, -0.015890374779701233, 0.037288784980773926, 0.02100549079477787, -0.14709416031837463], [-0.19582800567150116, 0.04520189017057419, -0.03734823688864708, 0.10788774490356445, 0.17218708992004395]], [[-0.1469256728887558, -0.03460075333714485, 0.11763176321983337, -0.10897347331047058, 0.1255406141281128], [0.017811216413974762, -0.1832926720380783, 0.12980933487415314, -0.05878594145178795, -0.10419053584337234], [0.11181643605232239, -0.14218145608901978, -0.006225927267223597, -0.08371762186288834, 0.0945466160774231], [0.007023524958640337, 0.15039263665676117, 0.10999167710542679, -0.004907877650111914, -0.08342748135328293], [-0.16691669821739197, 0.1870676577091217, 0.1041833758354187, 0.046802956610918045, -0.14154376089572906], [-0.015018744394183159, -0.1329239308834076, -0.14711043238639832, -0.15105514228343964, 0.05826893076300621], [-0.11405204981565475, -0.013636019080877304, -0.01819911226630211, 0.06773041933774948, -0.15721693634986877], [-0.14044244587421417, -0.10739696770906448, 0.09977449476718903, -0.1199827641248703, 0.1388711780309677]], [[0.05834551900625229, 0.1381978988647461, 0.062259841710329056, -0.008133233524858952, 0.07512498646974564], [-0.15940724313259125, -0.0661868155002594, -0.3134287893772125, -0.5093613862991333, -0.2922689914703369], [0.13668856024742126, -0.03620199114084244, 0.0053617628291249275, 0.21174481511116028, 0.08177337050437927], [-0.16928139328956604, -0.16492584347724915, 0.042284730821847916, -0.14655481278896332, -0.13918302953243256], [-0.23031826317310333, -0.13709448277950287, -0.11603778600692749, -0.1464557647705078, -0.09703379124403], [0.09843853861093521, 0.11915268003940582, -0.07204365730285645, -0.010665013454854488, -0.09813082963228226], [-0.04260752350091934, 0.17433878779411316, 0.17139527201652527, 0.1617736518383026, 0.0459241047501564], [0.13240410387516022, 0.20853592455387115, 0.033593952655792236, 0.21354402601718903, -0.049942538142204285]], [[0.019658450037240982, 0.003093617269769311, -0.007402326911687851, 0.07788045704364777, 0.15720833837985992], [-0.24156668782234192, -0.28418561816215515, -0.3480963408946991, -0.34211021661758423, -0.2311682403087616], [-0.0488220676779747, -0.10727489739656448, -0.058174069970846176, 0.1508089154958725, 0.08641470968723297], [-0.06066516414284706, -0.11866629868745804, -0.08814534544944763, -0.0238803718239069, -0.047468531876802444], [0.13095104694366455, -0.06154302507638931, -0.184219092130661, -0.14862747490406036, -0.22847095131874084], [-0.007237298414111137, -0.13106945157051086, -0.10580844432115555, 0.12543489038944244, -0.1539032757282257], [0.05114755034446716, 0.11914321780204773, 0.14097534120082855, -0.018614796921610832, -0.027333194389939308], [0.2002330869436264, 0.1034429669380188, 0.21192242205142975, 0.0347059890627861, 0.06734433025121689]], [[0.005103309638798237, 0.026759859174489975, -0.05924120172858238, 0.1270555853843689, 0.04089225456118584], [-0.1919938623905182, -0.07990031689405441, -0.33878856897354126, -0.16975075006484985, -0.24093687534332275], [0.10940025001764297, -0.0325421579182148, -0.07665270566940308, 0.09953749924898148, -0.0949660912156105], [-0.1287786066532135, -0.19313979148864746, -0.2027260810136795, -0.11111266911029816, -0.12527219951152802], [-0.03707502782344818, -0.12918078899383545, -0.20530784130096436, -0.048512108623981476, 0.03946112096309662], [-0.07407931983470917, 0.043226562440395355, 0.07434353977441788, -0.0547136627137661, -0.07380810379981995], [0.13823635876178741, 0.07611194252967834, 0.1257246732711792, 0.11143500357866287, -0.05150715261697769], [0.08468808978796005, 0.18303436040878296, 0.22740834951400757, 0.14765281975269318, 0.08809313923120499]], [[0.19747118651866913, -0.024803102016448975, -0.021751029416918755, -0.34706833958625793, 0.25789499282836914], [-0.009912432171404362, -0.4104403853416443, 0.22023265063762665, 0.11552625149488449, 0.11448077112436295], [0.20899517834186554, 0.17008304595947266, 0.12860535085201263, -0.29257452487945557, 0.20288920402526855], [0.15956231951713562, -0.13583636283874512, 0.1851138174533844, -0.16067203879356384, 0.3064350187778473], [-0.13668222725391388, 0.09105762094259262, -0.15159691870212555, 0.34594377875328064, -0.03269543498754501], [0.265011191368103, 0.031092967838048935, 0.1576252281665802, -0.3240121603012085, 0.20065492391586304], [-0.04558062553405762, 0.10389783978462219, -0.12190999835729599, -0.18767249584197998, -0.0849926695227623], [0.14768218994140625, 0.07740621268749237, 0.13255666196346283, 0.20392796397209167, -0.14635664224624634]]], [[[0.15786750614643097, 0.1666419506072998, 0.1527860462665558, 0.07780880481004715, 0.1589934527873993], [0.0474998839199543, 0.13993918895721436, 0.22725574672222137, 0.0920000746846199, 0.22787976264953613], [0.15015222132205963, 0.1964368224143982, -0.01945049874484539, 0.09731553494930267, 0.025496596470475197], [-0.007236317731440067, -0.038450006395578384, -0.033755846321582794, -0.08465082943439484, 0.1431020051240921], [0.20748209953308105, 0.04205714538693428, -0.035411935299634933, 0.17469021677970886, 0.20409195125102997], [0.06321407854557037, -0.05423946678638458, 0.19933681190013885, 0.1498749703168869, -0.13791941106319427], [0.20010848343372345, 0.1691150814294815, -0.02845182456076145, 0.14229176938533783, -0.052663058042526245], [-0.012722316198050976, 0.10243749618530273, 0.001169963856227696, -0.1227884441614151, -0.07331733405590057]], [[-0.048742473125457764, -0.027551086619496346, 0.01849263347685337, 0.1457531750202179, 0.0025172766763716936], [-0.06126939132809639, 0.06010901555418968, 0.14580538868904114, 0.1461813747882843, -0.06655426323413849], [0.006396281532943249, -0.07718757539987564, -0.04858068376779556, 0.06575117260217667, -0.0963163748383522], [0.12261679023504257, 0.18965034186840057, 0.04085325822234154, -0.06408834457397461, -0.009934160858392715], [0.0672948881983757, -0.05865666642785072, 0.14132842421531677, -0.07288219779729843, -0.07111137360334396], [-0.11937816441059113, 0.02215956151485443, 0.15601547062397003, -0.09123306721448898, 0.19101405143737793], [-0.037016287446022034, 0.087050661444664, 0.05021705478429794, -0.10471190512180328, 0.1821148544549942], [0.11229751259088516, 0.15994995832443237, 0.0723111629486084, -0.1036158949136734, 0.11960742622613907]], [[-0.04338812455534935, 0.019333574920892715, -0.17513011395931244, -0.09161722660064697, -0.2032078504562378], [-0.11074882745742798, -0.020161768421530724, -0.17177219688892365, -0.001693509635515511, 0.0338299423456192], [-0.04082900658249855, -0.16209733486175537, -0.17788773775100708, -0.10433219373226166, 0.00146418996155262], [0.11673115193843842, 0.054610561579465866, -0.056259673088788986, 0.018007157370448112, -0.19105201959609985], [-0.1781725287437439, 0.02178489789366722, -0.1310410499572754, 0.027644028887152672, -0.15527212619781494], [-0.12982052564620972, -0.17946144938468933, 0.028999846428632736, -0.06735968589782715, 0.06781486421823502], [0.044182732701301575, 0.031522706151008606, -0.04753433167934418, -0.10175201296806335, -0.08642755448818207], [-0.0280794445425272, 0.029198849573731422, -0.15364298224449158, 0.07744250446557999, -0.06352938711643219]], [[-0.10916629433631897, -0.2203637957572937, 0.0828181579709053, 0.07932247966527939, -0.13136771321296692], [0.11602295935153961, -0.031054237857460976, 0.019948581233620644, 0.0592438206076622, -0.01968088559806347], [0.2507339417934418, 0.2316712588071823, -0.04885419085621834, 0.013864870183169842, -0.10004783421754837], [0.03996431455016136, -0.0879625529050827, -0.09826092422008514, -0.001657521235756576, 0.007928467355668545], [-0.08729220926761627, -0.17018118500709534, 0.03750700503587723, -0.07268108427524567, 0.02858889475464821], [-0.12550956010818481, -0.05143796652555466, -0.24679657816886902, -0.19236774742603302, -0.09089773148298264], [0.11202679574489594, -0.22111260890960693, -0.14354851841926575, -0.12777067720890045, 0.09025201946496964], [0.1848336011171341, -0.015352067537605762, 0.29867708683013916, 0.10450126230716705, -0.06732380390167236]], [[-0.09760520607233047, -0.09658834338188171, -0.10860779881477356, 0.05538708344101906, -0.08042702078819275], [0.14025425910949707, 5.312991561368108e-05, -0.03448440879583359, 0.07998818159103394, 0.23331481218338013], [-0.09606583416461945, 0.07872532308101654, -0.12803083658218384, 0.1909063309431076, -0.228504940867424], [-0.09277703613042831, 0.25575119256973267, -0.07136852294206619, 0.09473881125450134, 0.01932990737259388], [-0.015363908372819424, -0.04801918938755989, 0.1578189730644226, 0.06326418370008469, -0.05485735461115837], [-0.07208260893821716, 0.0009414826636202633, 0.13475026190280914, -0.08303821086883545, 0.16733644902706146], [-0.012352907098829746, -0.03954903036355972, 0.16131094098091125, 0.1244293749332428, 0.1587761789560318], [0.2382027506828308, -0.19462615251541138, 0.03407410532236099, -0.04263107478618622, 0.13053303956985474]], [[-0.25476059317588806, -0.35302430391311646, 0.0036195802967995405, 0.10342748463153839, -0.18155576288700104], [-0.1625356823205948, -0.35669752955436707, -0.03922716900706291, -0.09795164316892624, -0.05477699264883995], [0.24562585353851318, 0.24808020889759064, 0.2032395452260971, 0.057826586067676544, 0.1183033362030983], [-0.1295841783285141, -0.03982139006257057, 0.09571945667266846, 0.04573991522192955, 0.024474579840898514], [-0.2548828721046448, 0.08900532126426697, 0.06594540923833847, 0.037556152790784836, -0.25807690620422363], [-0.1371937394142151, -0.32285577058792114, -0.003265828127041459, -0.14432191848754883, -0.13819287717342377], [0.058110564947128296, -0.3276696801185608, -0.2641007602214813, -0.12498490512371063, -0.09363022446632385], [-0.11474420875310898, 0.3894168734550476, 0.25988060235977173, 0.06178545951843262, 0.17068222165107727]], [[0.15906712412834167, 0.21900688111782074, 0.20776914060115814, 0.038762662559747696, -0.05252693593502045], [0.11531481146812439, -0.05858748406171799, -0.2711677849292755, -0.22056205570697784, -0.33752408623695374], [0.009362898766994476, 0.060424912720918655, 0.049544643610715866, 0.12580782175064087, 0.1327829211950302], [0.08887876570224762, 0.08977971971035004, 0.571628212928772, 0.197941854596138, 0.07658220082521439], [-0.13492731750011444, -0.27900373935699463, -0.22081229090690613, -0.17847569286823273, -0.45357683300971985], [0.0014040963724255562, -0.07174360752105713, -0.10240492224693298, -0.327627956867218, -0.1360824853181839], [-0.10986284166574478, 0.0936395525932312, -0.20485025644302368, -0.1773754507303238, -0.12123966962099075], [0.13616904616355896, 0.35719072818756104, 0.5374603271484375, 0.2669313848018646, 0.43744808435440063]], [[-0.16165724396705627, 0.14369376003742218, -0.14443868398666382, 0.003228071378543973, 0.0003351292107254267], [-0.15679532289505005, 0.11264646053314209, -0.221826434135437, -0.008656497113406658, 0.1885976493358612], [-0.021417956799268723, 0.04708721116185188, -0.16957010328769684, 0.1626754254102707, -0.4001619517803192], [-0.04151744395494461, 0.08482171595096588, -0.04875239357352257, 0.17366383969783783, 0.20395909249782562], [-0.09773099422454834, -0.02719491720199585, 0.11750613898038864, 0.020400797948241234, 0.18236082792282104], [-0.23148775100708008, -0.04603837803006172, -0.02426370419561863, 0.030963433906435966, 0.15737831592559814], [-0.1915704905986786, -0.20932775735855103, 0.08988994359970093, 0.08159878849983215, -0.028392065316438675], [-0.23984111845493317, -0.006517020985484123, 0.21135753393173218, -0.26095348596572876, -0.24328725039958954]]]], "biases": [[0.14118149876594543, -0.5116382837295532, 0.14577384293079376, -0.5408392548561096, -0.35823559761047363, -0.19792646169662476, 0.5832894444465637, 0.4822920858860016], [0.018467577174305916, 0.17108038067817688, -0.04695342108607292, 0.08008492738008499, 0.1019597053527832, 0.08697020262479782, -0.0006312492769211531, -0.07398853451013565], [0.21935778856277466, 0.025616778060793877, -0.00841254647821188, -0.04282330721616745, 0.03571523725986481, -0.02544151060283184, -0.19449251890182495, 0.08638565242290497]], "head_w": [-0.2109154462814331, 0.006330783013254404, -0.012051859870553017, 0.21376824378967285, -0.16998203098773956, 0.2684285342693329, 0.19878853857517242, -0.16279537975788116, -0.2883453369140625, -0.21580877900123596, 0.20575323700904846, -0.1299976408481598, -0.029396722093224525, 0.03226596117019653, 0.30248162150382996, 0.20041824877262115], "head_b": 0.10040033608675003}""")

CNN_WINDOW = 128
CNN_DILATIONS = (1, 4, 16)
CNN_KERNEL = 5


class InlineCnn:
    def __init__(self, data: dict) -> None:
        self.weights = [np.asarray(w) for w in data["weights"]]
        self.biases = [np.asarray(b) for b in data["biases"]]
        self.head_w = np.asarray(data["head_w"])
        self.head_b = float(data["head_b"])

    def forward(self, window: np.ndarray) -> float:
        h = window.reshape(1, -1).astype("float64")
        for weight, bias, dilation in zip(self.weights, self.biases, CNN_DILATIONS):
            pad = (CNN_KERNEL - 1) * dilation // 2
            padded = np.pad(h, ((0, 0), (pad, pad)))
            out = np.empty((weight.shape[0], h.shape[1]))
            for j in range(h.shape[1]):
                taps = padded[:, j : j + (CNN_KERNEL - 1) * dilation + 1 : dilation]
                out[:, j] = np.tensordot(weight, taps, axes=([1, 2], [0, 1])) + bias
            h = np.maximum(out, 0.0)
        pooled = np.concatenate([h.max(axis=1), h.mean(axis=1)])
        z = float(pooled @ self.head_w + self.head_b)
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


VIEWS = ("raw", "whitened", "absolute")


class Monitor:
    """One series watched one observation at a time: all fifty channels."""

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
        # The convolutional channel: a trailing window of standardised values
        # that may span the boundary -- the platform has the history in hand
        # too, and "the boundary looked ordinary" is itself a learnable shape.
        self.cnn = InlineCnn(CNN_WEIGHTS)
        tail = [
            self.norm.clip(self.norm.standardise(float(v), -len(history) + i))
            for i, v in enumerate(np.asarray(history, dtype="float64")[-CNN_WINDOW:])
        ]
        self._window = tail
        self._cnn_score = 0.5
        self._cnn_next = 1

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

        # The convolutional channel runs on the retro cadence: geometric early,
        # then sparse -- a forward pass per step would be quadratic in cost for
        # a score that changes little between adjacent late steps.
        self._window.append(z)
        if len(self._window) > CNN_WINDOW:
            self._window.pop(0)
        if self._step >= self._cnn_next:
            self._cnn_next = max(self._cnn_next + 1, int(self._cnn_next * 1.12))
            w = np.asarray(self._window, dtype="float64")
            if len(w) < CNN_WINDOW:
                w = np.concatenate([np.zeros(CNN_WINDOW - len(w)), w])
            self._cnn_score = self.cnn.forward(w)

        # The reverting counterparts, in the same detector order as `stream`:
        # the current statistic rather than the running peak. The peak cannot
        # recant a false alarm; the current value drains once the stream
        # behaves again, and the trees hold both.
        nows = [d.now for v in VIEWS for d in self.detectors[v]]

        return (
            stream
            + scales
            + [retro[c] for c in RETRO2_CHANNELS]
            + [self._cnn_score]
            + nows
        )


N_CHANNELS = 9 + 24 + len(RETRO2_CHANNELS) + 1 + 9


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
