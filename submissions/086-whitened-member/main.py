"""#39 with the whitened stream as a member: every test on innovations the history made i.i.d.

Assembled from the library by scripts/assemble_submission.py — edits belong in
src/structural_break/, never here.

The ensemble of #39 (cloud 0.6056) with one member added at a 0.30 share
at every step. The history fits an AR(p <= 12) by BIC, a conditional
scale whose memory is chosen by Gaussian quasi-likelihood, and the
empirical CDF of the standardised innovations; the online stream is mapped
through that fit to normal scores that are i.i.d. N(0,1) under the null
for a far wider family of histories -- real-world series included -- than
the AR(1) whitening every other channel used. On that stream: CUSUMs and
exponential averages for mean, scale, dependence and volatility
clustering, shape and tail frequencies against the history's own,
generalised likelihood ratios over dyadic windows 8-1024 for mean, scale
and lag-1 dependence, and KS / Cramer-von Mises / Anderson-Darling tests
against N(0,1) over windows 32-512 and the prefix; mean and dependence
extras on the scale-adaptive stream; the conditional scale process
against its history level; and the step. Ninety channels under a slow
learner: 0.6088 alone on fold 2 -- nearly the whole ensemble's worth --
and 0.6358 in the blend against 0.6251, gaining on every range of steps.
Streamed by the library's white module, verified against the training
matrix.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

import joblib
import json
import numpy as np
from scipy.special import gammaln, ndtr, ndtri

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


# --- bocpd.py ----------------------------------------------------


#: Prior hazard of a regime change per step, and the longest run length kept.
BOCPD_HAZARD = 1.0 / 50
BOCPD_RMAX = 600
BOCPD_CHANNELS = 6


def _student_logpdf(x, mu, kappa, alpha, beta):
    nu = 2 * alpha
    scale2 = beta * (kappa + 1) / (alpha * kappa)
    z2 = (x - mu) ** 2 / scale2
    return (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi * scale2)
            - (nu + 1) / 2 * np.log1p(z2 / nu))


class RunLengthMonitor:
    """Streaming BOCPD: one update per online point, six channels out."""

    def __init__(self, history: np.ndarray) -> None:
        hist = np.asarray(history, dtype="float64")
        self.m0 = float(hist.mean())
        v0 = float(hist.var()) + 1e-12
        # The prior behaves as five history points at the history's mean and variance.
        self.k0, self.a0 = 5.0, 2.5
        self.b0 = self.a0 * v0
        self.mu = np.array([self.m0]); self.kappa = np.array([self.k0])
        self.alpha = np.array([self.a0]); self.beta = np.array([self.b0])
        self.logR = np.array([0.0])
        self._t = 0

    def update(self, x: float) -> list[float]:
        lp = _student_logpdf(x, self.mu, self.kappa, self.alpha, self.beta)
        log_growth = self.logR + lp + np.log(1 - BOCPD_HAZARD)
        log_cp = np.logaddexp.reduce(self.logR + lp) + np.log(BOCPD_HAZARD)
        newR = np.concatenate([[log_cp], log_growth])
        evidence = np.logaddexp.reduce(newR)
        newR -= evidence
        mu_new = np.concatenate([[self.m0], (self.kappa * self.mu + x) / (self.kappa + 1)])
        kappa_new = np.concatenate([[self.k0], self.kappa + 1])
        alpha_new = np.concatenate([[self.a0], self.alpha + 0.5])
        beta_new = np.concatenate([[self.b0], self.beta + self.kappa * (x - self.mu) ** 2 / (2 * (self.kappa + 1))])
        if len(newR) > BOCPD_RMAX:
            tail = np.logaddexp.reduce(newR[BOCPD_RMAX:])
            newR = newR[:BOCPD_RMAX + 1].copy(); newR[BOCPD_RMAX] = tail
            mu_new, kappa_new, alpha_new, beta_new = (
                a[:BOCPD_RMAX + 1] for a in (mu_new, kappa_new, alpha_new, beta_new))
        self.logR, self.mu, self.kappa, self.alpha, self.beta = newR, mu_new, kappa_new, alpha_new, beta_new
        P = np.exp(self.logR)
        r = np.arange(len(P))
        out = [float(P[:5].sum()), float(P[:20].sum()), float(P[:60].sum()), float(P[:200].sum()),
               float((P * r).sum() / (self._t + 1)), float(-evidence)]
        self._t += 1
        return out


# --- mass.py -----------------------------------------------------


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


# --- freqdep.py --------------------------------------------------


#: Short and long spectral windows; band counts per window family.
SPECTRUM_SHORT = ((32, 64, 128, 256), 6)
SPECTRUM_LONG = ((64, 128, 256, 512, 1024), 10)
#: Autocorrelation lags and exponential windows (in points).
ACF_LAGS = (1, 2, 5, 10)
ACF_WINDOWS = (50.0, 200.0)
#: Channels emitted: 4*(6+2) + 5*(10+2) + 4*2.
FREQDEP_CHANNELS = 4 * 8 + 5 * 12 + len(ACF_LAGS) * len(ACF_WINDOWS)


def _band_edges(w: int, nb: int) -> np.ndarray:
    k = w // 2
    e = np.unique(np.round(np.geomspace(1, k, nb + 1)).astype(int))
    while len(e) < nb + 1:
        e = np.append(e, e[-1] + 1)
    return e[: nb + 1]


def _spec_feats(seg: np.ndarray, edges: np.ndarray, nb: int) -> np.ndarray:
    w = len(seg)
    sp = np.abs(np.fft.rfft((seg - seg.mean()) * np.hanning(w)))[1:] ** 2
    p = sp / (sp.sum() + 1e-12)
    k = len(p)
    bands = np.array([p[min(edges[i] - 1, k): min(edges[i + 1] - 1, k)].sum() for i in range(nb)])
    ent = -(p * np.log(p + 1e-12)).sum()
    cen = (p * (np.arange(1, k + 1) / w)).sum()
    return np.concatenate([np.log(bands + 1e-6), [ent, cen]])


class MultiSpectrum:
    """Rolling spectra on several windows, each against the history's band profile.

    ``short`` reproduces the 32-256 builder (history profile from
    non-overlapping windows); the long family uses half-window overlap with the
    window capped at the history's length and zero-pads the tail, because
    fifty-one histories are shorter than 1024 points.
    """

    def __init__(self, history: np.ndarray, windows, nb: int, short: bool) -> None:
        h = np.asarray(history, dtype="float64")
        self.mu, self.sd = float(h.mean()), float(h.std()) + 1e-12
        zh = (h - self.mu) / self.sd
        self.windows, self.nb = tuple(windows), nb
        self.edges = {w: _band_edges(w, nb) for w in self.windows}
        self.ref = {}
        for w in self.windows:
            if short:
                segs = [_spec_feats(zh[i: i + w], self.edges[w], nb) for i in range(0, len(zh) - w + 1, w)]
            else:
                we = min(w, len(zh)); step = max(we // 2, 1)
                segs = [_spec_feats(zh[i: i + we], self.edges[w], nb) for i in range(0, len(zh) - we + 1, step)]
                if len(segs) < 2:
                    segs = segs + segs
            S = np.array(segs)
            self.ref[w] = (S.mean(0), S.std(0) + 1e-3)
        m = max(self.windows)
        tail = zh[-m:]
        if not short and len(tail) < m:
            tail = np.concatenate([np.zeros(m - len(tail)), tail])
        self.buf = list(tail)
        self.cap = m

    def update(self, x: float) -> list[float]:
        self.buf.append((float(x) - self.mu) / self.sd)
        if len(self.buf) > self.cap:
            self.buf.pop(0)
        arr = np.asarray(self.buf)
        out = []
        for w in self.windows:
            f = _spec_feats(arr[-w:], self.edges[w], self.nb)
            mref, sref = self.ref[w]
            out.extend(np.clip((f - mref) / sref, -20, 20).tolist())
        return out


class RollingACF:
    """Exponentially-weighted autocorrelations at fixed lags, minus the history's."""

    def __init__(self, history: np.ndarray) -> None:
        h = np.asarray(history, dtype="float64")
        self.mu, self.sd = float(h.mean()), float(h.std()) + 1e-12
        zh = (h - self.mu) / self.sd
        self.hist_acf = {k: float(np.corrcoef(zh[:-k], zh[k:])[0, 1]) for k in ACF_LAGS}
        self.buf = list(zh[-max(ACF_LAGS):])
        self.num = {(k, w): 0.0 for k in ACF_LAGS for w in ACF_WINDOWS}
        self.den = {w: 1.0 for w in ACF_WINDOWS}

    def update(self, x: float) -> list[float]:
        z = (float(x) - self.mu) / self.sd
        for w in ACF_WINDOWS:
            a = 1.0 / w
            self.den[w] = (1 - a) * self.den[w] + a * z * z
        out = [0.0] * (len(ACF_LAGS) * len(ACF_WINDOWS))
        for j, k in enumerate(ACF_LAGS):
            zk = self.buf[-k] if len(self.buf) >= k else 0.0
            for i, w in enumerate(ACF_WINDOWS):
                a = 1.0 / w
                self.num[(k, w)] = (1 - a) * self.num[(k, w)] + a * z * zk
                r = self.num[(k, w)] / max(self.den[w], 1e-9)
                out[i * len(ACF_LAGS) + j] = float(np.clip(r, -1.5, 1.5) - self.hist_acf[k])
        self.buf.append(z)
        if len(self.buf) > max(ACF_LAGS) + 1:
            self.buf.pop(0)
        return out


class FreqDep:
    """The hundred channels of the independent member, in the training matrices' order."""

    def __init__(self, history: np.ndarray) -> None:
        self.short = MultiSpectrum(history, *SPECTRUM_SHORT, short=True)
        self.long = MultiSpectrum(history, *SPECTRUM_LONG, short=False)
        self.acf = RollingACF(history)

    def update(self, x: float) -> list[float]:
        return self.short.update(x) + self.long.update(x) + self.acf.update(x)


# --- novelty.py --------------------------------------------------


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


# --- depcusum.py -------------------------------------------------


DEP_ORDER = 5
DEP_LAGS = (1, 2, 3, 5, 10)
DEP_ABS_LAGS = (1, 2, 5)
DEP_DRIFT = 0.25
DEP_PORT = 10
DEP_CHANNELS = len(DEP_LAGS) * 3 + 2 + len(DEP_ABS_LAGS) * 2


class _Cusum2:
    """Page's two-sided CUSUM with drift; reports the larger arm."""

    __slots__ = ("up", "down")

    def __init__(self) -> None:
        self.up = self.down = 0.0

    def update(self, a: float) -> float:
        self.up = max(0.0, self.up + a - DEP_DRIFT)
        self.down = max(0.0, self.down - a - DEP_DRIFT)
        return max(self.up, self.down)


class DepCusum:
    """Streaming dependence-score CUSUM: one update per online point."""

    def __init__(self, history: np.ndarray) -> None:
        h = np.asarray(history, dtype="float64")
        self.mu, self.sd = float(h.mean()), float(h.std()) + 1e-12
        zh = (h - self.mu) / self.sd
        P = DEP_ORDER
        X = np.column_stack([zh[P - k - 1: len(zh) - k - 1] for k in range(P)])
        self.coef = np.linalg.lstsq(X, zh[P:], rcond=None)[0]
        e = zh[P:] - sum(self.coef[k] * zh[P - k - 1: len(zh) - k - 1] for k in range(P))
        self.sigma = float(e.std()) + 1e-9
        u = e / self.sigma
        au = np.abs(u)
        # Null moments of the lagged products, from the history's own residuals.
        self.m, self.s = {}, {}
        for k in range(1, DEP_PORT + 1):
            q = u[k:] * u[:-k]
            self.m[k], self.s[k] = float(q.mean()), float(q.std()) + 1e-9
        self.am, self.as_ = {}, {}
        for k in DEP_ABS_LAGS:
            r = au[k:] * au[:-k]
            self.am[k], self.as_[k] = float(r.mean()), float(r.std()) + 1e-9
        self.z_tail = list(zh[-P:])                 # last P standardised values
        self.u_tail = list(u[-DEP_PORT:])           # last ten standardised residuals
        self.cusum = {k: _Cusum2() for k in DEP_LAGS}
        self.e_fast = {k: 0.0 for k in DEP_LAGS}    # alpha 0.02
        self.e_slow = {k: 0.0 for k in DEP_LAGS}    # alpha 0.005
        self.port1 = {k: 0.0 for k in range(1, DEP_PORT + 1)}   # alpha 0.01
        self.port2 = {k: 0.0 for k in range(1, DEP_PORT + 1)}   # alpha 0.003
        self.acusum = {k: _Cusum2() for k in DEP_ABS_LAGS}
        self.a_ew = {k: 0.0 for k in DEP_ABS_LAGS}  # alpha 0.01

    def update(self, x: float) -> list[float]:
        z = (float(x) - self.mu) / self.sd
        e = z - sum(self.coef[k] * self.z_tail[-k - 1] for k in range(DEP_ORDER))
        self.z_tail.append(z)
        del self.z_tail[0]
        u = e / self.sigma
        out: list[float] = []
        tail = self.u_tail
        for k in range(1, DEP_PORT + 1):
            a = (u * tail[-k] - self.m[k]) / self.s[k]
            self.port1[k] += 0.01 * (a - self.port1[k])
            self.port2[k] += 0.003 * (a - self.port2[k])
            if k in self.cusum:
                self.e_fast[k] += 0.02 * (a - self.e_fast[k])
                self.e_slow[k] += 0.005 * (a - self.e_slow[k])
                out.extend((self.cusum[k].update(a), self.e_fast[k], self.e_slow[k]))
        out.append(199.0 * sum(v * v for v in self.port1.values()))
        out.append(666.0 * sum(v * v for v in self.port2.values()))
        au = abs(u)
        for k in DEP_ABS_LAGS:
            a = (au * abs(tail[-k]) - self.am[k]) / self.as_[k]
            self.a_ew[k] += 0.01 * (a - self.a_ew[k])
            out.extend((self.acusum[k].update(a), self.a_ew[k]))
        tail.append(u)
        del tail[0]
        return [float(np.clip(np.nan_to_num(v), -50, 50)) for v in out]


# --- white.py ----------------------------------------------------


WHITE_PMAX = 12
WHITE_LAMBDAS = (0.90, 0.94, 0.97, 0.99, 1.0)
WHITE_DRIFT = 0.25
WHITE_LAGS = (1, 2, 3, 5, 10)
WHITE_PORT = 10
WHITE_DYADIC = (8, 16, 32, 64, 128, 256, 512, 1024)
WHITE_TESTW = (32, 64, 128, 256, 512)
WHITE_FULL = 8 + len(WHITE_LAGS) * 2 + 1 + 2 + 7 + 3 * len(WHITE_DYADIC) + 3 + 3 * (len(WHITE_TESTW) + 1) + 3
WHITE_CHANNELS = WHITE_FULL + 10 + 3 + 1
_RING = max(WHITE_DYADIC) + 1


def _scores(u: np.ndarray, sorted_ref: np.ndarray) -> np.ndarray:
    return ndtri(np.clip((np.searchsorted(sorted_ref, u) + 0.5) / (len(sorted_ref) + 1), 1e-6, 1 - 1e-6))


def _one_sample_tests(w: np.ndarray):
    """KS, Cramer-von Mises and Anderson-Darling of a window against N(0,1)."""
    x = np.sort(w)
    m = len(x)
    F = ndtr(x)
    i = np.arange(1, m + 1)
    ks = np.sqrt(m) * max((i / m - F).max(), (F - (i - 1) / m).max())
    cvm = 1.0 / (12 * m) + ((F - (2 * i - 1) / (2 * m)) ** 2).sum()
    Fc = np.clip(F, 1e-10, 1 - 1e-10)
    ad = -m - ((2 * i - 1) * (np.log(Fc) + np.log(1 - Fc[::-1]))).sum() / m
    return ks, cvm, ad


class _Cusum2:
    __slots__ = ("up", "down")

    def __init__(self) -> None:
        self.up = self.down = 0.0

    def update(self, a: float) -> float:
        self.up = max(0.0, self.up + a - WHITE_DRIFT)
        self.down = max(0.0, self.down - a - WHITE_DRIFT)
        return max(self.up, self.down)


class _Ewma:
    __slots__ = ("alpha", "v")

    def __init__(self, alpha: float) -> None:
        self.alpha, self.v = alpha, 0.0

    def update(self, a: float) -> float:
        self.v += self.alpha * (a - self.v)
        return self.v


class _Ring:
    """The last _RING values of the extended stream (history scores, then online)."""

    def __init__(self, tail: np.ndarray) -> None:
        self.buf = np.zeros(_RING)
        self.n = 0
        for v in tail[-_RING:]:
            self.push(float(v))

    def push(self, v: float) -> None:
        if self.n < _RING:
            self.buf[self.n] = v
            self.n += 1
        else:
            self.buf[:-1] = self.buf[1:]
            self.buf[-1] = v

    def last(self, m: int) -> np.ndarray:
        return self.buf[max(self.n - m, 0): self.n]

    def before(self, m: int) -> float | None:
        j = self.n - m - 1
        return float(self.buf[j]) if j >= 0 else None


def _glr(ring: _Ring, scale_stats: bool):
    """Mean, (scale,) lag-1 dependence GLR over the dyadic windows, per scale and maximised."""
    means, scales, deps = [], [], []
    for m in WHITE_DYADIC:
        seg = ring.last(m)
        me = len(seg)
        s1 = seg.sum()
        s2 = (seg ** 2).sum()
        sx = float((seg[1:] * seg[:-1]).sum())
        prev = ring.before(me)
        if prev is not None:
            sx += seg[0] * prev
        var = max(s2 / me, 1e-6)
        r1 = float(np.clip(sx / max(s2, 1e-9), -0.99, 0.99))
        means.append(abs(s1) / np.sqrt(me))
        scales.append(0.5 * me * (var - 1 - np.log(var)))
        deps.append(-0.5 * me * np.log(1 - r1 ** 2))
    if scale_stats:
        per = [v for trio in zip(means, scales, deps) for v in trio]
        return per + [max(means), max(scales), max(deps)]
    return [max(means), max(deps)]


class _FullBattery:
    """The seventy-six channels on one normal-score stream."""

    def __init__(self, nh: np.ndarray) -> None:
        self.ring = _Ring(nh)
        self.hist_tail = list(nh[-WHITE_PORT:])
        self.prev = float(nh[-1])
        ah = np.abs(nh)
        prod = ah[1:] * ah[:-1]
        self.vol_m, self.vol_s = float(prod.mean()), float(prod.std()) + 1e-9
        self.p_hi, self.p_mid, self.p_lo = float((ah > 2.5).mean()), float((ah > 1.5).mean()), float((ah < 0.3).mean())
        self.p_sign = float((np.sign(nh[1:]) != np.sign(nh[:-1])).mean())
        self.abs_mean = float(ah.mean())
        self.c_mean, self.c_scale, self.c_vol = _Cusum2(), _Cusum2(), _Cusum2()
        self.e_mean = (_Ewma(0.02), _Ewma(0.005))
        self.e_scale = (_Ewma(0.02), _Ewma(0.005))
        self.c_lag = {k: _Cusum2() for k in WHITE_LAGS}
        self.port = {k: _Ewma(0.01) for k in range(1, WHITE_PORT + 1)}
        self.e_vol = _Ewma(0.01)
        self.e_shape = [_Ewma(0.01) for _ in range(7)]
        self.sum1 = self.sum2 = 0.0
        self.lagbuf = list(nh[-WHITE_PORT:])     # the last ten scores, history then online
        self.prefix: list[float] = []
        self.next_scan = 1
        self.tests = np.zeros(3 * (len(WHITE_TESTW) + 1) + 3)

    def update(self, n: float, t: int) -> list[float]:
        out = []
        self.sum1 += n
        self.sum2 += n * n
        out.append(self.c_mean.update(n))
        out.extend(e.update(n) for e in self.e_mean)
        out.append(self.sum1 / np.sqrt(t + 1))
        q = (n * n - 1) / np.sqrt(2.0)
        out.append(self.c_scale.update(q))
        out.extend(e.update(q) for e in self.e_scale)
        out.append((self.sum2 / (t + 1) - 1) * np.sqrt((t + 1) / 2.0))
        lb = self.lagbuf
        port_sq = 0.0
        for k in range(1, WHITE_PORT + 1):
            a = n * lb[-k]
            pv = self.port[k].update(a)
            port_sq += pv * pv
            if k in self.c_lag:
                out.append(self.c_lag[k].update(a))
                out.append(pv)
        out.append(199.0 * port_sq)
        a = (abs(n) * abs(lb[-1]) - self.vol_m) / self.vol_s
        out.append(self.c_vol.update(a))
        out.append(self.e_vol.update(a))
        an = abs(n)
        shape_in = (n ** 3, n ** 4 - 3, float(an > 2.5) - self.p_hi, float(an > 1.5) - self.p_mid,
                    float(an < 0.3) - self.p_lo, float(np.sign(n) != np.sign(lb[-1])) - self.p_sign, an - self.abs_mean)
        out.extend(e.update(v) for e, v in zip(self.e_shape, shape_in))
        lb.append(n)
        del lb[0]
        self.ring.push(n)
        out.extend(_glr(self.ring, scale_stats=True))
        self.prefix.append(n)
        if t + 1 >= self.next_scan:
            self.next_scan = max(self.next_scan + 1, int(self.next_scan * 1.12))
            vals = []
            for W in WHITE_TESTW:
                vals.extend(_one_sample_tests(self.ring.last(W)))
            vals.extend(_one_sample_tests(np.asarray(self.prefix)) if t + 1 >= 8 else (0.0, 0.0, 0.0))
            v = np.array(vals)
            self.tests = np.concatenate([v, v.reshape(-1, 3).max(0)])
        out.extend(self.tests.tolist())
        return out


class _CondExtras:
    """Mean and dependence extras on the conditional stream: ten channels."""

    def __init__(self, nh: np.ndarray) -> None:
        self.ring = _Ring(nh)
        self.c_mean, self.e_mean = _Cusum2(), _Ewma(0.02)
        self.c_lag = {k: _Cusum2() for k in (1, 2, 5)}
        self.port = {k: _Ewma(0.01) for k in range(1, WHITE_PORT + 1)}
        self.lagbuf = list(nh[-WHITE_PORT:])
        self.next_scan = 1
        self.tests = np.zeros(2)

    def update(self, n: float, t: int) -> list[float]:
        out = [self.c_mean.update(n), self.e_mean.update(n)]
        lb = self.lagbuf
        port_sq = 0.0
        for k in range(1, WHITE_PORT + 1):
            a = n * lb[-k]
            pv = self.port[k].update(a)
            port_sq += pv * pv
            if k in self.c_lag:
                out.append(self.c_lag[k].update(a))
        out.append(199.0 * port_sq)
        lb.append(n)
        del lb[0]
        self.ring.push(n)
        out.extend(_glr(self.ring, scale_stats=False))
        if t + 1 >= self.next_scan:
            self.next_scan = max(self.next_scan + 1, int(self.next_scan * 1.12))
            ks, ad = [], []
            for W in WHITE_TESTW:
                k_, _, a_ = _one_sample_tests(self.ring.last(W))
                ks.append(k_)
                ad.append(a_)
            self.tests = np.array([max(ks), max(ad)])
        out.extend(self.tests.tolist())
        return out


class WhiteMonitor:
    """The whitened-stream battery, streamed one online point at a time."""

    def __init__(self, history: np.ndarray) -> None:
        h = np.asarray(history, dtype="float64")
        self.mu, self.sd = float(h.mean()), float(h.std()) + 1e-12
        zh = (h - self.mu) / self.sd
        n = len(zh)
        P = WHITE_PMAX
        y = zh[P:]
        best = (np.inf, 0, np.zeros(0))
        for p in range(0, P + 1):
            if p == 0:
                e, coef = y, np.zeros(0)
            else:
                X = np.column_stack([zh[P - j - 1: n - j - 1] for j in range(p)])
                coef = np.linalg.lstsq(X, y, rcond=None)[0]
                e = y - X @ coef
            bic = len(e) * np.log((e ** 2).mean() + 1e-12) + p * np.log(len(e))
            if bic < best[0]:
                best = (bic, p, coef)
        p, self.coef = best[1], best[2]
        self.p = p
        e = zh[p:] - (np.column_stack([zh[p - j - 1: n - j - 1] for j in range(p)]) @ self.coef if p else 0.0)
        e2 = e ** 2
        v0 = max(float(e2.mean()), 1e-12)
        bestl = (-np.inf, 1.0, np.full(len(e), v0))
        for lam in WHITE_LAMBDAS:
            if lam >= 1.0:
                s2 = np.full(len(e), v0)
            else:
                s2 = np.empty(len(e))
                s2[0] = v0
                for i in range(1, len(e)):
                    s2[i] = lam * s2[i - 1] + (1 - lam) * e2[i - 1]
            s2 = np.maximum(s2, 1e-12)
            ql = -0.5 * float(np.sum(np.log(s2) + e2 / s2))
            if np.isfinite(ql) and ql > bestl[0]:
                bestl = (ql, lam, s2)
        self.lam, s2 = bestl[1], bestl[2]
        self.v0 = v0
        self.s2 = float(self.lam * s2[-1] + (1 - self.lam) * e2[-1]) if self.lam < 1.0 else v0
        uc, uu = e / np.sqrt(s2), e / np.sqrt(v0)
        self.suc, self.suu = np.sort(uc), np.sort(uu)
        l = np.log(s2 / v0)
        self.lm, self.ls = float(l.mean()), float(l.std()) + 1e-6
        self.full = _FullBattery(_scores(uu, self.suu))
        self.cond = _CondExtras(_scores(uc, self.suc))
        self.z_tail = list(zh[-P:])
        self.e_lz, self.c_lz = _Ewma(0.02), _Cusum2()
        self._t = 0

    def update(self, x: float) -> list[float]:
        z = (float(x) - self.mu) / self.sd
        e = z - sum(self.coef[j] * self.z_tail[-j - 1] for j in range(self.p))
        self.z_tail.append(z)
        del self.z_tail[0]
        s2 = max(self.s2, 1e-12)
        n_c = float(np.clip(_scores(np.array([e / np.sqrt(s2)]), self.suc)[0], -4.5, 4.5))
        n_u = float(np.clip(_scores(np.array([e / np.sqrt(self.v0)]), self.suu)[0], -4.5, 4.5))
        lz = (np.log(s2 / self.v0) - self.lm) / self.ls
        if self.lam < 1.0:
            self.s2 = self.lam * s2 + (1 - self.lam) * e * e
        t = self._t
        self._t += 1
        out = self.full.update(n_u, t) + self.cond.update(n_c, t)
        out.extend((lz, self.e_lz.update(lz), self.c_lz.update(lz)))
        return [float(np.clip(np.nan_to_num(v), -60, 60)) for v in out] + [float(t)]


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


class DualMonitor:
    """Two full pipelines over two compressions of the same stream.

    The second sees asinh(x): log-like in the tails, linear near zero, defined
    on negatives. It is not a derived channel -- its normalisation, trend,
    thresholds and detectors are all fitted on the compressed series, so where
    heavy tails fool the raw view the two disagree, and the disagreement is
    what the combiner feeds on. Adopted on grouped 5-fold CV: 0.5746 against
    0.5719 for the raw-only fifty, ahead on three folds with no meaningful
    loss on the other two.
    """

    def __init__(self, history: np.ndarray) -> None:
        history = np.asarray(history, dtype="float64")
        self.raw = Monitor(history)
        self.compressed = Monitor(np.arcsinh(history))

    def update(self, x: float) -> list[float]:
        return self.raw.update(float(x)) + self.compressed.update(float(np.arcsinh(x)))


FORECAST_K = 10       # trailing-mean window
FORECAST_LAGS = 8
FORECAST_CHANNELS = 4


def _deviation_features(d: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Lag rows for the forecaster: eight trailing deviations plus the mean
    and spread of the last five. Purely backward-looking."""
    lags = np.stack([d[idx - j] for j in range(1, FORECAST_LAGS + 1)], axis=1)
    roll5 = np.stack([d[idx - j] for j in range(1, 6)], axis=1)
    return np.hstack([lags, roll5.mean(1, keepdims=True), roll5.std(1, keepdims=True)])


class Forecaster:
    """Is the next value still predictable the way the history was?

    A small LightGBM regressor predicts each next deviation-from-trailing-mean
    of the standardised stream: pretrained on every training history, then
    finetuned here on this series' own break-free history, where its residual
    scale sigma is measured. The reported channels are the standardised
    prediction error smoothed fast and slow, its running peak, and the raw
    step error. The EWMAs rise while the series stops being forecastable and
    drain once it is again -- an organic, reversible alarm.
    """

    def __init__(self, base, norm: Normalisation, history: np.ndarray) -> None:
        import lightgbm as lgb

        n = len(history)
        z_hist = [norm.clip(norm.standardise(float(v), i - n)) for i, v in enumerate(history)]
        self._z = list(z_hist)          # full standardised past, history then online
        self._steps_done = 0
        d = self._deviations(np.asarray(self._z))
        idx = np.arange(FORECAST_LAGS + FORECAST_K, n)
        self.model, self.sigma = base, 1.0
        if len(idx) > 40:
            feats = _deviation_features(d, idx).astype("float32")
            targets = d[idx].astype("float32")
            self.model = lgb.train(
                dict(objective="l2", learning_rate=0.03, num_leaves=7,
                     min_data_in_leaf=20, verbose=-1, deterministic=True,
                     force_row_wise=True, num_threads=1, seed=0),
                lgb.Dataset(feats, targets), num_boost_round=30, init_model=base)
            resid = targets - self.model.predict(feats, num_threads=1)
            self.sigma = float(np.std(resid)) + 1e-6
        self.fast, self.slow, self.peak = 1.0, 1.0, 0.0

    @staticmethod
    def _deviations(z: np.ndarray) -> np.ndarray:
        csum = np.concatenate([[0.0], np.cumsum(z)])
        out = np.full(len(z), np.nan)
        for i in range(FORECAST_K, len(z)):
            out[i] = z[i] - (csum[i] - csum[i - FORECAST_K]) / FORECAST_K
        return out

    def update(self, z: float) -> list[float]:
        self._z.append(float(z))
        arr = np.asarray(self._z)
        d = self._deviations(arr[-(FORECAST_K + FORECAST_LAGS + 2):])
        i = len(d) - 1
        error = 0.0
        row = _deviation_features(d, np.asarray([i]))
        if np.isfinite(row).all() and np.isfinite(d[i]):
            pred = float(self.model.predict(row, num_threads=1)[0])
            error = abs(float(d[i]) - pred) / self.sigma
        e = min(error, 8.0)
        self.fast += 0.10 * (e - self.fast)
        self.slow += 0.02 * (e - self.slow)
        self.peak = max(self.peak, self.fast)
        return [self.fast, self.slow, self.peak, e]


BATTERY_QS = (0.05, 0.25, 0.50, 0.75, 0.95)


def _acf(v: np.ndarray, lag: int) -> float:
    if len(v) <= lag + 2:
        return 0.0
    a, b = v[:-lag], v[lag:]
    sa, sb = a.std(), b.std()
    if sa < 1e-9 or sb < 1e-9:
        return 0.0
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))


def _hist_summary(z: np.ndarray):
    return (np.sort(z), np.quantile(z, BATTERY_QS), float(z.mean()), float(z.std()),
            float(np.median(np.abs(z - np.median(z)))),
            _acf(z, 1), _acf(z, 2), _acf(z, 5), float(np.abs(z).mean()))


def _two_sample(summary, prefix: np.ndarray) -> list:
    """Drift-free two-sample statistics: the history against the prefix so
    far. No length multipliers anywhere -- under the null every statistic is
    centred regardless of how long the prefix has grown, so length is never
    read as evidence."""
    from scipy.special import erfinv

    hist_sorted, hq, hmean, hstd, hmad, hacf1, hacf2, hacf5, habs = summary
    p = prefix
    n = len(p)
    out = []
    out.extend((np.quantile(p, BATTERY_QS) - hq).tolist())
    out.append(float(p.mean() - hmean))
    out.append(float(np.log((p.std() + 1e-9) / (hstd + 1e-9))))
    c = p - p.mean()
    s2 = float((c ** 2).mean()) + 1e-12
    out.append(float((c ** 3).mean()) / s2 ** 1.5)
    out.append(float((c ** 4).mean()) / s2 ** 2 - 3.0)
    pos = np.searchsorted(hist_sorted, np.sort(p), side="right") / len(hist_sorted)
    out.append(float(np.max(np.abs(pos - (np.arange(1, n + 1) / n)))))
    pmad = float(np.median(np.abs(p - np.median(p)))) + 1e-9
    out.append(float(np.log(pmad / (hmad + 1e-9))))
    out.append(_acf(p, 1) - hacf1)
    out.append(_acf(p, 2) - hacf2)
    out.append(_acf(p, 5) - hacf5)
    out.append(float(np.abs(p).mean() - habs))
    u = (np.searchsorted(hist_sorted, p, side="left")
         + np.searchsorted(hist_sorted, p, side="right")) / 2.0
    u = np.clip((u + 0.5) / (len(hist_sorted) + 1.0), 1e-6, 1 - 1e-6)
    g = np.sqrt(2) * erfinv(2 * u - 1)
    out.append(float(g.mean()))
    out.append(float(np.log(g.std() + 1e-9)))
    tail = p[-max(n // 4, 5):]
    out.append(float(tail.mean() - hmean))
    out.append(float(np.log((tail.std() + 1e-9) / (hstd + 1e-9))))
    half = n // 2
    if half >= 3:
        out.append(float(p[half:].mean() - p[:half].mean()))
        out.append(float(np.log((p[half:].std() + 1e-9) / (p[:half].std() + 1e-9))))
    else:
        out.extend([0.0, 0.0])
    return out


def _acf1(v: np.ndarray) -> float:
    if len(v) <= 3:
        return 0.0
    a, b = v[:-1], v[1:]
    sa, sb = a.std(), b.std()
    if sa < 1e-9 or sb < 1e-9:
        return 0.0
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))


def _slope_t(v: np.ndarray) -> float:
    n = len(v)
    if n < 8:
        return 0.0
    t_ax = np.arange(n) - (n - 1) / 2.0
    denom = float((t_ax ** 2).sum())
    beta = float((t_ax * (v - v.mean())).sum()) / denom
    resid = v - v.mean() - beta * t_ax
    se = np.sqrt(float((resid ** 2).sum()) / max(n - 2, 1) / denom) + 1e-12
    return float(np.clip(beta / se, -12, 12))


def _hist_summary2(z: np.ndarray) -> dict:
    d = np.diff(z)
    dev = np.abs(z - np.median(z))
    return dict(
        sorted=np.sort(z), q10=float(np.quantile(z, 0.10)), q90=float(np.quantile(z, 0.90)),
        q05=float(np.quantile(z, 0.05)), q95=float(np.quantile(z, 0.95)),
        mean=float(z.mean()), std=float(z.std()),
        lev=float(dev.mean()), d_std=float(d.std()) if len(d) > 2 else 1.0,
        d_acf=_acf1(d), d_abs=float(np.abs(d).mean()) if len(d) else 0.0,
        sign=float((z > 0).mean()),
    )


def _two_sample2(h: dict, p: np.ndarray) -> list:
    n = len(p)
    out = []
    out.append(float(np.quantile(p, 0.10) - h["q10"]))
    out.append(float(np.quantile(p, 0.90) - h["q90"]))
    ps = np.sort(p)
    pos = np.searchsorted(h["sorted"], ps, side="right") / len(h["sorted"])
    ecdf = np.arange(1, n + 1) / n
    gap = pos - ecdf
    out.append(float(np.mean(gap ** 2)))
    w = np.clip(pos * (1 - pos), 1e-3, None)
    out.append(float(np.clip(np.mean(gap ** 2 / w), 0, 10)))
    dev = np.abs(p - np.median(p))
    out.append(float(np.log((dev.mean() + 1e-9) / (h["lev"] + 1e-9))))
    d = np.diff(p)
    if len(d) > 2:
        out.append(float(np.log((d.std() + 1e-9) / (h["d_std"] + 1e-9))))
        out.append(_acf1(d) - h["d_acf"])
        out.append(float(np.log((np.abs(d).mean() + 1e-9) / (h["d_abs"] + 1e-9))))
    else:
        out.extend([0.0, 0.0, 0.0])
    for wlen in (10, 25, 100):
        tail = p[-wlen:]
        out.append(float(tail.mean() - h["mean"]))
        out.append(float(np.log((tail.std() + 1e-9) / (h["std"] + 1e-9))))
    out.append(float((p > h["q95"]).mean() - 0.05))
    out.append(float((p < h["q05"]).mean() - 0.05))
    out.append(float((p > 0).mean() - h["sign"]))
    out.append(_acf1(np.sign(p)))
    out.append(_slope_t(p))
    out.append(_slope_t(np.abs(p)))
    return out


SPEC_NFFT = 64
SPEC_BANDS = 8


def _spectrum(v: np.ndarray):
    """Power in eight bands (normalised), spectral entropy, peak position,
    log total power — of the last NFFT points, Hann-windowed."""
    if len(v) < SPEC_NFFT:
        v = np.concatenate([np.zeros(SPEC_NFFT - len(v)), v])
    seg = v[-SPEC_NFFT:] * np.hanning(SPEC_NFFT)
    p = np.abs(np.fft.rfft(seg)) ** 2
    p = p[1:]
    tot = p.sum() + 1e-12
    pn = p / tot
    band = np.add.reduceat(pn, np.linspace(0, len(pn), SPEC_BANDS + 1)[:-1].astype(int))
    ent = float(-(pn * np.log(pn + 1e-12)).sum() / np.log(len(pn)))
    peak = float(np.argmax(pn)) / len(pn)
    return band, ent, peak, float(np.log(tot))


class Spectral:
    """The frequency-domain family: the prefix's spectrum against the
    history's own spectral profile.

    Every other channel this project owns lives in the time domain, and a
    break that rearranges *periodicity* -- the same variance redistributed
    across frequencies -- passes them unseen. The history supplies both the
    mean profile and its spread, so each band's deviation is read in that
    series' own sigmas. Recomputed on the geometric cadence, held between."""

    def __init__(self, history: np.ndarray, norm: "Normalisation") -> None:
        n = len(history)
        z = np.asarray([norm.clip(norm.standardise(float(v), i - n))
                        for i, v in enumerate(history)])
        bands, ents, peaks, tots = [], [], [], []
        stride = max(SPEC_NFFT // 2, 1)
        for i in range(SPEC_NFFT, len(z) + 1, stride):
            b, e, pk, lt = _spectrum(z[i - SPEC_NFFT:i])
            bands.append(b); ents.append(e); peaks.append(pk); tots.append(lt)
        if bands:
            B = np.stack(bands)
            self.hb, self.he = B.mean(0), float(np.mean(ents))
            self.hp, self.ht = float(np.mean(peaks)), float(np.mean(tots))
            self.hb_sd = B.std(0) + 1e-3
            self.he_sd = float(np.std(ents)) + 1e-3
        else:
            b, e, pk, lt = _spectrum(z)
            self.hb, self.he, self.hp, self.ht = b, e, pk, lt
            self.hb_sd, self.he_sd = np.ones(SPEC_BANDS) * 0.1, 0.1
        self.norm = norm
        self.buf = list(z[-SPEC_NFFT:])
        self._next_scan = 1
        self._step = 0
        self.current = [0.0] * 14

    def update(self, x: float) -> list:
        z = self.norm.clip(self.norm.standardise(float(x), self._step))
        self.buf.append(float(z))
        if len(self.buf) > SPEC_NFFT:
            self.buf.pop(0)
        self._step += 1
        if self._step >= self._next_scan:
            self._next_scan = max(self._next_scan + 1, int(self._next_scan * 1.12))
            b, e, pk, lt = _spectrum(np.asarray(self.buf))
            diff = (b - self.hb) / self.hb_sd
            self.current = list(diff) + [
                (e - self.he) / self.he_sd,
                pk - self.hp,
                lt - self.ht,
                float(np.abs(diff).max()),
                float(np.abs(b - self.hb).sum()),
                float(np.dot(b, self.hb) / (np.linalg.norm(b) * np.linalg.norm(self.hb) + 1e-12)),
            ]
        return list(self.current)


class PrefixBattery:
    """The per-prefix battery: 21 statistics per view, raw and asinh, held
    between geometric-cadence recomputes exactly like the retrospective
    scans. The prefix is standardised by the view's own history fit, unclipped
    -- the battery wants to see the tails the detectors are protected from."""

    def __init__(self, history: np.ndarray) -> None:
        history = np.asarray(history, dtype="float64")
        self.views = []
        for transform in (None, np.arcsinh):
            h = history if transform is None else transform(history)
            norm = Normalisation.fit(h)
            n = len(h)
            z_h = np.asarray([norm.standardise(float(v), i - n) for i, v in enumerate(h)])
            self.views.append([norm, _hist_summary(z_h), _hist_summary2(z_h), []])
        self._next_scan = 1
        self._step = 0
        self.current = [0.0] * 82

    def update(self, x: float) -> list:
        for k, view in enumerate(self.views):
            value = float(x) if k == 0 else float(np.arcsinh(x))
            view[3].append(view[0].standardise(value, self._step))
        self._step += 1
        if self._step >= self._next_scan:
            self._next_scan = max(self._next_scan + 1, int(self._next_scan * 1.12))
            v1, v2 = [], []
            for _norm, summary, summary2, prefix in self.views:
                arr = np.asarray(prefix)
                v1.extend(_two_sample(summary, arr))
                v2.extend(_two_sample2(summary2, arr))
            # Column order matches the training matrices: all of v1 (B40),
            # then all of v2 (B2).
            self.current = v1 + v2
        return list(self.current)


class TriMonitor:
    """The dual pipelines of 011 plus the forecaster of 013."""

    def __init__(self, history: np.ndarray, base_forecaster) -> None:
        history = np.asarray(history, dtype="float64")
        self.dual = DualMonitor(history)
        self.forecast = Forecaster(base_forecaster, self.dual.raw.norm, history)
        self.battery = PrefixBattery(history)
        self.spectral = Spectral(history, self.dual.raw.norm)
        # The run-length posterior rides as a six-channel suffix that only the
        # classifier reads (experiment 085: +0.0046 there, nothing elsewhere).
        self.regime = RunLengthMonitor(history)
        # The mass battery is a separate member's input, not a channel of this
        # monitor: appended to the two hundred it loses, trained on its own it
        # disagrees with the ensemble where that pays (114).
        self.mass = MassBattery(history)
        # The second independent member's input: frequency and dependence
        # channels the ensemble does not otherwise read (124-131b).
        self.freqdep = FreqDep(history)
        # The third view: the online window ranked against the history's own
        # windows (140). Read only together with the frequency channels, by
        # the union member, from step UNION_GATE on.
        self.novelty = Novelty(history)
        # The whitened stream (145): AR(p) by BIC, a conditional scale and the
        # innovation ECDF fitted on the history; every test on normal scores.
        # Alone 0.6088 on fold 2 -- nearly the whole ensemble's worth -- and
        # +0.0107 to the blend at a 0.30 share. Read by its own member.
        self.white = WhiteMonitor(history)
        self._step = 0

    def update(self, x: float) -> list[float]:
        channels = self.dual.update(float(x))
        z = self.dual.raw.norm.clip(
            self.dual.raw.norm.standardise(float(x), self._step))
        self._step += 1
        return (channels + self.forecast.update(z) + self.battery.update(float(x))
                + self.spectral.update(float(x)) + self.regime.update(float(x))
                + self.mass.update(float(x)) + self.freqdep.update(float(x))
                + self.novelty.update(float(x)) + self.white.update(float(x)))


N_CHANNELS = 2 * (9 + 24 + len(RETRO2_CHANNELS) + 1 + 9) + FORECAST_CHANNELS + 82 + 14 + BOCPD_CHANNELS
NET_CHANNELS = 200   # the trajectory networks read the first two hundred
RANK_CHANNELS = 200  # so do the rankers
CLF_CHANNELS = 206   # the classifier alone reads the run-length suffix
MASS_OFFSET = 206    # the mass battery follows, read only by its own member
FREQDEP_OFFSET = 296 # then the frequency/dependence channels, read only by theirs
NOVELTY_OFFSET = 396 # then the novelty channels; the union member reads 296: as one row
UNION_GATE = 100     # before this step the frequency member alone, after it the union
#: The blend from the gate on: the union's share and the mass member's, the
#: core taking the rest (140c: 0.6251 on fold 2 against 0.6232 for #38).
UNION_MASS, UNION_SHARE = 0.25, 0.25
WHITE_OFFSET = 416   # then the whitened-stream channels, read only by their member
WHITE_SHARE = 0.30   # the whitened member's share of the final blend at every step (145: 0.6358 vs 0.6251)


TCN_DILS = (1, 2, 4, 8, 16, 32)


def _gelu(x: np.ndarray) -> np.ndarray:
    from scipy.special import erf

    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


class BatchedStreamingChanTCN:
    """Twelve nets, one matrix multiply per step.

    The member weights are stacked along a leading axis, so a step costs one
    einsum per layer for the whole bag instead of twelve python loops --
    profiled at ~1 ms/step for all members together. Verified against the
    per-member forward to 1e-6."""

    def __init__(self, weight_dicts, mu, sd) -> None:
        self.n = len(weight_dicts)
        self.mu = np.asarray(mu, dtype="float64")
        self.sd = np.asarray(sd, dtype="float64")
        st = lambda key: np.stack([np.asarray(w[key], dtype="float64") for w in weight_dicts])
        self.w_inp = st("inp.weight")[:, :, :, 0]
        self.b_inp = st("inp.bias")
        self.conv_w, self.conv_b, self.mix_w, self.mix_b = [], [], [], []
        for li in range(len(TCN_DILS)):
            self.conv_w.append(st(f"blocks.{li}.conv.weight"))
            self.conv_b.append(st(f"blocks.{li}.conv.bias"))
            self.mix_w.append(st(f"blocks.{li}.mix.weight")[:, :, :, 0])
            self.mix_b.append(st(f"blocks.{li}.mix.bias"))
        self.w_head = st("head.weight")[:, 0, :, 0]
        self.b_head = st("head.bias")[:, 0]
        self.hist = [[] for _ in range(len(TCN_DILS) + 1)]

    def update(self, chan_vec) -> np.ndarray:
        x = (np.asarray(chan_vec, dtype="float64") - self.mu) / self.sd
        h = np.einsum("noi,i->no", self.w_inp, x) + self.b_inp
        self.hist[0].append(h)
        for li, d in enumerate(TCN_DILS):
            layer_in = self.hist[li]
            t_idx = len(layer_in) - 1
            acc = self.conv_b[li].copy()
            W = self.conv_w[li]
            for j, off in enumerate((2 * d, d, 0)):
                idx = t_idx - off
                if idx >= 0:
                    acc += np.einsum("nock,nc->no", W[:, :, :, j:j+1], layer_in[idx])[:, :]
            z = _gelu(acc)
            out = layer_in[t_idx] + np.einsum("noc,nc->no", self.mix_w[li], z) + self.mix_b[li]
            self.hist[li + 1].append(out)
        top = self.hist[-1][-1]
        return np.einsum("nc,nc->n", self.w_head, top) + self.b_head


class StreamingChanTCN:
    """The channel-trajectory network, streamed one step at a time.

    The boosted combiners see each step's 186 channels as an isolated
    snapshot; this network sees how they *move* -- ramp shapes, fronts,
    agreement across channels -- through six dilated causal convolutions
    (receptive field 127 steps). Trained with the per-step ranking loss; here
    only the numpy forward runs, incrementally: each step costs one new
    position per layer, verified against the torch forward to 2e-7."""

    def __init__(self, weights: dict, mu, sd) -> None:
        self.w = {k: np.asarray(v, dtype="float64") for k, v in weights.items()}
        self.mu = np.asarray(mu, dtype="float64")
        self.sd = np.asarray(sd, dtype="float64")
        self.hist = [[] for _ in range(len(TCN_DILS) + 1)]

    def update(self, chan_vec) -> float:
        x = (np.asarray(chan_vec, dtype="float64") - self.mu) / self.sd
        h = self.w["inp.weight"][:, :, 0] @ x + self.w["inp.bias"]
        self.hist[0].append(h)
        for li, d in enumerate(TCN_DILS):
            layer_in = self.hist[li]
            t_idx = len(layer_in) - 1
            W = self.w[f"blocks.{li}.conv.weight"]
            acc = self.w[f"blocks.{li}.conv.bias"].copy()
            for j, off in enumerate((2 * d, d, 0)):
                idx = t_idx - off
                if idx >= 0:
                    acc += W[:, :, j] @ layer_in[idx]
            z = _gelu(acc)
            out = (layer_in[t_idx]
                   + self.w[f"blocks.{li}.mix.weight"][:, :, 0] @ z
                   + self.w[f"blocks.{li}.mix.bias"])
            self.hist[li + 1].append(out)
        top = self.hist[-1][-1]
        return float(self.w["head.weight"][0, :, 0] @ top + self.w["head.bias"][0])


def train(
    datasets: List[Tuple[int, List[float], List[float], Optional[int]]],
    model_directory_path: str,
) -> None:
    """Full rebuild: pretrain the forecaster on every history, then walk every
    series through the TriMonitor and fit the combiner on all 104 channels."""
    import lightgbm as lgb

    feats_all, targ_all = [], []
    for _sid, x_hist, _x_online, _tau in datasets:
        history = np.asarray(x_hist, dtype="float64")
        norm = Normalisation.fit(history)
        n = len(history)
        z = np.asarray([norm.clip(norm.standardise(float(v), i - n))
                        for i, v in enumerate(history)])
        d = Forecaster._deviations(z)
        idx = np.arange(FORECAST_LAGS + FORECAST_K, n)
        if len(idx) > 8:
            feats_all.append(_deviation_features(d, idx)[::2])
            targ_all.append(d[idx][::2])
    base = lgb.train(
        dict(objective="l2", learning_rate=0.05, num_leaves=15,
             min_data_in_leaf=200, feature_fraction=0.9, bagging_fraction=0.8,
             bagging_freq=1, verbose=-1, deterministic=True,
             force_row_wise=True, num_threads=4, seed=0),
        lgb.Dataset(np.vstack(feats_all).astype("float32"),
                    np.concatenate(targ_all).astype("float32")),
        num_boost_round=200)

    rows: list[list[float]] = []
    targets: list[int] = []
    steps: list[int] = []
    for _sid, x_hist, x_online, tau in datasets:
        online = np.asarray(x_online, dtype="float64")
        if len(online) == 0:
            continue
        monitor = TriMonitor(np.asarray(x_hist, dtype="float64"), base)
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
        num_leaves=63,
        min_child_samples=500,
        subsample=0.8,
        subsample_freq=1,
        # Half the columns per tree: with 186 channels the strongest
        # decorrelator the resweep found -- each tree sees a different half
        # of the ensemble's eyes.
        colsample_bytree=0.5,
        reg_lambda=10.0,
        n_estimators=300,
        random_state=0,
        n_jobs=4,
        deterministic=True,
        force_row_wise=True,
        verbose=-1,
    )
    booster.fit(x, y, sample_weight=weights)

    # The ranking half of the blend: the same rows sorted by step, every
    # cross-section a group, trained to order it. Truncation 2000 is the
    # measured optimum (500 cuts the signal, 8000 dilutes the gradient).
    order = np.argsort(step_array, kind="stable")
    _, group_sizes = np.unique(step_array[order], return_counts=True)
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
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
        lambdarank_truncation_level=2000,
        label_gain=[0, 1],
    )
    ranker.fit(x[order], y[order], group=group_sizes)

    joblib.dump(
        {"booster": booster, "rankers": [ranker], "forecaster": base.model_to_string()},
        os.path.join(model_directory_path, "model.joblib"),
    )


def infer(
    datasets: Iterable[Tuple[List[float], Iterable[float]]],
    model_directory_path: str,
):
    import lightgbm as lgb

    model = joblib.load(os.path.join(model_directory_path, "model.joblib"))
    # Raw boosters with a single thread per call: the sklearn wrappers spawn
    # a thread pool on every one-row predict, and five million spawns across
    # eight workers deadlocked run #109121 into a 7-hour timeout (exit 124).
    clf_booster = model["booster"].booster_
    mass_booster = model["mass_classifier"].booster_
    freqdep_booster = model["freqdep_classifier"].booster_
    union_booster = model["union_classifier"].booster_
    white_booster = model["white_classifier"].booster_
    rank_boosters = [r.booster_ for r in model["rankers"]]
    base_forecaster = lgb.Booster(model_str=model["forecaster"])
    net_list = model.get("nets") or []
    net_mu, net_sd = model.get("net_mu"), model.get("net_sd")

    yield  # Signal readiness to the runner.

    for x_historical, x_online in datasets:
        monitor = TriMonitor(np.asarray(x_historical, dtype="float64"), base_forecaster)
        # The triple: ranker 0.36, classifier 0.24, channel-trajectory net
        # 0.40 -- fold-0 0.6068 against 0.6045 for the pair. The net is the
        # first genuinely unlike member: it reads the channels' motion, which
        # per-step trees cannot see.
        # The heavy-member A/B: the exact #19 weights (0.35 bagged rankers +
        # 0.15 classifier + 0.50 nets), with the four fold-nets replaced by
        # eight members trained on 88% of ALL data each, 16 epochs,
        # best-epoch by a private per-member holdout. Any cloud delta
        # against #19's 0.5877 is member quality and nothing else.
        bag_net = BatchedStreamingChanTCN(net_list, net_mu, net_sd) if net_list else None
        for step, point in enumerate(x_online):
            channels = np.asarray(monitor.update(float(point)), dtype="float64")
            clf_row = channels[:CLF_CHANNELS].reshape(1, -1)
            rank_row = channels[:RANK_CHANNELS].reshape(1, -1)
            net_row = channels[:NET_CHANNELS]
            mass_row = channels[MASS_OFFSET:FREQDEP_OFFSET].reshape(1, -1)
            mass_prob = float(mass_booster.predict(mass_row, num_threads=1)[0])
            prob = float(clf_booster.predict(clf_row, num_threads=1)[0])
            bag = sum(1.0 / (1.0 + math.exp(-float(rb.predict(rank_row, num_threads=1)[0])))
                      for rb in rank_boosters) / len(rank_boosters)
            trees = 0.7 * bag + 0.3 * prob
            if bag_net is None:
                yield trees
            else:
                # Twelve members: #28's six and six more under the last-epoch
                # rule (082b). Either six reads 0.616 on fold 2 in this blend;
                # together they read the same with half the member variance.
                logits = bag_net.update(net_row)
                net_sig = float(np.mean(1.0 / (1.0 + np.exp(-logits))))
                # Twenty-four networks here, twelve in #36: fold 2 cannot
                # separate them, the cloud reads this recipe within 0.001, and
                # halved member variance is the only thing a larger pool buys.
                # The ensemble as shipped, then a quarter weight on the mass
                # member: fold 2 rises from 0.6167 to 0.6205 at this share, and
                # the gain grows monotonically from 0.10 to 0.30 (114).
                # Two independent members on top of the core: the mass member
                # at a quarter (#36: +0.0039 in the cloud) and the frequency/
                # dependence member at a fifth (131b: +0.0025 on fold 2 at
                # correlation 0.37). Jointly tuned: fold 2 reads 0.6232 on a
                # flat plateau across 0.20-0.30 / 0.15-0.25.
                core = 0.45 * trees + 0.55 * net_sig
                white_row = channels[WHITE_OFFSET:].reshape(1, -1)
                white_prob = float(white_booster.predict(white_row, num_threads=1)[0])
                if step < UNION_GATE:
                    # #38 as shipped: the novelty windows are unfilled this
                    # early and the union reads noise there (140: -0.005 on
                    # steps 0-100), so the frequency member stands alone.
                    freqdep_row = channels[FREQDEP_OFFSET:NOVELTY_OFFSET].reshape(1, -1)
                    freqdep_prob = float(freqdep_booster.predict(freqdep_row, num_threads=1)[0])
                    blend = 0.55 * core + 0.25 * mass_prob + 0.20 * freqdep_prob
                else:
                    # From the gate on, the frequency channels and the
                    # novelty channels are read together by one classifier:
                    # a member that agrees with the frequency member at 0.21
                    # and lifts fold 2 from 0.6232 to 0.6251 (140c).
                    union_row = channels[FREQDEP_OFFSET:WHITE_OFFSET].reshape(1, -1)
                    union_prob = float(union_booster.predict(union_row, num_threads=1)[0])
                    blend = ((1.0 - UNION_MASS - UNION_SHARE) * core
                             + UNION_MASS * mass_prob + UNION_SHARE * union_prob)
                # #39 as shipped (cloud 0.6056), then the whitened member at
                # 0.30 at every step: it gains on every range of fold 2
                # (+0.011 early, +0.010 on 100-300, +0.013 on 300-700).
                yield (1.0 - WHITE_SHARE) * blend + WHITE_SHARE * white_prob
