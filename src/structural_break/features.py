"""Normalising a series before any detector sees it.

A detector tuned on independent, stationary, unit-variance noise will alarm
constantly on a series that merely drifts, or whose observations are correlated,
or whose scale differs from what the detector expects. None of those is a
structural break, and all three are common. So the historical segment is used
for more than a mean and a standard deviation: it fixes what "ordinary" looks
like for *this* series, along every axis a detector could mistake for a change.

Four normalisations, each removing one way of being fooled.

**Level and scale.** The obvious one. Everything downstream works in standard
errors of the historical segment.

**Trend.** A series with a slope has a running mean that climbs forever, and a
cumulative detector reads that as a permanent shift arriving continuously. The
historical slope is estimated and extended into the online segment, so a series
that keeps drifting exactly as it always did produces no signal. This is the
normalisation with the sharpest trade-off: subtracting a trend that is genuinely
part of the process removes a real break if the break *is* a change in slope,
so the slope residual is kept as its own detector input rather than thrown away.

**Serial dependence.** For an AR(1) process the variance of a mean of n
observations exceeds the independent case by roughly (1+rho)/(1-rho). A
detector that ignores it fires on dependence rather than on change. Two
treatments are offered and both are used: inflating the reference spread (cheap,
approximate) and whitening the series through its own fitted AR coefficient
(exact under the model, and destroys the signal if the break *is* a change in
dependence — so, again, kept alongside rather than instead).

**Distributional shape.** Heavy tails make a Gaussian-calibrated detector alarm
on ordinary observations. The historical excess kurtosis is measured, and the
winsorising threshold widened where the series is genuinely heavy-tailed rather
than broken.

The pattern throughout: normalise so the detector is not fooled, and keep what
was normalised away as its own channel, because a break may live in exactly the
quantity being removed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


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
