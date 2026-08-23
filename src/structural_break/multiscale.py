"""Multi-scale change evidence: the same question at six receptive fields.

A break is a disagreement between "recently" and "before" — but *recently* has
no single right size. A level shift of half a sigma needs a hundred observations
to stand out of the noise; a threefold variance jump shows in ten. A detector
with one window is tuned to one break size and blind on either side of it.

So the stream is watched through six exponential windows at once — effective
lengths 5, 10, 20, 50, 100 and 200 observations — and each produces two
standardised discrepancies against the break-free history: one in the mean, one
in the spread. This is a convolution bank in the exact signal-processing sense:
each EWMA is a causal exponential kernel, the six lengths are six receptive
fields, and the pair (mean, spread) are two feature maps per field. The learned
combiner downstream plays the role of the 1x1 convolution that mixes them.

Why exponential kernels rather than boxes
-----------------------------------------
A box window needs a ring buffer per scale; an EWMA is one number per scale and
one multiply per observation, so six scales cost twelve floats of state. On ten
thousand series of a thousand steps that difference is the run fitting the
platform's quota or not. The shapes are close enough that the scale ladder, not
the kernel shape, carries the information.

Standardisation
---------------
Each mean-discrepancy is scaled by the standard error of an EWMA of that length
on unit-variance noise, sqrt(alpha / (2 - alpha)) — so a value of 2.0 means the
same thing at every scale: this window's average sits two of *its own* standard
errors from where the history says it should. Without this the long windows
would always look quieter than the short ones and the combiner would have to
relearn the ladder from data. The spread-discrepancy is the log of the EWMA of
squared values, likewise scaled by its own null spread.

The running peak of each channel is kept alongside its current value. The
current value answers "does it look broken now"; the peak answers "has it ever
looked broken" — and the task's label is the second question.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

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
