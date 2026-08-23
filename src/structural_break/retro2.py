"""Retrospective verdicts that can confirm a missed break and cancel a false one.

The first retrospective class reported the best split statistic and nothing
else. This one reports what a person reviewing the whole seen segment would
actually use, and the two behaviours the review is *for* have their own
channels:

**Confirming late.** ``best_split`` statistics at several trailing depths —
compare the last w observations against everything before them, for w in
{10, 20, 50, 100} as well as the free scan over all splits. A break missed at
the time it happened grows more obvious as post-break data accumulates, and the
depth ladder says how long ago it must have been.

**Cancelling.** Two channels exist precisely to take evidence *away*:

* ``split_stability`` — where the best split has been landing across recent
  scans. A real break pins the argmax to one position; noise wanders it. The
  channel is the negative spread of recent argmax positions, squashed, so a
  wandering maximum reads as "probably nothing".
* ``tail_calm`` — the discrepancy of the observations *after* the best split
  against the history. A genuine break leaves the tail persistently different;
  a spike leaves a tail that looks exactly like the history again, and this
  channel notices and votes the alarm down.

Everything runs on the scan cadence of the first version — geometric schedule,
scores carried between scans — and stays O(t) per scan via prefix sums.
"""

from __future__ import annotations

import math

import numpy as np

from structural_break.detectors import squash
from structural_break.features import Normalisation
from structural_break.retrospective import (
    MIN_SIDE,
    _null_mean,
    _scan_mean_variance,
)

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
