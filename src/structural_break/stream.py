"""Running the detectors exactly as the platform will run them.

Calibrating on a series the model can see in full is the most direct way to
produce numbers that do not survive submission. The platform reveals the online
segment one observation at a time and demands a score after each; every
statistic here is therefore updated in O(1) from the observation just seen, and
nothing is ever re-derived from a window that includes the future.

This module is the harness that enforces that. Feeding it a series and reading
back per-step scores is the *only* way anything in this repository is measured,
so a mistake that would look like a leak on the platform looks like one here.

The channels
------------
Nine numbers per step, from three detector families over three views of the
data. The views are the point: each removes one way of being fooled and each
destroys one kind of break, so running all three keeps both properties.

* **raw** — detrended and standardised only. Sees a change in level, and is
  fooled by dependence.
* **whitened** — additionally passed through the historical AR coefficient.
  Immune to the dependence that was there before; loud when the dependence
  itself changes, since the wrong coefficient stops whitening.
* **absolute** — the magnitude of the whitened value, which turns a change in
  spread into a change in level and lets the mean-shift detectors find it.

Nine channels rather than one score, because the combination is then a fitted
question rather than an assumed one.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np

from structural_break.detectors import Cusum, PageHinkley, VarianceRatio
from structural_break.features import Normalisation, whiten

#: The three views, in the order their columns appear.
VIEWS = ("raw", "whitened", "absolute")

#: The three detector families.
FAMILIES = ("cusum", "page_hinkley", "variance_ratio")

#: Column names, one per (view, family) pair.
CHANNELS = tuple(f"{view}_{family}" for view in VIEWS for family in FAMILIES)


class Monitor:
    """One series being watched, one observation at a time."""

    def __init__(self, history: np.ndarray) -> None:
        self.norm = Normalisation.fit(history)
        # Detectors work on already-standardised input, so their own reference
        # is the unit Gaussian.
        self.detectors = {
            view: {
                "cusum": Cusum(),
                "page_hinkley": PageHinkley(),
                "variance_ratio": VarianceRatio(),
            }
            for view in VIEWS
        }
        self._previous_z = 0.0
        self._step = 0
        if len(history) > 0:
            last = float(history[-1])
            self._previous_z = self.norm.standardise(last, -1)

    def update(self, x: float) -> dict[str, float]:
        """One observation in, nine scores out."""
        z = self.norm.clip(self.norm.standardise(float(x), self._step))
        w = whiten(z, self._previous_z, self.norm.rho)
        # The dependence adjustment applies to the raw view only: the whitened
        # view has had the dependence removed, so inflating its threshold as
        # well would double-count and make it deaf.
        values = {
            "raw": z / self.norm.inflation,
            "whitened": w,
            "absolute": abs(w) - 0.7979,  # centred: E|N(0,1)| = sqrt(2/pi)
        }
        self._previous_z = z
        self._step += 1

        out: dict[str, float] = {}
        for view, value in values.items():
            for family, detector in self.detectors[view].items():
                out[f"{view}_{family}"] = detector.update(value)
        return out


def run(history: np.ndarray, online: Iterable[float]) -> np.ndarray:
    """Per-step channel scores for one series, shaped (steps, channels)."""
    monitor = Monitor(history)
    rows = []
    for value in online:
        step = monitor.update(float(value))
        rows.append([step[name] for name in CHANNELS])
    return np.asarray(rows, dtype="float64") if rows else np.empty((0, len(CHANNELS)))


def iter_series(x, y=None) -> Iterator[tuple[int, np.ndarray, np.ndarray, np.ndarray | None]]:
    """Walk a competition frame as (id, history, online, labels)."""
    for series_id, part in x.groupby(level="id"):
        history = part.loc[part["period"] == 1, "value"].to_numpy()
        online = part.loc[part["period"] == 2, "value"].to_numpy()
        if len(online) == 0:
            continue
        labels = None
        if y is not None and series_id in y.index.get_level_values(0):
            labels = y.loc[series_id, "target"].to_numpy()
        yield int(series_id), history, online, labels
