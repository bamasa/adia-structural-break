"""Turning 10,000 labelled series into rows a model can learn from.

Each series contributes one row per online step: the nine detector channels at
that step, a few quantities describing the series itself, and the label — has
the break already happened by now.

Two decisions that decide whether the training set is honest.

**Every step is a row, not every series.** The metric scores each step, and the
early ones — where almost nothing has accumulated — carry as much weight as the
late ones. Training on end-of-series rows only would fit the easy half of the
problem and score badly on the half that matters.

**The step index is a feature, and that is not a leak.** The model is allowed to
know how far into the online segment it is, because at inference time it knows
that too. What it is never given is the segment's *length*: on the platform the
end has not arrived yet, and a model that learned "series of length 300 break
around step 180" would be reading something unavailable.

Subsampling
-----------
Ten thousand series of up to a thousand steps is five million rows, which is
more than the fit needs and more than fits comfortably in memory alongside the
raw parquet. Rows are sampled on a geometric-ish grid — dense early, sparse
late — because that is where the metric's weight and the problem's difficulty
both are.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

from structural_break.features import Normalisation
from structural_break.stream import CHANNELS, Monitor

#: Columns describing the series rather than the moment.
CONTEXT = ("step", "log_step", "hist_rho", "hist_kurtosis", "hist_log_n", "hist_log_sd")

COLUMNS = (*CHANNELS, *CONTEXT)


def sample_steps(length: int, target: int = 60) -> np.ndarray:
    """Which steps of a series to keep, dense early and sparse late.

    Early steps are where the evidence is thinnest and the metric's weight is
    highest, so they are worth more rows than the tail — where consecutive
    steps are nearly identical anyway.
    """
    if length <= target:
        return np.arange(length)
    grid = np.unique(np.geomspace(1, length, num=target).astype(int) - 1)
    return grid[(grid >= 0) & (grid < length)]


def series_rows(
    history: np.ndarray,
    online: np.ndarray,
    labels: np.ndarray | None,
    *,
    subsample: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Channels and context for one series, optionally thinned."""
    monitor = Monitor(history)
    norm: Normalisation = monitor.norm
    context_tail = (
        norm.rho,
        norm.kurtosis,
        float(np.log(max(norm.n, 1))),
        float(np.log(max(norm.sd, 1e-9))),
    )

    keep = set(sample_steps(len(online)).tolist()) if subsample else None
    rows, targets = [], []
    for step, value in enumerate(online):
        scores = monitor.update(float(value))
        if keep is not None and step not in keep:
            continue
        rows.append(
            [scores[name] for name in CHANNELS]
            + [float(step), float(np.log1p(step)), *context_tail]
        )
        if labels is not None:
            targets.append(int(labels[step]))

    x = np.asarray(rows, dtype="float64") if rows else np.empty((0, len(COLUMNS)))
    y = np.asarray(targets, dtype="int64") if labels is not None else None
    return x, y


def build(
    frame: pd.DataFrame,
    labels: pd.DataFrame | None = None,
    *,
    limit: int | None = None,
    subsample: bool = True,
    on_progress=None,
) -> tuple[pd.DataFrame, np.ndarray | None, np.ndarray]:
    """Rows for every series in ``frame``.

    Returns the feature frame, the labels, and the series id each row came
    from — the last so that validation can split by *series* rather than by
    row. Splitting by row would put steps of the same series on both sides and
    make every score optimistic.
    """
    xs, ys, groups = [], [], []
    for count, (series_id, part) in enumerate(frame.groupby(level="id")):
        if limit is not None and count >= limit:
            break
        history = part.loc[part["period"] == 1, "value"].to_numpy()
        online = part.loc[part["period"] == 2, "value"].to_numpy()
        if len(online) == 0:
            continue
        target = None
        if labels is not None:
            target = labels.loc[series_id, "target"].to_numpy()
        rows, y = series_rows(history, online, target, subsample=subsample)
        if len(rows) == 0:
            continue
        xs.append(rows)
        if y is not None:
            ys.append(y)
        groups.append(np.full(len(rows), series_id))
        if on_progress is not None and count % 500 == 0:
            on_progress(f"  {count} series")

    x = pd.DataFrame(np.vstack(xs), columns=list(COLUMNS))
    y = np.concatenate(ys) if ys else None
    return x, y, np.concatenate(groups)


def iter_series(frame: pd.DataFrame) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    for series_id, part in frame.groupby(level="id"):
        yield (
            int(series_id),
            part.loc[part["period"] == 1, "value"].to_numpy(),
            part.loc[part["period"] == 2, "value"].to_numpy(),
        )
