"""Learning how to combine nine detector channels into one score.

The detectors answer "has this stream departed from its history", each in its
own way and each with a threshold calibrated against a simulated null. What none
of them knows is which departure matters *on this data*: the competition's
series break in ways the classical families were not designed around — shape,
tail behaviour, dependence — and ten thousand labelled series is exactly the
evidence needed to weigh them.

So the model is a combiner, not a detector. Its inputs are the nine channels
plus a little context; its output is the score submitted. That arrangement keeps
the streaming discipline intact — every input is O(1) and causal — while letting
the weighting be fitted rather than assumed.

Validation is by series, never by row
--------------------------------------
Each series contributes up to sixty rows, and consecutive steps of one series
are nearly identical. Splitting rows at random puts steps of the same series on
both sides of the split and turns the validation score into a memorisation
score. Every split here is on the series id.

The objective is not the metric
-------------------------------
The model is fitted on log-loss over rows, and judged on Time-Stratified AUC
over steps. Those disagree: log-loss rewards being right on the many easy late
rows, while the metric weighs every step equally and the hard rows are early.
The gap is handled by weighting rows so that each *step index* contributes
comparably, rather than by inventing a custom objective — a weighted proper
scoring rule stays calibrated, and a hand-rolled ranking loss on this much data
mostly finds its own bugs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def step_weights(steps: np.ndarray) -> np.ndarray:
    """Weight rows so early steps are not drowned by late ones.

    The metric averages per-step AUCs with equal weight per step, but a long
    series contributes many late rows and few early ones. Without correction the
    fit optimises the tail of the problem and neglects the part the metric
    actually rewards.
    """
    unique, counts = np.unique(steps, return_counts=True)
    lookup = dict(zip(unique.tolist(), counts.tolist(), strict=True))
    weights = np.array([1.0 / lookup[s] for s in steps.tolist()], dtype="float64")
    return weights * (len(weights) / weights.sum())


def split_by_series(groups: np.ndarray, *, folds: int = 5, seed: int = 0):
    """Fold assignment that keeps every row of a series together."""
    series = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(series)
    assignment = {sid: i % folds for i, sid in enumerate(series.tolist())}
    return np.array([assignment[g] for g in groups.tolist()], dtype="int64")


@dataclass
class Fitted:
    """A trained combiner and what it scored where it was not fitted."""

    booster: object
    columns: list[str]
    ts_auc: float
    baseline_ts_auc: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.booster.predict_proba(x)[:, 1]  # type: ignore[attr-defined]


def ts_auc(scores: np.ndarray, labels: np.ndarray, steps: np.ndarray) -> float:
    """Time-Stratified AUC, as the competition defines it."""
    weighted = total = 0.0
    for step in np.unique(steps):
        mask = steps == step
        y = labels[mask]
        positives, negatives = int(y.sum()), int((1 - y).sum())
        if positives == 0 or negatives == 0:
            continue
        s = scores[mask]
        ranks = s.argsort().argsort() + 1
        auc = (ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives)
        weight = float(positives * negatives)
        weighted += weight * auc
        total += weight
    return weighted / total if total else 0.5


def fit(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    columns: list[str],
    *,
    step_column: int,
    params: dict | None = None,
    seed: int = 0,
) -> Fitted:
    """Fit on four fifths of the series, score on the fifth."""
    import lightgbm as lgb

    folds = split_by_series(groups, folds=5, seed=seed)
    train, valid = folds != 0, folds == 0

    settings = {
        "objective": "binary",
        "learning_rate": 0.05,
        "num_leaves": 31,
        # Large leaves: rows within a series are near-duplicates, so a small
        # leaf memorises a handful of series rather than learning a rule.
        "min_child_samples": 500,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 5.0,
        "n_estimators": 400,
        "random_state": seed,
        "n_jobs": 1,  # LightGBM's own OpenMP conflicts with other runtimes
        "verbose": -1,
    }
    settings.update(params or {})

    booster = lgb.LGBMClassifier(**settings)
    booster.fit(
        x[train],
        y[train],
        sample_weight=step_weights(x[train][:, step_column]),
    )

    scores = booster.predict_proba(x[valid])[:, 1]
    steps = x[valid][:, step_column]
    # The strongest single channel on the same rows, as the thing to beat.
    best_channel = max(
        range(len(columns) - 6),
        key=lambda i: ts_auc(x[valid][:, i], y[valid], steps),
    )
    return Fitted(
        booster=booster,
        columns=columns,
        ts_auc=ts_auc(scores, y[valid], steps),
        baseline_ts_auc=ts_auc(x[valid][:, best_channel], y[valid], steps),
    )
