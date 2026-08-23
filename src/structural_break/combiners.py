"""Two ways of turning nine detector channels into one score.

The detectors each answer "has this stream departed from its history" in their
own way. Nine answers arrive per step and one number must be submitted, so
something has to combine them. Submission 001 assumed the maximum -- trust
whoever shouts loudest -- which is a guess, and ten thousand labelled series is
enough evidence to replace a guess with a measurement.

Two combiners, deliberately different in what they can express:

**Weighted** fits one coefficient per channel and adds them up. A channel is
worth the same everywhere: useful when it is useful, ignored when it is not,
but never conditional on what the other channels are saying. Logistic
regression on the logit of each score, so the combination stays a probability
and the coefficients stay readable -- a negative one says a channel is
*anti*-informative, which is worth knowing.

**Boosted** fits gradient-boosted trees, so the weight of a channel depends on
the others. It can express "the CUSUM matters when the dispersion channel is
quiet, and not otherwise", which the weighted form cannot say at all.

The first is the honest baseline for the second. If the trees do not beat the
line, their extra freedom bought nothing but variance -- an outcome this project
has measured before and should expect again.

Validation
----------
Both are scored the same way: fit on four fifths of the *series*, score on the
fifth, never splitting a series across the boundary. Consecutive steps of one
series are near-duplicates, so a random split of rows would put the same moment
on both sides and turn validation into memorisation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def logit(p: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Scores live in (0, 1); a linear model wants them unbounded."""
    q = np.clip(p, epsilon, 1.0 - epsilon)
    return np.log(q / (1.0 - q))


def step_weights(steps: np.ndarray) -> np.ndarray:
    """Weight rows so each step index contributes comparably.

    The metric averages per-step AUCs with equal weight per step, while a long
    series contributes many late rows and few early ones. Left uncorrected, the
    fit optimises the tail of the problem and neglects the front, which is
    where the evidence is thinnest and the score is decided.
    """
    unique, counts = np.unique(steps, return_counts=True)
    lookup = dict(zip(unique.tolist(), counts.tolist(), strict=True))
    weights = np.array([1.0 / lookup[int(s)] for s in steps], dtype="float64")
    return weights * (len(weights) / weights.sum())


def split_by_series(groups: np.ndarray, *, folds: int = 5, seed: int = 0) -> np.ndarray:
    """Fold assignment that keeps every row of a series together."""
    series = np.unique(groups)
    rng = np.random.default_rng(seed)
    order = series.copy()
    rng.shuffle(order)
    assignment = {int(sid): i % folds for i, sid in enumerate(order.tolist())}
    return np.array([assignment[int(g)] for g in groups], dtype="int8")


def ts_auc(scores: np.ndarray, labels: np.ndarray, steps: np.ndarray) -> float:
    """Time-Stratified AUC, exactly as the competition defines it.

    At each step, an ordinary AUC across every series alive there, averaged with
    weights equal to the number of positive-negative pairs available. Steps with
    only one class contribute nothing, which is most of the very late ones.
    """
    weighted = total = 0.0
    for step in np.unique(steps):
        mask = steps == step
        y = labels[mask]
        positives = int(y.sum())
        negatives = int(len(y) - positives)
        if positives == 0 or negatives == 0:
            continue
        ranks = scores[mask].argsort().argsort() + 1
        auc = (ranks[y == 1].sum() - positives * (positives + 1) / 2) / (
            positives * negatives
        )
        weight = float(positives * negatives)
        weighted += weight * auc
        total += weight
    return weighted / total if total else 0.5


@dataclass
class Weighted:
    """One coefficient per channel, fitted on the logit scale."""

    coefficients: np.ndarray
    intercept: float
    channels: list[str]

    @classmethod
    def fit(
        cls, x: np.ndarray, y: np.ndarray, channels: list[str], weights: np.ndarray | None
    ) -> Weighted:
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=1000, C=1.0)
        model.fit(logit(x), y, sample_weight=weights)
        return cls(
            coefficients=model.coef_[0].astype("float64"),
            intercept=float(model.intercept_[0]),
            channels=list(channels),
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = logit(x) @ self.coefficients + self.intercept
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def describe(self) -> str:
        pairs = sorted(
            zip(self.channels, self.coefficients, strict=True),
            key=lambda p: -abs(p[1]),
        )
        return ", ".join(f"{name}={value:+.2f}" for name, value in pairs)


@dataclass
class Boosted:
    """Gradient-boosted trees: a channel's weight depends on the others."""

    booster: object
    channels: list[str]

    @classmethod
    def fit(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        channels: list[str],
        weights: np.ndarray | None,
        **params: object,
    ) -> Boosted:
        import lightgbm as lgb

        settings: dict = {
            "objective": "binary",
            "learning_rate": 0.05,
            "num_leaves": 31,
            # Large leaves on purpose: rows within a series are near-duplicates,
            # so a small leaf memorises a handful of series rather than learning
            # anything that transfers.
            "min_child_samples": 2000,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_lambda": 10.0,
            "n_estimators": 300,
            "random_state": 0,
            # LightGBM ships its own OpenMP; single-threaded avoids the
            # segfault it causes beside other runtimes.
            "n_jobs": 1,
            "verbose": -1,
        }
        settings.update(params)
        model = lgb.LGBMClassifier(**settings)
        model.fit(x, y, sample_weight=weights)
        return cls(booster=model, channels=list(channels))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.booster.predict_proba(x)[:, 1]  # type: ignore[attr-defined]

    def describe(self) -> str:
        pairs = sorted(
            zip(self.channels, self.booster.feature_importances_, strict=True),  # type: ignore[attr-defined]
            key=lambda p: -p[1],
        )
        return ", ".join(f"{name}={value}" for name, value in pairs[:6])


def cross_validate(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    steps: np.ndarray,
    channels: list[str],
    *,
    folds: int = 5,
    seed: int = 0,
    weight_steps: bool = True,
) -> dict[str, list[float]]:
    """Both combiners and the maximum baseline, on every fold.

    The maximum is submission 001's rule and is the thing to beat. Reporting all
    three per fold rather than as averages is deliberate: a combiner that wins
    on average while losing on two folds of five has found something fragile,
    and the spread says so where a mean would hide it.
    """
    assignment = split_by_series(groups, folds=folds, seed=seed)
    out: dict[str, list[float]] = {"maximum": [], "weighted": [], "boosted": []}

    for fold in range(folds):
        train, valid = assignment != fold, assignment == fold
        weights = step_weights(steps[train]) if weight_steps else None

        out["maximum"].append(ts_auc(x[valid].max(axis=1), y[valid], steps[valid]))

        weighted = Weighted.fit(x[train], y[train], channels, weights)
        out["weighted"].append(ts_auc(weighted.predict(x[valid]), y[valid], steps[valid]))

        boosted = Boosted.fit(x[train], y[train], channels, weights)
        out["boosted"].append(ts_auc(boosted.predict(x[valid]), y[valid], steps[valid]))

    return out
