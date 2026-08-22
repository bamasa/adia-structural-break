"""Score the detectors on the competition's own reduced test set.

The metric is Time-Stratified AUC: at each online step, an ordinary AUC across
every series still alive at that step, averaged with weights equal to the number
of positive-negative pairs available there. Two consequences shape everything:

* Only the *ordering between series* matters at a given step, never the absolute
  level of any score. A detector that is uniformly pessimistic loses nothing.
* Early steps carry as much weight as late ones, and early steps are where
  almost no evidence has accumulated. A detector that is only right after two
  hundred observations scores poorly however certain it eventually becomes.

The second point is the one worth designing against, and it is why the null
normalisation matters: at step five, every series has a small statistic, and
what separates them is whether it is small *for its own history*.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import detectors as d  # noqa: E402


def load(directory: Path = Path("data")):
    x = pd.read_parquet(directory / "X_test.reduced.parquet")
    y = pd.read_parquet(directory / "y_test.reduced.parquet")
    return x, y


def score_series(history: np.ndarray, online: np.ndarray) -> dict[str, np.ndarray]:
    """Per-step scores for one series, one column per detector plus the combination."""
    state = d.build(history)
    out = {name: np.empty(len(online)) for name in d.DETECTORS}
    out["max"] = np.empty(len(online))
    for i, value in enumerate(online):
        step = {name: state[name].update(float(value)) for name in d.DETECTORS}
        for name, score in step.items():
            out[name][i] = score
        out["max"][i] = d.combine(step)
    return out


def ts_auc(frame: pd.DataFrame, column: str) -> float:
    """Time-stratified AUC, as the competition defines it."""
    weighted, total = 0.0, 0.0
    for _, group in frame.groupby("step"):
        labels = group["target"].to_numpy()
        positives, negatives = int(labels.sum()), int((1 - labels).sum())
        if positives == 0 or negatives == 0:
            continue
        scores = group[column].to_numpy()
        ranks = scores.argsort().argsort() + 1
        auc = (ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (
            positives * negatives
        )
        weight = float(positives * negatives)
        weighted += weight * auc
        total += weight
    return weighted / total if total else 0.5


def main() -> None:
    x, y = load()
    rows = []
    for series_id, part in x.groupby(level="id"):
        history = part.loc[part["period"] == 1, "value"].to_numpy()
        online = part.loc[part["period"] == 2, "value"].to_numpy()
        if len(online) == 0:
            continue
        scores = score_series(history, online)
        rows.append(
            pd.DataFrame({"id": series_id, "step": np.arange(len(online)), **scores})
        )

    predictions = pd.concat(rows, ignore_index=True)
    labels = y.reset_index()
    labels["step"] = labels.groupby("id").cumcount()
    merged = predictions.merge(labels[["id", "step", "target"]], on=["id", "step"], how="inner")

    print(f"{len(merged):,} scored steps across {merged['id'].nunique()} series\n")
    for column in [*d.DETECTORS, "max"]:
        print(f"  {column:16s} TS-AUC {ts_auc(merged, column):.4f}")

    # The organisers' own baseline, for a reference point on the same rows.
    baseline = []
    for series_id, part in x.groupby(level="id"):
        history = part.loc[part["period"] == 1, "value"].to_numpy()
        online = part.loc[part["period"] == 2, "value"].to_numpy()
        if len(online) == 0:
            continue
        mu, sd = float(history.mean()), max(float(history.std(ddof=1)), 1e-8)
        mu_ewma, n_eff, alpha = mu, 0.0, 0.05
        values = np.empty(len(online))
        for i, point in enumerate(online):
            mu_ewma = (1 - alpha) * mu_ewma + alpha * float(point)
            n_eff = (1 - alpha) * n_eff + 1.0
            se = sd / np.sqrt(max(n_eff, 1.0))
            values[i] = np.tanh(abs(mu_ewma - mu) / (se * 3.0))
        baseline.append(pd.DataFrame({"id": series_id, "step": np.arange(len(online)), "ewma": values}))
    merged = merged.merge(pd.concat(baseline, ignore_index=True), on=["id", "step"], how="left")
    print(f"\n  {'ewma (their baseline)':16s} TS-AUC {ts_auc(merged, 'ewma'):.4f}")


if __name__ == "__main__":
    main()
