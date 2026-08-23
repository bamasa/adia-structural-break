"""Generate and execute the model-inspection notebook.

A notebook in a repository rots the moment the model changes, unless the
notebook is itself generated and executed by a script that lives beside the
model. This is that script: it writes the cells, runs them against the current
artefact and the organisers' labelled hundred series, and commits the executed
result — figures embedded, numbers current.

The notebook answers four questions a person forms hypotheses from:

1. What does a series look like, raw and after normalisation, with the true
   break and the model's reaction on the same axis?
2. Where does the model do well, averagely, badly — sorted, so the eye goes
   straight to the failures?
3. Which *kinds* of series are hard: strong trend, heavy tails, high
   dependence, late breaks?
4. How does the score trajectory behave around a break — sharp, sluggish,
   or absent?
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "notebooks" / "model_inspection.ipynb"

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("md", text))


def code(text: str) -> None:
    CELLS.append(("code", text))


md("""# Inspecting the detector on the organisers' labelled test series

One hundred series with known break positions, scored by the 40-channel model
exactly as the platform would run it — one observation at a time, no lookahead.
Every figure follows the same layout: **raw series** with the true break in
red, **normalised stream** the detectors actually see, and the **model score**
with the break marked again. The vertical grey line is the history/online
boundary.

Sorted galleries first (best, median, worst), then slices by the character of
the series — trend, dispersion, dependence, tail weight, break position — which
is where hypotheses about the next model come from.""")

code("""import sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

WS = Path("../../structural-break-real-time-test")
sys.path.insert(0, str(WS))  # the submission module, exactly as shipped

import importlib.util
spec = importlib.util.spec_from_file_location(
    "submission", Path("../submissions/005-forty-channels/main.py")
)
submission = importlib.util.module_from_spec(spec)
# Registered before execution: the module defines dataclasses, and dataclass
# field resolution looks itself up in sys.modules -- an unregistered module
# crashes there with an AttributeError three frames deep.
sys.modules["submission"] = submission
spec.loader.exec_module(submission)

x = pd.read_parquet(WS / "data/X_test.reduced.parquet")
y = pd.read_parquet(WS / "data/y_test.reduced.parquet")
model = joblib.load("../../model005_backup.joblib")["booster"]
print(f"{x.index.get_level_values(0).nunique()} series, model with {model.n_features_} channels")""")

code("""# Score every series step by step, storing everything the figures need.
records = {}
for sid, part in x.groupby(level="id"):
    hist = part.loc[part.period == 1, "value"].to_numpy()
    online = part.loc[part.period == 2, "value"].to_numpy()
    if len(online) == 0:
        continue
    labels = y.loc[sid, "target"].to_numpy()
    monitor = submission.Monitor(hist)
    channels = np.asarray([monitor.update(float(v)) for v in online])
    scores = model.predict_proba(channels)[:, 1]
    norm = monitor.norm
    z_online = np.asarray(
        [norm.clip(norm.standardise(float(v), i)) for i, v in enumerate(online)]
    )
    z_hist = np.asarray(
        [norm.clip(norm.standardise(float(v), -len(hist) + i)) for i, v in enumerate(hist)]
    )
    tau = int(labels.argmax()) if labels.max() > 0 else None
    # Where the model itself calls the break: the first step at which the score
    # crosses half of its own maximum on this series. A display convention, not
    # part of the metric -- the metric never asks for a point -- but a figure
    # without it leaves the reader guessing where the purple line "decided".
    peak = float(scores.max())
    crossed = np.flatnonzero(scores >= 0.5 * peak) if peak > 0 else []
    detected = int(crossed[0]) if len(crossed) else None
    records[int(sid)] = dict(
        hist=hist, online=online, z=z_online, z_hist=z_hist, scores=scores,
        labels=labels, tau=tau, detected=detected,
        slope=norm.slope, sd=norm.sd, rho=norm.rho, kurt=norm.kurtosis,
    )
print(f"scored {len(records)} series")""")

code("""# Per-series quality. For a broken series: the within-series AUC of the score
# against the per-step label -- does the score rank post-break steps above
# pre-break ones. For an unbroken series: the false-alarm level, taken as the
# final score (lower is better). The two are different questions, so the
# galleries are sorted separately.
def within_auc(scores, labels):
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    ranks = np.concatenate([pos, neg]).argsort().argsort() + 1
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

rows = []
for sid, r in records.items():
    rows.append(dict(
        id=sid, broken=r["tau"] is not None, tau=r["tau"],
        n_online=len(r["online"]),
        quality=within_auc(r["scores"], r["labels"]) if r["tau"] is not None else np.nan,
        false_alarm=float(r["scores"][-1]) if r["tau"] is None else np.nan,
        trend=abs(r["slope"]) * len(r["hist"]) / r["sd"],
        sd=r["sd"], rho=r["rho"], kurt=r["kurt"],
    ))
table = pd.DataFrame(rows).set_index("id")
print(table.groupby("broken").size())
broken = table[table.broken].sort_values("quality", ascending=False)
clean = table[~table.broken].sort_values("false_alarm")
print(f"broken: median within-series AUC {broken.quality.median():.3f}")
print(f"clean:  median final score {clean.false_alarm.median():.3f}")""")

code('''def show(sid, title_extra=""):
    r = records[sid]
    hist, online, z, scores, tau = r["hist"], r["online"], r["z"], r["scores"], r["tau"]
    z_hist, detected = r["z_hist"], r["detected"]
    n_h = len(hist)
    fig, axes = plt.subplots(3, 1, figsize=(11, 6.5), sharex=True,
                             gridspec_kw=dict(height_ratios=[2, 1.4, 1.4]))
    t_hist = np.arange(-n_h, 0)
    t_on = np.arange(len(online))
    axes[0].plot(t_hist[-600:], hist[-600:], lw=0.6, color="#9aa0a6",
                 label="история (слома нет по условию)")
    axes[0].plot(t_on, online, lw=0.8, color="#1a73e8",
                 label="онлайн-часть (приходит по одной точке)")
    axes[0].set_ylabel("сырой ряд")
    axes[1].plot(t_hist[-600:], z_hist[-600:], lw=0.6, color="#9aa0a6",
                 label="история после нормировки (для сравнения масштаба)")
    axes[1].plot(t_on, z, lw=0.8, color="#188038",
                 label="онлайн после нормировки: минус тренд, минус масштаб, обрезка выбросов")
    axes[1].axhline(0, color="grey", lw=0.5)
    axes[1].set_ylabel("нормированный")
    axes[2].plot(t_on, scores, lw=1.4, color="#7b1fa2",
                 label="счёт модели: уверенность, что слом уже был (0..1)")
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].set_ylabel("счёт модели")
    axes[2].set_xlabel("шаг онлайн-части (история — при отрицательных t)")
    for k, ax in enumerate(axes):
        ax.axvline(0, color="grey", lw=1.0, alpha=0.6,
                   label="граница история/онлайн" if k == 0 else None)
        if tau is not None:
            ax.axvline(tau, color="#d93025", lw=1.2, ls="--",
                       label="ИСТИННЫЙ слом (разметка)" if k == 0 else None)
        if detected is not None:
            ax.axvline(detected, color="#f9ab00", lw=1.6, alpha=0.9,
                       label="МОДЕЛЬ решила: слом был (первое пересечение половины"
                             " своего максимума)" if k == 0 else None)
        ax.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.85)
    q = table.loc[sid]
    status = f"break at {tau}" if tau is not None else "no break"
    metric = (f"within-series AUC {q.quality:.3f}" if tau is not None
              else f"final score {q.false_alarm:.3f}")
    fig.suptitle(f"series {sid} — {status} — {metric}{title_extra}", y=0.995)
    plt.tight_layout()
    plt.show()''')

md("""## Broken series: best, median, worst

Red dashed line — the true break. The question for each figure: does the score
climb *at* the line, later, or never.""")

code("""ids = list(broken.index)
for sid in ids[:3]:
    show(sid, "  (best)")""")
code("""mid = len(ids) // 2
for sid in ids[mid - 1 : mid + 2]:
    show(sid, "  (median)")""")
code("""for sid in ids[-3:]:
    show(sid, "  (worst — where the model loses most)")""")

md("""## Unbroken series: the false alarms

No red line exists; anything the score does here is a mistake. The three
highest-scoring clean series are the model's worst false alarms.""")

code("""for sid in list(clean.index)[-3:]:
    show(sid, "  (false alarm)")""")

md("""## Which kinds of series are hard

Median within-series AUC of the broken series, split at the median of each
history property. A large gap names a hypothesis; a flat pair says that axis
does not matter.""")

code("""splits = {}
b = table[table.broken]
for col, name in [("trend", "trend strength"), ("sd", "dispersion"),
                  ("rho", "autocorrelation"), ("kurt", "tail weight"),
                  ("tau", "break position"), ("n_online", "online length")]:
    m = b[col].median()
    low, high = b[b[col] <= m], b[b[col] > m]
    splits[name] = (low.quality.median(), high.quality.median(), m)
frame = pd.DataFrame(splits, index=["low half", "high half", "split at"]).T
display(frame.round(3))

fig, ax = plt.subplots(figsize=(9, 4))
xpos = np.arange(len(frame))
ax.bar(xpos - 0.18, frame["low half"], width=0.36, label="low half", color="#9aa0a6")
ax.bar(xpos + 0.18, frame["high half"], width=0.36, label="high half", color="#1a73e8")
ax.set_xticks(xpos, frame.index, rotation=20)
ax.set_ylabel("медианный AUC внутри ряда; выше = лучше отделяет до/после")
ax.axhline(0.5, color="#d93025", lw=0.8, ls="--")
ax.legend(frameon=False)
ax.set_title("Where the detector is strong and where it is blind")
plt.tight_layout(); plt.show()""")

md("""## Reading list for the next model

The bars above are the hypothesis generator: every axis with a visible gap is a
question — what channel would close it — and every worst-gallery figure is a
concrete series to reason about. The notebook regenerates from
`scripts/build_analysis_notebook.py`; edits belong there.""")


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(text) if kind == "md" else nbf.v4.new_code_cell(text)
        for kind, text in CELLS
    ]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    TARGET.parent.mkdir(exist_ok=True)
    execute = "--no-exec" not in sys.argv
    if execute:
        from nbclient import NotebookClient

        client = NotebookClient(nb, timeout=1800, kernel_name="python3",
                                resources={"metadata": {"path": str(TARGET.parent)}})
        client.execute()
    nbf.write(nb, TARGET)
    print(f"{'executed and ' if execute else ''}written -> {TARGET}")


if __name__ == "__main__":
    main()
