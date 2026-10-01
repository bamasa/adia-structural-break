"""Generate and execute the model-inspection notebook.

A notebook in a repository rots the moment the model changes, unless the
notebook is itself generated and executed by a script that lives beside the
model. This is that script: it writes the cells, runs them against the current
artefacts and the organisers' labelled hundred series, and commits the executed
result — figures embedded, numbers current.

The notebook's audience is the project owner reading it as a report; every
cell of prose, every label and every title is written in plain English, like the
repository around it. Each figure stacks: the raw series, the normalised stream,
one narrow panel per strong model (with the model's behaviour on the
guaranteed-break-free history too), and the component channels.
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


md("""# Model breakdown on the labelled hundred series

A hundred series with known break positions, run exactly the way the
platform does it — one point at a time, with no look-ahead. All three
strong models are evaluated: **005** (40 channels), **006** (41, +convolutional channel),
**008** (50, +reverting channels — the current best).

Every figure is laid out the same way, top to bottom:

1. **raw series** — the grey history (no break by construction) and the blue online part;
2. **normalised series** — what the detectors actually see: trend removed,
   scale removed, outliers clipped;
3. **three narrow panels — one per model**: the score from 0 to 1, including on
   the history (dotted) — shows whether the model twitched where a break is
   guaranteed not to exist. The dot marks where the model "decided" a break happened; for 008
   the orange triangles are alarm cancellations;
4. **components** — the individual detectors the score is built from.

Vertical lines on all panels: grey — the history/online boundary, red
dashed — the true break from the labels.

The score on the history is an illustrative run: on the platform the model does not
score the history, and at the boundary (t = 0) the detectors start from a clean slate, as in production.
The convolutional channel is not recomputed on the history and sits at 0.5.

First the galleries by quality (best, middle, worst), then slices by
the character of the series — that is where the hypotheses for the next model come from.""")

code("""import sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

WS = Path("../../structural-break-real-time-test")
sys.path.insert(0, str(WS))

import importlib.util
spec = importlib.util.spec_from_file_location(
    "submission", Path("../submissions/008-reverting-channels/main.py")
)
submission = importlib.util.module_from_spec(spec)
# Register before executing: the module declares dataclasses, and resolving their
# fields looks the module up in sys.modules — without registration it fails with AttributeError.
sys.modules["submission"] = submission
spec.loader.exec_module(submission)

x = pd.read_parquet(WS / "data/X_test.reduced.parquet")
y = pd.read_parquet(WS / "data/y_test.reduced.parquet")
models = {
    "005": joblib.load("../../model005_backup.joblib")["booster"],    # 40 channels
    "006": joblib.load("../../model006.joblib")["booster"],           # 41, +CNN
    "008": joblib.load("../../resources008/model.joblib")["booster"], # 50, +reverting
}
model = models["008"]  # the best: sorting and metrics are computed on it
print(f"series: {x.index.get_level_values(0).nunique()}; channels per model:",
      {k: m.n_features_ for k, m in models.items()})""")

code("""# Run every series step by step and collect everything the figures need.
records = {}
for sid, part in x.groupby(level="id"):
    hist = part.loc[part.period == 1, "value"].to_numpy()
    online = part.loc[part.period == 2, "value"].to_numpy()
    if len(online) == 0:
        continue
    labels = y.loc[sid, "target"].to_numpy()

    # Online run — exactly as on the platform.
    monitor = submission.Monitor(hist)
    channels = np.asarray([monitor.update(float(v)) for v in online])
    # The first 40 columns are the 005 vector, the first 41 — 006: channels were only
    # ever appended at the end, so a single run feeds all the models.
    all_scores = {
        name: m.predict_proba(channels[:, : m.n_features_])[:, 1]
        for name, m in models.items()
    }
    scores = all_scores["008"]

    # Illustrative run over the history: the same detectors, steps -n..-1, the trend
    # is removed at its own point. The platform has no such run; the convolutional
    # channel is not recomputed here and sits at 0.5.
    playback = submission.Monitor(hist)
    playback._step = -len(hist)
    playback._previous_z = 0.0
    hist_channels = np.asarray([playback.update(float(v)) for v in hist])
    hist_scores = {
        name: m.predict_proba(hist_channels[:, : m.n_features_])[:, 1]
        for name, m in models.items()
    }

    norm = monitor.norm
    z_online = np.asarray(
        [norm.clip(norm.standardise(float(v), i)) for i, v in enumerate(online)]
    )
    z_hist = np.asarray(
        [norm.clip(norm.standardise(float(v), -len(hist) + i)) for i, v in enumerate(hist)]
    )
    tau = int(labels.argmax()) if labels.max() > 0 else None

    # Where the model itself "decided" a break happened: the first step at which its score
    # crossed half of its maximum on this series. A convention for the eye —
    # the metric does not ask for a point.
    detected_by = {}
    for name, sc in all_scores.items():
        peak = float(sc.max())
        crossed = np.flatnonzero(sc >= 0.5 * peak) if peak > 0 else []
        detected_by[name] = int(crossed[0]) if len(crossed) else None
    detected = detected_by["008"]

    # Alarm cancellation (only 008 can do it): the score drops below half of its
    # running maximum after the alarm had been raised in earnest.
    running = np.maximum.accumulate(scores)
    alarmed = running >= 0.6 * float(scores.max()) if scores.max() > 0 else running > 1
    below = scores < 0.5 * running
    cross_down = below & ~np.roll(below, 1) & alarmed
    cross_down[0] = False
    cancellations = np.flatnonzero(cross_down)

    records[int(sid)] = dict(
        hist=hist, online=online, z=z_online, z_hist=z_hist, scores=scores,
        channels=channels,
        labels=labels, tau=tau, detected=detected,
        all_scores=all_scores, hist_scores=hist_scores,
        detected_by=detected_by, cancellations=cancellations,
        slope=norm.slope, sd=norm.sd, rho=norm.rho, kurt=norm.kurtosis,
    )
print(f"series run: {len(records)}")""")

code("""# Per-series quality. For a series with a break: within-series AUC — does the score
# rank the steps after the break above the steps before. For a series without a break: the false-alarm
# level, taken as the final score (lower is better). The questions differ, so the galleries
# are sorted separately. Everything is computed on the best model (008).
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
# Drawdown depth: how far the 008 score can descend after a rise.
table["drawdown"] = [
    float((np.maximum.accumulate(records[sid]["scores"]) - records[sid]["scores"]).max())
    for sid in table.index
]
broken = table[table.broken].sort_values("quality", ascending=False)
clean = table[~table.broken].sort_values("false_alarm")
print(f"series with a break: {len(broken)}, without: {len(clean)}")
print(f"with a break: median within-series AUC {broken.quality.median():.3f}")
print(f"without a break: median final score {clean.false_alarm.median():.3f}")""")

md("""## How the competition metric works — and why it is not about "when it fired"

The metric (TS-AUC) compares **different series against each other at the same
step** — not a series against itself over time. At every step t the platform takes all
the series, splits them into "break already happened" and "no break yet", and asks: do
the former rank above the latter by score? That is the AUC of one step; the total is the mean over steps
(weighted by the number of compared pairs).

A toy example — one step, four series:

| series | break already happened? | our score |
|--------|-------------------------|-----------|
| A      | yes                     | 0.08      |
| B      | no                      | 0.03      |
| C      | no                      | 0.12      |
| D      | yes                     | 0.20      |

"Broken vs clean" pairs: A>B ✓, A>C ✗ (the clean C overtook the broken A!),
D>B ✓, D>C ✓ → step AUC = 3/4 = 0.75. We lost not because A "rose late
relative to itself", but because the jumpy clean C ranks above the honest
broken A. Two conclusions follow: score calibration does not matter (only the order),
and false alarms on clean series are a direct loss.

The within-series AUC from the galleries above is a different, diagnostic quantity: for
a single series the competition metric is undefined, like a "race position" for
a runner running alone. Below is the competition metric itself on our
hundred: per step and in total, for all three models.""")

code("""# The competition metric on the labelled hundred: the AUC of every step
# (broken vs clean at that step) and the total weighted by the number of pairs —
# exactly the same construction as on the platform and in our cross-validation.
def step_auc(scores_at_t, broken_at_t):
    pos = scores_at_t[broken_at_t]
    neg = scores_at_t[~broken_at_t]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan, 0
    ranks = np.concatenate([pos, neg]).argsort().argsort() + 1
    auc = (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return auc, len(pos) * len(neg)

max_t = max(len(r["online"]) for r in records.values())
fig, ax = plt.subplots(figsize=(11, 4.5))
totals = {}
colours = {"005": "#5f6368", "006": "#1a73e8", "008": "#7b1fa2"}
for name, colour in colours.items():
    aucs, weights = [], []
    for t_step in range(max_t):
        sc, br = [], []
        for r in records.values():
            if t_step < len(r["online"]):
                sc.append(r["all_scores"][name][t_step])
                br.append(r["tau"] is not None and t_step >= r["tau"])
        auc, w = step_auc(np.asarray(sc), np.asarray(br))
        aucs.append(auc); weights.append(w)
    aucs = np.asarray(aucs); weights = np.asarray(weights, dtype=float)
    ok = ~np.isnan(aucs)
    totals[name] = float((aucs[ok] * weights[ok]).sum() / weights[ok].sum())
    smooth = pd.Series(aucs).rolling(25, min_periods=1, center=True).mean()
    ax.plot(smooth, lw=1.4, color=colour,
            label=f"model {name}: total {totals[name]:.4f}")
ax.axhline(0.5, color="#d93025", lw=0.8, ls="--", label="0.5 — coin flip")
ax.set_xlabel("online-part step")
ax.set_ylabel("step AUC (smoothed, window 25)")
ax.set_title("The competition metric per step: exactly where the models earn and lose")
ax.legend(loc="lower right", fontsize=8, frameon=True)
plt.tight_layout(); plt.show()
print("total metric on the hundred (pair-weighted):",
      {k: round(v, 4) for k, v in totals.items()})""")

md("""## Where we earn and lose on the metric

A single series' contribution to the metric: at every step the series takes part in pairs against
the series from the opposite pile. A broken series "earns" when it ranks
above the clean ones; a clean one — when it ranks below the broken ones. The mean share of won
pairs over all steps is the series' earnings (0.5 is neutral, above it
feeds the metric, below it eats it). We compute it on the best model (008) and look at three
galleries: the best earners, the middle, the worst losses. Every figure shows all
three models — you can see which of them copes where.""")

code("""# Earnings of every series: the mean share of won pairs over steps.
earn = {sid: [] for sid in records}
for t_step in range(max(len(r["online"]) for r in records.values())):
    alive = [(sid, r) for sid, r in records.items() if t_step < len(r["online"])]
    broken_scores = [r["all_scores"]["008"][t_step] for _, r in alive
                     if r["tau"] is not None and t_step >= r["tau"]]
    clean_scores = [r["all_scores"]["008"][t_step] for _, r in alive
                    if not (r["tau"] is not None and t_step >= r["tau"])]
    if not broken_scores or not clean_scores:
        continue
    bs = np.asarray(broken_scores); cs = np.asarray(clean_scores)
    for sid, r in alive:
        sc = r["all_scores"]["008"][t_step]
        if r["tau"] is not None and t_step >= r["tau"]:
            earn[sid].append((sc > cs).mean() + 0.5 * (sc == cs).mean())
        else:
            earn[sid].append((sc < bs).mean() + 0.5 * (sc == bs).mean())
table["earn"] = [float(np.mean(earn[sid])) if earn[sid] else np.nan
                 for sid in table.index]
ranked = table.dropna(subset=["earn"]).sort_values("earn", ascending=False)
print("top earnings:", ranked.earn.head(3).round(3).to_dict())
print("worst losses:", ranked.earn.tail(3).round(3).to_dict())""")

code('''MODEL_STYLE = {
    "005": ("40 channels", "#5f6368"),
    "006": ("41 channels, +convolutional", "#1a73e8"),
    "008": ("50 channels, +reverting — BEST", "#7b1fa2"),
}

def show(sid, title_extra=""):
    r = records[sid]
    hist, online, z, scores, tau = r["hist"], r["online"], r["z"], r["scores"], r["tau"]
    z_hist, detected = r["z_hist"], r["detected"]
    all_scores, hist_scores = r["all_scores"], r["hist_scores"]
    detected_by, cancellations = r["detected_by"], r["cancellations"]
    n_h = len(hist)
    ch = r["channels"]
    fig, axes = plt.subplots(6, 1, figsize=(11, 11), sharex=True,
                             gridspec_kw=dict(height_ratios=[2, 1.2, 0.75, 0.75, 0.95, 1.5]))
    t_hist = np.arange(-n_h, 0)
    t_on = np.arange(len(online))

    axes[0].plot(t_hist[-600:], hist[-600:], lw=0.6, color="#9aa0a6",
                 label="history (no break by construction)")
    axes[0].plot(t_on, online, lw=0.8, color="#1a73e8",
                 label="online part (arrives one point at a time)")
    axes[0].set_ylabel("raw series")

    axes[1].plot(t_hist[-600:], z_hist[-600:], lw=0.6, color="#9aa0a6",
                 label="history after normalisation (for scale comparison)")
    axes[1].plot(t_on, z, lw=0.8, color="#188038",
                 label="online after normalisation: trend removed, scale removed, outliers clipped")
    axes[1].axhline(0, color="grey", lw=0.5)
    axes[1].set_ylabel("normalised")

    # One narrow panel per model: the history score dotted, online solid,
    # the dot — where this model fired, for 008 — the alarm-cancellation triangles.
    for k, (name, (label, colour)) in enumerate(MODEL_STYLE.items()):
        ax = axes[2 + k]
        sc, hs = all_scores[name], hist_scores[name]
        ax.plot(t_hist[-600:], hs[-600:], lw=0.8, color=colour, ls=":", alpha=0.7,
                label="on the history (illustrative)")
        ax.plot(t_on, sc, lw=1.3, color=colour, label=f"model {name}: {label}")
        if detected_by[name] is not None:
            d = detected_by[name]
            ax.plot(d, sc[d], "o", ms=6, color=colour, zorder=5,
                    label="the model fired here")
            ax.axvline(d, color=colour, lw=0.8, alpha=0.35)
        if name == "008" and len(cancellations):
            ax.plot(cancellations, sc[cancellations], "v", ms=8,
                    color="#f9ab00", zorder=6, label="alarm CANCELLATION")
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel(name)

    comp = [
        ("CUSUM (level shift, max over 3 kinds)", ch[:, [0, 3, 6]].max(axis=1), "#1a73e8"),
        ("Page-Hinkley (slow drift)", ch[:, [1, 4, 7]].max(axis=1), "#188038"),
        ("Variance-ratio (change in spread)", ch[:, [2, 5, 8]].max(axis=1), "#d93025"),
        ("Multiscale: current divergence", ch[:, 9:21].max(axis=1), "#f9ab00"),
        ("Multiscale: all-time peak", ch[:, 21:33].max(axis=1), "#9334e6"),
        ("Retro-scan of the best split", ch[:, 33], "#5f6368"),
    ]
    for name, series_c, colour in comp:
        axes[5].plot(t_on, series_c, lw=1.0, color=colour, label=name, alpha=0.9)
    axes[5].set_ylim(-0.02, 1.02)
    axes[5].set_ylabel("components")
    axes[5].set_xlabel("online-part step (history at negative t)")

    for k, ax in enumerate(axes):
        ax.axvline(0, color="grey", lw=1.0, alpha=0.6,
                   label="history/online boundary" if k == 0 else None)
        if tau is not None:
            ax.axvline(tau, color="#d93025", lw=1.2, ls="--",
                       label="TRUE break (labels)" if k == 0 else None)
        ax.legend(loc="upper left", fontsize=7, ncols=2 if k in (0, 1, 5) else 3,
                  frameon=True, framealpha=0.85)
    q = table.loc[sid]
    status = f"break at step {tau}" if tau is not None else "no break"
    metric = (f"within-series AUC {q.quality:.3f}" if tau is not None
              else f"final score {q.false_alarm:.3f}")
    fig.suptitle(f"series {sid} — {status} — {metric}{title_extra}", y=0.995)
    plt.tight_layout()
    plt.show()''')

md("""## Series with a break: best, middle, worst

The red dashed line is the true break. The question for every figure: does the score rise
*at* the red line, later — or not at all.""")

code("""ids = list(broken.index)
for sid in ids[:3]:
    show(sid, "  (best)")""")
code("""mid = len(ids) // 2
for sid in ids[mid - 1 : mid + 2]:
    show(sid, "  (middle)")""")
code("""for sid in ids[-3:]:
    show(sid, "  (worst — this is where the model loses the most)")""")

md("""## Series without a break: false alarms

There is no red line; everything the score does here is an error. The three clean
series with the highest final score are the model's worst false alarms. This is also where
the main skill of 008 shows: an alarm raised by mistake can be cancelled.""")

code("""for sid in list(clean.index)[-3:]:
    show(sid, "  (false alarm)")""")

md("""## Alarm cancellation in action

Series where the score of the best model (008) rose — and came back down: "ah no,
that was not a break". The old models (005/006) cannot do this: their channels remember
only the peak of suspicion, and on the same panels you can see their score getting stuck
at the top. This very difference gave +0.006 on cross-validation.""")

code("""# The deepest drawdown of the 008 score — three series without a break and, for contrast,
# one with a break: a drawdown before the real break does not prevent firing afterwards.
deep_clean = table[~table.broken].sort_values("drawdown", ascending=False)
for sid in list(deep_clean.index)[:3]:
    show(sid, "  (alarm cancellation)")
deep_broken = table[table.broken].sort_values("drawdown", ascending=False)
for sid in list(deep_broken.index)[:1]:
    show(sid, "  (drawdown, then a real break)")""")

md("""## Galleries by earnings: best, middle, losses

The caption of every figure is the earnings: the share of pairs this series won over all
steps. For the "losses", look at which model is to blame: if the 005/006 score got stuck
at the top on a clean series while 008 came down — that is alarm cancellation in action;
if all three are high — the series fools the preprocessing itself, and that is a task for
the next experiment.""")

code("""ids_e = list(ranked.index)
for sid in ids_e[:3]:
    show(sid, f"  (feeds the metric: earnings {ranked.earn[sid]:.3f})")""")
code("""mid_e = len(ids_e) // 2
for sid in ids_e[mid_e - 1 : mid_e + 2]:
    show(sid, f"  (middle: earnings {ranked.earn[sid]:.3f})")""")
code("""for sid in ids_e[-3:]:
    show(sid, f"  (eats the metric: earnings {ranked.earn[sid]:.3f})")""")

md("""## The point metric: did we catch the break, and when

The firing threshold is not picked out of thin air but taken from the clean series: it is the score level
that 90% of the clean series never cross. Crossed it — "the model declared a
break". Then for every series with a break we compute the delay: the declaration step
minus the true break. Categories:

- **false start** — declared before the true break;
- **on time** — within the first 10 steps after the break;
- **late** — later than 10 steps;
- **missed** — the threshold was not crossed before the end of the series.

Below — the relation between delay and earnings: how much the competition metric actually
pays for each category.""")

code("""threshold = float(np.quantile(
    [r["scores"].max() for r in records.values() if r["tau"] is None], 0.90))
print(f"firing threshold (held by 90% of the clean series): {threshold:.3f}")

def call_step(scores):
    hit = np.flatnonzero(scores >= threshold)
    return int(hit[0]) if len(hit) else None

cats, delays = {}, {}
for sid, r in records.items():
    if r["tau"] is None:
        continue
    d = call_step(r["scores"])
    if d is None:
        cats[sid] = "missed"; delays[sid] = np.nan
    elif d < r["tau"]:
        cats[sid] = "false start"; delays[sid] = d - r["tau"]
    elif d - r["tau"] <= 10:
        cats[sid] = "on time"; delays[sid] = d - r["tau"]
    else:
        cats[sid] = "late"; delays[sid] = d - r["tau"]
table["category"] = pd.Series(cats)
table["delay"] = pd.Series(delays)
b = table[table.broken]
summary = b.groupby("category").agg(
    series=("category", "size"),
    mean_delay=("delay", "mean"),
    mean_earnings=("earn", "mean"),
).round(2)
display(summary)

fig, ax = plt.subplots(figsize=(8, 4))
ok = b.dropna(subset=["delay", "earn"])
ax.scatter(ok.delay, ok.earn, s=28, color="#7b1fa2", alpha=0.75)
ax.axvline(0, color="grey", lw=0.8)
ax.axhline(0.5, color="#d93025", lw=0.8, ls="--", label="0.5 — neutral for the metric")
ax.set_xlabel("declaration delay, steps (negative — false start)")
ax.set_ylabel("series earnings in pairs")
ax.set_title("The later the break is declared, the less the series brings to the metric")
ax.legend(fontsize=8, frameon=False)
plt.tight_layout(); plt.show()
print("correlation of delay and earnings:",
      round(ok.delay.corr(ok.earn), 3))""")

md("""### Cases by category""")

code("""for cat in ["on time", "late", "false start", "missed"]:
    ids_c = list(b[b.category == cat].sort_values("delay").index)
    for sid in ids_c[:2]:
        d = table.delay[sid]
        extra = f"  ({cat}" + (f", delay {d:+.0f} steps)" if pd.notna(d) else ")")
        show(sid, extra)""")

md("""## Which series are hard

Median within-series AUC (over series with a break), with every axis split into
halves at the median. A large gap between the bars is a ready-made hypothesis;
equal bars mean this axis does not matter.""")

code("""splits = {}
b = table[table.broken]
for col, name in [("trend", "trend strength"), ("sd", "spread"),
                  ("rho", "autocorrelation"), ("kurt", "tail heaviness"),
                  ("tau", "break position"), ("n_online", "online-part length")]:
    m = b[col].median()
    low, high = b[b[col] <= m], b[b[col] > m]
    splits[name] = (low.quality.median(), high.quality.median(), m)
frame = pd.DataFrame(splits, index=["lower half", "upper half", "threshold (median)"]).T
display(frame.round(3))

fig, ax = plt.subplots(figsize=(9, 4))
xpos = np.arange(len(frame))
ax.bar(xpos - 0.18, frame["lower half"], width=0.36,
       label="lower half", color="#9aa0a6")
ax.bar(xpos + 0.18, frame["upper half"], width=0.36,
       label="upper half", color="#1a73e8")
ax.set_xticks(xpos, frame.index, rotation=20)
ax.set_ylabel("median within-series AUC; higher = better")
ax.axhline(0.5, color="#d93025", lw=0.8, ls="--")
ax.legend(frameon=False)
ax.set_title("Where the detector is strong and where it is blind")
plt.tight_layout(); plt.show()""")

md("""## What to read from this next

The bars above are a hypothesis generator: every axis with a visible gap is a question of
which channel will close it; every figure among the "worst" is a specific series worth
thinking about. The notebook is rebuilt by the script
`scripts/build_analysis_notebook.py`; edits go only there.""")


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
    print(f"executed and written -> {TARGET}" if execute else f"written -> {TARGET}")


if __name__ == "__main__":
    main()
