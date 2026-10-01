"""Fold-1 gallery: the worst earners by earnings (ranker OOF)."""
import sys
sys.path.insert(0, "repo/src")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from structural_break.combiners import split_by_series

y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
oof = np.load("oof_rank.npy").astype("float64")
assignment = split_by_series(g, folds=5, seed=0)
m1 = assignment == 1
yy, ss, gg, pp = y[m1], s[m1], g[m1], oof[m1]

# Series earnings: share of won pairs across steps (within the fold).
earn = {}
for t_step in np.unique(ss):
    mt = ss == t_step
    lab, sc, sid = yy[mt], pp[mt], gg[mt]
    b, c = sc[lab == 1], sc[lab == 0]
    if not len(b) or not len(c):
        continue
    for v, l, sd in zip(sc, lab, sid):
        opp = c if l else b
        win = (v > opp).mean() + 0.5 * (v == opp).mean() if l else (v < opp).mean() + 0.5 * (v == opp).mean()
        earn.setdefault(int(sd), []).append(win)
earn = {k: float(np.mean(v)) for k, v in earn.items()}
worst = sorted(earn, key=earn.get)[:6]
print("worst fold-1 series:", [(sid, round(earn[sid], 3)) for sid in worst])

x = pd.read_parquet("structural-break-real-time-test/data/X_train.parquet")
starts = np.flatnonzero(np.concatenate([[True], gg[1:] != gg[:-1]]))
bounds = np.append(starts, len(gg))
row_of = {int(gg[a]): (a, b) for a, b in zip(starts, bounds[1:])}

fig, axes = plt.subplots(6, 2, figsize=(13, 15), gridspec_kw=dict(width_ratios=[2, 1]))
for row, sid in enumerate(worst):
    part = x.loc[sid]
    hist = part.loc[part.period == 1, "value"].to_numpy()
    online = part.loc[part.period == 2, "value"].to_numpy()
    a, b = row_of[sid]
    lab = yy[a:b]; sc = pp[a:b]
    tau = int(lab.argmax()) if lab.max() else None
    t_on = np.arange(len(online))
    axL, axR = axes[row]
    axL.plot(np.arange(-min(len(hist), 300), 0), hist[-300:], lw=0.5, color="#9aa0a6")
    axL.plot(t_on, online, lw=0.7, color="#1a73e8")
    if tau is not None:
        axL.axvline(tau, color="#d93025", ls="--", lw=1.2)
    axL.set_ylabel(f"series {sid}\nearnings {earn[sid]:.2f}")
    axR.plot(t_on, 1/(1+np.exp(-sc)), lw=1.1, color="#7b1fa2")
    if tau is not None:
        axR.axvline(tau, color="#d93025", ls="--", lw=1.2)
    axR.set_ylim(0, 1)
    if row == 0:
        axL.set_title("series (grey — history tail, red — true break)", fontsize=9)
        axR.set_title("ranker score (sigmoid)", fontsize=9)
plt.tight_layout()
fig.savefig("/tmp/fold1_worst.png", dpi=115)
print("saved /tmp/fold1_worst.png")
