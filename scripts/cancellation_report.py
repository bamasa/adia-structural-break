"""Breakdown of model 008 alarm cancellations on the labelled hundred: three groups of cases."""
import sys
sys.path.insert(0, "structural-break-real-time-test")
import importlib.util
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

spec = importlib.util.spec_from_file_location("sub", "repo/submissions/008-reverting-channels/main.py")
sub = importlib.util.module_from_spec(spec)
sys.modules["sub"] = sub
spec.loader.exec_module(sub)
model = joblib.load("resources008/model.joblib")["booster"]

x = pd.read_parquet("structural-break-real-time-test/data/X_test.reduced.parquet")
y = pd.read_parquet("structural-break-real-time-test/data/y_test.reduced.parquet")

recs = {}
for sid, part in x.groupby(level="id"):
    hist = part.loc[part.period == 1, "value"].to_numpy()
    online = part.loc[part.period == 2, "value"].to_numpy()
    if len(online) == 0: continue
    m = sub.Monitor(hist)
    ch = np.asarray([m.update(float(v)) for v in online])
    sc = model.predict_proba(ch)[:, 1]
    lab = y.loc[sid, "target"].to_numpy()
    tau = int(lab.argmax()) if lab.max() > 0 else None
    recs[int(sid)] = dict(online=online, sc=sc, tau=tau)

# Cancellation events: the score dropped below half of the running maximum after
# the alarm had been raised in earnest (peak >= 0.3 in absolute terms).
dd_clean = [1 - (r["sc"][-1] / np.maximum.accumulate(r["sc"]).max())
            for r in recs.values() if r["tau"] is None and r["sc"].max() > 0.2]
print(f"clean series with peak>0.2: {len(dd_clean)}; median descent from peak to final: {np.median(dd_clean):.0%}")
cases = {"false_cancel": [], "bad_cancel": [], "good_cancel": []}
for sid, r in recs.items():
    sc, tau = r["sc"], r["tau"]
    run = np.maximum.accumulate(sc)
    alarmed = run >= 0.20
    below = sc < 0.7 * run
    ev = np.flatnonzero(below & ~np.roll(below, 1) & alarmed)
    ev = ev[ev > 0]
    if not len(ev): continue
    t0 = int(ev[0])
    peak_before = float(run[t0])
    peak_step = int(np.argmax(sc[:t0]))
    speed = t0 - peak_step
    after = sc[t0:]
    if tau is not None and t0 >= tau:
        # cancelled after a real break — a false cancellation
        loss = peak_before - float(after.min())
        cases["false_cancel"].append((loss, sid, t0))
    elif tau is None:
        rebound = float(after.max()) / peak_before if peak_before > 0 else 1.0
        final = float(sc[-1]) / peak_before if peak_before > 0 else 1.0
        if speed <= 60 and final < 0.75 and rebound < 0.98:
            cases["good_cancel"].append((-speed, sid, t0))
        else:
            cases["bad_cancel"].append((max(final, rebound), sid, t0))
    else:
        # cancelled a false alarm BEFORE the real break — check for re-triggering
        re_max = float(sc[tau:].max())
        if re_max >= 0.8 * peak_before:
            cases["good_cancel"].append((-1000 - re_max, sid, t0))  # priority: with re-arm
        else:
            cases["false_cancel"].append((peak_before - re_max, sid, t0))

TITLES = {
    "false_cancel": "FALSE cancellations: cancelled, but the break is real (or did not rise after it)",
    "bad_cancel": "BAD cancellations: the alarm is false, but the cancellation is slow / incomplete / the score came back",
    "good_cancel": "GOOD cancellations: fast descent; if a real break followed — triggered again",
}
for key, items in cases.items():
    items.sort(reverse=True)
    picked = items[:5]
    if not picked:
        print(key, ": no cases"); continue
    fig, axes = plt.subplots(len(picked), 1, figsize=(11, 2.3 * len(picked)), squeeze=False)
    for ax_row, (_, sid, t0) in zip(axes, picked):
        ax = ax_row[0]
        r = recs[sid]; sc, tau, online = r["sc"], r["tau"], r["online"]
        t = np.arange(len(sc))
        raw = (online - online.min()) / (np.ptp(online) + 1e-9)
        ax.plot(t, raw, lw=0.5, color="#b5d4f4", alpha=0.8, label="raw series (squashed to 0..1)")
        ax.plot(t, sc, lw=1.4, color="#7b1fa2", label="score 008")
        ax.plot(t, np.maximum.accumulate(sc), lw=0.8, ls="--", color="#9aa0a6", label="running maximum")
        run = np.maximum.accumulate(sc); below = sc < 0.7 * run
        ev = np.flatnonzero(below & ~np.roll(below, 1) & (run >= 0.20)); ev = ev[ev > 0]
        ax.plot(ev, sc[ev], "v", ms=8, color="#f9ab00", label="cancellation")
        if tau is not None:
            ax.axvline(tau, color="#d93025", lw=1.2, ls="--", label="TRUE break")
        ax.set_ylim(-0.02, 1.02); ax.set_ylabel(f"series {sid}")
        ax.legend(loc="upper left", fontsize=6, ncols=3, framealpha=0.8)
    axes[0][0].set_title(TITLES[key], fontsize=11, loc="left")
    axes[-1][0].set_xlabel("online-part step")
    plt.tight_layout()
    fig.savefig(f"/tmp/cancel_{key}.png", dpi=120)
    print(key, "->", len(picked), "cases:", [sid for _, sid, _ in picked])
