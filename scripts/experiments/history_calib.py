"""051: score calibration by the series history — subtract the "normal alarm level"."""
import sys, time
sys.path.insert(0, "structural-break-real-time-test")
sys.path.insert(0, "repo/src")
import importlib.util
import numpy as np
import pandas as pd
import joblib
from structural_break.combiners import split_by_series, ts_auc

spec = importlib.util.spec_from_file_location("sub", "repo/submissions/039-fold-ensemble/main.py")
sub = importlib.util.module_from_spec(spec)
sys.modules["sub"] = sub
spec.loader.exec_module(sub)

t0 = time.time()
art = joblib.load("resources039b/model.joblib")
clf_b = art["booster"].booster_
rank_bs = [r.booster_ for r in art["rankers"]]
import lightgbm as lgb
base_fc = lgb.Booster(model_str=art["forecaster"])

y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
assignment = split_by_series(g, folds=5, seed=0)
starts = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]]))
fold0_sids = set(int(g[a]) for a in starts if int(assignment[a]) == 0)

x = pd.read_parquet("structural-break-real-time-test/data/X_train.parquet")
hist_levels, online_rows, count = {}, {}, 0
for sid, part in x.groupby(level="id"):
    if int(sid) not in fold0_sids:
        continue
    hist = part.loc[part.period == 1, "value"].to_numpy()
    online = part.loc[part.period == 2, "value"].to_numpy()
    if len(online) == 0:
        continue
    # Pass over the history: the same monitor, steps -n..-1 (no break there by construction).
    pb = sub.TriMonitor(hist, base_fc)
    pb._step = -len(hist)
    pb.dual.raw._step = -len(hist)
    pb.dual.compressed._step = -len(hist)
    hch = []
    take = max(len(hist) // 3, 1)          # tail of the history — closer to the boundary
    for i, v in enumerate(hist):
        row = pb.update(float(v))
        if i >= len(hist) - take:
            hch.append(row)
    hch = np.asarray(hch)
    hs = np.mean([1/(1+np.exp(-rb.predict(hch, num_threads=8))) for rb in rank_bs], axis=0)
    hist_levels[int(sid)] = (float(np.median(hs)), float(np.quantile(hs, 0.9)))
    count += 1
    if count % 400 == 0:
        print(f"  {count} series [{time.time()-t0:.0f}s]", flush=True)

np.save("hist_levels_fold0.npy", np.array(
    [[k, v[0], v[1]] for k, v in hist_levels.items()], dtype="float64"))
print(f"history pass done: {count} series [{time.time()-t0:.0f}s]", flush=True)

# Apply the calibration to the saved online scores of fold 0.
m0 = assignment == 0
yf, sf, gf = y[m0], s[m0], g[m0]
rnk = 1.0/(1.0+np.exp(-np.load("oof_rank.npy")[m0].astype("float64")))
clf = np.load("oof_cfg5.npy")[m0].astype("float64")
blend = 0.6*rnk + 0.4*clf
print(f"pair reference: {ts_auc(blend, yf, sf):.4f}", flush=True)
med = np.array([hist_levels.get(int(sid), (0.5, 0.5))[0] for sid in gf])
q90 = np.array([hist_levels.get(int(sid), (0.5, 0.5))[1] for sid in gf])
for name, adj in (("minus history median", blend - med),
                  ("minus 0.5*median", blend - 0.5*med),
                  ("minus history q90", blend - q90),
                  ("divide by median", blend / np.maximum(med, 1e-3))):
    print(f"{name}: {ts_auc(adj, yf, sf):.4f}", flush=True)
print(f"total {time.time()-t0:.0f}s", flush=True)
