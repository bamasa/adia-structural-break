"""Evaluate the built submission on the platform's local test sample (100 labelled series)."""
import importlib.util, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, "repo/src")
from structural_break.combiners import ts_auc
sub_path, res_dir = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("sub", sub_path)
sub = importlib.util.module_from_spec(spec); sys.modules["sub"] = sub; spec.loader.exec_module(sub)
D = "structural-break-real-time-test/data/"
xte = pd.read_parquet(D + "X_test.reduced.parquet"); yte = pd.read_parquet(D + "y_test.reduced.parquet")
datasets, keys = [], []
for sid, part in xte.groupby(level="id"):
    hist = part.loc[part.period == 1, "value"].to_numpy("float64")
    online = part.loc[part.period == 2, "value"].to_numpy("float64")
    datasets.append((hist, online)); keys.append((int(sid), len(online)))
t0 = time.time()
gen = sub.infer(datasets, res_dir); next(gen)
scores, labels, steps = [], [], []
for (sid, n) in keys:
    lab = yte.loc[sid, "target"].to_numpy()
    assert len(lab) == n, (sid, len(lab), n)
    for t in range(n):
        scores.append(next(gen)); labels.append(int(lab[t])); steps.append(t)
scores, labels, steps = map(np.asarray, (scores, labels, steps))
print(f"{sub_path}: TS-AUC on test.reduced (100 series) = {ts_auc(scores, labels, steps):.4f}  [{time.time()-t0:.0f}s, {len(scores)} steps]", flush=True)
np.save(f"reduced_scores_{res_dir.rstrip('/').split('/')[-1]}.npy", scores)
