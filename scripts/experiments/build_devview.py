"""014: the full 50-channel pipeline applied to the deviation-from-trailing-mean series."""
import sys, time
sys.path.insert(0, "structural-break-real-time-test")
import importlib.util
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("sub", "repo/submissions/008-reverting-channels/main.py")
sub = importlib.util.module_from_spec(spec)
sys.modules["sub"] = sub
spec.loader.exec_module(sub)

K = 10
t0 = time.time()
x = pd.read_parquet("structural-break-real-time-test/data/X_train.parquet")
print(f"загрузка {time.time()-t0:.0f}s", flush=True)

def deviations(v):
    v = np.asarray(v, dtype="float64")
    csum = np.concatenate([[0.0], np.cumsum(v)])
    out = np.full(len(v), np.nan)
    for i in range(K, len(v)):
        out[i] = v[i] - (csum[i] - csum[i - K]) / K
    return out

rows, count = [], 0
for sid, part in x.groupby(level="id"):
    hist = part.loc[part.period == 1, "value"].to_numpy()
    online = part.loc[part.period == 2, "value"].to_numpy()
    if len(online) == 0:
        continue
    d = deviations(np.concatenate([hist, online]))
    n_h = len(hist)
    m = sub.Monitor(d[K:n_h])
    for v in d[n_h:]:
        rows.append(m.update(float(v)))
    count += 1
    if count % 2000 == 0:
        print(f"  {count} рядов, {time.time()-t0:.0f}s", flush=True)

a = np.asarray(rows, dtype="float32")
np.save("X50d.npy", a)
print(f"готово {time.time()-t0:.0f}s: {a.shape}", flush=True)
