"""031: the online-anchor view — the 50-channel pipeline referenced to the
online segment's own first 40 points instead of the history."""
import sys, time
sys.path.insert(0, "structural-break-real-time-test")
import importlib.util
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("sub", "repo/submissions/008-reverting-channels/main.py")
sub = importlib.util.module_from_spec(spec)
sys.modules["sub"] = sub
spec.loader.exec_module(sub)

ANCHOR = 40
t0 = time.time()
x = pd.read_parquet("structural-break-real-time-test/data/X_train.parquet")
print(f"loading {time.time()-t0:.0f}s", flush=True)

rows, count = [], 0
for sid, part in x.groupby(level="id"):
    online = part.loc[part.period == 2, "value"].to_numpy()
    if len(online) == 0:
        continue
    # Growing anchor, fully causal: the monitor is created at step 10 from
    # the first ten points and re-created at step 40 from the first forty.
    zeros = [0.5] * 50
    m = None
    for i in range(len(online)):
        if i in (10, ANCHOR):
            m = sub.Monitor(online[:i])
        rows.append(m.update(float(online[i])) if m is not None else list(zeros))
    count += 1
    if count % 2000 == 0:
        print(f"  {count} series, {time.time()-t0:.0f}s", flush=True)

a = np.asarray(rows, dtype="float32")
np.save("A50.npy", a)
print(f"done {time.time()-t0:.0f}s: {a.shape}", flush=True)
