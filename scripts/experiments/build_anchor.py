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
print(f"загрузка {time.time()-t0:.0f}s", flush=True)

rows, count = [], 0
for sid, part in x.groupby(level="id"):
    online = part.loc[part.period == 2, "value"].to_numpy()
    if len(online) == 0:
        continue
    # Растущий якорь, полностью причинный: монитор создаётся на шаге 10 из
    # первых десяти точек и пересоздаётся на шаге 40 из первых сорока.
    zeros = [0.5] * 50
    m = None
    for i in range(len(online)):
        if i in (10, ANCHOR):
            m = sub.Monitor(online[:i])
        rows.append(m.update(float(online[i])) if m is not None else list(zeros))
    count += 1
    if count % 2000 == 0:
        print(f"  {count} рядов, {time.time()-t0:.0f}s", flush=True)

a = np.asarray(rows, dtype="float32")
np.save("A50.npy", a)
print(f"готово {time.time()-t0:.0f}s: {a.shape}", flush=True)
