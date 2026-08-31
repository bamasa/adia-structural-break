"""065: аугментация обучающего набора — перенос границы истории вправо."""
import sys, time
sys.path.insert(0, "structural-break-real-time-test")
sys.path.insert(0, "repo/src")
import importlib.util
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("sub", "repo/submissions/053-spectral/main.py")
sub = importlib.util.module_from_spec(spec)
sys.modules["sub"] = sub
spec.loader.exec_module(sub)
import lightgbm as lgb, joblib
base_fc = lgb.Booster(model_str=joblib.load("resources053/model.joblib")["forecaster"])

t0 = time.time()
x = pd.read_parquet("structural-break-real-time-test/data/X_train.parquet")
yl = pd.read_parquet("structural-break-real-time-test/data/y_train.parquet")
print(f"загрузка {time.time()-t0:.0f}s", flush=True)

rng = np.random.default_rng(1)
rows, labels, groups, steps = [], [], [], []
count = 0
for sid, part in x.groupby(level="id"):
    hist = part.loc[part.period == 1, "value"].to_numpy()
    online = part.loc[part.period == 2, "value"].to_numpy()
    if len(online) < 60:
        continue
    lab = yl.loc[sid, "target"].to_numpy()
    tau = int(lab.argmax()) if lab.max() > 0 else None
    # Сдвигаем границу вправо: k из первой трети «чистого» участка.
    limit = tau if tau is not None else len(online)
    if limit < 30:
        continue
    hi = max(11, min(limit, len(online) // 2))
    cuts = sorted({int(c) for c in rng.integers(10, hi, size=3)})
    for j, k in enumerate(cuts):
        h2 = np.concatenate([hist, online[:k]])
        o2 = online[k:]
        if len(o2) < 20:
            continue
        lab2 = lab[k:]
        mon = sub.TriMonitor(h2, base_fc)
        for i, v in enumerate(o2):
            rows.append(mon.update(float(v)))
            labels.append(int(lab2[i]))
            groups.append(100000 + int(sid) * 10 + j)
            steps.append(i)
    count += 1
    if count % 1000 == 0:
        print(f"  {count} рядов, {time.time()-t0:.0f}s", flush=True)

np.save("AUG3_X.npy", np.asarray(rows, dtype="float32"))
np.save("AUG3_Y.npy", np.asarray(labels, dtype="int8"))
np.save("AUG3_G.npy", np.asarray(groups, dtype="int64"))
np.save("AUG3_S.npy", np.asarray(steps, dtype="int32"))
print(f"готово {time.time()-t0:.0f}s: {len(rows)} строк из {count} рядов", flush=True)
