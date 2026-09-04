"""Verify the assembled #073 against the training matrices and profile it."""
import importlib.util, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, "repo/src")
from structural_break.combiners import split_by_series
spec = importlib.util.spec_from_file_location("sub", "repo/submissions/074-aug3-full/main.py")
sub = importlib.util.module_from_spec(spec); sys.modules["sub"] = sub; spec.loader.exec_module(sub)
X = np.hstack([np.load("X40.npy"), np.load("C_cnn.npy")[:, None], np.load("N9.npy").astype("float32"),
               np.load("X50a.npy"), np.load("E4.npy"), np.load("B40.npy"), np.load("B2.npy"),
               np.load("SPEC14.npy")]).astype("float64")
g = np.load("G40.npy"); s = np.load("S40.npy")
xtr = pd.read_parquet("structural-break-real-time-test/data/X_train.parquet")
sid = 7
ser = xtr.loc[sid]
hist = ser[ser.period == 1].value.to_numpy(); online = ser[ser.period == 2].value.to_numpy()
rows = X[g == sid]; steps = s[g == sid]
gen = sub.infer([(hist, online)], "resources074")
next(gen)
t0 = time.time()
scores = [next(gen) for _ in range(len(online))]
dt = (time.time() - t0) / len(online) * 1000
print(f"series {sid}: {len(online)} steps, {dt:.2f} ms/step, score range {min(scores):.3f}-{max(scores):.3f}")
# Channels: recompute through the assembled monitor and compare row by row.
import joblib, lightgbm as lgb
model = joblib.load("resources074/model.joblib")
mon = sub.TriMonitor(hist.astype("float64"), lgb.Booster(model_str=model["forecaster"]))
chans = np.array([mon.update(float(p)) for p in online])
order = np.argsort(steps)
diff = np.abs(chans[:len(order)] - rows[order][:len(chans)]).max()
print(f"channel mismatch vs training matrix: {diff:.2e}  (nets {len(model['nets'])}, rankers {len(model['rankers'])})")
assert diff < 1e-4, "CHANNELS DISAGREE WITH THE TRAINING MATRIX"
assert dt < 6, "TOO SLOW"
print("OK")
