"""#075: channels vs matrices (200 + 6 BOCPD), speed, dry run."""
import importlib.util, sys, time
import numpy as np, pandas as pd
spec = importlib.util.spec_from_file_location("sub", "repo/submissions/082-mass-pool24/main.py")
sub = importlib.util.module_from_spec(spec); sys.modules["sub"] = sub; spec.loader.exec_module(sub)
X = np.hstack([np.load("X40.npy"), np.load("C_cnn.npy")[:, None], np.load("N9.npy").astype("float32"),
               np.load("X50a.npy"), np.load("E4.npy"), np.load("B40.npy"), np.load("B2.npy"),
               np.load("SPEC14.npy"), np.load("BOCPD6_H50.npy"), np.load("MASS90.npy")]).astype("float64")
g = np.load("G40.npy"); s = np.load("S40.npy")
xtr = pd.read_parquet("structural-break-real-time-test/data/X_train.parquet")
import joblib, lightgbm as lgb
model = joblib.load("resources082/model.joblib")
for sid in (7, 4242):
    ser = xtr.loc[sid]
    hist = ser[ser.period == 1].value.to_numpy(); online = ser[ser.period == 2].value.to_numpy()
    rows = X[g == sid][np.argsort(s[g == sid])]
    gen = sub.infer([(hist, online)], "resources082"); next(gen)
    t0 = time.time(); scores = [next(gen) for _ in range(len(online))]
    dt = (time.time() - t0) / len(online) * 1000
    mon = sub.TriMonitor(hist.astype("float64"), lgb.Booster(model_str=model["forecaster"]))
    chans = np.array([mon.update(float(p)) for p in online])
    d200 = np.abs(chans[:, :200] - rows[:, :200]).max(); d6 = np.abs(chans[:, 200:206] - rows[:, 200:206]).max(); dm = np.abs(chans[:, 206:] - rows[:, 206:]).max()
    print(f"series {sid}: {len(online)} steps, {dt:.2f} ms/step, score {min(scores):.3f}-{max(scores):.3f}; "
          f"mismatch 200: {d200:.1e}, BOCPD: {d6:.1e}, MASS: {dm:.1e}", flush=True)
    assert d200 < 1e-4 and d6 < 1e-4 and dm < 1e-4, "CHANNELS DISAGREE"
    assert dt < 6, "TOO SLOW"
print(f"OK (nets {len(model['nets'])}, rankers {len(model['rankers'])}, clf {model['booster'].booster_.num_feature()})")
