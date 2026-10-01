"""#086: channels vs matrices (206 + 90 + 100 + 20 + 90 whitened), speed, dry run."""
import importlib.util, sys, time
import numpy as np, pandas as pd
spec = importlib.util.spec_from_file_location("sub", "repo/submissions/086-whitened-member/main.py")
sub = importlib.util.module_from_spec(spec); sys.modules["sub"] = sub; spec.loader.exec_module(sub)
X = np.hstack([np.load("X40.npy"), np.load("C_cnn.npy")[:, None], np.load("N9.npy").astype("float32"),
               np.load("X50a.npy"), np.load("E4.npy"), np.load("B40.npy"), np.load("B2.npy"),
               np.load("SPEC14.npy"), np.load("BOCPD6_H50.npy"), np.load("MASS90.npy"), np.load("MSPEC32.npy"), np.load("MSPEC60.npy"), np.load("ACF8.npy"), np.load("KNN20.npy"), np.load("WHITE90.npy")]).astype("float64")
g = np.load("G40.npy"); s = np.load("S40.npy")
xtr = pd.read_parquet("structural-break-real-time-test/data/X_train.parquet")
import joblib, lightgbm as lgb
model = joblib.load("resources086/model.joblib")
for sid in (7, 4242):
    ser = xtr.loc[sid]
    hist = ser[ser.period == 1].value.to_numpy(); online = ser[ser.period == 2].value.to_numpy()
    rows = X[g == sid][np.argsort(s[g == sid])]
    gen = sub.infer([(hist, online)], "resources086"); next(gen)
    t0 = time.time(); scores = [next(gen) for _ in range(len(online))]
    dt = (time.time() - t0) / len(online) * 1000
    mon = sub.TriMonitor(hist.astype("float64"), lgb.Booster(model_str=model["forecaster"]))
    chans = np.array([mon.update(float(p)) for p in online])
    d200 = np.abs(chans[:, :200] - rows[:, :200]).max(); d6 = np.abs(chans[:, 200:206] - rows[:, 200:206]).max(); dm = np.abs(chans[:, 206:296] - rows[:, 206:296]).max(); dfq = np.abs(chans[:, 296:396] - rows[:, 296:396]).max(); dnv = np.abs(chans[:, 396:416] - rows[:, 396:416]).max(); dwh = np.abs(chans[:, 416:] - rows[:, 416:]).max()
    print(f"series {sid}: {len(online)} steps, {dt:.2f} ms/step, score {min(scores):.3f}-{max(scores):.3f}; "
          f"mismatch 200: {d200:.1e}, BOCPD: {d6:.1e}, MASS: {dm:.1e}, FREQDEP: {dfq:.1e}, NOVELTY: {dnv:.1e}, WHITE: {dwh:.1e}", flush=True)
    assert d200 < 1e-4 and d6 < 1e-4 and dm < 1e-4 and dfq < 1e-4 and dnv < 1e-4 and dwh < 1e-2, "CHANNELS DISAGREE"
    assert chans.shape[1] == 506 and sub.NOVELTY_OFFSET == 396 and sub.WHITE_OFFSET == 416, "WIDTH"
    assert dt < 8, "TOO SLOW"
print(f"OK (nets {len(model['nets'])}, rankers {len(model['rankers'])}, clf {model['booster'].booster_.num_feature()}, union {model['union_classifier'].booster_.num_feature()}, white {model['white_classifier'].booster_.num_feature()})")
