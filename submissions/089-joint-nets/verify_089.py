"""#088: channels versus matrices (206 + 90 + 100 + 20 + 90 whitened + 21 SR + 8 context), speed, end-to-end run."""
import importlib.util, sys, time
import numpy as np, pandas as pd
spec = importlib.util.spec_from_file_location("sub", "repo/submissions/089-joint-nets/main.py")
sub = importlib.util.module_from_spec(spec); sys.modules["sub"] = sub; spec.loader.exec_module(sub)
# The matrices are memory-mapped and only the two series' rows are gathered: the full
# float64 stack (5M x 535) is 21 GB and swapped the machine for twenty minutes.
MATS = ["X40.npy", "C_cnn.npy", "N9.npy", "X50a.npy", "E4.npy", "B40.npy", "B2.npy", "SPEC14.npy", "BOCPD6_H50.npy",
        "MASS90.npy", "MSPEC32.npy", "MSPEC60.npy", "ACF8.npy", "KNN20.npy", "WHITE90.npy", "SR22.npy", "HISTCTX8.npy"]
def rows_for(idx):
    parts = []
    for name in MATS:
        m = np.load(name, mmap_mode="r"); block = np.asarray(m[idx]).astype("float64")
        parts.append(block[:, None] if block.ndim == 1 else block)
    return np.hstack(parts)
g = np.load("G40.npy"); s = np.load("S40.npy")
xtr = pd.read_parquet("structural-break-real-time-test/data/X_train.parquet")
import joblib, lightgbm as lgb
model = joblib.load("resources089/model.joblib")
for sid in (7, 4242):
    ser = xtr.loc[sid]
    hist = ser[ser.period == 1].value.to_numpy(); online = ser[ser.period == 2].value.to_numpy()
    idx = np.flatnonzero(g == sid); rows = rows_for(idx)[np.argsort(s[idx])]
    gen = sub.infer([(hist, online)], "resources089"); next(gen)
    t0 = time.time(); scores = [next(gen) for _ in range(len(online))]
    dt = (time.time() - t0) / len(online) * 1000
    mon = sub.TriMonitor(hist.astype("float64"), lgb.Booster(model_str=model["forecaster"]))
    chans = np.array([mon.update(float(p)) for p in online])
    d200 = np.abs(chans[:, :200] - rows[:, :200]).max(); d6 = np.abs(chans[:, 200:206] - rows[:, 200:206]).max(); dm = np.abs(chans[:, 206:296] - rows[:, 206:296]).max(); dfq = np.abs(chans[:, 296:396] - rows[:, 296:396]).max(); dnv = np.abs(chans[:, 396:416] - rows[:, 396:416]).max(); dwh = np.abs(chans[:, 416:] - rows[:, 416:]).max()
    print(f"series {sid}: {len(online)} steps, {dt:.2f} ms/step, score {min(scores):.3f}-{max(scores):.3f}; "
          f"mismatch 200: {d200:.1e}, BOCPD: {d6:.1e}, MASS: {dm:.1e}, FREQDEP: {dfq:.1e}, NOVELTY: {dnv:.1e}, WHITE: {dwh:.1e}", flush=True)
    assert d200 < 1e-4 and d6 < 1e-4 and dm < 1e-4 and dfq < 1e-4 and dnv < 1e-4 and dwh < 1e-2, "CHANNELS DISAGREE"
    assert chans.shape[1] == 535 and sub.NOVELTY_OFFSET == 396 and sub.WHITE_OFFSET == 416 and sub.WHITE_CLF_WIDTH == 119, "WIDTH"
    assert dt < 12, "TOO SLOW"
print(f"OK (nets {len(model['nets'])}, rankers {len(model['rankers'])}, clf {model['booster'].booster_.num_feature()}, union {model['union_classifier'].booster_.num_feature()}, white clf {model['white3_classifier'].booster_.num_feature()}, ranks {model['white2_ranker'].booster_.num_feature()}/{model['white3_ranker_ctx'].booster_.num_feature()}, white nets {len(model['white_nets'])}, joint nets {len(model['joint_nets'])})")
