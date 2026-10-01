"""105: two new training sets from the competition series.

FLIP — mirrored series (x -> -x), same tau; id = 300000 + sid.
BACK — the boundary shifted back by m points: history = hist[:-m], online = hist[-m:] + online,
       tau' = tau + m (or -1); re-standardization by the new history, as the platform does; id = 400000 + sid.
       m ~ U[50, min(400, len(hist) - 1000, 999 - L)], series without enough slack are skipped.
Format as the competition data: *_series.parquet (id, time, value, period 1/2), *_index.parquet (tau_index).
"""
import numpy as np, pandas as pd
D = "structural-break-real-time-test/data/"
X = pd.read_parquet(D + "X_train.parquet"); yi = pd.read_parquet(D + "y_train_index.parquet")
rng = np.random.default_rng(105)
def pack(rows, idx, prefix):
    df = pd.DataFrame({"id": np.concatenate([np.full(len(v), i) for i, v, _ in rows]),
                       "time": np.concatenate([np.arange(len(v)) for _, v, _ in rows]),
                       "value": np.concatenate([v for _, v, _ in rows]),
                       "period": np.concatenate([p for _, _, p in rows])}).set_index(["id", "time"])
    df.to_parquet(f"{prefix}_series.parquet")
    ix = pd.DataFrame(idx, columns=["id", "tau_index", "online_len"]).set_index("id"); ix.to_parquet(f"{prefix}_index.parquet")
    b = ix[ix.tau_index >= 0]
    print(f"{prefix}: {len(ix)} series; with a break {(ix.tau_index >= 0).mean():.3f}; online {ix.online_len.min()}–{ix.online_len.max()} "
          f"(median {int(ix.online_len.median())}); tau/L median {(b.tau_index / b.online_len).median():.3f}", flush=True)
flip_rows, flip_idx, back_rows, back_idx = [], [], [], []
for sid, part in X.groupby(level="id"):
    v = part.value.to_numpy("float64"); p = part.period.to_numpy()
    hist, online = v[p == 1], v[p == 2]; tau = int(yi.loc[sid, "tau_index"]); L = len(online)
    flip_rows.append((300000 + int(sid), -v, p.copy())); flip_idx.append((300000 + int(sid), tau, L))
    hi = min(400, len(hist) - 1000, 999 - L)
    if hi >= 50:
        m = int(rng.integers(50, hi + 1))
        h2, o2 = hist[:-m], np.concatenate([hist[-m:], online])
        mu, sd = h2.mean(), h2.std() + 1e-12
        v2 = (np.concatenate([h2, o2]) - mu) / sd
        p2 = np.concatenate([np.ones(len(h2), int), np.full(len(o2), 2)])
        back_rows.append((400000 + int(sid), v2, p2)); back_idx.append((400000 + int(sid), tau + m if tau >= 0 else -1, len(o2)))
pack(flip_rows, flip_idx, "FLIP"); pack(back_rows, back_idx, "BACK")
