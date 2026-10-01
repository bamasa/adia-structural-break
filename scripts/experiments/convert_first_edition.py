"""091: first edition (2025) -> real-time format.

In the first edition the break (if any) sits exactly at the period 0 -> 1 boundary.
To get a uniform tau, as in real-time, the boundary is shifted back:
the last k points of the pre-segment become the start of the online part (tau = k),
the history is the remainder (>= 1000). k ~ U[0, L_online) as in real-time (tau/L ~ 0.49).
Series without a break: the same procedure with k=0..; tau = -1.
Output: FE_series.parquet (id, time, value, period 1/2) and FE_index.parquet (tau_index).
Usage: python convert_first_edition.py <first-edition workspace>/data
"""
import sys, numpy as np, pandas as pd
D = sys.argv[1].rstrip("/") + "/"
X = pd.read_parquet(D + "X_train.parquet"); y = pd.read_parquet(D + "y_train.parquet")
ycol = y.columns[0] if isinstance(y, pd.DataFrame) else None
lab = (y[ycol] if ycol else y).astype(bool)
rng = np.random.default_rng(2025)
out_val, out_per, out_id, out_t, idx = [], [], [], [], []
n_ok = 0
for sid, part in X.groupby(level="id"):
    v = part.value.to_numpy("float64"); p = part.period.to_numpy()
    pre, post = v[p == 0], v[p == 1]
    if len(pre) < 1010 or len(post) < 10:
        continue
    brk = bool(lab.loc[sid])
    # real-time online-part length: 10..999; keep the post length, capped at 999
    post = post[:999]
    L = len(post)
    # k — how many pre-points go into the online part (tau = k when there is a break)
    k_max = min(len(pre) - 1000, L - 1)
    k = int(rng.integers(0, k_max + 1)) if k_max > 0 else 0
    hist = pre[:len(pre) - k]; online = np.concatenate([pre[len(pre) - k:], post])[:999]
    # standardization by the history (as in real-time)
    mu, sd = hist.mean(), hist.std() + 1e-12
    hist = (hist - mu) / sd; online = (online - mu) / sd
    new_id = 200000 + int(sid)
    n_h, n_o = len(hist), len(online)
    out_val.append(np.concatenate([hist, online])); out_per.append(np.concatenate([np.ones(n_h, int), np.full(n_o, 2)]))
    out_id.append(np.full(n_h + n_o, new_id)); out_t.append(np.arange(n_h + n_o))
    idx.append((new_id, k if brk else -1, n_o))
    n_ok += 1
df = pd.DataFrame({"id": np.concatenate(out_id), "time": np.concatenate(out_t), "value": np.concatenate(out_val),
                   "period": np.concatenate(out_per)}).set_index(["id", "time"])
df.to_parquet("FE_series.parquet")
ix = pd.DataFrame(idx, columns=["id", "tau_index", "online_len"]).set_index("id"); ix.to_parquet("FE_index.parquet")
print(f"converted {n_ok} series; with a break {(ix.tau_index >= 0).mean():.3f}; online {ix.online_len.min()}–{ix.online_len.max()} (median {int(ix.online_len.median())}); "
      f"tau/L median {(ix[ix.tau_index>=0].tau_index / ix[ix.tau_index>=0].online_len).median():.3f}")
