"""091: первая редакция (2025) -> формат real-time.

В первой редакции слом (если есть) стоит ровно на границе period 0 -> 1.
Чтобы получить равномерный tau, как в real-time, граница сдвигается назад:
последние k точек pre-сегмента становятся началом онлайн-части (tau = k),
история — остаток (>= 1000). k ~ U[0, L_online) как в real-time (tau/L ~ 0.49).
Ряды без слома: та же процедура с k=0..; tau = -1.
Выход: FE_series.parquet (id, time, value, period 1/2) и FE_index.parquet (tau_index).
Использование: python convert_first_edition.py <workspace первой редакции>/data
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
    # длина онлайн-части real-time: 10..999; сохраняем длину post, ограниченную 999
    post = post[:999]
    L = len(post)
    # k — сколько pre-точек уходит в онлайн (tau = k при сломе)
    k_max = min(len(pre) - 1000, L - 1)
    k = int(rng.integers(0, k_max + 1)) if k_max > 0 else 0
    hist = pre[:len(pre) - k]; online = np.concatenate([pre[len(pre) - k:], post])[:999]
    # стандартизация по истории (как в real-time)
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
print(f"конвертировано {n_ok} рядов; со сломом {(ix.tau_index >= 0).mean():.3f}; онлайн {ix.online_len.min()}–{ix.online_len.max()} (медиана {int(ix.online_len.median())}); "
      f"tau/L медиана {(ix[ix.tau_index>=0].tau_index / ix[ix.tau_index>=0].online_len).median():.3f}")
