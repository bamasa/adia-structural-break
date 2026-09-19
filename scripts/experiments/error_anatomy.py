"""109: где ансамбль слеп — разбор по типам слома.

Для каждого ряда фолда-2 со сломом считаем, что именно изменилось (окна 300 до/после tau):
сдвиг среднего, отношение дисперсий, изменение AR(1), изменение формы (асимметрия/эксцесс).
Затем меряем per-series AUC ансамбля #30 внутри каждого типа: ряд со сломом против всех
рядов без слома на тех же шагах. Показывает, какой тип изменений мы не видим.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "repo/src")
from structural_break.combiners import split_by_series, ts_auc
g = np.load("G40.npy"); y = np.load("Y40.npy"); s = np.load("S40.npy")
m2 = split_by_series(g, folds=5, seed=0) == 2; gf, yf, sf = g[m2], y[m2], s[m2]
base = np.load("fold2_base30.npy")
D = "structural-break-real-time-test/data/"
X = pd.read_parquet(D + "X_train.parquet"); yi = pd.read_parquet(D + "y_train_index.parquet")
starts = np.flatnonzero(np.concatenate([[True], gf[1:] != gf[:-1]])); bounds = np.append(starts, len(gf))
def ar1(v): return np.corrcoef(v[:-1], v[1:])[0, 1] if len(v) > 5 else np.nan
rows = []
for a, b in zip(starts, bounds[1:]):
    sid = int(gf[a]); tau = int(yi.loc[sid, "tau_index"])
    part = X.loc[sid]; v = part.value.to_numpy("float64"); p = part.period.to_numpy()
    hist, online = v[p == 1], v[p == 2]
    r = dict(a=a, b=b, sid=sid, tau=tau, n=b - a)
    if tau >= 0 and (b - a) - tau >= 30:
        pre = np.concatenate([hist, online[:tau]])[-300:]; post = online[tau:][:300]
        r["dmean"] = abs(post.mean() - pre.mean()) / (pre.std() + 1e-12)
        r["vratio"] = post.std() / (pre.std() + 1e-12)
        r["dar1"] = abs(ar1(post) - ar1(pre))
        r["dshape"] = abs(pd.Series(post).kurt() - pd.Series(pre).kurt())
    rows.append(r)
df = pd.DataFrame(rows)
brk = df[df.tau >= 0].dropna(subset=["dmean"]).copy()
def kind(r):
    k = []
    if r.dmean > 0.3: k.append("среднее")
    if r.vratio < 0.85 or r.vratio > 1.18: k.append("дисперсия")
    if r.dar1 > 0.12: k.append("зависимости")
    if r.dshape > 1.5: k.append("форма")
    return "+".join(k) if k else "нет явного"
brk["kind"] = brk.apply(kind, axis=1)
# AUC каждого сломанного ряда против ВСЕХ несломанных на тех же шагах
clean = df[df.tau < 0]
clean_scores = {}
for _, c in clean.iterrows():
    seg = base[int(c.a):int(c.b)]
    for t, sc in enumerate(seg): clean_scores.setdefault(t, []).append(sc)
clean_arr = {t: np.array(v) for t, v in clean_scores.items()}
def series_auc(r):
    seg = base[int(r.a):int(r.b)]; wins = tot = 0.0
    for t in range(int(r.tau), int(r.n)):
        arr = clean_arr.get(t)
        if arr is None or len(arr) < 20: continue
        wins += (arr < seg[t]).sum() + 0.5 * (arr == seg[t]).sum(); tot += len(arr)
    return wins / tot if tot else np.nan
brk["auc"] = brk.apply(series_auc, axis=1)
print(f"рядов со сломом в фолде-2: {len(brk)}; общий TS-AUC ансамбля {ts_auc(base, yf, sf):.4f}\n")
print("по одиночным признакам изменения:")
for col, thr, name in (("dmean", 0.3, "сдвиг среднего > 0.3σ"), ("vratio", None, "дисперсия вне [0.85, 1.18]"),
                       ("dar1", 0.12, "изменение AR(1) > 0.12"), ("dshape", 1.5, "изменение эксцесса > 1.5")):
    m = (brk[col] > thr) if thr else ((brk.vratio < 0.85) | (brk.vratio > 1.18))
    print(f"  {name:32s}: {m.sum():4d} рядов, средний AUC {brk.auc[m].mean():.4f} | остальные {brk.auc[~m].mean():.4f}")
print("\nпо сочетаниям (что именно изменилось):")
for k, grp in sorted(brk.groupby("kind"), key=lambda kv: -len(kv[1])):
    if len(grp) >= 15: print(f"  {k:28s}: {len(grp):4d} рядов, AUC {grp.auc.mean():.4f}")
print(f"\nвсего 'нет явного изменения': {(brk.kind == 'нет явного').sum()} рядов из {len(brk)} "
      f"({(brk.kind == 'нет явного').mean():.0%}), их AUC {brk.auc[brk.kind == 'нет явного'].mean():.4f}")
