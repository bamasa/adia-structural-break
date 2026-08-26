"""020: battery v2 — twenty more drift-free two-sample statistics per view."""
import sys, time
sys.path.insert(0, "repo/src")
import numpy as np
import pandas as pd
from structural_break.features import Normalisation
from structural_break.stream import iter_series

t0 = time.time()
x = pd.read_parquet("structural-break-real-time-test/data/X_train.parquet")
y = pd.read_parquet("structural-break-real-time-test/data/y_train.parquet")
print(f"загрузка {time.time()-t0:.0f}s", flush=True)

def acf1(v):
    if len(v) <= 3:
        return 0.0
    a, b = v[:-1], v[1:]
    sa, sb = a.std(), b.std()
    if sa < 1e-9 or sb < 1e-9:
        return 0.0
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))

def slope_t(v):
    n = len(v)
    if n < 8:
        return 0.0
    t_ax = np.arange(n) - (n - 1) / 2.0
    denom = float((t_ax ** 2).sum())
    beta = float((t_ax * (v - v.mean())).sum()) / denom
    resid = v - v.mean() - beta * t_ax
    se = np.sqrt(float((resid ** 2).sum()) / max(n - 2, 1) / denom) + 1e-12
    return float(np.clip(beta / se, -12, 12))

def hist_summary2(z):
    d = np.diff(z)
    dev = np.abs(z - np.median(z))
    return dict(
        sorted=np.sort(z), q10=float(np.quantile(z, 0.10)), q90=float(np.quantile(z, 0.90)),
        q05=float(np.quantile(z, 0.05)), q95=float(np.quantile(z, 0.95)),
        mean=float(z.mean()), std=float(z.std()),
        lev=float(dev.mean()), d_std=float(d.std()) if len(d) > 2 else 1.0,
        d_acf=acf1(d), d_abs=float(np.abs(d).mean()) if len(d) else 0.0,
        sign=float((z > 0).mean()),
    )

def two_sample2(h, p):
    n = len(p)
    out = []
    out.append(float(np.quantile(p, 0.10) - h["q10"]))
    out.append(float(np.quantile(p, 0.90) - h["q90"]))
    ps = np.sort(p)
    pos = np.searchsorted(h["sorted"], ps, side="right") / len(h["sorted"])
    ecdf = np.arange(1, n + 1) / n
    gap = pos - ecdf
    out.append(float(np.mean(gap ** 2)))                                  # Крамер–фон Мизес
    w = np.clip(pos * (1 - pos), 1e-3, None)
    out.append(float(np.clip(np.mean(gap ** 2 / w), 0, 10)))              # Андерсон–Дарлинг (обрезан)
    dev = np.abs(p - np.median(p))
    out.append(float(np.log((dev.mean() + 1e-9) / (h["lev"] + 1e-9))))    # Левен
    d = np.diff(p)
    if len(d) > 2:
        out.append(float(np.log((d.std() + 1e-9) / (h["d_std"] + 1e-9))))
        out.append(acf1(d) - h["d_acf"])
        out.append(float(np.log((np.abs(d).mean() + 1e-9) / (h["d_abs"] + 1e-9))))
    else:
        out.extend([0.0, 0.0, 0.0])
    for wlen in (10, 25, 100):
        tail = p[-wlen:]
        out.append(float(tail.mean() - h["mean"]))
        out.append(float(np.log((tail.std() + 1e-9) / (h["std"] + 1e-9))))
    out.append(float((p > h["q95"]).mean() - 0.05))                       # выходы за хвосты истории
    out.append(float((p < h["q05"]).mean() - 0.05))
    out.append(float((p > 0).mean() - h["sign"]))                         # знаковые статистики
    sg = np.sign(p)
    out.append(acf1(sg))
    out.append(slope_t(p))                                                # t-статистика тренда
    out.append(slope_t(np.abs(p)))                                        # тренд амплитуды
    return out  # 20 признаков

rows, count = [], 0
for sid, hist, online, labels in iter_series(x, y):
    views = []
    for transform in (None, np.arcsinh):
        h = np.asarray(hist, dtype="float64") if transform is None else transform(np.asarray(hist, dtype="float64"))
        o = np.asarray(online, dtype="float64") if transform is None else transform(np.asarray(online, dtype="float64"))
        norm = Normalisation.fit(h)
        n_h = len(h)
        z_h = np.asarray([norm.standardise(float(v), i - n_h) for i, v in enumerate(h)])
        z_o = np.asarray([norm.standardise(float(v), i) for i, v in enumerate(o)])
        views.append((hist_summary2(z_h), z_o))
    next_scan, current = 1, [0.0] * 40
    for step in range(len(online)):
        if step + 1 >= next_scan:
            next_scan = max(next_scan + 1, int(next_scan * 1.12))
            current = []
            for h, z_o in views:
                current.extend(two_sample2(h, z_o[: step + 1]))
        rows.append(list(current))
    count += 1
    if count % 2000 == 0:
        print(f"  {count} рядов, {time.time()-t0:.0f}s", flush=True)

a = np.asarray(rows, dtype="float32")
np.save("B2.npy", a)
print(f"готово {time.time()-t0:.0f}s: {a.shape}", flush=True)
