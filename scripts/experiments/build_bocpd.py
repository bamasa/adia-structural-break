"""085: BOCPD-каналы — апостериор длины текущего режима (Adams & MacKay 2007).

Модель точки: нормаль с неизвестными средним и дисперсией, сопряжённый
Normal-Gamma-приор, подогнанный по истории. Хазард 1/200, длины режима до
600. Шесть каналов на шаг: P(r<5), P(r<20), P(r<60), P(r<200), E[r]/(t+1),
-log предиктивной плотности точки (неожиданность).
python build_bocpd.py <shard> <n> | merge <n>"""
import sys, time, os
import numpy as np
from scipy.special import gammaln

HAZARD = 1.0 / float(os.environ.get("BOCPD_H", "200"))
RMAX = 600
OUT = os.environ.get("BOCPD_OUT", "BOCPD6")
PARTS = f"{OUT.lower()}_parts"

def student_logpdf(x, mu, kappa, alpha, beta):
    # предиктивная Student-t для Normal-Gamma
    nu = 2 * alpha
    scale2 = beta * (kappa + 1) / (alpha * kappa)
    z2 = (x - mu) ** 2 / scale2
    return (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi * scale2)
            - (nu + 1) / 2 * np.log1p(z2 / nu))

def bocpd_channels(hist, online):
    m0 = hist.mean(); v0 = hist.var() + 1e-12
    # приор: как будто видели k0 точек истории со средним m0 и дисперсией v0
    k0, a0 = 5.0, 2.5
    b0 = a0 * v0
    n = len(online)
    out = np.empty((n, 6), dtype="float32")
    # параметры по длинам режима r = 0..R (r=0 — новый режим, только приор)
    mu = np.array([m0]); kappa = np.array([k0]); alpha = np.array([a0]); beta = np.array([b0])
    logR = np.array([0.0])  # log P(r_t = r)
    for t in range(n):
        x = online[t]
        lp = student_logpdf(x, mu, kappa, alpha, beta)
        # рост и смена режима
        log_growth = logR + lp + np.log(1 - HAZARD)
        log_cp = np.logaddexp.reduce(logR + lp) + np.log(HAZARD)
        newR = np.concatenate([[log_cp], log_growth])
        evidence = np.logaddexp.reduce(newR)
        newR -= evidence
        # обновление параметров: r=0 — приор; r>0 — приор/предыдущие + x
        mu_new = np.concatenate([[m0], (kappa * mu + x) / (kappa + 1)])
        kappa_new = np.concatenate([[k0], kappa + 1])
        alpha_new = np.concatenate([[a0], alpha + 0.5])
        beta_new = np.concatenate([[b0], beta + kappa * (x - mu) ** 2 / (2 * (kappa + 1))])
        if len(newR) > RMAX:
            tail = np.logaddexp.reduce(newR[RMAX:])
            newR = newR[:RMAX + 1].copy(); newR[RMAX] = tail
            mu_new, kappa_new, alpha_new, beta_new = (a[:RMAX + 1] for a in (mu_new, kappa_new, alpha_new, beta_new))
        logR, mu, kappa, alpha, beta = newR, mu_new, kappa_new, alpha_new, beta_new
        P = np.exp(logR)
        r = np.arange(len(P))
        out[t, 0] = P[:5].sum(); out[t, 1] = P[:20].sum(); out[t, 2] = P[:60].sum(); out[t, 3] = P[:200].sum()
        out[t, 4] = (P * r).sum() / (t + 1)
        out[t, 5] = -evidence
    return out

if sys.argv[1] == "merge":
    n = int(sys.argv[2])
    parts = [np.load(f"{PARTS}/part_{i}.npz", allow_pickle=True) for i in range(n)]
    if OUT.startswith("AUG3"):
        g = np.load("AUG3_G.npy"); s = np.load("AUG3_S.npy")
    else:
        g = np.load("G40.npy"); s = np.load("S40.npy")
    starts = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]])); bounds = np.append(starts, len(g))
    out = np.empty((len(g), 6), dtype="float32")
    block = {}
    for p in parts:
        for sid, arr in zip(p["sids"], p["arrs"]):
            block[int(sid)] = arr
    # arrs — object array; собираем по рядам
    for a, b in zip(starts, bounds[1:]):
        arr = block[int(g[a])]
        out[a:b] = arr[s[a:b]]
    np.save(f"{OUT}.npy", out)
    print(f"{OUT}: {out.shape}, nan {np.isnan(out).sum()}", flush=True)
    sys.exit(0)

shard, n_shards = int(sys.argv[1]), int(sys.argv[2])
import pandas as pd
t0 = time.time()
x = pd.read_parquet("structural-break-real-time-test/data/X_train.parquet")
ids = x.index.get_level_values("id")
x = x[(ids % n_shards) == shard]
sids, arrs = [], []
if OUT.startswith("AUG3"):
    # псевдоряды: history + online[:k], k = len(online) - длина псевдоряда; группы 100000+sid*10+j
    AG = np.load("AUG3_G.npy")
    a_starts = np.flatnonzero(np.concatenate([[True], AG[1:] != AG[:-1]])); a_bounds = np.append(a_starts, len(AG))
    series = {int(sid): (part.loc[part.period == 1, "value"].to_numpy("float64"),
                         part.loc[part.period == 2, "value"].to_numpy("float64"))
              for sid, part in x.groupby(level="id")}
    for i, (a, b) in enumerate(zip(a_starts, a_bounds[1:])):
        gid = int(AG[a]); sid = (gid - 100000) // 10
        if sid % n_shards != shard:
            continue
        hist, online = series[sid]
        k = len(online) - (b - a)
        sids.append(gid); arrs.append(bocpd_channels(np.concatenate([hist, online[:k]]), online[k:]))
        if len(sids) % 300 == 0:
            print(f"шард {shard}: {len(sids)} псевдорядов, {time.time()-t0:.0f}s", flush=True)
else:
    for i, (sid, part) in enumerate(x.groupby(level="id")):
        hist = part.loc[part.period == 1, "value"].to_numpy("float64")
        online = part.loc[part.period == 2, "value"].to_numpy("float64")
        sids.append(int(sid)); arrs.append(bocpd_channels(hist, online))
        if (i + 1) % 100 == 0:
            print(f"шард {shard}: {i+1} рядов, {time.time()-t0:.0f}s", flush=True)
os.makedirs(PARTS, exist_ok=True)
np.savez(f"{PARTS}/part_{shard}.npz", sids=np.array(sids), arrs=np.array(arrs, dtype=object), allow_pickle=True)
print(f"шард {shard} готов {time.time()-t0:.0f}s: {len(sids)} рядов", flush=True)
