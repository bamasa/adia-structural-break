"""096: BOCPD с моделью наблюдений AR(1) (байесовская регрессия x_t на x_{t-1}) + прямой апостериор P(r<=t).

Ловит смену структуры зависимостей, которую Normal-Gamma не видит. Семь каналов:
P(r<5), P(r<20), P(r<60), P(r<200), E[r]/(t+1), surprise, P(r<=t) — «слом уже был в онлайн-части».
python build_bocpd_ar.py <shard> <n>  |  merge <n>   (env BOCPD_H — хазард, по умолчанию 50)
"""
import sys, time, os, numpy as np
from scipy.special import gammaln
HAZARD = 1.0 / float(os.environ.get("BOCPD_H", "50")); RMAX = 1000
OUT = os.environ.get("BOCPD_OUT", "BOCPDAR7"); PARTS = f"{OUT.lower()}_parts"

def student_logpdf(y, loc, scale2, nu):
    z2 = (y - loc) ** 2 / scale2
    return gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi * scale2) - (nu + 1) / 2 * np.log1p(z2 / nu)

def bocpd_ar(hist, online):
    # приор из истории: AR(1) по МНК; сила приора k0 = 5 точек
    h = np.asarray(hist, float); X0 = np.column_stack([np.ones(len(h) - 1), h[:-1]]); y0 = h[1:]
    beta0, *_ = np.linalg.lstsq(X0, y0, rcond=None); resid = y0 - X0 @ beta0; s2 = resid.var() + 1e-12
    k0, a0 = 5.0, 2.5
    L0 = k0 * (X0.T @ X0) / len(y0)           # 2x2 точность приора
    m0 = beta0.copy(); b0 = a0 * s2
    # состояние по длинам режима r = 0..RMAX; индекс RMAX — «хвост», в нём с самого начала
    # живёт гипотеза «режим продолжается из истории» с параметрами, обученными на всей истории.
    Lh = L0 + X0.T @ X0; mh = np.linalg.solve(Lh, L0 @ m0 + X0.T @ y0)
    ah = a0 + 0.5 * len(y0); bh = b0 + 0.5 * (y0 @ y0 + m0 @ L0 @ m0 - mh @ Lh @ mh); bh = max(bh, 1e-12)
    m = np.vstack([m0[None, :]] + [m0[None, :]] * (RMAX - 1) + [mh[None, :]])
    Lam = np.concatenate([L0[None]] * RMAX + [Lh[None]]); a = np.concatenate([np.full(RMAX, a0), [ah]]); b = np.concatenate([np.full(RMAX, b0), [bh]])
    logR = np.full(RMAX + 1, -np.inf); logR[RMAX] = 0.0
    prev = h[-1]; out = np.empty((len(online), 7), dtype="float32")
    for t, x in enumerate(online):
        phi = np.array([1.0, prev])                                   # регрессор
        # предиктив: Student-t(nu=2a, loc=m·phi, scale2 = b/a·(1 + phiᵀ Lam⁻¹ phi))
        det = Lam[:, 0, 0] * Lam[:, 1, 1] - Lam[:, 0, 1] * Lam[:, 1, 0]
        inv = np.empty_like(Lam); inv[:, 0, 0] = Lam[:, 1, 1] / det; inv[:, 1, 1] = Lam[:, 0, 0] / det
        inv[:, 0, 1] = -Lam[:, 0, 1] / det; inv[:, 1, 0] = -Lam[:, 1, 0] / det
        q = phi @ inv @ phi                                            # (R,)
        loc = m @ phi; scale2 = b / a * (1 + q); lp = student_logpdf(x, loc, scale2, 2 * a)
        log_growth = logR + lp + np.log(1 - HAZARD); log_cp = np.logaddexp.reduce(logR + lp) + np.log(HAZARD)
        newR = np.concatenate([[log_cp], log_growth]); evidence = np.logaddexp.reduce(newR); newR -= evidence
        # обновление параметров (байесовская линейная регрессия, y = x)
        Lam_n = Lam + np.einsum("i,j->ij", phi, phi)[None]
        rhs = np.einsum("rij,rj->ri", Lam, m) + phi[None, :] * x
        det_n = Lam_n[:, 0, 0] * Lam_n[:, 1, 1] - Lam_n[:, 0, 1] * Lam_n[:, 1, 0]
        inv_n = np.empty_like(Lam_n); inv_n[:, 0, 0] = Lam_n[:, 1, 1] / det_n; inv_n[:, 1, 1] = Lam_n[:, 0, 0] / det_n
        inv_n[:, 0, 1] = -Lam_n[:, 0, 1] / det_n; inv_n[:, 1, 0] = -Lam_n[:, 1, 0] / det_n
        m_n = np.einsum("rij,rj->ri", inv_n, rhs)
        a_n = a + 0.5
        b_n = b + 0.5 * (x ** 2 + np.einsum("ri,rij,rj->r", m, Lam, m) - np.einsum("ri,rij,rj->r", m_n, Lam_n, m_n))
        b_n = np.maximum(b_n, 1e-12)
        m = np.vstack([m0[None, :], m_n]); Lam = np.concatenate([L0[None], Lam_n]); a = np.concatenate([[a0], a_n]); b = np.concatenate([[b0], b_n])
        if len(newR) > RMAX:
            tail = np.logaddexp.reduce(newR[RMAX:]); newR = newR[:RMAX + 1].copy(); newR[RMAX] = tail
            m, Lam, a, b = m[:RMAX + 1], Lam[:RMAX + 1], a[:RMAX + 1], b[:RMAX + 1]
        logR = newR; P = np.exp(logR); r = np.arange(len(P))
        out[t] = [P[:5].sum(), P[:20].sum(), P[:60].sum(), P[:200].sum(), (P[:RMAX] * r[:RMAX]).sum() / (t + 1), -evidence, 1.0 - P[RMAX]]
        prev = x
    return out

if __name__ == "__main__":
    if sys.argv[1] == "test":
        rng = np.random.default_rng(0); n = 1500
        # AR(1) с phi=0.1 -> phi=0.6 на шаге 700 онлайн-части, та же дисперсия
        def ar(phi, n, s): 
            x = np.zeros(n); e = rng.normal(0, s, n)
            for i in range(1, n): x[i] = phi * x[i-1] + e[i]
            return x
        hist = ar(0.1, 2000, 1.0); online = np.concatenate([ar(0.1, 700, 1.0), ar(0.6, 300, 0.8)])
        t0 = time.time(); o = bocpd_ar(hist, online); dt = (time.time()-t0)/len(online)*1000
        print(f"тест AR: P(r<=t) до слома (шаги 600-700) {o[600:700,6].mean():.3f}, после (720-800) {o[720:800,6].mean():.3f}; P(r<20) до {o[600:700,1].mean():.3f} после {o[705:725,1].mean():.3f}; {dt:.2f} мс/шаг")
        sys.exit(0)
    if sys.argv[1] == "merge":
        n = int(sys.argv[2]); parts = [np.load(f"{PARTS}/part_{i}.npz", allow_pickle=True) for i in range(n)]
        g = np.load("G40.npy"); s = np.load("S40.npy")
        starts = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]])); bounds = np.append(starts, len(g))
        out = np.empty((len(g), 7), dtype="float32"); block = {}
        for p in parts:
            for sid, arr in zip(p["sids"], p["arrs"]): block[int(sid)] = arr
        for a_, b_ in zip(starts, bounds[1:]): out[a_:b_] = block[int(g[a_])][s[a_:b_]]
        np.save(f"{OUT}.npy", out); print(f"{OUT}: {out.shape}, nan {np.isnan(out).sum()}", flush=True); sys.exit(0)
    shard, n_shards = int(sys.argv[1]), int(sys.argv[2])
    import pandas as pd
    t0 = time.time(); x = pd.read_parquet("structural-break-real-time-test/data/X_train.parquet")
    ids = x.index.get_level_values("id"); x = x[(ids % n_shards) == shard]
    sids, arrs = [], []
    for i, (sid, part) in enumerate(x.groupby(level="id")):
        hist = part.loc[part.period == 1, "value"].to_numpy("float64"); online = part.loc[part.period == 2, "value"].to_numpy("float64")
        sids.append(int(sid)); arrs.append(bocpd_ar(hist, online))
        if (i + 1) % 100 == 0: print(f"шард {shard}: {i+1} рядов, {time.time()-t0:.0f}s", flush=True)
    os.makedirs(PARTS, exist_ok=True)
    np.savez(f"{PARTS}/part_{shard}.npz", sids=np.array(sids), arrs=np.array(arrs, dtype=object), allow_pickle=True)
    print(f"шард {shard} готов {time.time()-t0:.0f}s: {len(sids)} рядов", flush=True)
