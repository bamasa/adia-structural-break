"""Task 2d: ARMA(p,q) order selection on histories, p,q in 0..3, via Hannan-Rissanen (long AR(25) residuals, then OLS),
BIC with (p+q+1) params. Also records the HR coefficient estimates of the selected order and residual variance.
-> t2d_orders.csv"""
import os, sys, time, numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from util import ar_ols
OUT = os.path.dirname(os.path.abspath(__file__))
vals = off = meta = None
def init():
    global vals, off, meta
    vals = np.load(os.path.join(OUT, "values.npy"), mmap_mode="r"); off = np.load(os.path.join(OUT, "offsets.npy")); meta = pd.read_csv(os.path.join(OUT, "meta.csv"))

def hr_fit(x, e, p, q, m=25):
    n = len(x); Y = x[m:]
    cols = [x[m - j - 1: n - j - 1] for j in range(p)] + [e[m - j - 1: n - j - 1] for j in range(q)] + [np.ones(len(Y))]
    X = np.column_stack(cols); beta, *_ = np.linalg.lstsq(X, Y, rcond=None); r = Y - X @ beta
    return beta, r.var()

def feats(k):
    a = off[k]; h = int(meta.hist_len[k]); x = np.asarray(vals[a:a + h], float); n = len(x)
    coef, c, e25, s25 = ar_ols(x, 25); e = np.concatenate([np.zeros(25), e25])
    best = (np.inf, 0, 0, None, None); tab = {}
    for p in range(4):
        for q in range(4):
            beta, v = hr_fit(x, e, p, q)
            bic = (n - 25) * np.log(v) + (p + q + 1) * np.log(n - 25); tab[(p, q)] = bic
            if bic < best[0]: best = (bic, p, q, beta, v)
    bic, p, q, beta, v = best
    r = dict(id=k, hr_p=p, hr_q=q, hr_s2=v, hr_bic=bic, hr_bic_wn=tab[(0, 0)], hr_bic_ar1=tab[(1, 0)], hr_bic_ma1=tab[(0, 1)], hr_bic_arma11=tab[(1, 1)])
    for j in range(3): r[f"hr_ar{j+1}"] = beta[j] if j < p else 0.0
    for j in range(3): r[f"hr_ma{j+1}"] = beta[p + j] if j < q else 0.0
    # second best margin
    srt = sorted(tab.values()); r["hr_margin"] = srt[1] - srt[0]
    return r

if __name__ == "__main__":
    init(); N = len(meta); t0 = time.time()
    with Pool(4, initializer=init) as pool:
        rows = list(pool.imap(feats, range(N), chunksize=100))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "t2d_orders.csv"), index=False); print("done", round(time.time() - t0, 1), "s")
