"""Shared helpers: text histograms, quantile tables, AR fitting, spectral tools."""
import numpy as np, pandas as pd

def thist(x, bins=20, rng=None, width=50, log=False, label=""):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if log: x = np.log10(np.maximum(x, 1e-12))
    if rng is None: rng = (np.quantile(x, 0.005), np.quantile(x, 0.995))
    c, e = np.histogram(x, bins=bins, range=rng)
    out = [f"-- {label} n={len(x)} (range {rng[0]:.4g}..{rng[1]:.4g}{' log10' if log else ''}; below={int((x<rng[0]).sum())} above={int((x>rng[1]).sum())})"]
    m = c.max() if c.max() > 0 else 1
    for i in range(bins):
        out.append(f"{e[i]:>10.4g} {e[i+1]:>10.4g} | {'#' * int(round(width * c[i] / m)):<{width}} {c[i]}")
    return "\n".join(out)

def qtab(df, cols, qs=(0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)):
    rows = []
    for c in cols:
        x = df[c].to_numpy(float); x = x[np.isfinite(x)]
        rows.append([c, len(x), x.mean(), x.std()] + [np.quantile(x, q) for q in qs])
    return pd.DataFrame(rows, columns=["stat", "n", "mean", "sd"] + [f"q{q}" for q in qs]).to_string(index=False, float_format=lambda v: f"{v:.4g}")

def acf(x, nlags):
    x = np.asarray(x, float); x = x - x.mean(); n = len(x)
    v = np.dot(x, x) / n
    if v <= 0: return np.zeros(nlags + 1)
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[: nlags + 1] / n / v
    return ac

def pacf_from_acf(r, nlags):
    """Durbin-Levinson; r = acf[0..nlags]"""
    pac = np.zeros(nlags + 1); pac[0] = 1.0
    phi = np.zeros((nlags + 1, nlags + 1))
    phi[1, 1] = r[1]; pac[1] = r[1]
    for k in range(2, nlags + 1):
        num = r[k] - np.dot(phi[k - 1, 1:k], r[k - 1:0:-1])
        den = 1.0 - np.dot(phi[k - 1, 1:k], r[1:k])
        phi[k, k] = num / den if den > 1e-12 else 0.0
        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]
        pac[k] = phi[k, k]
    return pac

def ar_ols(x, p, with_const=True):
    """least-squares AR(p): returns coef (p,), const, residuals, sigma2"""
    x = np.asarray(x, float); n = len(x)
    if n <= p + 5:
        return np.full(p, np.nan), np.nan, np.array([]), np.nan
    Y = x[p:]
    X = np.column_stack([x[p - j - 1: n - j - 1] for j in range(p)])
    if with_const: X = np.column_stack([X, np.ones(len(Y))])
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    res = Y - X @ beta
    coef = beta[:p]; c = beta[p] if with_const else 0.0
    return coef, c, res, res.var()

def periodogram(x):
    x = np.asarray(x, float); x = x - x.mean(); n = len(x)
    f = np.fft.rfft(x)
    P = (np.abs(f) ** 2) / n
    freqs = np.fft.rfftfreq(n)
    return freqs[1:], P[1:]

def spectral_slope(x, fmin=0.01, fmax=0.5):
    """slope of log-periodogram (smoothed in log-freq bins) vs log-frequency"""
    fr, P = periodogram(x)
    m = (fr >= fmin) & (fr <= fmax) & (P > 0)
    if m.sum() < 10: return np.nan
    lf, lp = np.log(fr[m]), np.log(P[m])
    # bin into 20 log-spaced bins to reduce periodogram noise
    edges = np.linspace(lf.min(), lf.max(), 21)
    idx = np.clip(np.digitize(lf, edges) - 1, 0, 19)
    bx, by = [], []
    for b in range(20):
        s = idx == b
        if s.sum() >= 2: bx.append(lf[s].mean()); by.append(np.log(np.exp(lp[s]).mean()))
    if len(bx) < 5: return np.nan
    return np.polyfit(bx, by, 1)[0]

def hill(x, k_frac=0.05):
    a = np.sort(np.abs(np.asarray(x, float)))[::-1]
    a = a[a > 0]
    k = max(10, int(k_frac * len(a)))
    if len(a) <= k + 1: return np.nan
    return 1.0 / np.mean(np.log(a[:k] / a[k]))
