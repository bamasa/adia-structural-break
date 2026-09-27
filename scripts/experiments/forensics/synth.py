"""Synthetic generator draft for the ADIA structural-break real-time data (forensics-based).

Generating model (per series), bootstrapped from synth_params.csv (one row = one real series' fitted parameters):
  x_t = ARMA(p,q) filter of eps_t;  eps_t = sigma_t * z_t
  z_t ~ N(0,1) | Student-t(df) scaled to unit variance | Uniform (light tails); df = min(quantile df, 4+6/kurt) (v2)
  sigma_t^2 = (1-a-b) + a*eps_{t-1}^2 + b*sigma_{t-1}^2   (GARCH(1,1), only for vol != const; else sigma=1)
  history length ~ U{1000..5000}, online length ~ U{10..999}; break with prob 0.5 at tau_index ~ U{0..onl-1}.
Break menu (estimated from data; magnitudes continuous, no minimum):
  variance: with prob P_UP the innovation sd is multiplied by 2^E, E ~ Exponential(mean MU_UP) (increase);
            with prob P_DN divided by 2^E, E ~ Exponential(MU_DN);
  dependence (only non-WN families): with prob P_DEP each AR/MA coefficient gets N(0, S_DEP^2) added, then re-stabilised;
  mean: with prob P_MEAN a level shift of sign*U(0.2, 1.0) (in pre-sd units).
The whole series is standardised by the HISTORY mean/sd (as in the real data).
Usage: python synth.py N SEED -> writes synth_values.npy, synth_offsets.npy, synth_meta.csv next to this file.
"""
import os, sys, numpy as np, pandas as pd
from scipy import signal
OUT = os.path.dirname(os.path.abspath(__file__))
P_UP, MU_UP, P_DN, MU_DN = 0.20, 0.50, 0.04, 0.50
P_DEP, S_DEP = 0.35, 0.10
P_MEAN = 0.02
BURN = 600

def stabilise(ar, ma):
    """shrink coefficients until AR roots are outside the unit circle (|1/z| < 0.98) and MA invertible."""
    ar = ar.copy(); ma = ma.copy()
    for _ in range(60):
        pa = np.r_[1, -ar]
        if np.any(ar != 0) and np.max(np.abs(1 / np.roots(pa[::-1]))) >= 0.98: ar *= 0.95
        else: break
    for _ in range(60):
        pm = np.r_[1, ma]
        if np.any(ma != 0) and np.max(np.abs(1 / np.roots(pm[::-1]))) >= 0.98: ma *= 0.95
        else: break
    return ar, ma

def eff_df(df, res_kurt):
    """v2: effective t df = min(quantile-based df, kurtosis-based df 4+6/kurt); the quantile estimator alone misses the very heavy tails."""
    dk = 4 + 6 / res_kurt if res_kurt > 0.3 else 1e6
    return float(np.clip(min(df, dk), 2.3, 1e6))

def innovations(n, innov, df, rng):
    if innov == "light": return rng.uniform(-np.sqrt(3), np.sqrt(3), n)
    if df >= 30: return rng.standard_normal(n)
    d = max(df, 2.3); return rng.standard_t(d, n) / np.sqrt(d / (d - 2))

def garch(z, a, b, rng):
    n = len(z); e = np.empty(n); s2 = 1.0; om = max(1 - a - b, 1e-4)
    for t in range(n):
        e[t] = np.sqrt(s2) * z[t]; s2 = om + a * e[t] ** 2 + b * s2
    return e

def gen_one(row, rng):
    h = int(rng.integers(1000, 5001)); o = int(rng.integers(10, 1000)); n = BURN + h + o
    brk = rng.random() < 0.5; tau = int(rng.integers(0, o)) if brk else -1
    p, q = int(row.p), int(row.q)
    ar = np.array([row[f"ar{j+1}"] for j in range(max(p, 1))]); ma = np.array([row[f"ma{j+1}"] for j in range(max(q, 1))])
    ar, ma = stabilise(ar, ma)
    z = innovations(n, row.innov, eff_df(row.df, row.res_kurt), rng)
    e = garch(z, row.alpha, row.beta, rng) if row.vol != "const" else z
    kinds = []
    if brk:
        t0 = BURN + h + tau
        u = rng.random()
        if u < P_UP: e[t0:] *= 2 ** rng.exponential(MU_UP); kinds.append("V+")
        elif u < P_UP + P_DN: e[t0:] /= 2 ** rng.exponential(MU_DN); kinds.append("V-")
    ar2, ma2 = ar, ma
    if brk and (p + q > 0) and rng.random() < P_DEP:
        ar2 = ar + rng.normal(0, S_DEP, len(ar)) * (np.arange(len(ar)) < max(p, 1)); ma2 = ma + rng.normal(0, S_DEP, len(ma)) * (np.arange(len(ma)) < max(q, 1))
        ar2, ma2 = stabilise(ar2, ma2); kinds.append("D")
    if brk and kinds and kinds[-1] == "D":
        t0 = BURN + h + tau
        x1 = signal.lfilter(np.r_[1, ma], np.r_[1, -ar], e[:t0])
        # continue with new coefficients from the state of the old filter
        zi = signal.lfiltic(np.r_[1, ma2], np.r_[1, -ar2], x1[::-1][: max(len(ar2), len(ma2))], e[:t0][::-1][: max(len(ar2), len(ma2))])
        x2, _ = signal.lfilter(np.r_[1, ma2], np.r_[1, -ar2], e[t0:], zi=zi); x = np.r_[x1, x2]
    else:
        x = signal.lfilter(np.r_[1, ma], np.r_[1, -ar], e)
    if brk and rng.random() < P_MEAN:
        x[BURN + h + tau:] += rng.choice([-1, 1]) * rng.uniform(0.2, 1.0) * x[BURN: BURN + h].std(); kinds.append("M")
    x = x[BURN:]; hist = x[:h]; x = (x - hist.mean()) / hist.std()
    return x.astype(np.float32), h, o, tau, "+".join(kinds) if kinds else ("none" if brk else "")

def generate(N, seed=0, params=None):
    rng = np.random.default_rng(seed)
    if params is None: params = pd.read_csv(os.path.join(OUT, "synth_params.csv"))
    vals, offs, meta = [], [0], []
    for i in range(N):
        row = params.iloc[int(rng.integers(0, len(params)))]
        x, h, o, tau, kind = gen_one(row, rng)
        vals.append(x); offs.append(offs[-1] + len(x)); meta.append(dict(id=i, hist_len=h, onl_len=o, tau_index=tau, kind=kind, src=int(row.id), lin=row.lin, vol=row.vol, innov=row.innov))
    return np.concatenate(vals), np.array(offs, np.int64), pd.DataFrame(meta)

if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 500; seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    v, o, m = generate(N, seed)
    np.save(os.path.join(OUT, "synth_values.npy"), v); np.save(os.path.join(OUT, "synth_offsets.npy"), o); m.to_csv(os.path.join(OUT, "synth_meta.csv"), index=False)
    print("generated", N, "series;", int((m.tau_index >= 0).sum()), "with break; kinds:", m.kind.value_counts().to_dict())
