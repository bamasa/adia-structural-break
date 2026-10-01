"""157e: Student-t PIT instead of the empirical CDF — df by maximum likelihood on a grid, scale re-estimated per df."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_white as bw
from scipy.stats import t as student_t, norm
DFS = (2.5, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50, np.inf)
def fit_t(u):
    best = (-np.inf, np.inf, 1.0)
    for df in DFS:
        if np.isinf(df): sc = u.std() + 1e-9; ll = norm.logpdf(u, scale=sc).sum()
        else:
            sc = u.std() * np.sqrt(max(df - 2, 0.1) / df) + 1e-9   # variance-matched scale, then one refinement
            for _ in range(2):
                w = (df + 1) / (df + (u / sc) ** 2); sc = np.sqrt((w * u ** 2).mean()) + 1e-9
            ll = student_t.logpdf(u, df, scale=sc).sum()
        if ll > best[0]: best = (ll, df, sc)
    return best[1], best[2]
def scores_t(u, df, sc):
    p = norm.cdf(u / sc) if np.isinf(df) else student_t.cdf(u / sc, df)
    return norm.ppf(np.clip(p, 1e-6, 1 - 1e-6))
_orig_fit = bw.fit_history
def fit_history(h):
    fit = _orig_fit(h)
    # recover the history innovations from the stored sorted arrays is not possible; refit cheaply: reuse the AR order and lambda
    P, p, coef, lam = bw.PMAX, fit["p"], fit["coef"], fit["lam"]; n = len(h)
    e = h[p:] - (np.column_stack([h[p - j - 1: n - j - 1] for j in range(p)]) @ coef if p else 0.0); e2 = e ** 2; v0 = fit["v0"]
    s2 = np.full(len(e), v0) if lam >= 1.0 else np.concatenate([[v0], bw.ewma(e2[:-1], 1 - lam, init=v0)])
    uc, uu = e / np.sqrt(s2), e / np.sqrt(v0)
    fit["tc"], fit["tu"] = fit_t(uc), fit_t(uu)
    fit["nhc"], fit["nhu"] = scores_t(uc, *fit["tc"]), scores_t(uu, *fit["tu"])
    return fit
def normal_scores_online(fit, h, zo):
    p, coef, lam = fit["p"], fit["coef"], fit["lam"]; P = bw.PMAX
    full = np.concatenate([h[-P:], zo]); m = len(full)
    e = full[P:] - (np.column_stack([full[P - j - 1: m - j - 1] for j in range(p)]) @ coef if p else 0.0); e2 = e ** 2
    s2 = np.full(len(e), fit["v0"]) if lam >= 1.0 else np.concatenate([[fit["s2_last"]], bw.ewma(e2[:-1], 1 - lam, init=fit["s2_last"])])
    nc = np.clip(scores_t(e / np.sqrt(s2), *fit["tc"]), -4.5, 4.5); nu = np.clip(scores_t(e / np.sqrt(fit["v0"]), *fit["tu"]), -4.5, 4.5)
    return nc, nu, (np.log(s2 / fit["v0"]) - fit["lm"]) / fit["ls"]
bw.fit_history = fit_history; bw.normal_scores_online = normal_scores_online
bw.OUT = "SCREEN_TPIT"; bw.PARTS = "screen_tpit_parts"
if __name__ == "__main__":
    sys.argv = [sys.argv[0]] + sys.argv[1:]
    exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "build_white.py")).read().split('if __name__ == "__main__":')[1].replace("\n    ", "\n"), {**bw.__dict__, "sys": sys, "time": __import__("time"), "os": os, "np": np, "OUT": "SCREEN_TPIT", "PARTS": "screen_tpit_parts", "white_channels": bw.white_channels, "NCH": bw.NCH})
