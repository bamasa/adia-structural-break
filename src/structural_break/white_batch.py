"""Batch builder of the whitened channels: the reference the stream is checked against.

:class:`structural_break.white.WhiteMonitor` streams one online point at a time.
The training matrices (WHITE90, SR22, HISTCTX8) were built by vectorised code
over whole series -- cumulative sums, ``lfilter`` exponential averages, running
minima for the CUSUMs, prefix sums for the dyadic windows. This module is that
code, moved out of ``scripts/experiments/build_white.py``, ``build_sr.py`` and
``build_histctx.py`` so that the two computations can be compared inside the
library: :func:`white_channels` returns, column for column, what
``WhiteMonitor(history, odds=odds, context=context).update(x)`` returns for each
online point (90, 111 or 119 columns), in float64.

The history fit and the online normal scores are injectable (``fit``,
``online_scores``) so that a variant whitening can reuse the battery; the
Student-t PIT of experiment 157e does exactly that from its script.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.signal import lfilter
from scipy.special import ndtr, ndtri

from .white import (
    CONTEXT_CHANNELS,
    SR_CHANNELS,
    SR_MEANS,
    SR_PHIS,
    SR_VAR_DOWN,
    SR_VAR_UP,
    WHITE_CHANNELS,
    WHITE_DRIFT,
    WHITE_DYADIC,
    WHITE_FULL,
    WHITE_LAGS,
    WHITE_LAMBDAS,
    WHITE_PMAX,
    WHITE_PORT,
    WHITE_TESTW,
)

__all__ = [
    "ewma",
    "cusum_peak",
    "fit_history",
    "normal_scores_online",
    "one_sample_tests",
    "log_sr",
    "odds_channels",
    "sr_channels",
    "history_context",
    "white_channels",
]


def ewma(a: np.ndarray, alpha: float, init: float = 0.0) -> np.ndarray:
    """Exponential average v_t = v_{t-1} + alpha (a_t - v_{t-1}), started at ``init``."""
    return lfilter([alpha], [1, -(1 - alpha)], a, zi=[(1 - alpha) * init])[0]


def cusum_peak(a: np.ndarray, drift: float = WHITE_DRIFT) -> np.ndarray:
    """Two-sided CUSUM with drift, as a running sum minus its running minimum."""
    cp = np.concatenate([[0.0], np.cumsum(a - drift)])
    cm = np.concatenate([[0.0], np.cumsum(-a - drift)])
    return np.maximum(cp[1:] - np.minimum.accumulate(cp)[1:], cm[1:] - np.minimum.accumulate(cm)[1:])


def _scores(u: np.ndarray, sorted_ref: np.ndarray) -> np.ndarray:
    return ndtri(np.clip((np.searchsorted(sorted_ref, u) + 0.5) / (len(sorted_ref) + 1), 1e-6, 1 - 1e-6))


def fit_history(h: np.ndarray) -> dict:
    """AR order by BIC on a common sample, conditional-scale lambda by quasi-likelihood, innovation ECDF.

    ``h`` is the standardised history. Returns the fit the online scoring needs:
    ``p, coef, lam, v0, s2_last, suc, suu`` (sorted conditional and unconditional
    standardised innovations), ``nhc, nhu`` (their normal scores, in time order)
    and ``lm, ls`` (mean and spread of the history's log conditional variance).
    """
    n = len(h)
    y = h[WHITE_PMAX:]
    best = (np.inf, 0, np.zeros(0))
    for p in range(0, WHITE_PMAX + 1):
        if p == 0:
            e, coef = y, np.zeros(0)
        else:
            X = np.column_stack([h[WHITE_PMAX - j - 1: n - j - 1] for j in range(p)])
            coef = np.linalg.lstsq(X, y, rcond=None)[0]
            e = y - X @ coef
        bic = len(e) * np.log((e ** 2).mean() + 1e-12) + p * np.log(len(e))
        if bic < best[0]:
            best = (bic, p, coef)
    p, coef = best[1], best[2]
    e = h[p:] - (np.column_stack([h[p - j - 1: n - j - 1] for j in range(p)]) @ coef if p else 0.0)
    e2 = e ** 2
    v0 = max(float(e2.mean()), 1e-12)
    bestl = (-np.inf, 1.0, np.full(len(e), v0))
    for lam in WHITE_LAMBDAS:
        s2 = np.full(len(e), v0) if lam >= 1.0 else np.concatenate([[v0], ewma(e2[:-1], 1 - lam, init=v0)])
        s2 = np.maximum(s2, 1e-12)
        ql = -0.5 * np.sum(np.log(s2) + e2 / s2)
        if np.isfinite(ql) and ql > bestl[0]:
            bestl = (ql, lam, s2)
    lam, s2 = bestl[1], bestl[2]
    uc, uu = e / np.sqrt(s2), e / np.sqrt(v0)
    suc, suu = np.sort(uc), np.sort(uu)
    l = np.log(s2 / v0)
    return dict(
        p=p, coef=coef, lam=lam,
        s2_last=float(lam * s2[-1] + (1 - lam) * e2[-1]) if lam < 1.0 else float(v0),
        v0=float(v0), suc=suc, suu=suu, nhc=_scores(uc, suc), nhu=_scores(uu, suu),
        lm=float(l.mean()), ls=float(l.std()) + 1e-6,
    )


def normal_scores_online(fit: dict, h: np.ndarray, zo: np.ndarray):
    """Conditional and unconditional normal scores of the online points, and the standardised log scale."""
    p, coef, lam = fit["p"], fit["coef"], fit["lam"]
    full = np.concatenate([h[-WHITE_PMAX:], zo])
    m = len(full)
    e = full[WHITE_PMAX:] - (np.column_stack([full[WHITE_PMAX - j - 1: m - j - 1] for j in range(p)]) @ coef if p else 0.0)
    e2 = e ** 2
    if lam >= 1.0:
        s2 = np.full(len(e), fit["v0"])
    else:
        s2 = np.concatenate([[fit["s2_last"]], ewma(e2[:-1], 1 - lam, init=fit["s2_last"])])

    def scores(u, su):
        return np.clip(_scores(u, su), -4.5, 4.5)

    return (scores(e / np.sqrt(s2), fit["suc"]), scores(e / np.sqrt(fit["v0"]), fit["suu"]),
            (np.log(s2 / fit["v0"]) - fit["lm"]) / fit["ls"])


def one_sample_tests(w: np.ndarray):
    """KS, Cramer-von Mises and Anderson-Darling of a window against N(0,1)."""
    x = np.sort(w)
    m = len(x)
    F = ndtr(x)
    i = np.arange(1, m + 1)
    ks = np.sqrt(m) * max((i / m - F).max(), (F - (i - 1) / m).max())
    cvm = 1.0 / (12 * m) + ((F - (2 * i - 1) / (2 * m)) ** 2).sum()
    Fc = np.clip(F, 1e-10, 1 - 1e-10)
    ad = -m - ((2 * i - 1) * (np.log(Fc) + np.log(1 - Fc[::-1]))).sum() / m
    return ks, cvm, ad


def _extended(n_on: np.ndarray, nh: np.ndarray):
    """The extended stream (history scores, then online) and the online offset into it."""
    tail = nh[-max(WHITE_DYADIC):]
    return np.concatenate([tail, n_on]), len(tail)


def _dyadic_glr(ext: np.ndarray, off: int, T: int):
    """Mean, scale and lag-1 dependence GLR over the dyadic windows of the extended stream, per window."""
    C1 = np.concatenate([[0.0], np.cumsum(ext)])
    C2 = np.concatenate([[0.0], np.cumsum(ext ** 2)])
    Cx = np.concatenate([[0.0], np.cumsum(ext[1:] * ext[:-1])])
    idx = np.arange(off, off + T) + 1          # exclusive end index into the prefix sums
    means, scales, deps = [], [], []
    for m in WHITE_DYADIC:
        lo = np.maximum(idx - m, 0)
        me = idx - lo                          # the window is the last m values, or all there are
        s1 = C1[idx] - C1[lo]
        s2 = C2[idx] - C2[lo]
        sx = Cx[idx - 1] - Cx[np.maximum(lo - 1, 0)]
        var = np.maximum(s2 / me, 1e-6)
        r1 = np.clip(sx / np.maximum(s2, 1e-9), -0.99, 0.99)
        means.append(np.abs(s1) / np.sqrt(me))
        scales.append(0.5 * me * (var - 1 - np.log(var)))
        deps.append(-0.5 * me * np.log(1 - r1 ** 2))
    return means, scales, deps


def _battery(n_on: np.ndarray, nh: np.ndarray, out: np.ndarray, c: int) -> int:
    """The full battery (76 channels) on one normal-score stream, written into ``out[:, c:]``; returns the next column."""
    T = len(n_on)
    ext, off = _extended(n_on, nh)
    n = n_on
    o = out[:, c:]
    # mean
    o[:, 0] = cusum_peak(n)
    o[:, 1] = ewma(n, 0.02)
    o[:, 2] = ewma(n, 0.005)
    o[:, 3] = np.cumsum(n) / np.sqrt(np.arange(1, T + 1))
    # scale
    q = (n ** 2 - 1) / np.sqrt(2)
    o[:, 4] = cusum_peak(q)
    o[:, 5] = ewma(q, 0.02)
    o[:, 6] = ewma(q, 0.005)
    o[:, 7] = (np.cumsum(n ** 2) / np.arange(1, T + 1) - 1) * np.sqrt(np.arange(1, T + 1) / 2)
    k_ = 8
    # dependence
    port = []
    for k in range(1, WHITE_PORT + 1):
        a = ext[off:] * ext[off - k: len(ext) - k]
        port.append(ewma(a, 0.01))
        if k in WHITE_LAGS:
            o[:, k_] = cusum_peak(a)
            o[:, k_ + 1] = port[-1]
            k_ += 2
    o[:, k_] = 199 * (np.stack(port) ** 2).sum(0)
    k_ += 1
    # volatility clustering, against the history's own level
    ah = np.abs(nh)
    mh, sh = (ah[1:] * ah[:-1]).mean(), (ah[1:] * ah[:-1]).std() + 1e-9
    a = (np.abs(ext[off:]) * np.abs(ext[off - 1: len(ext) - 1]) - mh) / sh
    o[:, k_] = cusum_peak(a)
    o[:, k_ + 1] = ewma(a, 0.01)
    k_ += 2
    # shape and tails, against the history's own frequencies
    o[:, k_] = ewma(n ** 3, 0.01)
    o[:, k_ + 1] = ewma(n ** 4 - 3, 0.01)
    for j, (ev_on, ev_h) in enumerate(((np.abs(n) > 2.5, ah > 2.5), (np.abs(n) > 1.5, ah > 1.5), (np.abs(n) < 0.3, ah < 0.3))):
        o[:, k_ + 2 + j] = ewma(ev_on.astype(float) - ev_h.mean(), 0.01)
    sc = (np.sign(ext[off:]) != np.sign(ext[off - 1: len(ext) - 1])).astype(float)
    o[:, k_ + 5] = ewma(sc - (np.sign(nh[1:]) != np.sign(nh[:-1])).mean(), 0.01)
    o[:, k_ + 6] = ewma(np.abs(n) - ah.mean(), 0.01)
    k_ += 7
    # GLR over dyadic windows on the extended stream, per window and maximised
    means, scales, deps = _dyadic_glr(ext, off, T)
    for g_mean, g_scale, g_dep in zip(means, scales, deps):
        o[:, k_] = g_mean
        o[:, k_ + 1] = g_scale
        o[:, k_ + 2] = g_dep
        k_ += 3
    for stat in (means, scales, deps):
        o[:, k_] = np.max(np.stack(stat), 0)
        k_ += 1
    # distribution tests at the geometric cadence, held between
    cur = np.zeros(3 * (len(WHITE_TESTW) + 1) + 3)
    nxt = 1
    for t in range(T):
        if t + 1 >= nxt:
            nxt = max(nxt + 1, int(nxt * 1.12))
            vals = []
            for W in WHITE_TESTW:
                vals.extend(one_sample_tests(ext[off + t + 1 - W: off + t + 1]))
            vals.extend(one_sample_tests(n[:t + 1]) if t + 1 >= 8 else (0.0, 0.0, 0.0))
            v = np.array(vals)
            cur = np.concatenate([v, v.reshape(-1, 3).max(0)])
        o[t, k_: k_ + len(cur)] = cur
    k_ += len(cur)
    return c + k_


def _cond_extras(n_on: np.ndarray, nh: np.ndarray, out: np.ndarray, c: int) -> int:
    """Mean and dependence extras (10 channels) on the conditional stream, written into ``out[:, c:]``.

    The conditional stream is sharper for mean and dependence when the history is
    heteroskedastic and blind to scale by construction (the scale estimate adapts),
    so it carries no scale statistics.
    """
    T = len(n_on)
    ext, off = _extended(n_on, nh)
    n = n_on
    out[:, c] = cusum_peak(n)
    out[:, c + 1] = ewma(n, 0.02)
    c += 2
    port = []
    for k in range(1, WHITE_PORT + 1):
        a = ext[off:] * ext[off - k: len(ext) - k]
        port.append(ewma(a, 0.01))
        if k in (1, 2, 5):
            out[:, c] = cusum_peak(a)
            c += 1
    out[:, c] = 199 * (np.stack(port) ** 2).sum(0)
    c += 1
    means, _, deps = _dyadic_glr(ext, off, T)
    out[:, c] = np.max(np.stack(means), 0)
    out[:, c + 1] = np.max(np.stack(deps), 0)
    c += 2
    cur = np.zeros(2)
    nxt = 1
    for t in range(T):
        if t + 1 >= nxt:
            nxt = max(nxt + 1, int(nxt * 1.12))
            ks, ad = [], []
            for W in WHITE_TESTW:
                k_, _, a_ = one_sample_tests(ext[off + t + 1 - W: off + t + 1])
                ks.append(k_)
                ad.append(a_)
            cur = np.array([max(ks), max(ad)])
        out[t, c: c + 2] = cur
    return c + 2


def log_sr(ell: np.ndarray) -> np.ndarray:
    """log Shiryaev-Roberts statistic from per-point log likelihood ratios.

    log R_t = L_t + log sum_{tau <= t} exp(-L_{tau-1}) with L the cumulative sum of
    ``ell``: the posterior odds of a change at some tau <= t under a uniform prior.
    """
    L = np.cumsum(ell)
    Lprev = np.concatenate([[0.0], L[:-1]])
    return L + np.logaddexp.accumulate(-Lprev)


def odds_channels(n_u: np.ndarray, prev0: float) -> np.ndarray:
    """The Shiryaev-Roberts odds (147) from the unconditional normal scores: 21 channels.

    ``prev0`` is the last history score, the lag-1 neighbour of the first online
    point for the dependence alternatives. Per alternative one column, then the
    equal-weight mixture of each family: variance up, variance down, mean, dependence.
    """
    n = np.asarray(n_u, dtype="float64")
    prev = np.concatenate([[prev0], n[:-1]])
    T = len(n)
    out = np.zeros((T, SR_CHANNELS))
    c = 0
    fam = {}
    for v in SR_VAR_UP + SR_VAR_DOWN:
        out[:, c] = log_sr(-0.5 * np.log(v) - 0.5 * n ** 2 * (1.0 / v - 1.0))
        c += 1
    fam["var_up"] = out[:, :len(SR_VAR_UP)].copy()
    fam["var_down"] = out[:, len(SR_VAR_UP):c].copy()
    for d in SR_MEANS:
        for sign in (1.0, -1.0):
            out[:, c] = log_sr(sign * d * n - 0.5 * d * d)
            c += 1
    fam["mean"] = out[:, c - 2 * len(SR_MEANS):c].copy()
    for phi in SR_PHIS:
        for sign in (1.0, -1.0):
            p = sign * phi
            out[:, c] = log_sr(-0.5 * np.log(1 - p * p) - 0.5 * ((n - p * prev) ** 2 / (1 - p * p) - n ** 2))
            c += 1
    fam["dep"] = out[:, c - 2 * len(SR_PHIS):c].copy()
    for key in ("var_up", "var_down", "mean", "dep"):
        F = fam[key]
        out[:, c] = np.logaddexp.reduce(F, axis=1) - np.log(F.shape[1])
        c += 1
    assert c == SR_CHANNELS, (c, SR_CHANNELS)
    return np.clip(np.nan_to_num(out), -50, 200)


def history_context(fit: dict, zh: np.ndarray) -> list[float]:
    """The history's family as eight constants (153): AR order, first coefficient,
    conditional-scale memory, log innovation variance, innovation excess kurtosis
    and skew, log history length, largest |z|."""
    u = fit["suu"]
    m2 = (u ** 2).mean() + 1e-12
    ctx = [float(fit["p"]), float(fit["coef"][0]) if fit["p"] else 0.0, float(fit["lam"]), float(np.log(fit["v0"])),
           float((u ** 4).mean() / m2 ** 2 - 3.0), float((u ** 3).mean() / m2 ** 1.5),
           float(np.log(len(zh))), float(np.abs(zh).max())]
    assert len(ctx) == CONTEXT_CHANNELS
    return ctx


def _standardise(hist, online):
    h = np.asarray(hist, dtype="float64")
    mu, sd = h.mean(), h.std() + 1e-12
    return (h - mu) / sd, (np.asarray(online, dtype="float64") - mu) / sd


def sr_channels(
    hist,
    online,
    *,
    fit: Callable[[np.ndarray], dict] = fit_history,
    online_scores: Callable = normal_scores_online,
) -> np.ndarray:
    """The 21 Shiryaev-Roberts odds channels of a series, from its raw history and online part."""
    zh, zo = _standardise(hist, online)
    f = fit(zh)
    _, n_u, _ = online_scores(f, zh, zo)
    return odds_channels(n_u, float(f["nhu"][-1]))


def white_channels(
    hist,
    online,
    odds: bool = False,
    context: bool = False,
    *,
    fit: Callable[[np.ndarray], dict] = fit_history,
    online_scores: Callable = normal_scores_online,
) -> np.ndarray:
    """The whitened channels of one series, computed in batch over its online part.

    Returns a float64 array of shape ``(len(online), width)`` whose rows are what
    ``WhiteMonitor(hist, odds=odds, context=context).update(x)`` returns for the
    successive online points: 90 columns, 111 with the Shiryaev-Roberts odds,
    119 with the eight history-context constants as well.
    """
    zh, zo = _standardise(hist, online)
    f = fit(zh)
    n_c, n_u, lz = online_scores(f, zh, zo)
    T = len(zo)
    out = np.zeros((T, WHITE_CHANNELS))
    c = _battery(n_u, f["nhu"], out, 0)
    assert c == WHITE_FULL, (c, WHITE_FULL)
    c = _cond_extras(n_c, f["nhc"], out, c)
    # The scale process itself: a persistent shift of the conditional variance against the history's level.
    out[:, c] = lz
    out[:, c + 1] = ewma(lz, 0.02)
    out[:, c + 2] = cusum_peak(lz)
    c += 3
    out[:, c] = np.arange(T)
    c += 1
    assert c == WHITE_CHANNELS, (c, WHITE_CHANNELS)
    out[:, :-1] = np.clip(out[:, :-1], -60, 60)          # the step index stays as it is
    out = np.nan_to_num(out)
    blocks = [out]
    if odds:
        blocks.append(odds_channels(n_u, float(f["nhu"][-1])))
    if context:
        blocks.append(np.tile(np.asarray(history_context(f, zh)), (T, 1)))
    return np.hstack(blocks) if len(blocks) > 1 else out
