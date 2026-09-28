"""The whitened stream: every test on innovations the history has made i.i.d.

Every channel this project owned read the raw stream, its asinh, or an AR(1)
whitening of it. The public survey (27 September) found the strongest
real-time recipe rests on whitening the stream *properly*: the history fits
an AR(p <= 12) by BIC, a conditional scale (an exponentially weighted
variance, its memory chosen by Gaussian quasi-likelihood), and the empirical
CDF of the standardised innovations; the online stream is mapped through
that fit to normal scores that are i.i.d. N(0,1) under the null for a far
wider family of histories -- real-world series included -- than an AR(1)
covers, and every departure from i.i.d. N(0,1) is a break of some kind.

Two innovation streams are kept. The unconditional one (history innovation
variance) carries the full battery: CUSUMs and exponential averages for
mean, scale, dependence (lags 1, 2, 3, 5, 10 and a portmanteau), volatility
clustering, shape and tails; generalised likelihood ratios for mean, scale
and lag-1 dependence over dyadic windows 8..1024, per scale and maximised;
and one-sample KS, Cramer-von Mises and Anderson-Darling tests against
N(0,1) over windows 32..512 and the online prefix, recomputed on the
geometric cadence. The conditional stream -- sharper for mean and dependence
when the history is heteroskedastic, blind to scale by construction -- adds
mean and dependence extras, and the conditional scale process itself is
read against its history level. Plus the step index: ninety channels
(experiment 145: 0.6088 alone, +0.0107 to the blend on fold 2).

With ``odds=True`` the Shiryaev-Roberts odds of 147 ride along: twenty-one
more channels, one hundred and eleven in all.

Reproduces the batch builders to 1e-4 on the float32 matrices.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr, ndtri

WHITE_PMAX = 12
WHITE_LAMBDAS = (0.90, 0.94, 0.97, 0.99, 1.0)
WHITE_DRIFT = 0.25
WHITE_LAGS = (1, 2, 3, 5, 10)
WHITE_PORT = 10
WHITE_DYADIC = (8, 16, 32, 64, 128, 256, 512, 1024)
WHITE_TESTW = (32, 64, 128, 256, 512)
WHITE_FULL = 8 + len(WHITE_LAGS) * 2 + 1 + 2 + 7 + 3 * len(WHITE_DYADIC) + 3 + 3 * (len(WHITE_TESTW) + 1) + 3
WHITE_CHANNELS = WHITE_FULL + 10 + 3 + 1
_RING = max(WHITE_DYADIC) + 1

#: Shiryaev-Roberts odds on the unconditional stream (147): the accumulated
#: posterior odds of a change at some tau <= t under a uniform prior on tau,
#: for a grid of alternatives per break family, plus an equal-weight mixture
#: per family. R_t = (1 + R_{t-1}) * likelihood ratio of the point: one
#: recursion per alternative per step.
SR_VAR_UP = (1.25, 1.5, 2.0, 3.0, 5.0)
SR_VAR_DOWN = (0.7, 0.5)
SR_MEANS = (0.3, 0.6, 1.0)
SR_PHIS = (0.2, 0.4)
SR_CHANNELS = len(SR_VAR_UP) + len(SR_VAR_DOWN) + 2 * len(SR_MEANS) + 2 * len(SR_PHIS) + 4
WHITE_ODDS_CHANNELS = WHITE_CHANNELS + SR_CHANNELS
#: The history's family as static context (153): AR order, first coefficient,
#: conditional-scale memory, log innovation variance, innovation excess
#: kurtosis and skew, log history length, largest |z|. Constant per series.
CONTEXT_CHANNELS = 8


class _ShiryaevRoberts:
    """Log odds per alternative, streamed; the families' mixtures appended."""

    def __init__(self, prev: float) -> None:
        self.prev = prev
        n_alt = len(SR_VAR_UP) + len(SR_VAR_DOWN) + 2 * len(SR_MEANS) + 2 * len(SR_PHIS)
        self.logR = np.full(n_alt, -np.inf)
        self.fam = ((0, len(SR_VAR_UP)), (len(SR_VAR_UP), len(SR_VAR_UP) + len(SR_VAR_DOWN)),
                    (len(SR_VAR_UP) + len(SR_VAR_DOWN), len(SR_VAR_UP) + len(SR_VAR_DOWN) + 2 * len(SR_MEANS)),
                    (n_alt - 2 * len(SR_PHIS), n_alt))

    def update(self, n: float) -> list[float]:
        ell = []
        for v in SR_VAR_UP + SR_VAR_DOWN:
            ell.append(-0.5 * np.log(v) - 0.5 * n * n * (1.0 / v - 1.0))
        for d in SR_MEANS:
            for sign in (1.0, -1.0):
                ell.append(sign * d * n - 0.5 * d * d)
        for phi in SR_PHIS:
            for sign in (1.0, -1.0):
                p = sign * phi
                ell.append(-0.5 * np.log(1 - p * p) - 0.5 * ((n - p * self.prev) ** 2 / (1 - p * p) - n * n))
        self.logR = np.logaddexp(0.0, self.logR) + np.asarray(ell)
        self.prev = n
        out = self.logR.tolist()
        for a, b in self.fam:
            out.append(float(np.logaddexp.reduce(self.logR[a:b]) - np.log(b - a)))
        return [float(np.clip(np.nan_to_num(v), -50, 200)) for v in out]


def _scores(u: np.ndarray, sorted_ref: np.ndarray) -> np.ndarray:
    return ndtri(np.clip((np.searchsorted(sorted_ref, u) + 0.5) / (len(sorted_ref) + 1), 1e-6, 1 - 1e-6))


def _one_sample_tests(w: np.ndarray):
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


class _Cusum2:
    __slots__ = ("up", "down")

    def __init__(self) -> None:
        self.up = self.down = 0.0

    def update(self, a: float) -> float:
        self.up = max(0.0, self.up + a - WHITE_DRIFT)
        self.down = max(0.0, self.down - a - WHITE_DRIFT)
        return max(self.up, self.down)


class _Ewma:
    __slots__ = ("alpha", "v")

    def __init__(self, alpha: float) -> None:
        self.alpha, self.v = alpha, 0.0

    def update(self, a: float) -> float:
        self.v += self.alpha * (a - self.v)
        return self.v


class _Ring:
    """The last _RING values of the extended stream (history scores, then online)."""

    def __init__(self, tail: np.ndarray) -> None:
        self.buf = np.zeros(_RING)
        self.n = 0
        for v in tail[-_RING:]:
            self.push(float(v))

    def push(self, v: float) -> None:
        if self.n < _RING:
            self.buf[self.n] = v
            self.n += 1
        else:
            self.buf[:-1] = self.buf[1:]
            self.buf[-1] = v

    def last(self, m: int) -> np.ndarray:
        return self.buf[max(self.n - m, 0): self.n]

    def before(self, m: int) -> float | None:
        j = self.n - m - 1
        return float(self.buf[j]) if j >= 0 else None


def _glr(ring: _Ring, scale_stats: bool):
    """Mean, (scale,) lag-1 dependence GLR over the dyadic windows, per scale and maximised."""
    means, scales, deps = [], [], []
    for m in WHITE_DYADIC:
        seg = ring.last(m)
        me = len(seg)
        s1 = seg.sum()
        s2 = (seg ** 2).sum()
        sx = float((seg[1:] * seg[:-1]).sum())
        prev = ring.before(me)
        if prev is not None:
            sx += seg[0] * prev
        var = max(s2 / me, 1e-6)
        r1 = float(np.clip(sx / max(s2, 1e-9), -0.99, 0.99))
        means.append(abs(s1) / np.sqrt(me))
        scales.append(0.5 * me * (var - 1 - np.log(var)))
        deps.append(-0.5 * me * np.log(1 - r1 ** 2))
    if scale_stats:
        per = [v for trio in zip(means, scales, deps) for v in trio]
        return per + [max(means), max(scales), max(deps)]
    return [max(means), max(deps)]


class _FullBattery:
    """The seventy-six channels on one normal-score stream."""

    def __init__(self, nh: np.ndarray) -> None:
        self.ring = _Ring(nh)
        self.hist_tail = list(nh[-WHITE_PORT:])
        self.prev = float(nh[-1])
        ah = np.abs(nh)
        prod = ah[1:] * ah[:-1]
        self.vol_m, self.vol_s = float(prod.mean()), float(prod.std()) + 1e-9
        self.p_hi, self.p_mid, self.p_lo = float((ah > 2.5).mean()), float((ah > 1.5).mean()), float((ah < 0.3).mean())
        self.p_sign = float((np.sign(nh[1:]) != np.sign(nh[:-1])).mean())
        self.abs_mean = float(ah.mean())
        self.c_mean, self.c_scale, self.c_vol = _Cusum2(), _Cusum2(), _Cusum2()
        self.e_mean = (_Ewma(0.02), _Ewma(0.005))
        self.e_scale = (_Ewma(0.02), _Ewma(0.005))
        self.c_lag = {k: _Cusum2() for k in WHITE_LAGS}
        self.port = {k: _Ewma(0.01) for k in range(1, WHITE_PORT + 1)}
        self.e_vol = _Ewma(0.01)
        self.e_shape = [_Ewma(0.01) for _ in range(7)]
        self.sum1 = self.sum2 = 0.0
        self.lagbuf = list(nh[-WHITE_PORT:])     # the last ten scores, history then online
        self.prefix: list[float] = []
        self.next_scan = 1
        self.tests = np.zeros(3 * (len(WHITE_TESTW) + 1) + 3)

    def update(self, n: float, t: int) -> list[float]:
        out = []
        self.sum1 += n
        self.sum2 += n * n
        out.append(self.c_mean.update(n))
        out.extend(e.update(n) for e in self.e_mean)
        out.append(self.sum1 / np.sqrt(t + 1))
        q = (n * n - 1) / np.sqrt(2.0)
        out.append(self.c_scale.update(q))
        out.extend(e.update(q) for e in self.e_scale)
        out.append((self.sum2 / (t + 1) - 1) * np.sqrt((t + 1) / 2.0))
        lb = self.lagbuf
        port_sq = 0.0
        for k in range(1, WHITE_PORT + 1):
            a = n * lb[-k]
            pv = self.port[k].update(a)
            port_sq += pv * pv
            if k in self.c_lag:
                out.append(self.c_lag[k].update(a))
                out.append(pv)
        out.append(199.0 * port_sq)
        a = (abs(n) * abs(lb[-1]) - self.vol_m) / self.vol_s
        out.append(self.c_vol.update(a))
        out.append(self.e_vol.update(a))
        an = abs(n)
        shape_in = (n ** 3, n ** 4 - 3, float(an > 2.5) - self.p_hi, float(an > 1.5) - self.p_mid,
                    float(an < 0.3) - self.p_lo, float(np.sign(n) != np.sign(lb[-1])) - self.p_sign, an - self.abs_mean)
        out.extend(e.update(v) for e, v in zip(self.e_shape, shape_in))
        lb.append(n)
        del lb[0]
        self.ring.push(n)
        out.extend(_glr(self.ring, scale_stats=True))
        self.prefix.append(n)
        if t + 1 >= self.next_scan:
            self.next_scan = max(self.next_scan + 1, int(self.next_scan * 1.12))
            vals = []
            for W in WHITE_TESTW:
                vals.extend(_one_sample_tests(self.ring.last(W)))
            vals.extend(_one_sample_tests(np.asarray(self.prefix)) if t + 1 >= 8 else (0.0, 0.0, 0.0))
            v = np.array(vals)
            self.tests = np.concatenate([v, v.reshape(-1, 3).max(0)])
        out.extend(self.tests.tolist())
        return out


class _CondExtras:
    """Mean and dependence extras on the conditional stream: ten channels."""

    def __init__(self, nh: np.ndarray) -> None:
        self.ring = _Ring(nh)
        self.c_mean, self.e_mean = _Cusum2(), _Ewma(0.02)
        self.c_lag = {k: _Cusum2() for k in (1, 2, 5)}
        self.port = {k: _Ewma(0.01) for k in range(1, WHITE_PORT + 1)}
        self.lagbuf = list(nh[-WHITE_PORT:])
        self.next_scan = 1
        self.tests = np.zeros(2)

    def update(self, n: float, t: int) -> list[float]:
        out = [self.c_mean.update(n), self.e_mean.update(n)]
        lb = self.lagbuf
        port_sq = 0.0
        for k in range(1, WHITE_PORT + 1):
            a = n * lb[-k]
            pv = self.port[k].update(a)
            port_sq += pv * pv
            if k in self.c_lag:
                out.append(self.c_lag[k].update(a))
        out.append(199.0 * port_sq)
        lb.append(n)
        del lb[0]
        self.ring.push(n)
        out.extend(_glr(self.ring, scale_stats=False))
        if t + 1 >= self.next_scan:
            self.next_scan = max(self.next_scan + 1, int(self.next_scan * 1.12))
            ks, ad = [], []
            for W in WHITE_TESTW:
                k_, _, a_ = _one_sample_tests(self.ring.last(W))
                ks.append(k_)
                ad.append(a_)
            self.tests = np.array([max(ks), max(ad)])
        out.extend(self.tests.tolist())
        return out


class WhiteMonitor:
    """The whitened-stream battery, streamed one online point at a time."""

    def __init__(self, history: np.ndarray, odds: bool = False, context: bool = False) -> None:
        h = np.asarray(history, dtype="float64")
        self.mu, self.sd = float(h.mean()), float(h.std()) + 1e-12
        zh = (h - self.mu) / self.sd
        n = len(zh)
        P = WHITE_PMAX
        y = zh[P:]
        best = (np.inf, 0, np.zeros(0))
        for p in range(0, P + 1):
            if p == 0:
                e, coef = y, np.zeros(0)
            else:
                X = np.column_stack([zh[P - j - 1: n - j - 1] for j in range(p)])
                coef = np.linalg.lstsq(X, y, rcond=None)[0]
                e = y - X @ coef
            bic = len(e) * np.log((e ** 2).mean() + 1e-12) + p * np.log(len(e))
            if bic < best[0]:
                best = (bic, p, coef)
        p, self.coef = best[1], best[2]
        self.p = p
        e = zh[p:] - (np.column_stack([zh[p - j - 1: n - j - 1] for j in range(p)]) @ self.coef if p else 0.0)
        e2 = e ** 2
        v0 = max(float(e2.mean()), 1e-12)
        bestl = (-np.inf, 1.0, np.full(len(e), v0))
        for lam in WHITE_LAMBDAS:
            if lam >= 1.0:
                s2 = np.full(len(e), v0)
            else:
                s2 = np.empty(len(e))
                s2[0] = v0
                for i in range(1, len(e)):
                    s2[i] = lam * s2[i - 1] + (1 - lam) * e2[i - 1]
            s2 = np.maximum(s2, 1e-12)
            ql = -0.5 * float(np.sum(np.log(s2) + e2 / s2))
            if np.isfinite(ql) and ql > bestl[0]:
                bestl = (ql, lam, s2)
        self.lam, s2 = bestl[1], bestl[2]
        self.v0 = v0
        self.s2 = float(self.lam * s2[-1] + (1 - self.lam) * e2[-1]) if self.lam < 1.0 else v0
        uc, uu = e / np.sqrt(s2), e / np.sqrt(v0)
        self.suc, self.suu = np.sort(uc), np.sort(uu)
        l = np.log(s2 / v0)
        self.lm, self.ls = float(l.mean()), float(l.std()) + 1e-6
        nhu = _scores(uu, self.suu)
        m2 = float((uu ** 2).mean()) + 1e-12
        self.context = [float(p), float(self.coef[0]) if p else 0.0, float(self.lam), float(np.log(v0)),
                        float((uu ** 4).mean()) / m2 ** 2 - 3.0, float((uu ** 3).mean()) / m2 ** 1.5,
                        float(np.log(n)), float(np.abs(zh).max())]
        self.full = _FullBattery(nhu)
        self.cond = _CondExtras(_scores(uc, self.suc))
        self.odds = _ShiryaevRoberts(float(nhu[-1])) if odds else None
        self.with_context = context
        self.z_tail = list(zh[-P:])
        self.e_lz, self.c_lz = _Ewma(0.02), _Cusum2()
        self._t = 0

    def update(self, x: float) -> list[float]:
        z = (float(x) - self.mu) / self.sd
        e = z - sum(self.coef[j] * self.z_tail[-j - 1] for j in range(self.p))
        self.z_tail.append(z)
        del self.z_tail[0]
        s2 = max(self.s2, 1e-12)
        n_c = float(np.clip(_scores(np.array([e / np.sqrt(s2)]), self.suc)[0], -4.5, 4.5))
        n_u = float(np.clip(_scores(np.array([e / np.sqrt(self.v0)]), self.suu)[0], -4.5, 4.5))
        lz = (np.log(s2 / self.v0) - self.lm) / self.ls
        if self.lam < 1.0:
            self.s2 = self.lam * s2 + (1 - self.lam) * e * e
        t = self._t
        self._t += 1
        out = self.full.update(n_u, t) + self.cond.update(n_c, t)
        out.extend((lz, self.e_lz.update(lz), self.c_lz.update(lz)))
        res = [float(np.clip(np.nan_to_num(v), -60, 60)) for v in out] + [float(t)]
        if self.odds is not None:
            res.extend(self.odds.update(n_u))
        if self.with_context:
            res.extend(self.context)
        return res
