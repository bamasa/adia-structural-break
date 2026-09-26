"""Dependence-score CUSUM: the sequential test for a change in temporal dependence.

The history fits an AR(5); its residuals are white by construction. If the
online dependence changes, residuals computed with the history's coefficients
become autocorrelated, and the score for a change in the lag-k coefficient is
the sum of lagged residual products. Page's two-sided CUSUM on the standardised
products (drift 0.25) is the sequential test with the best power for a small
sustained change -- what a CUSUM on squares is for variance, this is for
dependence. Per lag 1, 2, 3, 5, 10: the CUSUM peak and two exponential
averages of the product; a portmanteau over lags 1..10 at two horizons;
absolute products at lags 1, 2, 5 (volatility clustering) with CUSUM peak and
average. Twenty-three channels (experiment 144), read by their own member.
Reproduces the batch builder to 1e-5 on the float32 matrix.
"""

from __future__ import annotations

import numpy as np

DEP_ORDER = 5
DEP_LAGS = (1, 2, 3, 5, 10)
DEP_ABS_LAGS = (1, 2, 5)
DEP_DRIFT = 0.25
DEP_PORT = 10
DEP_CHANNELS = len(DEP_LAGS) * 3 + 2 + len(DEP_ABS_LAGS) * 2


class _Cusum2:
    """Page's two-sided CUSUM with drift; reports the larger arm."""

    __slots__ = ("up", "down")

    def __init__(self) -> None:
        self.up = self.down = 0.0

    def update(self, a: float) -> float:
        self.up = max(0.0, self.up + a - DEP_DRIFT)
        self.down = max(0.0, self.down - a - DEP_DRIFT)
        return max(self.up, self.down)


class DepCusum:
    """Streaming dependence-score CUSUM: one update per online point."""

    def __init__(self, history: np.ndarray) -> None:
        h = np.asarray(history, dtype="float64")
        self.mu, self.sd = float(h.mean()), float(h.std()) + 1e-12
        zh = (h - self.mu) / self.sd
        P = DEP_ORDER
        X = np.column_stack([zh[P - k - 1: len(zh) - k - 1] for k in range(P)])
        self.coef = np.linalg.lstsq(X, zh[P:], rcond=None)[0]
        e = zh[P:] - sum(self.coef[k] * zh[P - k - 1: len(zh) - k - 1] for k in range(P))
        self.sigma = float(e.std()) + 1e-9
        u = e / self.sigma
        au = np.abs(u)
        # Null moments of the lagged products, from the history's own residuals.
        self.m, self.s = {}, {}
        for k in range(1, DEP_PORT + 1):
            q = u[k:] * u[:-k]
            self.m[k], self.s[k] = float(q.mean()), float(q.std()) + 1e-9
        self.am, self.as_ = {}, {}
        for k in DEP_ABS_LAGS:
            r = au[k:] * au[:-k]
            self.am[k], self.as_[k] = float(r.mean()), float(r.std()) + 1e-9
        self.z_tail = list(zh[-P:])                 # last P standardised values
        self.u_tail = list(u[-DEP_PORT:])           # last ten standardised residuals
        self.cusum = {k: _Cusum2() for k in DEP_LAGS}
        self.e_fast = {k: 0.0 for k in DEP_LAGS}    # alpha 0.02
        self.e_slow = {k: 0.0 for k in DEP_LAGS}    # alpha 0.005
        self.port1 = {k: 0.0 for k in range(1, DEP_PORT + 1)}   # alpha 0.01
        self.port2 = {k: 0.0 for k in range(1, DEP_PORT + 1)}   # alpha 0.003
        self.acusum = {k: _Cusum2() for k in DEP_ABS_LAGS}
        self.a_ew = {k: 0.0 for k in DEP_ABS_LAGS}  # alpha 0.01

    def update(self, x: float) -> list[float]:
        z = (float(x) - self.mu) / self.sd
        e = z - sum(self.coef[k] * self.z_tail[-k - 1] for k in range(DEP_ORDER))
        self.z_tail.append(z)
        del self.z_tail[0]
        u = e / self.sigma
        out: list[float] = []
        tail = self.u_tail
        for k in range(1, DEP_PORT + 1):
            a = (u * tail[-k] - self.m[k]) / self.s[k]
            self.port1[k] += 0.01 * (a - self.port1[k])
            self.port2[k] += 0.003 * (a - self.port2[k])
            if k in self.cusum:
                self.e_fast[k] += 0.02 * (a - self.e_fast[k])
                self.e_slow[k] += 0.005 * (a - self.e_slow[k])
                out.extend((self.cusum[k].update(a), self.e_fast[k], self.e_slow[k]))
        out.append(199.0 * sum(v * v for v in self.port1.values()))
        out.append(666.0 * sum(v * v for v in self.port2.values()))
        au = abs(u)
        for k in DEP_ABS_LAGS:
            a = (au * abs(tail[-k]) - self.am[k]) / self.as_[k]
            self.a_ew[k] += 0.01 * (a - self.a_ew[k])
            out.extend((self.acusum[k].update(a), self.a_ew[k]))
        tail.append(u)
        del tail[0]
        return [float(np.clip(np.nan_to_num(v), -50, 50)) for v in out]
