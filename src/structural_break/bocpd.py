"""Bayesian online change-point detection as a streaming channel family.

Adams & MacKay's run-length filter over a Normal-Gamma model of the point,
prior fitted to the history, hazard 1/50, run lengths tracked to 600. Six
channels per step: the posterior mass on run lengths below 5, 20, 60 and 200
(how sure the filter is that a regime started recently), the expected run
length relative to the steps seen, and the point's surprise — the negative
log predictive density under the current mixture.

The trees' classifier gained 0.0046 on the untouched fold from these six;
the ranker and the trajectory networks did not want them (experiment 085),
so the monitor exposes them as a suffix the interface hands only to the
classifier. Verified against the batch builder to 1e-6.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln

#: Prior hazard of a regime change per step, and the longest run length kept.
BOCPD_HAZARD = 1.0 / 50
BOCPD_RMAX = 600
BOCPD_CHANNELS = 6


def _student_logpdf(x, mu, kappa, alpha, beta):
    nu = 2 * alpha
    scale2 = beta * (kappa + 1) / (alpha * kappa)
    z2 = (x - mu) ** 2 / scale2
    return (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi * scale2)
            - (nu + 1) / 2 * np.log1p(z2 / nu))


class RunLengthMonitor:
    """Streaming BOCPD: one update per online point, six channels out."""

    def __init__(self, history: np.ndarray) -> None:
        hist = np.asarray(history, dtype="float64")
        self.m0 = float(hist.mean())
        v0 = float(hist.var()) + 1e-12
        # The prior behaves as five history points at the history's mean and variance.
        self.k0, self.a0 = 5.0, 2.5
        self.b0 = self.a0 * v0
        self.mu = np.array([self.m0]); self.kappa = np.array([self.k0])
        self.alpha = np.array([self.a0]); self.beta = np.array([self.b0])
        self.logR = np.array([0.0])
        self._t = 0

    def update(self, x: float) -> list[float]:
        lp = _student_logpdf(x, self.mu, self.kappa, self.alpha, self.beta)
        log_growth = self.logR + lp + np.log(1 - BOCPD_HAZARD)
        log_cp = np.logaddexp.reduce(self.logR + lp) + np.log(BOCPD_HAZARD)
        newR = np.concatenate([[log_cp], log_growth])
        evidence = np.logaddexp.reduce(newR)
        newR -= evidence
        mu_new = np.concatenate([[self.m0], (self.kappa * self.mu + x) / (self.kappa + 1)])
        kappa_new = np.concatenate([[self.k0], self.kappa + 1])
        alpha_new = np.concatenate([[self.a0], self.alpha + 0.5])
        beta_new = np.concatenate([[self.b0], self.beta + self.kappa * (x - self.mu) ** 2 / (2 * (self.kappa + 1))])
        if len(newR) > BOCPD_RMAX:
            tail = np.logaddexp.reduce(newR[BOCPD_RMAX:])
            newR = newR[:BOCPD_RMAX + 1].copy(); newR[BOCPD_RMAX] = tail
            mu_new, kappa_new, alpha_new, beta_new = (
                a[:BOCPD_RMAX + 1] for a in (mu_new, kappa_new, alpha_new, beta_new))
        self.logR, self.mu, self.kappa, self.alpha, self.beta = newR, mu_new, kappa_new, alpha_new, beta_new
        P = np.exp(self.logR)
        r = np.arange(len(P))
        out = [float(P[:5].sum()), float(P[:20].sum()), float(P[:60].sum()), float(P[:200].sum()),
               float((P * r).sum() / (self._t + 1)), float(-evidence)]
        self._t += 1
        return out
