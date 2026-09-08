"""The streaming run-length monitor must reproduce the batch filter exactly.

The batch reference here is the builder that produced the training channels
(scripts/experiments/build_bocpd.py); the shipped module streams one point at
a time. Any drift between the two would put the classifier on channels it was
never trained on — the failure class the channel check in the assembler was
built to catch. Run with:  python -m unittest discover tests
"""

import unittest

import numpy as np
from scipy.special import gammaln

from structural_break.bocpd import BOCPD_HAZARD, BOCPD_RMAX, RunLengthMonitor


def _student_logpdf(x, mu, kappa, alpha, beta):
    nu = 2 * alpha
    scale2 = beta * (kappa + 1) / (alpha * kappa)
    z2 = (x - mu) ** 2 / scale2
    return (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi * scale2)
            - (nu + 1) / 2 * np.log1p(z2 / nu))


def batch_channels(hist, online):
    """The batch filter, as in the channel builder."""
    m0 = hist.mean(); v0 = hist.var() + 1e-12
    k0, a0 = 5.0, 2.5
    b0 = a0 * v0
    out = np.empty((len(online), 6))
    mu = np.array([m0]); kappa = np.array([k0]); alpha = np.array([a0]); beta = np.array([b0])
    logR = np.array([0.0])
    for t, x in enumerate(online):
        lp = _student_logpdf(x, mu, kappa, alpha, beta)
        log_growth = logR + lp + np.log(1 - BOCPD_HAZARD)
        log_cp = np.logaddexp.reduce(logR + lp) + np.log(BOCPD_HAZARD)
        newR = np.concatenate([[log_cp], log_growth])
        evidence = np.logaddexp.reduce(newR)
        newR -= evidence
        mu_new = np.concatenate([[m0], (kappa * mu + x) / (kappa + 1)])
        kappa_new = np.concatenate([[k0], kappa + 1])
        alpha_new = np.concatenate([[a0], alpha + 0.5])
        beta_new = np.concatenate([[b0], beta + kappa * (x - mu) ** 2 / (2 * (kappa + 1))])
        if len(newR) > BOCPD_RMAX:
            tail = np.logaddexp.reduce(newR[BOCPD_RMAX:])
            newR = newR[:BOCPD_RMAX + 1].copy(); newR[BOCPD_RMAX] = tail
            mu_new, kappa_new, alpha_new, beta_new = (a[:BOCPD_RMAX + 1] for a in (mu_new, kappa_new, alpha_new, beta_new))
        logR, mu, kappa, alpha, beta = newR, mu_new, kappa_new, alpha_new, beta_new
        P = np.exp(logR); r = np.arange(len(P))
        out[t] = [P[:5].sum(), P[:20].sum(), P[:60].sum(), P[:200].sum(), (P * r).sum() / (t + 1), -evidence]
    return out


class StreamingMatchesBatch(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.hist = rng.normal(0.0, 1.0, 400)
        # A mean shift at step 300 of a 900-step online part, long enough to
        # exercise the run-length cap at 600.
        self.online = np.concatenate([rng.normal(0.0, 1.0, 300), rng.normal(1.5, 1.0, 600)])

    def test_channels_agree(self):
        mon = RunLengthMonitor(self.hist)
        streamed = np.array([mon.update(float(x)) for x in self.online])
        batch = batch_channels(self.hist, self.online)
        self.assertEqual(streamed.shape, (900, 6))
        np.testing.assert_allclose(streamed, batch, rtol=0, atol=1e-9)

    def test_break_shows_in_recent_run_mass(self):
        mon = RunLengthMonitor(self.hist)
        streamed = np.array([mon.update(float(x)) for x in self.online])
        before = streamed[250:300, 1].mean()   # P(r < 20) just before the shift
        after = streamed[302:312, 1].mean()    # and just after it
        self.assertGreater(after, before + 0.3)

    def test_probabilities_are_probabilities(self):
        mon = RunLengthMonitor(self.hist)
        streamed = np.array([mon.update(float(x)) for x in self.online])
        self.assertTrue(np.all(streamed[:, :4] >= 0) and np.all(streamed[:, :4] <= 1 + 1e-12))
        self.assertTrue(np.all(np.diff(streamed[:, :4], axis=1) >= -1e-12))  # nested thresholds
        self.assertTrue(np.all(np.isfinite(streamed)))


if __name__ == "__main__":
    unittest.main()
