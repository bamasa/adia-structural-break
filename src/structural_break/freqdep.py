"""Frequency and dependence channels: the ensemble's independent member.

Every strong member of the ensemble reads level and scale over windows, and
every attempt to add another such member landed at 0.8-0.95 correlation with
the ones already there (experiments 115-123). What the ensemble does not
read is the frequency content and the serial dependence of the online part —
the properties that 109 found behind the breaks it misses. Members built on
those correlate 0.12-0.35 with the ensemble (124, 129): independent, and
individually too weak to pay.

Combined under one slow, shallow classifier they pay (131b): rolling spectra
on four short windows and five long ones, and exponentially-weighted
autocorrelations at four lags over two windows — one hundred channels, each
compared with the history's own profile. This module streams them one point
at a time; the classes reproduce the batch builders that made the training
matrices to 1e-6.
"""

from __future__ import annotations

import numpy as np

#: Short and long spectral windows; band counts per window family.
SPECTRUM_SHORT = ((32, 64, 128, 256), 6)
SPECTRUM_LONG = ((64, 128, 256, 512, 1024), 10)
#: Autocorrelation lags and exponential windows (in points).
ACF_LAGS = (1, 2, 5, 10)
ACF_WINDOWS = (50.0, 200.0)
#: Channels emitted: 4*(6+2) + 5*(10+2) + 4*2.
FREQDEP_CHANNELS = 4 * 8 + 5 * 12 + len(ACF_LAGS) * len(ACF_WINDOWS)


def _band_edges(w: int, nb: int) -> np.ndarray:
    k = w // 2
    e = np.unique(np.round(np.geomspace(1, k, nb + 1)).astype(int))
    while len(e) < nb + 1:
        e = np.append(e, e[-1] + 1)
    return e[: nb + 1]


def _spec_feats(seg: np.ndarray, edges: np.ndarray, nb: int) -> np.ndarray:
    w = len(seg)
    sp = np.abs(np.fft.rfft((seg - seg.mean()) * np.hanning(w)))[1:] ** 2
    p = sp / (sp.sum() + 1e-12)
    k = len(p)
    bands = np.array([p[min(edges[i] - 1, k): min(edges[i + 1] - 1, k)].sum() for i in range(nb)])
    ent = -(p * np.log(p + 1e-12)).sum()
    cen = (p * (np.arange(1, k + 1) / w)).sum()
    return np.concatenate([np.log(bands + 1e-6), [ent, cen]])


class MultiSpectrum:
    """Rolling spectra on several windows, each against the history's band profile.

    ``short`` reproduces the 32-256 builder (history profile from
    non-overlapping windows); the long family uses half-window overlap with the
    window capped at the history's length and zero-pads the tail, because
    fifty-one histories are shorter than 1024 points.
    """

    def __init__(self, history: np.ndarray, windows, nb: int, short: bool) -> None:
        h = np.asarray(history, dtype="float64")
        self.mu, self.sd = float(h.mean()), float(h.std()) + 1e-12
        zh = (h - self.mu) / self.sd
        self.windows, self.nb = tuple(windows), nb
        self.edges = {w: _band_edges(w, nb) for w in self.windows}
        self.ref = {}
        for w in self.windows:
            if short:
                segs = [_spec_feats(zh[i: i + w], self.edges[w], nb) for i in range(0, len(zh) - w + 1, w)]
            else:
                we = min(w, len(zh)); step = max(we // 2, 1)
                segs = [_spec_feats(zh[i: i + we], self.edges[w], nb) for i in range(0, len(zh) - we + 1, step)]
                if len(segs) < 2:
                    segs = segs + segs
            S = np.array(segs)
            self.ref[w] = (S.mean(0), S.std(0) + 1e-3)
        m = max(self.windows)
        tail = zh[-m:]
        if not short and len(tail) < m:
            tail = np.concatenate([np.zeros(m - len(tail)), tail])
        self.buf = list(tail)
        self.cap = m

    def update(self, x: float) -> list[float]:
        self.buf.append((float(x) - self.mu) / self.sd)
        if len(self.buf) > self.cap:
            self.buf.pop(0)
        arr = np.asarray(self.buf)
        out = []
        for w in self.windows:
            f = _spec_feats(arr[-w:], self.edges[w], self.nb)
            mref, sref = self.ref[w]
            out.extend(np.clip((f - mref) / sref, -20, 20).tolist())
        return out


class RollingACF:
    """Exponentially-weighted autocorrelations at fixed lags, minus the history's."""

    def __init__(self, history: np.ndarray) -> None:
        h = np.asarray(history, dtype="float64")
        self.mu, self.sd = float(h.mean()), float(h.std()) + 1e-12
        zh = (h - self.mu) / self.sd
        self.hist_acf = {k: float(np.corrcoef(zh[:-k], zh[k:])[0, 1]) for k in ACF_LAGS}
        self.buf = list(zh[-max(ACF_LAGS):])
        self.num = {(k, w): 0.0 for k in ACF_LAGS for w in ACF_WINDOWS}
        self.den = {w: 1.0 for w in ACF_WINDOWS}

    def update(self, x: float) -> list[float]:
        z = (float(x) - self.mu) / self.sd
        for w in ACF_WINDOWS:
            a = 1.0 / w
            self.den[w] = (1 - a) * self.den[w] + a * z * z
        out = [0.0] * (len(ACF_LAGS) * len(ACF_WINDOWS))
        for j, k in enumerate(ACF_LAGS):
            zk = self.buf[-k] if len(self.buf) >= k else 0.0
            for i, w in enumerate(ACF_WINDOWS):
                a = 1.0 / w
                self.num[(k, w)] = (1 - a) * self.num[(k, w)] + a * z * zk
                r = self.num[(k, w)] / max(self.den[w], 1e-9)
                out[i * len(ACF_LAGS) + j] = float(np.clip(r, -1.5, 1.5) - self.hist_acf[k])
        self.buf.append(z)
        if len(self.buf) > max(ACF_LAGS) + 1:
            self.buf.pop(0)
        return out


class FreqDep:
    """The hundred channels of the independent member, in the training matrices' order."""

    def __init__(self, history: np.ndarray) -> None:
        self.short = MultiSpectrum(history, *SPECTRUM_SHORT, short=True)
        self.long = MultiSpectrum(history, *SPECTRUM_LONG, short=False)
        self.acf = RollingACF(history)

    def update(self, x: float) -> list[float]:
        return self.short.update(x) + self.long.update(x) + self.acf.update(x)
