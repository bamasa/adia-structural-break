"""054: spectrum v2 — windows 32/64/128, spectrum of increments, band ratios."""
import sys, time
sys.path.insert(0, "repo/src")
import numpy as np
import pandas as pd
from structural_break.features import Normalisation
from structural_break.stream import iter_series

t0 = time.time()
x = pd.read_parquet("structural-break-real-time-test/data/X_train.parquet")
y = pd.read_parquet("structural-break-real-time-test/data/y_train.parquet")
print(f"loading {time.time()-t0:.0f}s", flush=True)

WINS = (32, 128)
BANDS = 6

def spec_feats(v, nfft):
    if len(v) < nfft:
        v = np.concatenate([np.zeros(nfft - len(v)), v])
    seg = v[-nfft:] * np.hanning(nfft)
    p = np.abs(np.fft.rfft(seg)) ** 2
    p = p[1:]
    tot = p.sum() + 1e-12
    pn = p / tot
    band = np.add.reduceat(pn, np.linspace(0, len(pn), BANDS + 1)[:-1].astype(int))
    half = len(pn) // 2
    lo_hi = float(np.log((pn[:half].sum() + 1e-9) / (pn[half:].sum() + 1e-9)))
    ent = float(-(pn * np.log(pn + 1e-12)).sum() / np.log(len(pn)))
    return band, lo_hi, ent

def profile(z, nfft):
    B, LH, E = [], [], []
    stride = max(nfft // 2, 1)
    for i in range(nfft, len(z) + 1, stride):
        b, lh, e = spec_feats(z[i - nfft:i], nfft)
        B.append(b); LH.append(lh); E.append(e)
    if not B:
        b, lh, e = spec_feats(z, nfft)
        return b, lh, e, np.ones(BANDS) * 0.1, 0.3, 0.1
    Bm = np.stack(B)
    return (Bm.mean(0), float(np.mean(LH)), float(np.mean(E)),
            Bm.std(0) + 1e-3, float(np.std(LH)) + 1e-3, float(np.std(E)) + 1e-3)

rows, count = [], 0
NFEAT = len(WINS) * 2 * (BANDS + 2)      # windows x (level, increments) x (bands+2)
for sid, hist, online, labels in iter_series(x, y):
    norm = Normalisation.fit(hist)
    n = len(hist)
    z_h = np.asarray([norm.clip(norm.standardise(float(v), i - n)) for i, v in enumerate(hist)])
    z_o = np.asarray([norm.clip(norm.standardise(float(v), i)) for i, v in enumerate(online)])
    d_h = np.diff(z_h, prepend=z_h[0])
    profiles = {}
    for w in WINS:
        profiles[(w, "lvl")] = profile(z_h, w)
        profiles[(w, "dif")] = profile(d_h, w)
    buf_l = list(z_h[-max(WINS):])
    buf_d = list(d_h[-max(WINS):])
    cur = [0.0] * NFEAT
    next_scan, step_i, prev = 1, 0, float(z_h[-1])
    for v in z_o:
        buf_l.append(float(v)); buf_d.append(float(v) - prev); prev = float(v)
        if len(buf_l) > max(WINS):
            buf_l.pop(0); buf_d.pop(0)
        step_i += 1
        if step_i >= next_scan:
            next_scan = max(next_scan + 1, int(next_scan * 1.12))
            cur = []
            al, ad = np.asarray(buf_l), np.asarray(buf_d)
            for w in WINS:
                for kind, arr in (("lvl", al), ("dif", ad)):
                    hb, hlh, he, hb_sd, hlh_sd, he_sd = profiles[(w, kind)]
                    b, lh, e = spec_feats(arr, w)
                    cur.extend(((b - hb) / hb_sd).tolist())
                    cur.append((lh - hlh) / hlh_sd)
                    cur.append((e - he) / he_sd)
        rows.append(list(cur))
    count += 1
    if count % 2000 == 0:
        print(f"  {count} series, {time.time()-t0:.0f}s", flush=True)

S = np.asarray(rows, dtype="float32")
np.save("SPEC2.npy", S)
print(f"done {time.time()-t0:.0f}s: {S.shape}", flush=True)
