"""084: two raw channels for the nets — the point's z-score against the history and asinh(z).

The feature channels are summaries; the nets have never seen the series itself. For the AUG3
pseudo-series the history is extended by online[:k]; k is recovered as
len(online) - pseudo-series length. Output: RAW2.npy (aligned with X40 rows) and
AUG3_RAW2.npy (aligned with AUG3_X rows)."""
import time, numpy as np, pandas as pd
t0 = time.time()
x = pd.read_parquet("structural-break-real-time-test/data/X_train.parquet")
series = {}
for sid, part in x.groupby(level="id"):
    series[int(sid)] = (part.loc[part.period == 1, "value"].to_numpy("float64"),
                        part.loc[part.period == 2, "value"].to_numpy("float64"))
del x
print(f"series loaded [{time.time()-t0:.0f}s]", flush=True)

def raw2(hist, online, steps):
    mu, sd = hist.mean(), hist.std() + 1e-9
    z = (online - mu) / sd
    out = np.empty((len(steps), 2), dtype="float32")
    out[:, 0] = z[steps]; out[:, 1] = np.arcsinh(z[steps])
    return out

# Originals
g = np.load("G40.npy"); s = np.load("S40.npy")
starts = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]])); bounds = np.append(starts, len(g))
R = np.empty((len(g), 2), dtype="float32")
for a, b in zip(starts, bounds[1:]):
    hist, online = series[int(g[a])]
    assert len(online) == b - a, (g[a], len(online), b - a)
    R[a:b] = raw2(hist, online, s[a:b])
np.save("RAW2.npy", R)
print(f"RAW2: {R.shape}, |z| max {np.abs(R[:,0]).max():.0f} [{time.time()-t0:.0f}s]", flush=True)

# AUG3 pseudo-series: k = len(online) - pseudo-series length
AG = np.load("AUG3_G.npy"); AS = np.load("AUG3_S.npy")
a_starts = np.flatnonzero(np.concatenate([[True], AG[1:] != AG[:-1]])); a_bounds = np.append(a_starts, len(AG))
AR = np.empty((len(AG), 2), dtype="float32")
ks = []
for a, b in zip(a_starts, a_bounds[1:]):
    sid = (int(AG[a]) - 100000) // 10
    hist, online = series[sid]
    k = len(online) - (b - a)
    assert 10 <= k < len(online), (sid, k)
    ks.append(k)
    h2 = np.concatenate([hist, online[:k]]); o2 = online[k:]
    AR[a:b] = raw2(h2, o2, AS[a:b])
np.save("AUG3_RAW2.npy", AR)
print(f"AUG3_RAW2: {AR.shape}, k in [{min(ks)}, {max(ks)}], median {int(np.median(ks))} [{time.time()-t0:.0f}s]", flush=True)
