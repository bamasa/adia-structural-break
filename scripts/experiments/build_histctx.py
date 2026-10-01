"""153: the history's own family as static context for the whitened member — AR order, first coefficient,
conditional-scale memory, innovation variance, innovation kurtosis and skew, history length, spike size.
Eight constants per series, broadcast to its rows. python build_histctx.py"""
import sys, time, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from structural_break.white_batch import fit_history, history_context
t0 = time.time(); x = pd.read_parquet("structural-break-real-time-test/data/X_train.parquet")
g = np.load("G40.npy"); st = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]])); bd = np.append(st, len(g)); pos = {int(g[a]): (a, b) for a, b in zip(st, bd[1:])}
out = np.zeros((len(g), 8), dtype="float32")
for i, (sid, part) in enumerate(x.groupby(level="id")):
    h = part.loc[part.period == 1, "value"].to_numpy("float64"); zh = (h - h.mean()) / (h.std() + 1e-12)
    a, b = pos[int(sid)]
    out[a:b] = history_context(fit_history(zh), zh)
    if (i + 1) % 2500 == 0: print(f"{i+1} series, {time.time()-t0:.0f}s", flush=True)
np.save("HISTCTX8.npy", out); print(f"HISTCTX8: {out.shape} [{time.time()-t0:.0f}s]")
