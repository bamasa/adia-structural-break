"""147: Shiryaev-Roberts statistics on the whitened stream — the accumulated posterior odds of a
change at some tau <= t (uniform tau prior) for the break types the forensics found: innovation
variance (increase and decrease), mean, and lag-1 dependence, each over a grid of alternatives
with a mixture over the grid. log R_t = L_t + logcumsumexp_{tau<=t}(-L_{tau-1}), L = cumsum of
per-point log likelihood ratios; fully vectorised per series. 22 channels.
python build_sr.py <shard> <n> | merge <n>"""
import sys, time, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from structural_break import white_batch as _batch
from structural_break.white_batch import log_sr  # noqa: F401
from structural_break.white import SR_VAR_UP as VAR_UP, SR_VAR_DOWN as VAR_DOWN, SR_MEANS as MEANS, SR_PHIS as PHIS, SR_CHANNELS as NCH  # noqa: F401
OUT = os.environ.get("SR_OUT", "SR22"); PARTS = f"{OUT.lower()}_parts"

def sr_channels(hist, online):
    """The Shiryaev-Roberts channels of one series as float32, as the SR22 matrix stores them."""
    return _batch.sr_channels(hist, online).astype("float32")

if __name__ == "__main__":
    if sys.argv[1] == "merge":
        n = int(sys.argv[2]); g = np.load("G40.npy"); s = np.load("S40.npy")
        st = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]])); bd = np.append(st, len(g))
        out = np.empty((len(g), NCH), dtype="float32"); block = {}
        for i in range(n):
            p = np.load(f"{PARTS}/part_{i}.npz", allow_pickle=True)
            for sid, arr in zip(p["sids"], p["arrs"]): block[int(sid)] = arr
        for a_, b_ in zip(st, bd[1:]): out[a_:b_] = block[int(g[a_])][s[a_:b_]]
        np.save(f"{OUT}.npy", out); print(f"{OUT}: {out.shape}, nan {np.isnan(out).sum()}", flush=True); sys.exit(0)
    shard, n_shards = int(sys.argv[1]), int(sys.argv[2])
    import pandas as pd
    t0 = time.time(); x = pd.read_parquet("structural-break-real-time-test/data/X_train.parquet")
    ids = x.index.get_level_values("id"); x = x[(ids % n_shards) == shard]; sids, arrs = [], []
    for i, (sid, part) in enumerate(x.groupby(level="id")):
        hist = part.loc[part.period == 1, "value"].to_numpy("float64"); online = part.loc[part.period == 2, "value"].to_numpy("float64")
        sids.append(int(sid)); arrs.append(sr_channels(hist, online))
    os.makedirs(PARTS, exist_ok=True)
    np.savez(f"{PARTS}/part_{shard}.npz", sids=np.array(sids), arrs=np.array(arrs, dtype=object), allow_pickle=True)
    print(f"shard {shard} done {time.time()-t0:.0f}s: {len(sids)} series", flush=True)
