"""146: the mass battery referenced to the recent tail of the history (last 512 points) — a second reference frame.
python build_mass_tail.py <shard> <n> | merge <n>"""
import sys, time, os, numpy as np
sys.path.insert(0, "repo/src")
from structural_break.mass import MassBattery, MASS_CHANNELS
TAIL = int(os.environ.get("MASS_TAIL", "512")); OUT = f"MASSTAIL{TAIL}"; PARTS = f"{OUT.lower()}_parts"; NCH = MASS_CHANNELS
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
        mb = MassBattery(hist[-TAIL:]); arrs.append(np.clip(np.nan_to_num(np.array([mb.update(float(v)) for v in online], dtype="float32")), -50, 50)); sids.append(int(sid))
        if (i + 1) % 250 == 0: print(f"shard {shard}: {i+1} series, {time.time()-t0:.0f}s", flush=True)
    os.makedirs(PARTS, exist_ok=True)
    np.savez(f"{PARTS}/part_{shard}.npz", sids=np.array(sids), arrs=np.array(arrs, dtype=object), allow_pickle=True)
    print(f"shard {shard} done {time.time()-t0:.0f}s: {len(sids)} series", flush=True)
