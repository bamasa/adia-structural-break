"""145: the Rosenblatt-whitened stream as a member's input.

The history fits AR(p<=12) by BIC, a conditional scale (EWMA variance, lambda by
Gaussian quasi-likelihood), and the empirical CDF of the standardised innovations;
the online stream is mapped through the same fit to normal scores n_t that are
i.i.d. N(0,1) under the null for a far wider family of histories than an AR(1)
whitening covers. On that stream: CUSUMs and EWMAs for mean, scale, dependence
(lags 1,2,3,5,10 + portmanteau), volatility clustering, shape and tails; GLR over
dyadic windows 8..1024 for mean, scale and lag-1 dependence (per scale and max);
one-sample KS/CvM/AD against N(0,1) over windows 32..512 and the prefix, at the
geometric cadence -- all on the unconditional stream (history innovation variance),
where scale breaks stay visible; mean/dependence/test extras on the conditional stream;
the conditional scale process against its history level; and the step index. 90 channels.
python build_white.py <shard> <n> | merge <n> | test

The computation lives in the library, structural_break.white_batch; this script is the
command-line builder of the WHITE90 matrix and re-exports the batch functions under their
old names for the sibling scripts (build_sr, build_histctx, build_whitez, build_wstream,
build_white_tpit).
"""
import sys, time, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from structural_break import white_batch as _batch
from structural_break.white_batch import ewma, cusum_peak, fit_history, normal_scores_online, one_sample_tests, _battery  # noqa: F401
from structural_break.white import (WHITE_PMAX as PMAX, WHITE_LAMBDAS as LAMBDAS, WHITE_DRIFT as DRIFT, WHITE_LAGS as LAGS,  # noqa: F401
                                    WHITE_PORT as PORT, WHITE_DYADIC as DYADIC, WHITE_TESTW as TESTW, WHITE_FULL as NFULL,
                                    WHITE_CHANNELS as NCH)
OUT = os.environ.get("WHITE_OUT", "WHITE90"); PARTS = f"{OUT.lower()}_parts"

def white_channels(hist, online):
    """The 90 channels of one series as float32, as the WHITE90 matrix stores them.

    The history fit and the online scoring are read from this module's namespace at call
    time, so a script that replaces them here (157e, build_white_tpit) is honoured.
    """
    return _batch.white_channels(hist, online, fit=fit_history, online_scores=normal_scores_online).astype("float32")

if __name__ == "__main__":
    if sys.argv[1] == "test":
        rng = np.random.default_rng(0)
        def ar(phi, n, sd=1.0):
            x = np.zeros(n); e = rng.normal(0, sd, n)
            for i in range(1, n): x[i] = phi * x[i - 1] + e[i]
            return x
        hist = ar(0.6, 3000) * (1 + 0.5 * np.sin(np.arange(3000) / 300))   # heteroskedastic AR history
        online = np.concatenate([ar(0.6, 300), ar(0.6, 300, sd=1.5)])
        t0 = time.time(); o = white_channels(hist, online); dt = (time.time() - t0) * 1000
        print(f"{o.shape[1]} channels (expected {NCH}), {dt:.0f} ms per series; scale break: scale CUSUM {o[250:300,4].mean():.2f} -> {o[450:600,4].mean():.2f}, scale GLR-max {o[250:300,53].mean():.1f} -> {o[450:600,53].mean():.1f}, KS-max {o[250:300,73].mean():.2f} -> {o[450:600,73].mean():.2f}, log-scale {o[250:300,86].mean():+.2f} -> {o[450:600,86].mean():+.2f}")
        online2 = np.concatenate([ar(0.6, 300), ar(0.9, 300, sd=np.sqrt((1 - 0.9 ** 2) / (1 - 0.6 ** 2)))]); o2 = white_channels(hist, online2)
        print(f"dependence break at equal variance: CUSUM lag1 {o2[250:300,8].mean():.2f} -> {o2[450:600,8].mean():.2f}, dependence GLR-max {o2[250:300,54].mean():.1f} -> {o2[450:600,54].mean():.1f}")
        sys.exit(0)
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
        sids.append(int(sid)); arrs.append(white_channels(hist, online))
        if (i + 1) % 250 == 0: print(f"shard {shard}: {i+1} series, {time.time()-t0:.0f}s", flush=True)
    os.makedirs(PARTS, exist_ok=True)
    np.savez(f"{PARTS}/part_{shard}.npz", sids=np.array(sids), arrs=np.array(arrs, dtype=object), allow_pickle=True)
    print(f"shard {shard} done {time.time()-t0:.0f}s: {len(sids)} series", flush=True)
