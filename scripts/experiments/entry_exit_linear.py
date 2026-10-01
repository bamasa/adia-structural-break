"""092b: a trainable verifier over score trajectories (trees and nets) — CV by series within fold 2."""
import sys, numpy as np, lightgbm as lgb
sys.path.insert(0, "repo/src")
from structural_break.combiners import split_by_series, ts_auc
g = np.load("G40.npy"); y = np.load("Y40.npy"); s = np.load("S40.npy")
m2 = split_by_series(g, folds=5, seed=0) == 2
gf, yf, sf = g[m2], y[m2], s[m2]
sig = lambda a: 1/(1+np.exp(-a.astype("float64")))
trees = 0.7*sig(np.load("fold2_rank_bocpd_200.npy")) + 0.3*np.load("fold2_clf_bocpdh50_206.npy").astype("float64")
net = np.mean([np.load(f"fold2_sig_nets_aug3_member_p{i}.pt.npy") for i in range(6)] + [np.load(f"fold2_sig_nets_aug3_last_member_z{i}.pt.npy") for i in range(6)], 0)
base = 0.45*trees + 0.55*net
starts = np.flatnonzero(np.concatenate([[True], gf[1:] != gf[:-1]])); bounds = np.append(starts, len(gf))

def traj_feats(x):
    n = len(x); c = np.cumsum(np.insert(x, 0, 0.0)); idx = np.arange(1, n + 1)
    def rmean(w): lo = np.maximum(0, idx - w); return (c[idx] - c[lo]) / (idx - lo)
    cm = np.maximum.accumulate(x); argmax_t = np.zeros(n); best = -1; bt = 0
    for t in range(n):
        if x[t] > best: best, bt = x[t], t
        argmax_t[t] = t - bt
    m20, m60, m200 = rmean(20), rmean(60), rmean(200)
    return np.column_stack([x, cm, m20, m60, m200, x - m20, m20 - m60, m60 - m200, cm - x, argmax_t, np.log1p(idx)])

F = np.empty((len(base), 3 * 11 + 1))
for a, b in zip(starts, bounds[1:]):
    F[a:b] = np.hstack([traj_feats(base[a:b]), traj_feats(trees[a:b]), traj_feats(net[a:b]), (trees[a:b] - net[a:b])[:, None]])
print(f"features {F.shape[1]}, rows {len(F)}; base {ts_auc(base, yf, sf):.4f}", flush=True)

# CV by series within fold 2 (5 parts)
series_ids = gf[starts]; rng = np.random.default_rng(0); part_of = {int(sid): i % 5 for i, sid in enumerate(rng.permutation(series_ids))}
part = np.array([part_of[int(v)] for v in gf])

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
oof = np.zeros(len(base))
for k in range(5):
    tr, te = part != k, part == k
    sc = StandardScaler().fit(F[tr]); m = LogisticRegression(C=0.1, max_iter=300).fit(sc.transform(F[tr]), yf[tr])
    oof[te] = m.decision_function(sc.transform(F[te]))
print(f"linear verifier (34 trajectory features): OOF {ts_auc(oof, yf, sf):.4f} | rank blend 0.5 with the base {ts_auc(0.5*base + 0.5/(1+np.exp(-oof)), yf, sf):.4f}", flush=True)
# only the current scores of the three sources (linearity control)
oof = np.zeros(len(base)); cols = [0, 11, 22, 33]
for k in range(5):
    tr, te = part != k, part == k
    sc = StandardScaler().fit(F[tr][:, cols]); m = LogisticRegression(C=0.1, max_iter=300).fit(sc.transform(F[tr][:, cols]), yf[tr])
    oof[te] = m.decision_function(sc.transform(F[te][:, cols]))
print(f"control: linear blend of the current scores (trees, nets, difference): {ts_auc(oof, yf, sf):.4f}")
