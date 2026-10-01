import sys, glob, re, numpy as np
sys.path.insert(0, "repo/src")
from structural_break.combiners import split_by_series, ts_auc
g = np.load("G40.npy"); y = np.load("Y40.npy"); s = np.load("S40.npy")
m2 = split_by_series(g, folds=5, seed=0) == 2
yf, sf = y[m2], s[m2]
sig = lambda a: 1/(1+np.exp(-a.astype("float64")))
pair = 0.35*sig(np.load("fold2_rank_aug.npy")) + 0.35*sig(np.load("fold2_rank_aug3.npy")) + 0.3*np.load("fold2_clf_reference.npy").astype("float64")
hold = {}
for src, pat, folder in (("heavy200.log", r"heavy (\d+): private holdout ([0-9.]+)", "heavy200"),
                         ("nets_aug.log", r"aug-net (\d+): holdout ([0-9.]+)", "nets_aug")):
    for line in open(src, errors="ignore"):
        m = re.match(pat, line)
        if m: hold[f"{folder}/member_{m.group(1)}.pt"] = float(m.group(2))
plain12 = sorted(hold, key=hold.get, reverse=True)[:12]
diff6 = sorted(glob.glob("nets_aug_diff/member_*.pt"))
aug3p = sorted(glob.glob("nets_aug3/member_p*.pt"))
S = lambda p: np.load(f"fold2_sig_{p.replace('/', '_')}.npy")
def score(base, ext, w=0.55, ew=0.5):
    net = np.mean([S(p) for p in base], axis=0) if not ext else \
          (1-ew)*np.mean([S(p) for p in base], axis=0) + ew*np.mean([S(p) for p in ext], axis=0)
    return ts_auc((1-w)*pair + w*net, yf, sf)
print(f"aug3 plain: {len(aug3p)} members")
print(f"only aug3 in the blend (weight 0.5):        {score(aug3p, [], w=0.5):.4f}")
print(f"only aug3 in the blend (weight 0.55):       {score(aug3p, [], w=0.55):.4f}")
print(f"only aug3 in the blend (weight 0.6):        {score(aug3p, [], w=0.6):.4f}")
best = (0, "")
for ew in (0.5, 0.6, 0.7, 0.8):
    for w in (0.5, 0.55, 0.6, 0.65):
        v = score(plain12+diff6, aug3p, w=w, ew=ew)
        if v > best[0]: best = (v, f"old pool + aug3 as a group {ew}, net weight {w}")
print(f"BEST: {best[0]:.4f} — {best[1]}   (target 0.617, bar 0.6154)")
