"""121: nets on the mass battery — fold 2 alone, correlation with #36, weight in the blend."""
import sys, glob, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "repo/src")
from structural_break.combiners import split_by_series, ts_auc
DEV = torch.device("mps")
class Block(nn.Module):
    def __init__(self, ch, dil):
        super().__init__(); self.conv = nn.Conv1d(ch, ch, 3, dilation=dil); self.mix = nn.Conv1d(ch, ch, 1); self.drop = nn.Dropout(0.1); self.dil = dil
    def forward(self, h):
        r = h; h = F.pad(h, (2 * self.dil, 0)); h = self.drop(F.gelu(self.conv(h))); return r + self.mix(h)
class ChanTCN(nn.Module):
    def __init__(self, n_in, ch=64):
        super().__init__(); self.inp = nn.Conv1d(n_in, ch, 1); self.blocks = nn.ModuleList([Block(ch, d) for d in (1, 2, 4, 8, 16, 32)]); self.head = nn.Conv1d(ch, 1, 1)
    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks: h = b(h)
        return self.head(h).squeeze(1)
X = np.load("MASS90.npy").astype("float32"); mu = np.load("mu90.npy"); sd = np.load("sd90.npy")
g = np.load("G40.npy"); y = np.load("Y40.npy"); s = np.load("S40.npy")
assignment = split_by_series(g, folds=5, seed=0); m2 = assignment == 2; yf, sf = y[m2], s[m2]
starts = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]])); bounds = np.append(starts, len(g))
rows2 = sorted([(a, b) for a, b in zip(starts, bounds[1:]) if int(assignment[a]) == 2], key=lambda ab: ab[1] - ab[0])
def fold_scores(model):
    out = {}
    with torch.no_grad():
        for k in range(0, len(rows2), 96):
            chunk = rows2[k:k + 96]; L = max(b - a for a, b in chunk); Xb = torch.zeros(len(chunk), 90, L)
            for i, (a, b) in enumerate(chunk): Xb[i, :, L - (b - a):] = torch.from_numpy(((X[a:b] - mu) / sd).T)
            o = model(Xb.to(DEV)).cpu().numpy()
            for i, (a, b) in enumerate(chunk): out[a] = o[i, L - (b - a):]
    return np.concatenate([out[a] for a, b in sorted(rows2)])
sigs = []
for p in sorted(glob.glob("nets_mass/member_q?.pt")):
    m = ChanTCN(90).to(DEV); m.load_state_dict(torch.load(p, map_location=DEV)); m.eval()
    sg = 1 / (1 + np.exp(-fold_scores(m).astype("float64"))); np.save(f"fold2_sig_{p.replace('/', '_')}.npy", sg); sigs.append(sg)
    print(f"{p}: fold 2 alone {ts_auc(sg, yf, sf):.4f}", flush=True)
net = np.mean(sigs, 0)
base = np.load("fold2_base30.npy"); mass = np.load(glob.glob("fold2_clf_mass_90*.npy")[0])
cur = 0.75 * base + 0.25 * mass
print(f"\n{len(sigs)} mass nets together: {ts_auc(net, yf, sf):.4f} | correlation with #36 {pd.Series(net).corr(pd.Series(cur), method='spearman'):.3f}, "
      f"with the mass classifier {pd.Series(net).corr(pd.Series(mass), method='spearman'):.3f}")
print(f"#36: {ts_auc(cur, yf, sf):.4f}")
best = (0, None)
for w in (0.10, 0.15, 0.20, 0.25, 0.30):
    v = ts_auc((1 - w) * cur + w * net, yf, sf)
    if v > best[0]: best = (v, w)
    print(f"  #36 + {w:.2f}·mass nets: {v:.4f}  ({v - ts_auc(cur, yf, sf):+.4f})")
print(f"BEST: {best[0]:.4f} at weight {best[1]}")
