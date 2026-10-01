"""045: a bag of 24 nets — averaging on folds 0 and 1, contribution to the ensemble."""
import sys, time
sys.path.insert(0, "repo/src")
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from structural_break.combiners import split_by_series, ts_auc

DEVICE = torch.device("mps")
t0 = time.time()
X = np.hstack([np.load("X40.npy"), np.load("C_cnn.npy")[:, None],
               np.load("N9.npy").astype("float32"), np.load("X50a.npy"),
               np.load("E4.npy"), np.load("B40.npy"), np.load("B2.npy")]).astype("float32")
y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
assignment = split_by_series(g, folds=5, seed=0)
starts = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]]))
bounds = np.append(starts, len(g))
mu = np.load("net_bag_mac/mu.npy"); sd = np.load("net_bag_mac/sd.npy")

class Block(nn.Module):
    def __init__(self, ch, dil):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, 3, dilation=dil)
        self.mix = nn.Conv1d(ch, ch, 1)
        self.drop = nn.Dropout(0.1)
        self.dil = dil
    def forward(self, h):
        r = h
        h = F.pad(h, (2 * self.dil, 0))
        h = self.drop(F.gelu(self.conv(h)))
        return r + self.mix(h)

class ChanTCN(nn.Module):
    def __init__(self, ch=64):
        super().__init__()
        self.inp = nn.Conv1d(186, ch, 1)
        self.blocks = nn.ModuleList([Block(ch, d) for d in (1, 2, 4, 8, 16, 32)])
        self.head = nn.Conv1d(ch, 1, 1)
    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = b(h)
        return self.head(h).squeeze(1)

folds_rows = {}
for f in (0, 1):
    rows = [(a, b) for a, b in zip(starts, bounds[1:]) if int(assignment[a]) == f]
    rows.sort(key=lambda ab: ab[1] - ab[0])
    folds_rows[f] = rows

def fold_scores(model, f):
    out = {}
    with torch.no_grad():
        rows = folds_rows[f]
        for k in range(0, len(rows), 96):
            chunk = rows[k:k + 96]
            L = max(b - a for a, b in chunk)
            Xb = torch.zeros(len(chunk), 186, L)
            for i, (a, b) in enumerate(chunk):
                fseq = (X[a:b] - mu) / sd
                Xb[i, :, L - (b - a):] = torch.from_numpy(fseq.T)
            o = model(Xb.to(DEVICE)).cpu().numpy()
            for i, (a, b) in enumerate(chunk):
                out[a] = o[i, L - (b - a):]
    return np.concatenate([out[a] for a, b in sorted(folds_rows[f])])

sigs = {0: [], 1: []}
for member in range(24):
    model = ChanTCN().to(DEVICE)
    model.load_state_dict(torch.load(f"net_bag_mac/member_{member}.pt", map_location=DEVICE))
    model.eval()
    for f in (0, 1):
        sigs[f].append(1.0 / (1.0 + np.exp(-fold_scores(model, f).astype("float64"))))
    if member % 6 == 5:
        print(f"  {member+1}/24 [{time.time()-t0:.0f}s]", flush=True)

np.save("bag24_members_f0.npy", np.stack(sigs[0]))
np.save("bag24_members_f1.npy", np.stack(sigs[1]))
for f in (0, 1):
    m = assignment == f
    yf, sf = y[m], s[m]
    ens = np.mean(sigs[f], axis=0)
    np.save(f"bag24_fold{f}.npy", ens)
    print(f"fold {f}: bag-24 alone {ts_auc(ens, yf, sf):.4f}", flush=True)
    for n in (8, 12, 16):
        sub = np.mean(sigs[f][:n], axis=0)
        print(f"  first {n} members: {ts_auc(sub, yf, sf):.4f}", flush=True)

# Ensemble on fold 0: replace the old four with the bag.
m0 = assignment == 0
yf, sf = y[m0], s[m0]
clf = np.load("oof_cfg5.npy")[m0].astype("float64")
rnk = np.load("oof_rank.npy")[m0].astype("float64")
blend = 0.6/(1+np.exp(-rnk)) + 0.4*clf
bag_r = np.load("rank_foldbag_fold0.npy").astype("float64")
ens0 = np.load("bag24_fold0.npy")
old4 = np.load("tcn_foldens_fold0.npy").astype("float64")
print(f"old four alone (reference): {ts_auc(old4, yf, sf):.4f}", flush=True)
for w in (0.5, 0.6):
    full = (1-w)*(0.7*bag_r + 0.3*clf) + w*ens0
    print(f"ensemble with bag-24 (weight {w}): {ts_auc(full, yf, sf):.4f} (nine-member reference 0.6099)", flush=True)
print(f"total {time.time()-t0:.0f}s", flush=True)
