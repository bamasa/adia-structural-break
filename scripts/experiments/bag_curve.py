"""046: bag scaling curve — 12/24/48/72 members on folds 0 and 1."""
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
    def __init__(self, ch=64, dils=(1, 2, 4, 8, 16, 32)):
        super().__init__()
        self.inp = nn.Conv1d(186, ch, 1)
        self.blocks = nn.ModuleList([Block(ch, d) for d in dils])
        self.head = nn.Conv1d(ch, 1, 1)
    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = b(h)
        return self.head(h).squeeze(1)

folds_rows = {}
for f in (0, 1):
    rows = sorted([(a, b) for a, b in zip(starts, bounds[1:]) if int(assignment[a]) == f],
                  key=lambda ab: ab[1] - ab[0])
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
# old 24 (ch64/6dils)
for m in range(24):
    model = ChanTCN().to(DEVICE)
    model.load_state_dict(torch.load(f"net_bag_mac/member_{m}.pt", map_location=DEVICE))
    model.eval()
    for f in (0, 1):
        sigs[f].append(1.0 / (1.0 + np.exp(-fold_scores(model, f).astype("float64"))))
print(f"old 24 run [{time.time()-t0:.0f}s]", flush=True)
# new 48 (config in the checkpoint)
for m in range(48):
    ck = torch.load(f"net_bag_mac/grow_{m}.pt", map_location=DEVICE)
    model = ChanTCN(**ck["cfg"]).to(DEVICE)
    model.load_state_dict(ck["state"])
    model.eval()
    for f in (0, 1):
        sigs[f].append(1.0 / (1.0 + np.exp(-fold_scores(model, f).astype("float64"))))
    if m % 12 == 11:
        print(f"  new {m+1}/48 [{time.time()-t0:.0f}s]", flush=True)

for f in (0, 1):
    mset = assignment == f
    yf, sf = y[mset], s[mset]
    for n in (12, 24, 48, 72):
        ens = np.mean(sigs[f][:n], axis=0)
        print(f"fold {f}, {n} members: {ts_auc(ens, yf, sf):.4f}", flush=True)
    np.save(f"bag72_fold{f}.npy", np.mean(sigs[f], axis=0))
    np.save(f"bag72_members_f{f}.npy", np.stack(sigs[f]).astype("float32"))

m0 = assignment == 0
yf, sf = y[m0], s[m0]
clf = np.load("oof_cfg5.npy")[m0].astype("float64")
bag_r = np.load("rank_foldbag_fold0.npy").astype("float64")
trees = 0.7 * bag_r + 0.3 * clf
for n in (12, 72):
    ens = np.mean(sigs[0][:n], axis=0)
    for w in (0.7, 0.8):
        print(f"ensemble of {n} members, weight {w}: fold 0 {ts_auc((1-w)*trees + w*ens, yf, sf):.4f} (reference #21: 0.6201)", flush=True)
print(f"total {time.time()-t0:.0f}s", flush=True)
