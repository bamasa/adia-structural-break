"""Measurement: the ensemble with the updated set of nets (including the record holder)."""
import sys, time, re, os, glob
sys.path.insert(0, "repo/src")
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from structural_break.combiners import split_by_series, ts_auc

DEVICE = torch.device("mps")
t0 = time.time()
X = np.hstack([np.load("X40.npy"), np.load("C_cnn.npy")[:, None],
               np.load("N9.npy").astype("float32"), np.load("X50a.npy"),
               np.load("E4.npy"), np.load("B40.npy"), np.load("B2.npy"),
               np.load("SPEC14.npy")]).astype("float32")
y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
assignment = split_by_series(g, folds=5, seed=0)
starts = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]]))
bounds = np.append(starts, len(g))
mu = np.load("mu200.npy"); sd = np.load("sd200.npy")

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
        self.inp = nn.Conv1d(200, ch, 1)
        self.blocks = nn.ModuleList([Block(ch, d) for d in (1, 2, 4, 8, 16, 32)])
        self.head = nn.Conv1d(ch, 1, 1)
    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = b(h)
        return self.head(h).squeeze(1)

rows2 = sorted([(a, b) for a, b in zip(starts, bounds[1:]) if int(assignment[a]) == 2],
               key=lambda ab: ab[1] - ab[0])
def fold_scores(model):
    out = {}
    with torch.no_grad():
        for k in range(0, len(rows2), 96):
            chunk = rows2[k:k + 96]
            L = max(b - a for a, b in chunk)
            Xb = torch.zeros(len(chunk), 200, L)
            for i, (a, b) in enumerate(chunk):
                Xb[i, :, L - (b - a):] = torch.from_numpy(((X[a:b] - mu) / sd).T)
            o = model(Xb.to(DEVICE)).cpu().numpy()
            for i, (a, b) in enumerate(chunk):
                out[a] = o[i, L - (b - a):]
    return np.concatenate([out[a] for a, b in sorted(rows2)])

hold = {}
for src, pat, folder in (("heavy200.log", r"heavy (\d+): private holdout ([0-9.]+)", "heavy200"),
                         ("nets_aug.log", r"aug[- ]net (\d+): holdout ([0-9.]+)", "nets_aug")):
    for line in open(src, errors="ignore"):
        m = re.match(pat, line)
        if m:
            p = f"{folder}/member_{m.group(1)}.pt"
            if os.path.exists(p):
                hold[p] = float(m.group(2))
print("pool:", len(hold), "members", flush=True)

sigs = {}
for p in hold:
    model = ChanTCN().to(DEVICE)
    model.load_state_dict(torch.load(p, map_location=DEVICE))
    model.eval()
    sigs[p] = 1.0 / (1.0 + np.exp(-fold_scores(model).astype("float64")))
print(f"scored [{time.time()-t0:.0f}s]", flush=True)

m2 = assignment == 2
yf, sf = y[m2], s[m2]
r_aug = 1/(1+np.exp(-np.load("fold2_rank_aug.npy").astype("float64")))
r_aug3 = 1/(1+np.exp(-np.load("fold2_rank_aug3.npy").astype("float64")))
c_plain = np.load("fold2_clf_reference.npy").astype("float64")
pair = 0.35*r_aug + 0.35*r_aug3 + 0.3*c_plain
order = sorted(hold, key=hold.get, reverse=True)
best = (0, None)
for k in (4, 6, 8, 10, 12):
    if k > len(order):
        break
    net = np.mean([sigs[p] for p in order[:k]], axis=0)
    solo = ts_auc(net, yf, sf)
    line = f"{k:2d} best nets: alone {solo:.4f}"
    for w in (0.4, 0.45, 0.5):
        v = ts_auc((1-w)*pair + w*net, yf, sf)
        line += f" | weight {w}: {v:.4f}"
        if v > best[0]:
            best = (v, f"{k} nets, weight {w}")
    print(line, flush=True)
print(f"\nBEST: {best[0]:.4f} — {best[1]}  (shipped as #27 at 0.6106)", flush=True)
