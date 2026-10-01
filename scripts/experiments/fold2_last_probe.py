"""Quick check of the ready 076 members on fold 2."""
import sys, os, glob
sys.path.insert(0, "repo/src")
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from structural_break.combiners import split_by_series, ts_auc

DEVICE = torch.device("mps")
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
    def __init__(self, n_in, ch=64):
        super().__init__()
        self.inp = nn.Conv1d(n_in, ch, 1)
        self.blocks = nn.ModuleList([Block(ch, d) for d in (1, 2, 4, 8, 16, 32)])
        self.head = nn.Conv1d(ch, 1, 1)
    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = b(h)
        return self.head(h).squeeze(1)

def with_diffs(seg):
    d1 = np.zeros_like(seg); d1[1:] = seg[1:] - seg[:-1]
    d10 = np.zeros_like(seg)
    if len(seg) > 10:
        d10[10:] = seg[10:] - seg[:-10]
    return np.hstack([seg, d1 * 3.0, d10 * 1.5]).astype("float32")

rows2 = sorted([(a, b) for a, b in zip(starts, bounds[1:]) if int(assignment[a]) == 2],
               key=lambda ab: ab[1] - ab[0])
m2 = assignment == 2
yf, sf = y[m2], s[m2]

def fold_scores(model, diffs):
    out = {}
    with torch.no_grad():
        for k in range(0, len(rows2), 64):
            chunk = rows2[k:k + 64]
            L = max(b - a for a, b in chunk)
            Xb = torch.zeros(len(chunk), 600 if diffs else 200, L)
            for i, (a, b) in enumerate(chunk):
                seg = (X[a:b] - mu) / sd
                if diffs:
                    seg = with_diffs(seg)
                Xb[i, :, L - (b - a):] = torch.from_numpy(seg.T)
            o = model(Xb.to(DEVICE)).cpu().numpy()
            for i, (a, b) in enumerate(chunk):
                out[a] = o[i, L - (b - a):]
    return np.concatenate([out[a] for a, b in sorted(rows2)])

for p in sorted(glob.glob("nets_aug3_last/member_z*.pt")):
    diffs = "_d" in p
    model = ChanTCN(600 if diffs else 200).to(DEVICE)
    model.load_state_dict(torch.load(p, map_location=DEVICE))
    model.eval()
    sig = 1.0 / (1.0 + np.exp(-fold_scores(model, diffs).astype("float64")))
    np.save(f"fold2_sig_{p.replace('/', '_')}.npy", sig)
    print(f"{p}: fold 2 alone {ts_auc(sig, yf, sf):.4f}", flush=True)
