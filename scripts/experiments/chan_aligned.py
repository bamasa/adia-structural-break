"""033b: выровненные скоры сети-по-каналам + ансамбль."""
import sys, time
sys.path.insert(0, "repo/src")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
mu, sd = X.mean(0), X.std(0) + 1e-6

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

model = ChanTCN().to(DEVICE)
model.load_state_dict(torch.load("tcn_chan10.pt", map_location=DEVICE))
model.eval()

scores = []
with torch.no_grad():
    for a, b in zip(starts, bounds[1:]):
        if int(assignment[a]) != 0:
            continue
        f = ((X[a:b] - mu) / sd)
        xb = torch.from_numpy(f.T).unsqueeze(0).to(DEVICE)
        scores.append(model(xb).cpu().numpy()[0])
net = np.concatenate(scores)
mask = assignment == 0
yf, sf = y[mask], s[mask]
np.save("tcn_chan_aligned.npy", net)
print(f"сеть соло (выровнено): {ts_auc(net, yf, sf):.4f}  [{time.time()-t0:.0f}s]", flush=True)

clf = np.load("oof_cfg5.npy")[mask].astype("float64")
rnk = np.load("oof_rank.npy")[mask].astype("float64")
sig_r = 1.0 / (1.0 + np.exp(-rnk))
sig_n = 1.0 / (1.0 + np.exp(-net.astype("float64")))
blend = 0.6 * sig_r + 0.4 * clf
print(f"смесь эталон: {ts_auc(blend, yf, sf):.4f}", flush=True)
for w in (0.1, 0.2, 0.3, 0.4):
    print(f"смесь + {w:.0%} сети: {ts_auc((1-w)*blend + w*sig_n, yf, sf):.4f}", flush=True)
tri = 0.45 * sig_r + 0.3 * clf + 0.25 * sig_n
print(f"тройка 45/30/25: {ts_auc(tri, yf, sf):.4f}", flush=True)
