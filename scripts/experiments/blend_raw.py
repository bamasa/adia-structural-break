"""Вклад сырой сети в ансамбль: корреляция с #30 и смесь. Считает на CPU, чтобы не мешать обучению."""
import sys, numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "repo/src")
from structural_break.combiners import split_by_series, ts_auc
CTX = 512; DILS = (1,2,4,8,16,32,64,128,256,512); CH = 48
class Block(nn.Module):
    def __init__(self, ch, d):
        super().__init__(); self.conv = nn.Conv1d(ch,ch,3,dilation=d); self.mix = nn.Conv1d(ch,ch,1); self.drop = nn.Dropout(0.1); self.d = d
    def forward(self, h):
        r = h; h = F.pad(h, (2*self.d, 0)); h = self.drop(F.gelu(self.conv(h))); return r + self.mix(h)
class RawTCN(nn.Module):
    def __init__(self):
        super().__init__(); self.inp = nn.Conv1d(2, CH, 1); self.blocks = nn.ModuleList([Block(CH,d) for d in DILS]); self.head = nn.Conv1d(CH,1,1)
    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks: h = b(h)
        return self.head(h).squeeze(1)
g = np.load("G40.npy"); y = np.load("Y40.npy"); s = np.load("S40.npy")
assignment = split_by_series(g, folds=5, seed=0); m2 = assignment == 2; gf, yf, sf = g[m2], y[m2], s[m2]
D = "structural-break-real-time-test/data/"
X = pd.read_parquet(D + "X_train.parquet")
starts = np.flatnonzero(np.concatenate([[True], gf[1:] != gf[:-1]])); bounds = np.append(starts, len(gf))
model = RawTCN(); model.load_state_dict(torch.load("nets_raw/member_w0.pt", map_location="cpu")); model.eval()
raw = np.empty(len(gf))
with torch.no_grad():
    for a, b in zip(starts, bounds[1:]):
        sid = int(gf[a]); part = X.loc[sid]
        v = part.value.to_numpy("float64"); p = part.period.to_numpy()
        hist, online = v[p == 1], v[p == 2]; mu, sd = hist.mean(), hist.std() + 1e-12
        z = np.concatenate([(hist[-CTX:] - mu)/sd, (online - mu)/sd]).astype("float32")
        feat = torch.from_numpy(np.stack([z, np.arcsinh(z)]))[None]
        o = model(feat)[0, -(b-a):].numpy().astype("float64")
        raw[a:b] = 1/(1+np.exp(-o))
np.save("fold2_raw_w0.npy", raw)
sig = lambda a: 1/(1+np.exp(-a.astype("float64")))
rank = sig(np.load("fold2_rank_bocpd_200.npy")); clf = np.load("fold2_clf_bocpdh50_206.npy").astype("float64")
net = np.mean([np.load(f"fold2_sig_nets_aug3_member_p{i}.pt.npy") for i in range(6)] + [np.load(f"fold2_sig_nets_aug3_last_member_z{i}.pt.npy") for i in range(6)], 0)
base = 0.45*(0.7*rank + 0.3*clf) + 0.55*net
print(f"сырая сеть соло: {ts_auc(raw, yf, sf):.4f} | ансамбль #30: {ts_auc(base, yf, sf):.4f}")
print(f"корреляция Спирмена с ансамблем: {pd.Series(raw).corr(pd.Series(base), method='spearman'):.3f} | с сетями: {pd.Series(raw).corr(pd.Series(net), method='spearman'):.3f}")
best = (0, 0)
for w in (0.02, 0.05, 0.10, 0.15, 0.25):
    v = ts_auc((1-w)*base + w*raw, yf, sf)
    if v > best[0]: best = (v, w)
    print(f"  доля сырой сети {w:.2f}: смесь {v:.4f}  ({v - ts_auc(base, yf, sf):+.4f})")
print(f"ЛУЧШЕЕ: {best[0]:.4f} при доле {best[1]}")
