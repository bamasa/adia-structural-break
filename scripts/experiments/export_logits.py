"""Recompute fold-2 logits for saved net members (the lean scripts' export had taken the last point only)."""
import sys, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "repo/src")
from structural_break.combiners import split_by_series
kind, outdir, members = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
DEVICE = torch.device("mps")
if kind == "a":
    X200 = np.load("X200.npy", mmap_mode="r"); W = np.load("W111.npy", mmap_mode="r"); mu, sd = np.load("mu_a.npy"), np.load("sd_a.npy"); N_IN = 311
    gather = lambda a, b: ((np.hstack([np.asarray(X200[a:b]), np.asarray(W[a:b])]) - mu) / sd).astype("float32")
else:
    W = np.load("WSTREAM5.npy", mmap_mode="r"); mu, sd = np.load("mu_s.npy"), np.load("sd_s.npy"); N_IN = 5
    gather = lambda a, b: ((np.asarray(W[a:b]) - mu) / sd).astype("float32")
g = np.load("G40.npy"); f = split_by_series(g, folds=5, seed=0)
starts = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]])); bounds = np.append(starts, len(g))
rows = [(int(a), int(b)) for a, b in zip(starts, bounds[1:]) if f[a] == 2]
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
order = sorted(range(len(rows)), key=lambda i: rows[i][1] - rows[i][0])
for m in members:
    model = ChanTCN(N_IN).to(DEVICE); model.load_state_dict(torch.load(f"{outdir}/member_{kind}{m}.pt", map_location="cpu")); model.eval()
    sc = [None] * len(rows)
    with torch.no_grad():
        for k in range(0, len(order), 96):
            chunk = order[k:k + 96]; segs = [gather(*rows[i]) for i in chunk]; L = max(len(x) for x in segs)
            Xb = torch.zeros(len(chunk), N_IN, L)
            for j, x in enumerate(segs): Xb[j, :, L - len(x):] = torch.from_numpy(x.T)
            out = model(Xb.to(DEVICE)).cpu().numpy()
            for row, i, x in zip(out, chunk, segs): sc[i] = row[-len(x):]
    arr = np.concatenate(sc); np.save(f"{outdir}/fold2_logits_{kind}{m}.npy", arr); print(f"{kind}{m}: {len(arr)} logits", flush=True)
