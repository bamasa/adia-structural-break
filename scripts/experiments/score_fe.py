"""#30 on the first-revision series: an honest cross-dataset measurement (torch-only: the nets; trees separately)."""
import sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "repo/src")
from structural_break.combiners import ts_auc
X = np.load("FE_X206.npy"); y = np.load("FE_Y.npy"); g = np.load("FE_G.npy"); s = np.load("FE_S.npy")
mu = np.load("mu200.npy"); sd = np.load("sd200.npy")
class Block(nn.Module):
    def __init__(self, ch, dil):
        super().__init__(); self.conv = nn.Conv1d(ch, ch, 3, dilation=dil); self.mix = nn.Conv1d(ch, ch, 1); self.drop = nn.Dropout(0.1); self.dil = dil
    def forward(self, h):
        r = h; h = F.pad(h, (2 * self.dil, 0)); h = self.drop(F.gelu(self.conv(h))); return r + self.mix(h)
class ChanTCN(nn.Module):
    def __init__(self, n_in=200, ch=64):
        super().__init__(); self.inp = nn.Conv1d(n_in, ch, 1); self.blocks = nn.ModuleList([Block(ch, d) for d in (1, 2, 4, 8, 16, 32)]); self.head = nn.Conv1d(ch, 1, 1)
    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks: h = b(h)
        return self.head(h).squeeze(1)
DEV = torch.device("mps")
starts = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]])); bounds = np.append(starts, len(g))
rows = sorted(zip(starts, bounds[1:]), key=lambda ab: ab[1] - ab[0])
paths = [f"nets_aug3/member_p{i}.pt" for i in range(6)] + [f"nets_aug3_last/member_z{i}.pt" for i in range(6)]
net_sig = np.zeros(len(y))
for p in paths:
    m = ChanTCN().to(DEV); m.load_state_dict(torch.load(p, map_location=DEV)); m.eval()
    with torch.no_grad():
        for k in range(0, len(rows), 64):
            chunk = rows[k:k+64]; L = max(b - a for a, b in chunk)
            Xb = torch.zeros(len(chunk), 200, L)
            for i, (a, b) in enumerate(chunk):
                Xb[i, :, L-(b-a):] = torch.from_numpy(((X[a:b, :200] - mu) / sd).T)
            o = m(Xb.to(DEV)).cpu().numpy()
            for i, (a, b) in enumerate(chunk):
                net_sig[a:b] += 1/(1+np.exp(-o[i, L-(b-a):].astype("float64")))
net_sig /= len(paths)
np.save("fe_net_sig.npy", net_sig)
print(f"nets (12) on the first revision: TS-AUC {ts_auc(net_sig, y, s):.4f}", flush=True)
