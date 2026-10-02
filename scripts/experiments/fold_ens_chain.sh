HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/../../.."                       # the workspace: the repository's parent directory
S="${S:-$HERE}"                           # where the companion tcn_* scripts live (a scratch folder when first run)
for d in 1 2 3 4; do
  .venv/bin/python $S/tcn_folddrop$d.py > tcn_fd$d.log 2>&1
done
.venv/bin/python - <<'PY'
import sys, numpy as np, torch
sys.path.insert(0, "repo/src")
import torch.nn as nn, torch.nn.functional as F
from structural_break.combiners import split_by_series, ts_auc

DEVICE = torch.device("mps")
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

def aligned(path):
    model = ChanTCN().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    out = []
    with torch.no_grad():
        for a, b in zip(starts, bounds[1:]):
            if int(assignment[a]) != 0:
                continue
            f = ((X[a:b] - mu) / sd)
            out.append(model(torch.from_numpy(f.T).unsqueeze(0).to(DEVICE)).cpu().numpy()[0])
    return np.concatenate(out)

mask = assignment == 0
yf, sf = y[mask], s[mask]
sigs = []
for d in (1, 2, 3, 4):
    v = aligned(f"tcn_fd{d}.pt").astype("float64")
    print(f"without fold {d}: alone {ts_auc(v, yf, sf):.4f}", flush=True)
    sigs.append(1.0 / (1.0 + np.exp(-v)))
ens = np.mean(sigs, axis=0)
print(f"fold ensemble (4 nets): alone {ts_auc(ens, yf, sf):.4f} (single: 0.5939)", flush=True)
np.save("tcn_foldens_fold0.npy", ens)
clf = np.load("oof_cfg5.npy")[mask].astype("float64")
rnk = np.load("oof_rank.npy")[mask].astype("float64")
blend = 0.6/(1.0+np.exp(-rnk)) + 0.4*clf
for w in (0.4, 0.5, 0.6):
    print(f"triple with {w:.0%} fold ensemble: {ts_auc((1-w)*blend + w*ens, yf, sf):.4f} (reference #18: 0.6068)", flush=True)
PY
