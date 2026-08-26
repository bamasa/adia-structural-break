"""023 verdict: aligned TCN fold-0 scores, ensemble with the stack's OOF."""
import sys, time
sys.path.insert(0, "repo/src")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from structural_break.features import Normalisation
from structural_break.stream import iter_series
from structural_break.combiners import split_by_series, ts_auc

DEVICE = torch.device("mps")
HIST_CAP = 384
t0 = time.time()

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

class TCN(nn.Module):
    def __init__(self, ch=48):
        super().__init__()
        self.inp = nn.Conv1d(3, ch, 1)
        self.blocks = nn.ModuleList([Block(ch, d) for d in (1, 2, 4, 8, 16, 32, 64, 128)])
        self.head = nn.Conv1d(ch, 1, 1)
    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = b(h)
        return self.head(h).squeeze(1)

model = TCN().to(DEVICE)
model.load_state_dict(torch.load("tcn_v2.pt", map_location=DEVICE))
model.eval()

x = pd.read_parquet("structural-break-real-time-test/data/X_train.parquet")
yp = pd.read_parquet("structural-break-real-time-test/data/y_train.parquet")
g = np.load("G40.npy"); y = np.load("Y40.npy"); s = np.load("S40.npy")
assignment = split_by_series(g, folds=5, seed=0)
starts = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]]))
sid_fold = {int(g[st]): int(assignment[st]) for st in starts}

# Скоры сети в порядке строк G40 (sid-порядок, шаги подряд).
scores = []
with torch.no_grad():
    for sid, hist, online, labels in iter_series(x, yp):
        if sid_fold[int(sid)] != 0:
            continue
        norm = Normalisation.fit(hist)
        n = len(hist)
        z_h = np.asarray([norm.clip(norm.standardise(float(v), i - n)) for i, v in enumerate(hist)],
                         dtype="float32")[-HIST_CAP:]
        z_o = np.asarray([norm.clip(norm.standardise(float(v), i)) for i, v in enumerate(online)],
                         dtype="float32")
        seq = np.concatenate([z_h, z_o])
        X = torch.zeros(1, 3, len(seq))
        X[0, 0] = torch.from_numpy(seq)
        X[0, 1] = torch.from_numpy(np.abs(seq))
        X[0, 2, len(z_h):] = 1.0
        out = model(X.to(DEVICE)).cpu().numpy()[0]
        scores.append(out[len(z_h):])
tcn = np.concatenate(scores)
mask = assignment == 0
assert len(tcn) == mask.sum(), (len(tcn), int(mask.sum()))
np.save("tcn_fold0_aligned.npy", tcn)
yf, sf = y[mask], s[mask]
print(f"сеть на фолде 0 (выровнено): {ts_auc(tcn, yf, sf):.4f}  [{time.time()-t0:.0f}s]", flush=True)

stack = np.load("oof_cfg5.npy")[mask]
print(f"стек (классификатор cfg5) фолд 0: {ts_auc(stack, yf, sf):.4f}", flush=True)

# Ансамбль: пошаговые кросс-секционные ранги, взвешенное среднее.
def step_ranks(v):
    out = np.empty_like(v, dtype="float64")
    for t_step in np.unique(sf):
        m = sf == t_step
        r = v[m].argsort().argsort().astype("float64")
        out[m] = r / max(len(r) - 1, 1)
    return out
r_stack, r_tcn = step_ranks(stack), step_ranks(tcn)
for w in (0.1, 0.2, 0.3):
    mix = (1 - w) * r_stack + w * r_tcn
    print(f"ансамбль стек+{w:.0%} сети: {ts_auc(mix, yf, sf):.4f}", flush=True)
print(f"всего {time.time()-t0:.0f}s", flush=True)
