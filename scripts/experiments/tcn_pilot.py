"""017 pilot: a small causal TCN scoring every online step, holdout = fold 0.

Input per series: the standardised stream (history capped at the last 512
points, then the online part) plus an is-online flag channel. Loss: BCE on
online steps only, each series weighted equally. Val: TS-AUC on fold-0
series -- the 104-channel model scores 0.5874 there.
"""
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

torch.manual_seed(0)
np.random.seed(0)
HIST_CAP, DEVICE = 256, torch.device("mps")
t0 = time.time()

x = pd.read_parquet("structural-break-real-time-test/data/X_train.parquet")
y = pd.read_parquet("structural-break-real-time-test/data/y_train.parquet")
g = np.load("G40.npy")
assignment = split_by_series(g, folds=5, seed=0)
sid_fold = {}
starts = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]]))
for st in starts:
    sid_fold[int(g[st])] = int(assignment[st])

series = []
for sid, hist, online, labels in iter_series(x, y):
    norm = Normalisation.fit(hist)
    n = len(hist)
    z_h = np.asarray([norm.clip(norm.standardise(float(v), i - n)) for i, v in enumerate(hist)],
                     dtype="float32")[-HIST_CAP:]
    z_o = np.asarray([norm.clip(norm.standardise(float(v), i)) for i, v in enumerate(online)],
                     dtype="float32")
    lab = labels.astype("float32")
    series.append((int(sid), z_h, z_o, lab, sid_fold[int(sid)]))
print(f"preparation {time.time()-t0:.0f}s: {len(series)} series", flush=True)

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
    def __init__(self, ch=24):
        super().__init__()
        self.inp = nn.Conv1d(3, ch, 1)
        self.blocks = nn.ModuleList([Block(ch, d) for d in (1, 2, 4, 8, 16, 32, 64)])
        self.head = nn.Conv1d(ch, 1, 1)
    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = b(h)
        return self.head(h).squeeze(1)

model = TCN().to(DEVICE)
print("parameters:", sum(p.numel() for p in model.parameters()), flush=True)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

train_set = [r for r in series if r[4] != 0]
val_set = [r for r in series if r[4] == 0]

def make_batch(rows):
    L = max(len(z_h) + len(z_o) for _, z_h, z_o, _, _ in rows)
    X = torch.zeros(len(rows), 3, L)
    M = torch.zeros(len(rows), L)
    Y = torch.zeros(len(rows), L)
    for i, (_, z_h, z_o, lab, _) in enumerate(rows):
        n = len(z_h) + len(z_o)
        seq = np.concatenate([z_h, z_o])
        X[i, 0, L - n:] = torch.from_numpy(seq)
        X[i, 1, L - n:] = torch.from_numpy(np.abs(seq))
        X[i, 2, L - len(z_o):] = 1.0
        M[i, L - len(z_o):] = 1.0 / max(len(z_o), 1)
        Y[i, L - len(z_o):] = torch.from_numpy(lab)
    return X.to(DEVICE), M.to(DEVICE), Y.to(DEVICE)

def validate():
    model.eval()
    scores, labels, steps = [], [], []
    with torch.no_grad():
        order = sorted(range(len(val_set)), key=lambda i: len(val_set[i][1]) + len(val_set[i][2]))
        for k in range(0, len(order), 64):
            rows = [val_set[i] for i in order[k:k + 64]]
            X, M, _ = make_batch(rows)
            out = torch.sigmoid(model(X)).cpu().numpy()
            for row, (_, z_h, z_o, lab, _) in zip(out, rows):
                sc = row[-len(z_o):]
                scores.append(sc); labels.append(lab); steps.append(np.arange(len(z_o)))
    return ts_auc(np.concatenate(scores), np.concatenate(labels).astype(int),
                  np.concatenate(steps))

order = sorted(range(len(train_set)), key=lambda i: len(train_set[i][1]) + len(train_set[i][2]))
batches = [[train_set[i] for i in order[k:k + 32]] for k in range(0, len(order), 32)]
for epoch in range(8):
    model.train()
    rng = np.random.default_rng(epoch)
    perm = rng.permutation(len(batches))
    total = 0.0
    for bi in perm:
        X, M, Y = make_batch(batches[bi])
        opt.zero_grad()
        logits = model(X)
        loss = (F.binary_cross_entropy_with_logits(logits, Y, reduction="none") * M).sum() / M.sum()
        loss.backward()
        opt.step()
        total += float(loss)
    auc = validate()
    print(f"epoch {epoch}: loss {total/len(batches):.4f}, TS-AUC fold 0 {auc:.4f}  [{time.time()-t0:.0f}s]", flush=True)
    torch.save(model.state_dict(), "tcn_pilot.pt")
print("done", flush=True)
