"""023: the TCN with a ranking loss and boundary augmentation, holdout fold 0."""
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
DEVICE = torch.device("mps")
HIST_CAP = 384
t0 = time.time()

x = pd.read_parquet("structural-break-real-time-test/data/X_train.parquet")
y = pd.read_parquet("structural-break-real-time-test/data/y_train.parquet")
g = np.load("G40.npy")
assignment = split_by_series(g, folds=5, seed=0)
starts = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]]))
sid_fold = {int(g[st]): int(assignment[st]) for st in starts}

raw = []
for sid, hist, online, labels in iter_series(x, y):
    raw.append((int(sid), np.asarray(hist, dtype="float64"),
                np.asarray(online, dtype="float64"), labels.astype("float32"),
                sid_fold[int(sid)]))
print(f"loading {time.time()-t0:.0f}s", flush=True)

def standardise(hist, online):
    norm = Normalisation.fit(hist)
    n = len(hist)
    z_h = np.asarray([norm.clip(norm.standardise(float(v), i - n)) for i, v in enumerate(hist)],
                     dtype="float32")
    z_o = np.asarray([norm.clip(norm.standardise(float(v), i)) for i, v in enumerate(online)],
                     dtype="float32")
    return z_h, z_o

# Validation is fixed; the training set is augmented on every call.
val_fixed = []
train_raw = []
for sid, hist, online, lab, fold in raw:
    if fold == 0:
        z_h, z_o = standardise(hist, online)
        val_fixed.append((z_h[-HIST_CAP:], z_o, lab))
    else:
        train_raw.append((hist, online, lab))
print(f"validation preparation {time.time()-t0:.0f}s", flush=True)

def augment(hist, online, lab, rng):
    """Trim the history, truncate the online part, shift the boundary for clean series."""
    h, o, l = hist, online, lab
    if l.max() == 0 and len(o) > 60 and rng.random() < 0.5:
        move = rng.integers(10, min(len(o) // 2, 200))
        h = np.concatenate([h, o[:move]]); o = o[move:]; l = l[move:]
    if len(h) > 96:
        keep = int(rng.integers(96, min(len(h), 640)))
        h = h[-keep:]
    if l.max() == 0 and len(o) > 40 and rng.random() < 0.3:
        cut = int(rng.integers(30, len(o)))
        o = o[:cut]; l = l[:cut]
    return h, o, l

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
    def __init__(self, ch=96):
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
print("parameters:", sum(p.numel() for p in model.parameters()), flush=True)
EPOCHS = 40
opt = torch.optim.Adam(model.parameters(), lr=1.2e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

def batch_tensors(rows):
    L = max(len(a) + len(b) for a, b, _ in rows)
    X = torch.zeros(len(rows), 3, L)
    onmask = torch.zeros(len(rows), L, dtype=torch.bool)
    Y = torch.zeros(len(rows), L)
    for i, (z_h, z_o, lab) in enumerate(rows):
        n = len(z_h) + len(z_o)
        seq = np.concatenate([z_h, z_o])
        X[i, 0, L - n:] = torch.from_numpy(seq)
        X[i, 1, L - n:] = torch.from_numpy(np.abs(seq))
        X[i, 2, L - len(z_o):] = 1.0
        onmask[i, L - len(z_o):] = True
        Y[i, L - len(z_o):] = torch.from_numpy(np.ascontiguousarray(lab))
    return X.to(DEVICE), onmask.to(DEVICE), Y.to(DEVICE)

def rank_loss(logits, onmask, Y, rng):
    """Pairwise ranking loss on matching steps within the batch: at every
    selected step the broken series (label 1) must rank above the clean ones (label 0)."""
    B, L = logits.shape
    total, count = logits.new_zeros(()), 0
    steps = rng.choice(L, size=min(48, L), replace=False)
    for t in steps:
        alive = onmask[:, t]
        if alive.sum() < 2:
            continue
        s = logits[alive, t]; lab = Y[alive, t]
        pos, neg = s[lab > 0.5], s[lab < 0.5]
        if len(pos) == 0 or len(neg) == 0:
            continue
        diff = neg.unsqueeze(0) - pos.unsqueeze(1)
        total = total + F.softplus(diff).mean()
        count += 1
    return total / max(count, 1)

def validate():
    model.eval()
    scores, labels, steps = [], [], []
    with torch.no_grad():
        order = sorted(range(len(val_fixed)), key=lambda i: len(val_fixed[i][0]) + len(val_fixed[i][1]))
        for k in range(0, len(order), 64):
            rows = [val_fixed[i] for i in order[k:k + 64]]
            X, _, _ = batch_tensors(rows)
            out = model(X).cpu().numpy()
            for row, (_z_h, z_o, lab) in zip(out, rows):
                scores.append(row[-len(z_o):]); labels.append(lab); steps.append(np.arange(len(z_o)))
    sc = np.concatenate(scores)
    np.save("tcn_v3_val_scores.npy", sc)
    return ts_auc(sc, np.concatenate(labels).astype(int), np.concatenate(steps))

for epoch in range(EPOCHS):
    model.train()
    rng = np.random.default_rng(1000 + epoch)
    prepared = []
    for h, o, l in train_raw:
        h2, o2, l2 = augment(h, o, l, rng)
        z_h, z_o = standardise(h2, o2)
        prepared.append((z_h[-HIST_CAP:], z_o, l2))
    order = sorted(range(len(prepared)), key=lambda i: len(prepared[i][0]) + len(prepared[i][1]))
    batches = [[prepared[i] for i in order[k:k + 24]] for k in range(0, len(order), 24)]
    perm = rng.permutation(len(batches))
    total = 0.0
    for bi in perm:
        X, onmask, Y = batch_tensors(batches[bi])
        opt.zero_grad()
        logits = model(X)
        bce = (F.binary_cross_entropy_with_logits(logits, Y, reduction="none")[onmask]).mean()
        loss = rank_loss(logits, onmask, Y, rng) + 0.3 * bce
        loss.backward()
        opt.step()
        total += float(loss)
    sched.step()
    auc = validate()
    torch.save(model.state_dict(), "tcn_v3.pt")
    print(f"epoch {epoch}: loss {total/len(batches):.4f}, TS-AUC fold 0 {auc:.4f}  [{time.time()-t0:.0f}s]", flush=True)
print("done", flush=True)
