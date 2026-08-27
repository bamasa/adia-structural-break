"""033: сеть поверх траекторий каналов (186-мерная последовательность на вход)."""
import sys, time
sys.path.insert(0, "repo/src")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from structural_break.combiners import split_by_series, ts_auc

torch.manual_seed(1)
np.random.seed(1)
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
series = []
for a, b in zip(starts, bounds[1:]):
    fold = int(assignment[a])
    series.append((((X[a:b] - mu) / sd), y[a:b].astype("float32"), fold))
del X
print(f"подготовка {time.time()-t0:.0f}s: {len(series)} рядов", flush=True)

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
print("параметров:", sum(p.numel() for p in model.parameters()), flush=True)
EPOCHS = 10
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

train_set = [r for r in series if r[2] != 0]
val_set = [r for r in series if r[2] == 0]

def batch_tensors(rows):
    L = max(len(f) for f, _, _ in rows)
    Xb = torch.zeros(len(rows), 186, L)
    M = torch.zeros(len(rows), L, dtype=torch.bool)
    Yb = torch.zeros(len(rows), L)
    for i, (f, lab, _) in enumerate(rows):
        n = len(f)
        Xb[i, :, L - n:] = torch.from_numpy(f.T)
        M[i, L - n:] = True
        Yb[i, L - n:] = torch.from_numpy(lab)
    return Xb.to(DEVICE), M.to(DEVICE), Yb.to(DEVICE)

def rank_loss(logits, onmask, Y, rng):
    B, L = logits.shape
    total, count = logits.new_zeros(()), 0
    for t in rng.choice(L, size=min(48, L), replace=False):
        alive = onmask[:, t]
        if alive.sum() < 2:
            continue
        sc, lab = logits[alive, t], Y[alive, t]
        pos, neg = sc[lab > 0.5], sc[lab < 0.5]
        if len(pos) == 0 or len(neg) == 0:
            continue
        total = total + F.softplus(neg.unsqueeze(0) - pos.unsqueeze(1)).mean()
        count += 1
    return total / max(count, 1)

def validate():
    model.eval()
    scores, labels, steps = [], [], []
    with torch.no_grad():
        order = sorted(range(len(val_set)), key=lambda i: len(val_set[i][0]))
        for k in range(0, len(order), 48):
            rows = [val_set[i] for i in order[k:k + 48]]
            Xb, _, _ = batch_tensors(rows)
            out = model(Xb).cpu().numpy()
            for row, (f, lab, _) in zip(out, rows):
                scores.append(row[-len(f):]); labels.append(lab.astype(int)); steps.append(np.arange(len(f)))
    sc = np.concatenate(scores)
    np.save("tcn_chan_s1_val.npy", sc)
    return ts_auc(sc, np.concatenate(labels), np.concatenate(steps))

order = sorted(range(len(train_set)), key=lambda i: len(train_set[i][0]))
batches = [[train_set[i] for i in order[k:k + 24]] for k in range(0, len(order), 24)]
for epoch in range(EPOCHS):
    model.train()
    rng = np.random.default_rng(2000 + epoch)
    total = 0.0
    for bi in rng.permutation(len(batches)):
        Xb, M, Yb = batch_tensors(batches[bi])
        opt.zero_grad()
        logits = model(Xb)
        bce = (F.binary_cross_entropy_with_logits(logits, Yb, reduction="none")[M]).mean()
        loss = rank_loss(logits, M, Yb, rng) + 0.3 * bce
        loss.backward()
        opt.step()
        total += float(loss)
    sched.step()
    auc = validate()
    torch.save(model.state_dict(), "tcn_chan_s1.pt")
    print(f"эпоха {epoch}: loss {total/len(batches):.4f}, TS-AUC фолд-0 {auc:.4f}  [{time.time()-t0:.0f}s]", flush=True)
print("готово", flush=True)
