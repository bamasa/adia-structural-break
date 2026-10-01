"""046: bag scaling law — 48 new members with extended diversity."""
import sys, time
sys.path.insert(0, "repo/src")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from structural_break.combiners import split_by_series

torch.manual_seed(7)
DEVICE = torch.device("mps")
t0 = time.time()

X = np.hstack([np.load("X40.npy"), np.load("C_cnn.npy")[:, None],
               np.load("N9.npy").astype("float32"), np.load("X50a.npy"),
               np.load("E4.npy"), np.load("B40.npy"), np.load("B2.npy")]).astype("float32")
y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
assignment = split_by_series(g, folds=5, seed=0)
starts = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]]))
bounds = np.append(starts, len(g))
mu = np.load("net_bag_mac/mu.npy"); sd = np.load("net_bag_mac/sd.npy")
series = [(((X[a:b] - mu) / sd), y[a:b].astype("float32"), int(assignment[a]))
          for a, b in zip(starts, bounds[1:])]
print(f"{len(series)} series ready [{time.time()-t0:.0f}s]", flush=True)

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
    def __init__(self, ch, dils):
        super().__init__()
        self.inp = nn.Conv1d(186, ch, 1)
        self.blocks = nn.ModuleList([Block(ch, d) for d in dils])
        self.head = nn.Conv1d(ch, 1, 1)
    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = b(h)
        return self.head(h).squeeze(1)

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
    L = logits.shape[1]
    total, count = logits.new_zeros(()), 0
    for t in rng.choice(L, size=min(48, L), replace=False):
        alive = onmask[:, t]
        if alive.sum() < 2:
            continue
        sc, lab = logits[alive, t], Y[alive, t]
        pos, neg = sc[lab > 0.5], sc[lab < 0.5]
        if not len(pos) or not len(neg):
            continue
        total = total + F.softplus(neg.unsqueeze(0) - pos.unsqueeze(1)).mean()
        count += 1
    return total / max(count, 1)

CONFIGS = [
    dict(ch=64, dils=(1, 2, 4, 8, 16, 32)),
    dict(ch=48, dils=(1, 2, 4, 8, 16, 32, 64)),
    dict(ch=80, dils=(1, 2, 4, 8, 16)),
]
FRACS = (0.5, 0.6, 0.7)
pool = [r for r in series if r[2] not in (0, 1)]

for member in range(48):
    rng = np.random.default_rng(500 + member)
    torch.manual_seed(500 + member)
    cfg = CONFIGS[member % 3]
    frac = FRACS[(member // 3) % 3]
    take = rng.choice(len(pool), size=int(frac * len(pool)), replace=False)
    train_set = [pool[i] for i in take]
    model = ChanTCN(**cfg).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
    order = sorted(range(len(train_set)), key=lambda i: len(train_set[i][0]))
    batches = [[train_set[i] for i in order[k:k + 24]] for k in range(0, len(order), 24)]
    for epoch in range(10):
        model.train()
        for bi in rng.permutation(len(batches)):
            Xb, M, Yb = batch_tensors(batches[bi])
            opt.zero_grad()
            logits = model(Xb)
            bce = (F.binary_cross_entropy_with_logits(logits, Yb, reduction="none")[M]).mean()
            loss = rank_loss(logits, M, Yb, rng) + 0.3 * bce
            loss.backward()
            opt.step()
        sched.step()
    torch.save({"cfg": cfg, "state": model.state_dict()}, f"net_bag_mac/grow_{member}.pt")
    print(f"member {member} (ch{cfg['ch']}, dils{len(cfg['dils'])}, frac{frac}) done [{time.time()-t0:.0f}s]", flush=True)
print("done", flush=True)
