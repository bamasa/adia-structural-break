"""076: nets on TRIPLE augmentation — three times more pseudo-series.

The nets are data-bound: single augmentation gave records to both architectures.
Only the rankers have seen the triple one (AUG3, three random cuts per series).
AUG3_X is 8.3 GB — read via mmap series by series. Half of the members are plain (200
inputs), half diff (600). Kill: holdouts no higher than the pools on single augmentation."""
import sys, time, os
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
               np.load("E4.npy"), np.load("B40.npy"), np.load("B2.npy"),
               np.load("SPEC14.npy")]).astype("float32")
y = np.load("Y40.npy"); g = np.load("G40.npy")
assignment = split_by_series(g, folds=5, seed=0)
starts = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]]))
bounds = np.append(starts, len(g))
mu = np.load("mu200.npy"); sd = np.load("sd200.npy")

# Original series (fold 2 is kept aside — it is the yardstick).
orig, orig_fold = [], []
for a, b in zip(starts, bounds[1:]):
    orig.append((((X[a:b] - mu) / sd), y[a:b].astype("float32")))
    orig_fold.append(int(assignment[a]))
del X

# Triple augmentation: 8.3 GB, read via mmap series by series.
AX = np.load("AUG3_X.npy", mmap_mode="r")
AY = np.load("AUG3_Y.npy"); AG = np.load("AUG3_G.npy")
a_starts = np.flatnonzero(np.concatenate([[True], AG[1:] != AG[:-1]]))
a_bounds = np.append(a_starts, len(AG))
fold_by_sid = {int(g[a]): int(assignment[a]) for a in starts}
aug = []
for a, b in zip(a_starts, a_bounds[1:]):
    sid = (int(AG[a]) - 100000) // 10
    # a pseudo-series inherits its parent's fold; fold 2 is excluded
    if fold_by_sid.get(sid, 0) != 2:
        aug.append((((np.asarray(AX[a:b]) - mu) / sd).astype("float32"),
                    AY[a:b].astype("float32")))
del AX, AY
print(f"original {len(orig)}, augmentation {len(aug)} [{time.time()-t0:.0f}s]", flush=True)

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

def with_diffs(seg):
    d1 = np.zeros_like(seg); d1[1:] = seg[1:] - seg[:-1]
    d10 = np.zeros_like(seg)
    if len(seg) > 10:
        d10[10:] = seg[10:] - seg[:-10]
    return np.hstack([seg, d1 * 3.0, d10 * 1.5]).astype("float32")

class ChanTCN(nn.Module):
    def __init__(self, n_in, ch=64):
        super().__init__()
        self.inp = nn.Conv1d(n_in, ch, 1)
        self.blocks = nn.ModuleList([Block(ch, d) for d in (1, 2, 4, 8, 16, 32)])
        self.head = nn.Conv1d(ch, 1, 1)
    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = b(h)
        return self.head(h).squeeze(1)

def batch_tensors(rows, use_diffs):
    L = max(len(f) for f, _ in rows)
    Xb = torch.zeros(len(rows), 600 if use_diffs else 200, L)
    M = torch.zeros(len(rows), L, dtype=torch.bool)
    Yb = torch.zeros(len(rows), L)
    for i, (f, lab) in enumerate(rows):
        n = len(f)
        Xb[i, :, L - n:] = torch.from_numpy((with_diffs(f) if use_diffs else f).T)
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

def holdout_auc(model, rows, use_diffs):
    model.eval()
    sc_, lb_, st_ = [], [], []
    with torch.no_grad():
        rs = sorted(rows, key=lambda r: len(r[0]))
        for k in range(0, len(rs), 96):
            chunk = rs[k:k + 96]
            Xb, _, _ = batch_tensors(chunk, use_diffs)
            out = model(Xb).cpu().numpy()
            for row, (f, lab) in zip(out, chunk):
                sc_.append(row[-len(f):]); lb_.append(lab.astype(int)); st_.append(np.arange(len(f)))
    return ts_auc(np.concatenate(sc_), np.concatenate(lb_), np.concatenate(st_))

os.makedirs("nets_aug3", exist_ok=True)
pool_orig = [orig[i] for i in range(len(orig)) if orig_fold[i] != 2]
# After the d0 probe (holdout 0.6543 -> fold 2 0.5875) the diff members were removed
# from the queue: they do not transfer, and cost 4.4 hours each.
JOBS = [("p", 0, False), ("d", 0, True), ("p", 1, False),
        ("p", 2, False), ("p", 3, False), ("p", 4, False), ("p", 5, False)]
for tag, member, use_diffs in JOBS:
    path = f"nets_aug3/member_{tag}{member}.pt"
    if os.path.exists(path):
        continue
    seed = (51000 if tag == "p" else 52000) + member
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    idx = rng.permutation(len(pool_orig))
    hold_n = int(0.08 * len(pool_orig))
    hold = [pool_orig[i] for i in idx[:hold_n]]
    train_set = [pool_orig[i] for i in idx[hold_n:]] + aug
    rng.shuffle(train_set)
    model = ChanTCN(600 if use_diffs else 200).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
    order = sorted(range(len(train_set)), key=lambda i: len(train_set[i][0]))
    batches = [[train_set[i] for i in order[k:k + 24]] for k in range(0, len(order), 24)]
    best = (0.0, None)
    for epoch in range(10):
        model.train()
        for bi in rng.permutation(len(batches)):
            Xb, M, Yb = batch_tensors(batches[bi], use_diffs)
            opt.zero_grad()
            logits = model(Xb)
            bce = (F.binary_cross_entropy_with_logits(logits, Yb, reduction="none")[M]).mean()
            loss = rank_loss(logits, M, Yb, rng) + 0.3 * bce
            loss.backward()
            opt.step()
        sched.step()
        a = holdout_auc(model, hold, use_diffs)
        if a > best[0]:
            best = (a, {k: v.cpu().clone() for k, v in model.state_dict().items()})
    torch.save(best[1], path)
    print(f"aug3-net {tag}{member}: holdout {best[0]:.4f}  [{time.time()-t0:.0f}s]", flush=True)
print("done", flush=True)
