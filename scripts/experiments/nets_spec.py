"""128: nets on the multi-window SPECTRUM channels (32 inputs) — strengthening the most independent member (124: corr. 0.12, alone 0.5285).

In 121 the net on the mass channels was stronger than the mass classifier (0.5914 versus 0.5830). If the same happens with the spectrum,
the member with correlation 0.12 may become strong enough to pay off.

Original description of 121:

#36 showed in the cloud: an independently built member pays off (+0.0039). The mass member is
trees on 90 channels at a single step. Here the same net as in the ensemble (ChanTCN,
rank loss, last epoch, triple augmentation), but it reads the 90 battery channels
as trajectories. A different family on independent features: it should diverge both from
the ensemble trees and from the mass classifier. Three members, 082b seeds.

082 (15 epochs) gave 0.5936 on seed 51000 versus 0.6036 for 10 epochs: the long
schedule drifts into memorisation. Here ten epochs, the six #28 seeds,
fold 2 at the last epoch; ensemble comparison against #28 (0.6163).

081d: fold 2 grows monotonically up to the last epochs; the holdout is noise in antiphase.
Here six members with the #28 seeds, no holdout (+8% data), cosine over 15
epochs; fold 2 is evaluated at epochs 9 and 14, both are saved. Answers: (a)
last epoch@10 versus the leaky selection of #28, (b) 15 epochs versus 10.

081 with a clean holdout gave 0.5623 on fold 2 versus 0.6048 for the same
seed with a leaky one. Here a single member, seed 51000, and per epoch: clean holdout,
leaky holdout (the same series, but their pseudo-series are in training — as in #28) and
fold 2. Answers which epoch-selection criterion transfers.

In all pools the best epoch was chosen by a holdout whose pseudo-series were
in training (080b: holdout 0.7255 with fold 2 at 0.5678). Epoch selection was
biased towards memorisation. Here the holdout is honest. Otherwise the #28 recipe.
Kill: alone on fold 2 no higher than 0.605.
Original 076: nets on TRIPLE augmentation — three times more pseudo-series.

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
X = np.load("MSPEC32.npy").astype("float32")
y = np.load("Y40.npy"); g = np.load("G40.npy")
assignment = split_by_series(g, folds=5, seed=0)
starts = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]]))
bounds = np.append(starts, len(g))
mu = X.mean(0); sd = X.std(0) + 1e-6
np.save("mu32s.npy", mu); np.save("sd32s.npy", sd)

# Original series (fold 2 is kept aside — it is the yardstick).
orig, orig_fold = [], []
for a, b in zip(starts, bounds[1:]):
    orig.append((((X[a:b] - mu) / sd), y[a:b].astype("float32")))
    orig_fold.append(int(assignment[a]))
del X

# Triple augmentation: 8.3 GB, read via mmap series by series.
AX = np.load("AUG3_MSPEC32.npy", mmap_mode="r")
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
                    AY[a:b].astype("float32"), sid))
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
    Xb = torch.zeros(len(rows), 600 if use_diffs else 32, L)
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

os.makedirs("nets_spec", exist_ok=True)
fold2_rows = [orig[i] for i in range(len(orig)) if orig_fold[i] == 2]
pool_orig = [orig[i] for i in range(len(orig)) if orig_fold[i] != 2]
pool_sid = [int(g[starts[i]]) for i in range(len(orig)) if orig_fold[i] != 2]
# After the d0 probe (holdout 0.6543 -> fold 2 0.5875) the diff members were removed
# from the queue: they do not transfer, and cost 4.4 hours each.
JOBS = [("s", i, False) for i in range(3)]
for tag, member, use_diffs in JOBS:
    path = f"nets_spec/member_{tag}{member}.pt"
    if os.path.exists(path):
        continue
    seed = 51000 + member  # the same seeds as p0–p2: paired comparison
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    idx = rng.permutation(len(pool_orig))
    hold_n = int(0.08 * len(pool_orig))
    train_set = list(pool_orig) + [(f, lab) for f, lab, sid in aug]  # 082: no holdout
    rng.shuffle(train_set)
    print(f"  member {tag}{member}: training {len(train_set)} series, no holdout", flush=True)
    rng.shuffle(train_set)
    model = ChanTCN(600 if use_diffs else 32).to(DEVICE)
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
        if epoch == 9:
            f2 = holdout_auc(model, fold2_rows, use_diffs)
            print(f"    member {tag}{member} epoch {epoch}: fold 2 {f2:.4f}  [{time.time()-t0:.0f}s]", flush=True)
            torch.save({k: v.cpu().clone() for k, v in model.state_dict().items()},
                       f"nets_spec/member_{tag}{member}_epoch{epoch}.pt")
            best = (f2, {k: v.cpu().clone() for k, v in model.state_dict().items()})
    torch.save(best[1], path)
    print(f"spec-net {tag}{member}: fold 2 at the last epoch {best[0]:.4f}  [{time.time()-t0:.0f}s]", flush=True)
print("done", flush=True)
