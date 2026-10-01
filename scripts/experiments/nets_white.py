"""152: trajectory networks over the whitened channels (90 + 21 odds), 082b recipe.

The same ChanTCN, ranking loss + 0.3 BCE, ten epochs, last epoch, no holdout, triple boundary
augmentation -- but the input is the whitened member's 111 channels (WHITE90 + SR22 for the
originals, AUG3_WHITE111 for the pseudo-series) instead of the 200. Two kinds of member: "w"
reads the 111 alone; "a" reads the 200 of the core and the 111 together (311 inputs).
"""
import sys, time, os
sys.path.insert(0, "repo/src")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from structural_break.combiners import split_by_series, ts_auc

DEVICE = torch.device("mps")
t0 = time.time()
KIND = os.environ.get("NET_KIND", "w")            # "w": 111 whitened inputs; "a": 200 + 111
MEMBERS = [int(m) for m in os.environ.get("NET_MEMBERS", "0,1,2").split(",")]
OUTDIR = f"nets_white_{KIND}"
W = np.hstack([np.load("WHITE90.npy"), np.load("SR22.npy")]).astype("float32")
if KIND == "a":
    X200 = np.hstack([np.load("X40.npy"), np.load("C_cnn.npy")[:, None], np.load("N9.npy").astype("float32"), np.load("X50a.npy"),
                      np.load("E4.npy"), np.load("B40.npy"), np.load("B2.npy"), np.load("SPEC14.npy")]).astype("float32")
    W = np.hstack([X200, W]); del X200
N_IN = W.shape[1]
y = np.load("Y40.npy"); g = np.load("G40.npy")
assignment = split_by_series(g, folds=5, seed=0)
starts = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]]))
bounds = np.append(starts, len(g))
train_rows = assignment != 2
mu = W[train_rows].mean(0); sd = W[train_rows].std(0) + 1e-6
np.save(f"mu_{KIND}.npy", mu); np.save(f"sd_{KIND}.npy", sd)
orig, orig_fold = [], []
for a, b in zip(starts, bounds[1:]):
    orig.append((((W[a:b] - mu) / sd).astype("float32"), y[a:b].astype("float32")))
    orig_fold.append(int(assignment[a]))
del W
AW = np.load("AUG3_WHITE111.npy", mmap_mode="r")
AX = np.load("AUG3_X.npy", mmap_mode="r") if KIND == "a" else None
AY = np.load("AUG3_Y.npy"); AG = np.load("AUG3_G.npy")
a_starts = np.flatnonzero(np.concatenate([[True], AG[1:] != AG[:-1]]))
a_bounds = np.append(a_starts, len(AG))
fold_by_sid = {int(g[a]): int(assignment[a]) for a in starts}
aug = []
for a, b in zip(a_starts, a_bounds[1:]):
    sid = (int(AG[a]) - 100000) // 10
    if fold_by_sid.get(sid, 0) != 2:
        seg = np.asarray(AW[a:b]) if KIND == "w" else np.hstack([np.asarray(AX[a:b]), np.asarray(AW[a:b])])
        aug.append((((seg - mu) / sd).astype("float32"), AY[a:b].astype("float32"), sid))
del AW, AY
print(f"input {N_IN}, originals {len(orig)}, augmentation {len(aug)} [{time.time()-t0:.0f}s]", flush=True)

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
    Xb = torch.zeros(len(rows), 3 * N_IN if use_diffs else N_IN, L)
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

os.makedirs(OUTDIR, exist_ok=True)
fold2_rows = [orig[i] for i in range(len(orig)) if orig_fold[i] == 2]
pool_orig = [orig[i] for i in range(len(orig)) if orig_fold[i] != 2]
pool_sid = [int(g[starts[i]]) for i in range(len(orig)) if orig_fold[i] != 2]
# After the d0 probe (holdout 0.6543 -> fold 2 0.5875) the diff members were removed
# from the queue: they do not transfer, and cost 4.4 hours each.
JOBS = [(KIND, i, False) for i in MEMBERS]
for tag, member, use_diffs in JOBS:
    path = f"{OUTDIR}/member_{tag}{member}.pt"
    if os.path.exists(path):
        continue
    seed = 51000 + member  # the same seeds as p0–p2: paired comparison
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    idx = rng.permutation(len(pool_orig))
    hold_n = int(0.08 * len(pool_orig))
    train_set = list(pool_orig) + [(f, lab) for f, lab, sid in aug]  # 082: no holdout
    rng.shuffle(train_set)
    print(f"  member {tag}{member}: training on {len(train_set)} series, no holdout", flush=True)
    rng.shuffle(train_set)
    model = ChanTCN(3 * N_IN if use_diffs else N_IN).to(DEVICE)
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
            # fold-2 logits of this member, in row order, for the blend caches
            model.eval(); sc = []
            with torch.no_grad():
                for f, lab in fold2_rows:
                    Xb, _, _ = batch_tensors([(f, lab)], use_diffs); sc.append(model(Xb).cpu().numpy()[0, -len(f):])
            np.save(f"{OUTDIR}/fold2_logits_{tag}{member}.npy", np.concatenate(sc))
            torch.save({k: v.cpu().clone() for k, v in model.state_dict().items()},
                       f"{OUTDIR}/member_{tag}{member}_epoch{epoch}.pt")
            best = (f2, {k: v.cpu().clone() for k, v in model.state_dict().items()})
    torch.save(best[1], path)
    print(f"last net {tag}{member}: fold 2 at the last epoch {best[0]:.4f}  [{time.time()-t0:.0f}s]", flush=True)
print("done", flush=True)
