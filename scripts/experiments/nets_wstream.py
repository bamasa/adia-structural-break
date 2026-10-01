"""158: trajectory networks over the whitened innovation STREAM itself (5 inputs: n_u, n_c, n_u^2-1, lag-1 product, log scale),
082b recipe, with a memory-lean loader: rows are gathered from memory-mapped matrices per batch, never
materialised (the materialised version needed 20 GB and was cancelled)."""
import sys, time, os
sys.path.insert(0, "repo/src")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from structural_break.combiners import split_by_series, ts_auc

DEVICE = torch.device("mps")
t0 = time.time()
KIND = "s"; MEMBERS = [int(m) for m in os.environ.get("NET_MEMBERS", "0,1,2").split(",")]
OUTDIR = "nets_wstream"
X200 = None; W111 = np.load("WSTREAM5.npy", mmap_mode="r")
AX = None; AW = np.load("AUG3_WSTREAM5.npy", mmap_mode="r")
N_IN = 5
y = np.load("Y40.npy"); g = np.load("G40.npy")
assignment = split_by_series(g, folds=5, seed=0)
starts = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]])); bounds = np.append(starts, len(g))
train_rows = assignment != 2
# normalisation over the training rows, in chunks
acc1 = np.zeros(N_IN); acc2 = np.zeros(N_IN); cnt = 0
for a in range(0, len(g), 500000):
    b = min(a + 500000, len(g)); m = train_rows[a:b]
    blk = np.asarray(W111[a:b])[m].astype("float64"); acc1 += blk.sum(0); acc2 += (blk ** 2).sum(0); cnt += len(blk)
mu = acc1 / cnt; sd = np.sqrt(np.maximum(acc2 / cnt - mu ** 2, 0)) + 1e-6
np.save("mu_s.npy", mu.astype("float32")); np.save("sd_s.npy", sd.astype("float32")); mu32, sd32 = mu.astype("float32"), sd.astype("float32")
# rows are (source, a, b, labels): gathered per batch
orig, orig_fold = [], []
for a, b in zip(starts, bounds[1:]):
    orig.append(("o", int(a), int(b), y[a:b].astype("float32"))); orig_fold.append(int(assignment[a]))
AY = np.load("AUG3_Y.npy"); AG = np.load("AUG3_G.npy")
a_starts = np.flatnonzero(np.concatenate([[True], AG[1:] != AG[:-1]])); a_bounds = np.append(a_starts, len(AG))
fold_by_sid = {int(g[a]): int(assignment[a]) for a in starts}
aug = [("a", int(a), int(b), AY[a:b].astype("float32"), (int(AG[a]) - 100000) // 10) for a, b in zip(a_starts, a_bounds[1:]) if fold_by_sid.get((int(AG[a]) - 100000) // 10, 0) != 2]
del AY
def gather(row):
    src, a, b = row[0], row[1], row[2]
    seg = np.asarray(W111[a:b]) if src == "o" else np.asarray(AW[a:b])
    return ((seg - mu32) / sd32).astype("float32")
print(f"вход {N_IN}, оригинал {len(orig)}, аугментация {len(aug)} [{time.time()-t0:.0f}s]", flush=True)

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
    rows = [(gather(r), r[3]) for r in rows]
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
        rs = sorted(rows, key=lambda r: r[2] - r[1])
        for k in range(0, len(rs), 96):
            chunk = rs[k:k + 96]
            Xb, _, _ = batch_tensors(chunk, use_diffs)
            out = model(Xb).cpu().numpy()
            for row, r in zip(out, chunk):
                n = r[2] - r[1]; sc_.append(row[-n:]); lb_.append(r[3].astype(int)); st_.append(np.arange(n))
    return ts_auc(np.concatenate(sc_), np.concatenate(lb_), np.concatenate(st_))

os.makedirs(OUTDIR, exist_ok=True)
fold2_rows = [orig[i] for i in range(len(orig)) if orig_fold[i] == 2]
pool_orig = [orig[i] for i in range(len(orig)) if orig_fold[i] != 2]
pool_sid = [int(g[starts[i]]) for i in range(len(orig)) if orig_fold[i] != 2]
# После пробы d0 (холдаут 0.6543 -> фолд-2 0.5875) diff-члены из очереди
# убраны: они не переносятся, а стоят по 4.4 часа каждый.
JOBS = [(KIND, i, False) for i in MEMBERS]
for tag, member, use_diffs in JOBS:
    path = f"{OUTDIR}/member_{tag}{member}.pt"
    if os.path.exists(path):
        continue
    seed = 51000 + member  # те же сиды, что p0–p2: парное сравнение
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    idx = rng.permutation(len(pool_orig))
    hold_n = int(0.08 * len(pool_orig))
    train_set = list(pool_orig) + [r[:4] for r in aug]  # 082: без холдаута
    rng.shuffle(train_set)
    print(f"  член {tag}{member}: обучение {len(train_set)} рядов, холдаута нет", flush=True)
    rng.shuffle(train_set)
    model = ChanTCN(3 * N_IN if use_diffs else N_IN).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
    order = sorted(range(len(train_set)), key=lambda i: train_set[i][2] - train_set[i][1])
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
            print(f"    член {tag}{member} эпоха {epoch}: фолд-2 {f2:.4f}  [{time.time()-t0:.0f}s]", flush=True)
            model.eval(); sc = [None] * len(fold2_rows)
            with torch.no_grad():
                idx_sorted = sorted(range(len(fold2_rows)), key=lambda i: fold2_rows[i][2] - fold2_rows[i][1])
                for k in range(0, len(idx_sorted), 96):
                    chunk = idx_sorted[k:k + 96]; Xb, _, _ = batch_tensors([fold2_rows[i] for i in chunk], use_diffs); out = model(Xb).cpu().numpy()
                    for row, i in zip(out, chunk): sc[i] = row[-(fold2_rows[i][2] - fold2_rows[i][1]):]
            np.save(f"{OUTDIR}/fold2_logits_{tag}{member}.npy", np.concatenate(sc))
            # fold-2 logits of this member, in row order, for the blend caches
            model.eval(); sc = [None] * len(fold2_rows)
            with torch.no_grad():
                idx_sorted = sorted(range(len(fold2_rows)), key=lambda i: len(fold2_rows[i][0]))
                for k in range(0, len(idx_sorted), 96):          # chunks of similar length: few distinct shapes for MPS
                    chunk = idx_sorted[k:k + 96]
                    Xb, _, _ = batch_tensors([fold2_rows[i] for i in chunk], use_diffs); out = model(Xb).cpu().numpy()
                    for row, i in zip(out, chunk): sc[i] = row[-(fold2_rows[i][2] - fold2_rows[i][1]):]
            np.save(f"{OUTDIR}/fold2_logits_{tag}{member}.npy", np.concatenate(sc))
            torch.save({k: v.cpu().clone() for k, v in model.state_dict().items()},
                       f"{OUTDIR}/member_{tag}{member}_epoch{epoch}.pt")
            best = (f2, {k: v.cpu().clone() for k, v in model.state_dict().items()})
    torch.save(best[1], path)
    print(f"last-сеть {tag}{member}: фолд-2 на последней эпохе {best[0]:.4f}  [{time.time()-t0:.0f}s]", flush=True)
print("done", flush=True)
