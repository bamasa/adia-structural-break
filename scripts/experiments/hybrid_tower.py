"""048: hybrid — two-tower net (raw series + channel trajectories)."""
import sys, time, os
sys.path.insert(0, "repo/src")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from structural_break.features import Normalisation
from structural_break.stream import iter_series
from structural_break.combiners import ts_auc

DEVICE = torch.device("mps")
HIST_CAP = 256
t0 = time.time()

X = np.hstack([np.load("X40.npy"), np.load("C_cnn.npy")[:, None],
               np.load("N9.npy").astype("float32"), np.load("X50a.npy"),
               np.load("E4.npy"), np.load("B40.npy"), np.load("B2.npy")]).astype("float32")
y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
starts = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]]))
bounds = np.append(starts, len(g))
mu = np.load("net_bag_mac/mu.npy"); sd = np.load("net_bag_mac/sd.npy")

# Raw sequences: on-disk cache so the normalisation is computed once.
if os.path.exists("raw_seq_cache.npz"):
    z_cache = np.load("raw_seq_cache.npz", allow_pickle=True)
    raw_hist = list(z_cache["hist"]); raw_online = list(z_cache["online"])
    print(f"raw cache loaded [{time.time()-t0:.0f}s]", flush=True)
else:
    x = pd.read_parquet("structural-break-real-time-test/data/X_train.parquet")
    yp = pd.read_parquet("structural-break-real-time-test/data/y_train.parquet")
    raw_hist, raw_online = [], []
    for sid, hist, online, labels in iter_series(x, yp):
        norm = Normalisation.fit(hist)
        n = len(hist)
        raw_hist.append(np.asarray([norm.clip(norm.standardise(float(v), i - n))
                                    for i, v in enumerate(hist)], dtype="float32")[-HIST_CAP:])
        raw_online.append(np.asarray([norm.clip(norm.standardise(float(v), i))
                                      for i, v in enumerate(online)], dtype="float32"))
    np.savez("raw_seq_cache.npz",
             hist=np.array(raw_hist, dtype=object), online=np.array(raw_online, dtype=object))
    print(f"raw cache built [{time.time()-t0:.0f}s]", flush=True)

series = []
for k, (a, b) in enumerate(zip(starts, bounds[1:])):
    series.append((((X[a:b] - mu) / sd), raw_hist[k], raw_online[k], y[a:b].astype("float32")))
print(f"{len(series)} series [{time.time()-t0:.0f}s]", flush=True)

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

class Hybrid(nn.Module):
    """Channel tower + raw tower (sees the history), fused per step."""
    def __init__(self, ch_chan=64, ch_raw=32):
        super().__init__()
        self.chan_inp = nn.Conv1d(186, ch_chan, 1)
        self.chan_blocks = nn.ModuleList([Block(ch_chan, d) for d in (1, 2, 4, 8, 16, 32)])
        self.raw_inp = nn.Conv1d(3, ch_raw, 1)
        self.raw_blocks = nn.ModuleList([Block(ch_raw, d) for d in (1, 2, 4, 8, 16, 32, 64)])
        self.head = nn.Conv1d(ch_chan + ch_raw, 1, 1)
    def forward(self, xc, xr, off):
        h = self.chan_inp(xc)
        for b in self.chan_blocks:
            h = b(h)
        r = self.raw_inp(xr)
        for b in self.raw_blocks:
            r = b(r)
        r = r[:, :, off:]
        return self.head(torch.cat([h, r], dim=1)).squeeze(1)

def batch_tensors(rows):
    Lc = max(len(c) for c, _, _, _ in rows)
    Lr = max(len(h) + len(o) for _, h, o, _ in rows)
    off = Lr - Lc
    Xc = torch.zeros(len(rows), 186, Lc)
    Xr = torch.zeros(len(rows), 3, Lr)
    M = torch.zeros(len(rows), Lc, dtype=torch.bool)
    Y = torch.zeros(len(rows), Lc)
    for i, (c, h, o, lab) in enumerate(rows):
        n = len(c)
        Xc[i, :, Lc - n:] = torch.from_numpy(c.T)
        seq = np.concatenate([h, o])
        Xr[i, 0, Lr - len(seq):] = torch.from_numpy(seq)
        Xr[i, 1, Lr - len(seq):] = torch.from_numpy(np.abs(seq))
        Xr[i, 2, Lr - len(o):] = 1.0
        M[i, Lc - n:] = True
        Y[i, Lc - n:] = torch.from_numpy(lab)
    return Xc.to(DEVICE), Xr.to(DEVICE), off, M.to(DEVICE), Y.to(DEVICE)

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

def holdout_auc(model, rows):
    model.eval()
    scores, labels, steps = [], [], []
    with torch.no_grad():
        rows_sorted = sorted(rows, key=lambda r: len(r[0]))
        for k in range(0, len(rows_sorted), 48):
            chunk = rows_sorted[k:k + 48]
            Xc, Xr, off, _, _ = batch_tensors(chunk)
            out = model(Xc, Xr, off).cpu().numpy()
            for row, (c, h, o, lab) in zip(out, chunk):
                scores.append(row[-len(c):]); labels.append(lab.astype(int)); steps.append(np.arange(len(c)))
    return ts_auc(np.concatenate(scores), np.concatenate(labels), np.concatenate(steps))

os.makedirs("hybrid", exist_ok=True)
for member in range(4):
    path = f"hybrid/member_{member}.pt"
    if os.path.exists(path):
        continue
    rng = np.random.default_rng(4000 + member)
    torch.manual_seed(4000 + member)
    idx = rng.permutation(len(series))
    hold_n = int(0.08 * len(series))
    hold = [series[i] for i in idx[:hold_n]]
    train_set = [series[i] for i in idx[hold_n:int(0.96 * len(series))]]
    model = Hybrid().to(DEVICE)
    print(f"member {member}: parameters {sum(p.numel() for p in model.parameters())}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=14)
    order = sorted(range(len(train_set)), key=lambda i: len(train_set[i][0]))
    batches = [[train_set[i] for i in order[k:k + 20]] for k in range(0, len(order), 20)]
    best = (0.0, None)
    for epoch in range(14):
        model.train()
        for bi in rng.permutation(len(batches)):
            Xc, Xr, off, M, Yb = batch_tensors(batches[bi])
            opt.zero_grad()
            logits = model(Xc, Xr, off)
            bce = (F.binary_cross_entropy_with_logits(logits, Yb, reduction="none")[M]).mean()
            loss = rank_loss(logits, M, Yb, rng) + 0.3 * bce
            loss.backward()
            opt.step()
        sched.step()
        a = holdout_auc(model, hold)
        if a > best[0]:
            best = (a, {k: v.cpu().clone() for k, v in model.state_dict().items()})
        print(f"  member {member} epoch {epoch}: holdout {a:.4f} (best {best[0]:.4f}) [{time.time()-t0:.0f}s]", flush=True)
    torch.save(best[1], path)
    print(f"member {member} done: {best[0]:.4f}", flush=True)
print("done", flush=True)
