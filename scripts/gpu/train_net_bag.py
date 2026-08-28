"""The 3080 programme, part 1: a bag of short channel-trajectory nets.

Trains N small causal TCNs over the 186-channel trajectories, each on its own
subsample of training series (a withheld fold, a random 60%, or a cleaned
subset), with the per-step ranking loss. Every lesson of the laptop era is
baked in: short schedules (overfitting beats augmentation past ~10 epochs),
diversity through data (seeds converge, subsamples do not), best-epoch
checkpoints, and two-fold validation (folds 0 and 1) because fold-0 alone
stopped transferring at #19.

Usage on the Linux box:
    python scripts/gpu/train_net_bag.py --data-dir /path/to/matrices --members 16

Expects in --data-dir: X40.npy, C_cnn.npy, N9.npy, X50a.npy, E4.npy,
B40.npy, B2.npy, Y40.npy, G40.npy, S40.npy  (~2 GB total, rsync from the
laptop's workspace root).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def split_by_series(groups: np.ndarray, folds: int = 5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sids = np.unique(groups)
    assignment = rng.integers(0, folds, size=len(sids))
    lookup = dict(zip(sids.tolist(), assignment.tolist()))
    return np.asarray([lookup[g] for g in groups.tolist()])


def ts_auc(scores, labels, steps):
    total_w, total = 0.0, 0.0
    for t in np.unique(steps):
        m = steps == t
        lab, sc = labels[m], scores[m]
        pos, neg = sc[lab == 1], sc[lab == 0]
        if not len(pos) or not len(neg):
            continue
        ranks = np.concatenate([pos, neg]).argsort().argsort() + 1
        auc = (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
        w = len(pos) * len(neg)
        total += auc * w
        total_w += w
    return total / total_w


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
    def __init__(self, ch=64, dils=(1, 2, 4, 8, 16, 32)):
        super().__init__()
        self.inp = nn.Conv1d(186, ch, 1)
        self.blocks = nn.ModuleList([Block(ch, d) for d in dils])
        self.head = nn.Conv1d(ch, 1, 1)

    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = b(h)
        return self.head(h).squeeze(1)


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("net_bag"))
    ap.add_argument("--members", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device)
    args.out_dir.mkdir(exist_ok=True)

    d = args.data_dir
    X = np.hstack([np.load(d / "X40.npy"), np.load(d / "C_cnn.npy")[:, None],
                   np.load(d / "N9.npy").astype("float32"), np.load(d / "X50a.npy"),
                   np.load(d / "E4.npy"), np.load(d / "B40.npy"),
                   np.load(d / "B2.npy")]).astype("float32")
    y = np.load(d / "Y40.npy"); g = np.load(d / "G40.npy"); s = np.load(d / "S40.npy")
    assignment = split_by_series(g)
    starts = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]]))
    bounds = np.append(starts, len(g))
    mu, sd = X.mean(0), X.std(0) + 1e-6
    np.save(args.out_dir / "mu.npy", mu)
    np.save(args.out_dir / "sd.npy", sd)
    series = [(((X[a:b] - mu) / sd), y[a:b].astype("float32"), int(assignment[a]))
              for a, b in zip(starts, bounds[1:])]
    print(f"{len(series)} series ready", flush=True)

    val = {f: [r for r in series if r[2] == f] for f in (0, 1)}

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
        return Xb.to(device), M.to(device), Yb.to(device)

    def validate(model, fold):
        model.eval()
        scores, labels, steps = [], [], []
        with torch.no_grad():
            rows_all = sorted(val[fold], key=lambda r: len(r[0]))
            for k in range(0, len(rows_all), 96):
                rows = rows_all[k:k + 96]
                Xb, _, _ = batch_tensors(rows)
                out = model(Xb).cpu().numpy()
                for row, (f, lab, _) in zip(out, rows):
                    scores.append(row[-len(f):])
                    labels.append(lab.astype(int))
                    steps.append(np.arange(len(f)))
        return ts_auc(np.concatenate(scores), np.concatenate(labels), np.concatenate(steps))

    master = np.random.default_rng(7)
    for member in range(args.members):
        rng = np.random.default_rng(100 + member)
        torch.manual_seed(100 + member)
        # Разнообразие через данные: случайные 60% обучающих рядов (фолды 0/1
        # всегда вне обучения — они валидация).
        pool = [r for r in series if r[2] not in (0, 1)]
        take = rng.choice(len(pool), size=int(0.6 * len(pool)), replace=False)
        train_set = [pool[i] for i in take]
        model = ChanTCN().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        order = sorted(range(len(train_set)), key=lambda i: len(train_set[i][0]))
        batches = [[train_set[i] for i in order[k:k + args.batch]]
                   for k in range(0, len(order), args.batch)]
        best = (0.0, None)
        t0 = time.time()  # noqa: F841 -- per-member timer
        for epoch in range(args.epochs):
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
            a0 = validate(model, 0)
            if a0 > best[0]:
                best = (a0, {k: v.cpu().clone() for k, v in model.state_dict().items()})
        a1 = None
        if best[1] is not None:
            model.load_state_dict(best[1])
            a1 = validate(model, 1)
            torch.save(best[1], args.out_dir / f"member_{member}.pt")
        print(f"member {member}: fold0 {best[0]:.4f}, fold1 {a1}  [{time.time()-t0:.0f}s]",
              flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
