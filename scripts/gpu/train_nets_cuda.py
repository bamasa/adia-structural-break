"""The GPU job: train the strong half of the ensemble — channel-trajectory nets.

Everything the laptop era established is baked in: the ranking loss the metric
actually measures, per-member private holdouts (the CV folds are burnt by
selection and must not be used), short schedules with best-epoch checkpoints
(past ~12 epochs the nets overfit through any augmentation), and diversity
through data rather than seeds. Two input variants are trained side by side —
plain 200 channels, and 600 with explicit first- and tenth-order differences,
the variant whose members reached the best holdouts on the laptop.

Usage on the Linux box (CUDA):

    python scripts/gpu/train_nets_cuda.py --data-dir /path/to/matrices \
        --out-dir nets_cuda --members 24 --epochs 14

Expects in --data-dir the ten matrices listed in scripts/gpu/README.md.
Members land as out-dir/{plain,diff}_member_N.pt with their holdout scores
printed per line; copy the whole out-dir back to the laptop, where the
submission assembler turns them into a shipped ensemble.
"""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def ts_auc(scores, labels, steps):
    total, total_w = 0.0, 0.0
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


def with_diffs(seg):
    d1 = np.zeros_like(seg)
    d1[1:] = seg[1:] - seg[:-1]
    d10 = np.zeros_like(seg)
    if len(seg) > 10:
        d10[10:] = seg[10:] - seg[:-10]
    return np.hstack([seg, d1 * 3.0, d10 * 1.5]).astype("float32")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("nets_cuda"))
    ap.add_argument("--members", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device)
    args.out_dir.mkdir(exist_ok=True)
    t0 = time.time()

    d = args.data_dir
    X = np.hstack([np.load(d / "X40.npy"), np.load(d / "C_cnn.npy")[:, None],
                   np.load(d / "N9.npy").astype("float32"), np.load(d / "X50a.npy"),
                   np.load(d / "E4.npy"), np.load(d / "B40.npy"),
                   np.load(d / "B2.npy"), np.load(d / "SPEC14.npy")]).astype("float32")
    y = np.load(d / "Y40.npy")
    g = np.load(d / "G40.npy")
    starts = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]]))
    bounds = np.append(starts, len(g))
    mu, sd = X.mean(0), X.std(0) + 1e-6
    np.save(args.out_dir / "mu200.npy", mu)
    np.save(args.out_dir / "sd200.npy", sd)
    series = [(((X[a:b] - mu) / sd), y[a:b].astype("float32"))
              for a, b in zip(starts, bounds[1:])]
    print(f"{len(series)} series ready [{time.time()-t0:.0f}s]", flush=True)
    del X

    def batch_tensors(rows, n_in):
        L = max(len(f) for f, _ in rows)
        Xb = torch.zeros(len(rows), n_in, L)
        M = torch.zeros(len(rows), L, dtype=torch.bool)
        Yb = torch.zeros(len(rows), L)
        for i, (f, lab) in enumerate(rows):
            seg = f if n_in == 200 else with_diffs(f)
            n = len(seg)
            Xb[i, :, L - n:] = torch.from_numpy(seg.T)
            M[i, L - n:] = True
            Yb[i, L - n:] = torch.from_numpy(lab)
        return Xb.to(device), M.to(device), Yb.to(device)

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

    def holdout_auc(model, rows, n_in):
        model.eval()
        scores, labels, steps = [], [], []
        with torch.no_grad():
            rows_sorted = sorted(rows, key=lambda r: len(r[0]))
            for k in range(0, len(rows_sorted), 96):
                chunk = rows_sorted[k:k + 96]
                Xb, _, _ = batch_tensors(chunk, n_in)
                out = model(Xb).cpu().numpy()
                for row, (f, lab) in zip(out, chunk):
                    scores.append(row[-len(f):])
                    labels.append(lab.astype(int))
                    steps.append(np.arange(len(f)))
        return ts_auc(np.concatenate(scores), np.concatenate(labels), np.concatenate(steps))

    for member in range(args.members):
        variant, n_in = ("plain", 200) if member % 2 == 0 else ("diff", 600)
        path = args.out_dir / f"{variant}_member_{member}.pt"
        if path.exists():
            continue
        rng = np.random.default_rng(20000 + member)
        torch.manual_seed(20000 + member)
        idx = rng.permutation(len(series))
        hold_n = int(0.08 * len(series))
        hold = [series[i] for i in idx[:hold_n]]
        train_set = [series[i] for i in idx[hold_n:]]
        model = ChanTCN(n_in).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        order = sorted(range(len(train_set)), key=lambda i: len(train_set[i][0]))
        batches = [[train_set[i] for i in order[k:k + args.batch]]
                   for k in range(0, len(order), args.batch)]
        best = (0.0, None)
        for epoch in range(args.epochs):
            model.train()
            for bi in rng.permutation(len(batches)):
                Xb, M, Yb = batch_tensors(batches[bi], n_in)
                opt.zero_grad()
                logits = model(Xb)
                bce = (F.binary_cross_entropy_with_logits(logits, Yb, reduction="none")[M]).mean()
                loss = rank_loss(logits, M, Yb, rng) + 0.3 * bce
                loss.backward()
                opt.step()
            sched.step()
            a = holdout_auc(model, hold, n_in)
            if a > best[0]:
                best = (a, {k: v.cpu().clone() for k, v in model.state_dict().items()})
        torch.save(best[1], path)
        print(f"{variant} member {member}: holdout {best[0]:.4f}  [{time.time()-t0:.0f}s]",
              flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
