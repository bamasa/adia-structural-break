"""049: самообучение чан-сети (прогноз следующего вектора каналов) + дообучение."""
import sys, time, os
sys.path.insert(0, "repo/src")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from structural_break.combiners import ts_auc

DEVICE = torch.device("mps")
t0 = time.time()
X = np.hstack([np.load("X40.npy"), np.load("C_cnn.npy")[:, None],
               np.load("N9.npy").astype("float32"), np.load("X50a.npy"),
               np.load("E4.npy"), np.load("B40.npy"), np.load("B2.npy")]).astype("float32")
y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
starts = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]]))
bounds = np.append(starts, len(g))
mu = np.load("net_bag_mac/mu.npy"); sd = np.load("net_bag_mac/sd.npy")
series = [(((X[a:b] - mu) / sd), y[a:b].astype("float32")) for a, b in zip(starts, bounds[1:])]
print(f"{len(series)} рядов [{time.time()-t0:.0f}s]", flush=True)

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

class Backbone(nn.Module):
    def __init__(self, ch=64):
        super().__init__()
        self.inp = nn.Conv1d(186, ch, 1)
        self.blocks = nn.ModuleList([Block(ch, d) for d in (1, 2, 4, 8, 16, 32)])
    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = b(h)
        return h

def batch_tensors(rows):
    L = max(len(f) for f, _ in rows)
    Xb = torch.zeros(len(rows), 186, L)
    M = torch.zeros(len(rows), L, dtype=torch.bool)
    Yb = torch.zeros(len(rows), L)
    for i, (f, lab) in enumerate(rows):
        n = len(f)
        Xb[i, :, L - n:] = torch.from_numpy(f.T)
        M[i, L - n:] = True
        Yb[i, L - n:] = torch.from_numpy(lab)
    return Xb.to(DEVICE), M.to(DEVICE), Yb.to(DEVICE)

# --- Этап 1: самообучение — прогноз следующего вектора каналов ---
if not os.path.exists("pretrained_backbone.pt"):
    backbone = Backbone().to(DEVICE)
    pred_head = nn.Conv1d(64, 186, 1).to(DEVICE)
    opt = torch.optim.Adam(list(backbone.parameters()) + list(pred_head.parameters()),
                           lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=12)
    rng = np.random.default_rng(0)
    order = sorted(range(len(series)), key=lambda i: len(series[i][0]))
    batches = [[series[i] for i in order[k:k + 24]] for k in range(0, len(order), 24)]
    for epoch in range(12):
        backbone.train()
        total, nb = 0.0, 0
        for bi in rng.permutation(len(batches)):
            Xb, M, _ = batch_tensors(batches[bi])
            opt.zero_grad()
            h = backbone(Xb)
            pred = pred_head(h)
            # предсказываем канал-вектор следующего шага
            tgt = Xb[:, :, 1:]
            prd = pred[:, :, :-1]
            msk = M[:, 1:].unsqueeze(1)
            loss = F.huber_loss(prd * msk, tgt * msk)
            loss.backward()
            opt.step()
            total += float(loss); nb += 1
        sched.step()
        print(f"предобучение, эпоха {epoch}: huber {total/nb:.4f}  [{time.time()-t0:.0f}s]", flush=True)
    torch.save(backbone.state_dict(), "pretrained_backbone.pt")
    print("хребет сохранён", flush=True)

# --- Этап 2: дообучение ранговым лоссом, рецепт тяжёлых членов ---
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

class Scorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.head = nn.Conv1d(64, 1, 1)
    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(1)

def holdout_auc(model, rows):
    model.eval()
    scores, labels, steps = [], [], []
    with torch.no_grad():
        rows_sorted = sorted(rows, key=lambda r: len(r[0]))
        for k in range(0, len(rows_sorted), 96):
            chunk = rows_sorted[k:k + 96]
            Xb, _, _ = batch_tensors(chunk)
            out = model(Xb).cpu().numpy()
            for row, (f, lab) in zip(out, chunk):
                scores.append(row[-len(f):]); labels.append(lab.astype(int)); steps.append(np.arange(len(f)))
    return ts_auc(np.concatenate(scores), np.concatenate(labels), np.concatenate(steps))

os.makedirs("pretrained", exist_ok=True)
for member in range(3):
    path = f"pretrained/member_{member}.pt"
    if os.path.exists(path):
        continue
    rng = np.random.default_rng(6000 + member)
    torch.manual_seed(6000 + member)
    idx = rng.permutation(len(series))
    hold_n = int(0.08 * len(series))
    hold = [series[i] for i in idx[:hold_n]]
    train_set = [series[i] for i in idx[hold_n:int(0.96 * len(series))]]
    model = Scorer().to(DEVICE)
    model.backbone.load_state_dict(torch.load("pretrained_backbone.pt", map_location=DEVICE))
    opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=12)
    order = sorted(range(len(train_set)), key=lambda i: len(train_set[i][0]))
    batches = [[train_set[i] for i in order[k:k + 24]] for k in range(0, len(order), 24)]
    best = (0.0, None)
    for epoch in range(12):
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
        a = holdout_auc(model, hold)
        if a > best[0]:
            best = (a, {k: v.cpu().clone() for k, v in model.state_dict().items()})
        print(f"  дообучение {member}, эпоха {epoch}: холдаут {a:.4f} (лучший {best[0]:.4f}) [{time.time()-t0:.0f}s]", flush=True)
    torch.save(best[1], path)
    print(f"дообученный член {member}: {best[0]:.4f}", flush=True)
print("done", flush=True)
