"""108: сеть по СЫРОМУ сигналу — независимая модальность, не видит наших 200 каналов.

084 подмешивал сырой ряд к каналам и терял; здесь сеть учится представлению с нуля,
а её выход входит в ансамбль как отдельный член. Вход 2 канала: z по истории и asinh(z).
Контекст: хвост истории CTX=512 точек + вся онлайн-часть; лосс только на онлайн-части.
Поле зрения 2047 шагов (дилатации 1..512). Парные сиды к 082b (0.6057/0.6065/0.6051).
"""
import os, sys, time, numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "repo/src")
from structural_break.combiners import split_by_series, ts_auc
DEVICE = torch.device("mps"); CTX = 512; DILS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512); CH = 48
t0 = time.time()
D = "structural-break-real-time-test/data/"
X = pd.read_parquet(D + "X_train.parquet"); yi = pd.read_parquet(D + "y_train_index.parquet")
g = np.load("G40.npy"); starts = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]]))
assignment = split_by_series(g, folds=5, seed=0); fold_of = {int(g[a]): int(assignment[a]) for a in starts}
series = []
for sid, part in X.groupby(level="id"):
    v = part.value.to_numpy("float64"); p = part.period.to_numpy()
    hist, online = v[p == 1], v[p == 2]
    mu, sd = hist.mean(), hist.std() + 1e-12
    z = np.concatenate([(hist[-CTX:] - mu) / sd, (online - mu) / sd]).astype("float32")
    feat = np.stack([z, np.arcsinh(z)])                      # (2, ctx+L)
    tau = int(yi.loc[sid, "tau_index"]); lab = np.zeros(len(online), dtype="float32")
    if tau >= 0: lab[tau:] = 1.0
    series.append((feat, lab, len(online), fold_of[int(sid)]))
del X
train = [s for s in series if s[3] != 2]; fold2 = [s for s in series if s[3] == 2]
print(f"обучение {len(train)} рядов, фолд-2 {len(fold2)} [{time.time()-t0:.0f}s]", flush=True)

class Block(nn.Module):
    def __init__(self, ch, d):
        super().__init__(); self.conv = nn.Conv1d(ch, ch, 3, dilation=d); self.mix = nn.Conv1d(ch, ch, 1); self.drop = nn.Dropout(0.1); self.d = d
    def forward(self, h):
        r = h; h = F.pad(h, (2 * self.d, 0)); h = self.drop(F.gelu(self.conv(h))); return r + self.mix(h)
class RawTCN(nn.Module):
    def __init__(self):
        super().__init__(); self.inp = nn.Conv1d(2, CH, 1)
        self.blocks = nn.ModuleList([Block(CH, d) for d in DILS]); self.head = nn.Conv1d(CH, 1, 1)
    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks: h = b(h)
        return self.head(h).squeeze(1)

def batch(rows):
    L = max(f.shape[1] for f, _, _, _ in rows)
    Xb = torch.zeros(len(rows), 2, L); M = torch.zeros(len(rows), L, dtype=torch.bool); Yb = torch.zeros(len(rows), L)
    for i, (f, lab, n, _) in enumerate(rows):
        w = f.shape[1]; Xb[i, :, L - w:] = torch.from_numpy(f)
        M[i, L - n:] = True; Yb[i, L - n:] = torch.from_numpy(lab)   # лосс только на онлайн-части
    return Xb.to(DEVICE), M.to(DEVICE), Yb.to(DEVICE)

def rank_loss(logits, onmask, Y, rng):
    L = logits.shape[1]; total, count = logits.new_zeros(()), 0
    for t in rng.choice(L, size=min(48, L), replace=False):
        alive = onmask[:, t]
        if alive.sum() < 2: continue
        sc, lab = logits[alive, t], Y[alive, t]
        pos, neg = sc[lab > 0.5], sc[lab < 0.5]
        if not len(pos) or not len(neg): continue
        total = total + F.softplus(neg.unsqueeze(0) - pos.unsqueeze(1)).mean(); count += 1
    return total / max(count, 1)

def fold2_auc(model):
    model.eval(); sc, lb, st = [], [], []
    rows = sorted(fold2, key=lambda r: r[0].shape[1])
    with torch.no_grad():
        for k in range(0, len(rows), 32):
            chunk = rows[k:k + 32]; Xb, M, Yb = batch(chunk); o = model(Xb).cpu().numpy()
            for i, (f, lab, n, _) in enumerate(chunk):
                sc.append(o[i, -n:]); lb.append(lab.astype(int)); st.append(np.arange(n))
    model.train()
    return ts_auc(np.concatenate(sc), np.concatenate(lb), np.concatenate(st))

os.makedirs("nets_raw", exist_ok=True)
for member in range(3):
    path = f"nets_raw/member_w{member}.pt"
    if os.path.exists(path): continue
    seed = 51000 + member; rng = np.random.default_rng(seed); torch.manual_seed(seed)
    model = RawTCN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
    order = sorted(range(len(train)), key=lambda i: train[i][0].shape[1])
    batches = [[train[i] for i in order[k:k + 16]] for k in range(0, len(order), 16)]
    print(f"  член w{member}: {len(train)} рядов, {len(batches)} батчей, параметров "
          f"{sum(p.numel() for p in model.parameters())/1000:.0f}k", flush=True)
    for epoch in range(10):
        for bi in rng.permutation(len(batches)):
            Xb, M, Yb = batch(batches[bi])
            opt.zero_grad(); logits = model(Xb)
            bce = (F.binary_cross_entropy_with_logits(logits, Yb, reduction="none")[M]).mean()
            (rank_loss(logits, M, Yb, rng) + 0.3 * bce).backward(); opt.step()
        sched.step()
        if epoch in (4, 9):
            print(f"    член w{member} эпоха {epoch}: фолд-2 {fold2_auc(model):.4f}  [{time.time()-t0:.0f}s]", flush=True)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, path)
    print(f"raw-сеть w{member}: фолд-2 на последней эпохе {fold2_auc(model):.4f}  [{time.time()-t0:.0f}s]", flush=True)
print("done", flush=True)
