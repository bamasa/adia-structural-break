"""Совместный свип весов: доли ранкеров/классификатора и вес сетевой половины."""
import sys, re, os, itertools
sys.path.insert(0, "repo/src")
import numpy as np
from structural_break.combiners import split_by_series, ts_auc

g = np.load("G40.npy"); y = np.load("Y40.npy"); s = np.load("S40.npy")
assignment = split_by_series(g, folds=5, seed=0)
m2 = assignment == 2
yf, sf = y[m2], s[m2]

r_aug = 1/(1+np.exp(-np.load("fold2_rank_aug.npy").astype("float64")))
r_aug3 = 1/(1+np.exp(-np.load("fold2_rank_aug3.npy").astype("float64")))
c_plain = np.load("fold2_clf_эталон.npy").astype("float64")

# Сетевая половина: топ-10 равными весами (лучшая конфигурация из 073).
hold, sigs = {}, {}
starts = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]]))
for src, pat, folder in (("heavy200.log", r"heavy (\d+): приватный холдаут ([0-9.]+)", "heavy200"),
                         ("nets_aug.log", r"aug-сеть (\d+): холдаут ([0-9.]+)", "nets_aug")):
    for line in open(src, errors="ignore"):
        m = re.match(pat, line)
        if m and os.path.exists(f"{folder}/member_{m.group(1)}.pt"):
            hold[f"{folder}/member_{m.group(1)}.pt"] = float(m.group(2))
top10 = sorted(hold, key=hold.get, reverse=True)[:10]
print("топ-10:", [f"{hold[p]:.4f}" for p in top10])

# Пер-членные счёта фолда-2 уже прогнаны в 073? Нет — там считалось на лету.
# Пересчитывать дорого; вместо этого использую кэш связки из fold2_newnets:
# восстановить нельзя, поэтому прогоняю сети заново одним батчем.
import torch, torch.nn as nn, torch.nn.functional as F
DEVICE = torch.device("mps")
X = np.hstack([np.load("X40.npy"), np.load("C_cnn.npy")[:, None],
               np.load("N9.npy").astype("float32"), np.load("X50a.npy"),
               np.load("E4.npy"), np.load("B40.npy"), np.load("B2.npy"),
               np.load("SPEC14.npy")]).astype("float32")
mu = np.load("mu200.npy"); sd = np.load("sd200.npy")
bounds = np.append(starts, len(g))

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
    def __init__(self, ch=64):
        super().__init__()
        self.inp = nn.Conv1d(200, ch, 1)
        self.blocks = nn.ModuleList([Block(ch, d) for d in (1, 2, 4, 8, 16, 32)])
        self.head = nn.Conv1d(ch, 1, 1)
    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = b(h)
        return self.head(h).squeeze(1)

rows2 = sorted([(a, b) for a, b in zip(starts, bounds[1:]) if int(assignment[a]) == 2],
               key=lambda ab: ab[1] - ab[0])
def fold_scores(model):
    out = {}
    with torch.no_grad():
        for k in range(0, len(rows2), 96):
            chunk = rows2[k:k + 96]
            L = max(b - a for a, b in chunk)
            Xb = torch.zeros(len(chunk), 200, L)
            for i, (a, b) in enumerate(chunk):
                Xb[i, :, L - (b - a):] = torch.from_numpy(((X[a:b] - mu) / sd).T)
            o = model(Xb.to(DEVICE)).cpu().numpy()
            for i, (a, b) in enumerate(chunk):
                out[a] = o[i, L - (b - a):]
    return np.concatenate([out[a] for a, b in sorted(rows2)])

acc = None
for p in top10:
    model = ChanTCN().to(DEVICE)
    model.load_state_dict(torch.load(p, map_location=DEVICE))
    model.eval()
    sig = 1.0/(1.0+np.exp(-fold_scores(model).astype("float64")))
    acc = sig if acc is None else acc + sig
net = acc / len(top10)
np.save("fold2_net_top10.npy", net)  # кэш на будущее
print("сети прогнаны, кэш сохранён", flush=True)

best = (0, None)
# Доли древесной половины по сетке 0.05, вес сетей 0.40–0.60.
for a in np.arange(0.0, 0.75, 0.05):
    for b in np.arange(0.0, 0.75 - a + 1e-9, 0.05):
        c = 1.0 - a - b
        if c < -1e-9 or c > 0.6:
            continue
        pair = a*r_aug + b*r_aug3 + c*c_plain
        for w in (0.40, 0.45, 0.50, 0.55, 0.60):
            v = ts_auc((1-w)*pair + w*net, yf, sf)
            if v > best[0]:
                best = (v, (round(a,2), round(b,2), round(c,2), w))
                print(f"новый максимум {v:.4f} при r_aug={a:.2f} r_aug3={b:.2f} clf={c:.2f} net={w}", flush=True)
print(f"\nИТОГ: {best[0]:.4f} {best[1]}  (текущий рекорд 0.6116 при 0.35/0.35/0.30, net 0.5)", flush=True)
