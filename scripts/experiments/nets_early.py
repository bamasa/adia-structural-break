"""087: ранние специалисты — ранговый лосс только на первых 200 шагах ряда.

На шагах <200 деревья и сети почти слепы (0.54–0.58), а это треть веса
метрики. Член, обученный только на ранних шагах, может выучить раннюю
геометрию слома. Оценка: фолд-2 целиком и отдельно на шагах <200.
Kill: ранний AUC не выше базовых членов на тех же шагах.

082 (15 эпох) дал 0.5936 на сиде 51000 против 0.6036 у 10 эпох: длинное
расписание уходит в запоминание. Здесь десять эпох, шесть сидов #28,
фолд-2 на последней эпохе; сравнение ансамбля с #28 (0.6163).

081d: фолд-2 растёт монотонно до последних эпох, холдаут — шум в противофазе.
Здесь шесть членов с сидами #28, без холдаута (+8% данных), косинус на 15
эпох; фолд-2 оценивается на эпохах 9 и 14, обе сохраняются. Отвечает: (а)
последняя эпоха@10 против дырявого выбора #28, (б) 15 эпох против 10.

081 с чистым холдаутом дал 0.5623 на фолде-2 против 0.6048 у того же
сида с дырявым. Здесь один член, сид 51000, и по эпохам: чистый холдаут,
дырявый холдаут (те же ряды, но их псевдоряды в обучении — как в #28) и
фолд-2. Ответ на вопрос, какой критерий выбора эпохи переносится.

Во всех пулах лучшая эпоха выбиралась по холдауту, чьи псевдоряды лежали
в обучении (080b: холдаут 0.7255 при фолде-2 0.5678). Выбор эпохи был
смещён к запоминанию. Здесь холдаут честный. Рецепт #28 в остальном.
Kill: соло на фолде-2 не выше 0.605.
Исходный 076: сети на ТРОЙНОЙ аугментации — втрое больше псевдорядов.

Сети data-bound: одинарная аугментация дала рекорды обоим архитектурам.
Тройную (AUG3, три случайных среза на ряд) видели только ранкеры.
AUG3_X 8.3 ГБ — читаем через mmap по-рядно. Половина членов plain (200
входов), половина diff (600). Kill: холдауты не выше пулов на одинарной."""
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

# Оригинальные ряды (фолд-2 держим в стороне — он линейка).
orig, orig_fold = [], []
for a, b in zip(starts, bounds[1:]):
    orig.append((((X[a:b] - mu) / sd), y[a:b].astype("float32")))
    orig_fold.append(int(assignment[a]))
del X

# Тройная аугментация: 8.3 ГБ, читаем mmap по-рядно.
AX = np.load("AUG3_X.npy", mmap_mode="r")
AY = np.load("AUG3_Y.npy"); AG = np.load("AUG3_G.npy")
a_starts = np.flatnonzero(np.concatenate([[True], AG[1:] != AG[:-1]]))
a_bounds = np.append(a_starts, len(AG))
fold_by_sid = {int(g[a]): int(assignment[a]) for a in starts}
aug = []
for a, b in zip(a_starts, a_bounds[1:]):
    sid = (int(AG[a]) - 100000) // 10
    # псевдоряд наследует фолд родителя; фолд-2 исключаем
    if fold_by_sid.get(sid, 0) != 2:
        aug.append((((np.asarray(AX[a:b]) - mu) / sd).astype("float32"),
                    AY[a:b].astype("float32"), sid))
del AX, AY
print(f"оригинал {len(orig)}, аугментация {len(aug)} [{time.time()-t0:.0f}s]", flush=True)

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

EARLY = 200
def rank_loss(logits, onmask, Y, rng):
    L = logits.shape[1]
    total, count = logits.new_zeros(()), 0
    # 087: шаг относительно начала своего ряда; берём только первые EARLY шагов
    first = onmask.float().argmax(dim=1)                      # позиция первого живого шага
    for t in rng.choice(L, size=min(48, L), replace=False):
        alive = onmask[:, t] & ((t - first) < EARLY)
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

os.makedirs("nets_early", exist_ok=True)
fold2_rows = [orig[i] for i in range(len(orig)) if orig_fold[i] == 2]
pool_orig = [orig[i] for i in range(len(orig)) if orig_fold[i] != 2]
pool_sid = [int(g[starts[i]]) for i in range(len(orig)) if orig_fold[i] != 2]
# После пробы d0 (холдаут 0.6543 -> фолд-2 0.5875) diff-члены из очереди
# убраны: они не переносятся, а стоят по 4.4 часа каждый.
JOBS = [("e", i, False) for i in range(2)]
for tag, member, use_diffs in JOBS:
    path = f"nets_early/member_{tag}{member}.pt"
    if os.path.exists(path):
        continue
    seed = 51000 + member  # те же сиды, что p0–p2: парное сравнение
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    idx = rng.permutation(len(pool_orig))
    hold_n = int(0.08 * len(pool_orig))
    train_set = list(pool_orig) + [(f, lab) for f, lab, sid in aug]  # 082: без холдаута
    rng.shuffle(train_set)
    print(f"  член {tag}{member}: обучение {len(train_set)} рядов, холдаута нет", flush=True)
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
        if epoch == 9:
            f2 = holdout_auc(model, fold2_rows, use_diffs)
            f2e = holdout_auc(model, [(f[:EARLY], lab[:EARLY]) for f, lab in fold2_rows], use_diffs)
            print(f"    член {tag}{member}: фолд-2 ранние(<200) {f2e:.4f}", flush=True)
            print(f"    член {tag}{member} эпоха {epoch}: фолд-2 {f2:.4f}  [{time.time()-t0:.0f}s]", flush=True)
            torch.save({k: v.cpu().clone() for k, v in model.state_dict().items()},
                       f"nets_early/member_{tag}{member}_epoch{epoch}.pt")
            best = (f2, {k: v.cpu().clone() for k, v in model.state_dict().items()})
    torch.save(best[1], path)
    print(f"early-сеть {tag}{member}: фолд-2 на последней эпохе {best[0]:.4f}  [{time.time()-t0:.0f}s]", flush=True)
print("done", flush=True)
