"""The combination sweep: channel sets, combiners, asymmetric post-processing."""
import sys, time
sys.path.insert(0, "repo/src")
import numpy as np
from structural_break.combiners import Boosted, Weighted, split_by_series, step_weights, ts_auc

t0 = time.time()
X100 = np.hstack([np.load("X40.npy"), np.load("C_cnn.npy")[:, None],
                  np.load("N9.npy").astype("float32"), np.load("X50a.npy")])
E4 = np.load("E4.npy"); E8 = np.load("E8.npy"); D50 = np.load("X50d.npy")
y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
assignment = split_by_series(g, folds=5, seed=0)

def cv_boosted(mat, tag, save_oof=False):
    names = [f"c{i}" for i in range(mat.shape[1])]
    scores, oof = [], np.zeros(len(y), dtype="float32")
    for fold in range(5):
        tr, va = assignment != fold, assignment == fold
        m = Boosted.fit(mat[tr], y[tr], names, step_weights(s[tr]))
        pred = m.predict(mat[va])
        if save_oof:
            oof[va] = pred
        scores.append(ts_auc(pred, y[va], s[va]))
    print(f"{tag}: {np.mean(scores):.4f}  " + " ".join(f"{v:.4f}" for v in scores)
          + f"   [{time.time()-t0:.0f}s]", flush=True)
    return np.mean(scores), oof

# 1) соло-качество новых блоков
print("соло D-вид, максимум по 50:", f"{ts_auc(D50.max(axis=1), y, s):.4f}", flush=True)
print("соло E8, пик горизонта 5:  ", f"{ts_auc(E8[:, 5], y, s):.4f}", flush=True)

# 2) наборы каналов
best_mean, best_mat, best_tag, best_oof = 0.5779, None, "104 (базовая 013)", None
for tag, mat in [
    ("108 (100 + E8-прогнозист)", np.hstack([X100, E8])),
    ("154 (104 + D-вид)", np.hstack([X100, E4, D50])),
    ("162 (100 + E8 + D-вид)", np.hstack([X100, E8, D50])),
]:
    mean, oof = cv_boosted(mat, tag, save_oof=True)
    if mean > best_mean:
        best_mean, best_mat, best_tag, best_oof = mean, mat, tag, oof
print(f"лучший набор: {best_tag} ({best_mean:.4f})", flush=True)

if best_mat is None:
    print("новые наборы не побили 104 — постобработку меряем на 104", flush=True)
    _, best_oof = cv_boosted(np.hstack([X100, E4]), "104 (пересчёт для OOF)", save_oof=True)
    best_mat = np.hstack([X100, E4])

# 3) взвешенная сумма и максимум на лучшем наборе (один фолд для скорости, потом полный если близко)
names = [f"c{i}" for i in range(best_mat.shape[1])]
tr, va = assignment != 0, assignment == 0
w_model = Weighted.fit(best_mat[tr], y[tr], names, step_weights(s[tr]))
print(f"взвешенная сумма, фолд 0: {ts_auc(w_model.predict(best_mat[va]), y[va], s[va]):.4f}", flush=True)
print(f"максимум каналов, фолд 0: {ts_auc(best_mat[va].max(axis=1), y[va], s[va]):.4f}", flush=True)

# 4) асимметричная постобработка OOF лучшей модели
starts = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]]))
bounds = np.append(starts, len(g))
fast_pred = E8[:, 0]  # быстрая EWMA ошибки прогноза: ~1 = ряд предсказуем

def fold_scores(p):
    return [ts_auc(p[assignment == f], y[assignment == f], s[assignment == f]) for f in range(5)]

base_scores = fold_scores(best_oof)
print(f"постобработка, база: {np.mean(base_scores):.4f}", flush=True)

def peak_hold(p, alpha):
    out = np.empty_like(p)
    for a, b in zip(bounds[:-1], bounds[1:]):
        acc = 0.0
        for i in range(a, b):
            acc = max(float(p[i]), alpha * acc)
            out[i] = acc
    return out

def gated(p, a_calm, a_loud, thr):
    out = np.empty_like(p)
    for a, b in zip(bounds[:-1], bounds[1:]):
        acc = 0.0
        for i in range(a, b):
            alpha = a_calm if fast_pred[i] < thr else a_loud
            acc = max(float(p[i]), alpha * acc)
            out[i] = acc
    return out

for alpha in (0.98, 0.995, 0.999):
    sc = fold_scores(peak_hold(best_oof, alpha))
    print(f"пик-холд α={alpha}: {np.mean(sc):.4f}  " + " ".join(f"{v:.4f}" for v in sc), flush=True)
for a_calm, a_loud, thr in [(0.95, 0.999, 1.05), (0.90, 0.999, 1.10), (0.97, 1.0, 1.05)]:
    sc = fold_scores(gated(best_oof, a_calm, a_loud, thr))
    print(f"гейт прогнозистом ({a_calm}/{a_loud}, порог {thr}): {np.mean(sc):.4f}  "
          + " ".join(f"{v:.4f}" for v in sc), flush=True)
print(f"всего {time.time()-t0:.0f}s", flush=True)
