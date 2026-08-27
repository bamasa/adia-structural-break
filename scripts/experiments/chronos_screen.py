"""030 (фолд-0): ошибка Chronos-2 как одиночный канал — соло-скрининг.

Планка: наш дообучаемый прогнозист соло 0.5586 (пик ошибки, полная выборка);
здесь скрин на рядах фолда-0. На каденс-точках просим Chronos спрогнозировать
следующие 8 шагов от префикса, меряем нормированный промах факта; уровень
нормировки — такие же прогнозы внутри истории (там слома нет по условию).
"""
import sys, time
sys.path.insert(0, "repo/src")
import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline
from structural_break.features import Normalisation
from structural_break.stream import iter_series
from structural_break.combiners import split_by_series, ts_auc

t0 = time.time()
pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="mps",
                                           torch_dtype=torch.float32)
print(f"модель загружена [{time.time()-t0:.0f}s]", flush=True)

x = pd.read_parquet("structural-break-real-time-test/data/X_train.parquet")
y = pd.read_parquet("structural-break-real-time-test/data/y_train.parquet")
g = np.load("G40.npy")
assignment = split_by_series(g, folds=5, seed=0)
starts = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]]))
sid_fold = {int(g[st]): int(assignment[st]) for st in starts}

H = 8          # горизонт прогноза
CTX = 256      # максимум контекста

def cadence(n):
    pts, nxt = [], 10
    while nxt < n:
        pts.append(nxt)
        nxt = max(nxt + H, int(nxt * 1.25))
    return pts

series_data = []
for sid, hist, online, labels in iter_series(x, y):
    if sid_fold[int(sid)] != 0:
        continue
    norm = Normalisation.fit(hist)
    n = len(hist)
    z_h = np.asarray([norm.clip(norm.standardise(float(v), i - n)) for i, v in enumerate(hist)],
                     dtype="float32")
    z_o = np.asarray([norm.clip(norm.standardise(float(v), i)) for i, v in enumerate(online)],
                     dtype="float32")
    series_data.append((z_h, z_o, labels))
print(f"рядов фолда-0: {len(series_data)}, подготовка {time.time()-t0:.0f}s", flush=True)

# Собираем все запросы одной пачкой: (контекст, факт h шагов).
requests, owners = [], []   # owner: (series_idx, "hist"/"online", позиция каденса)
for si, (z_h, z_o, labels) in enumerate(series_data):
    full = np.concatenate([z_h, z_o])
    n_h = len(z_h)
    # опорные прогнозы внутри истории — уровень нормальной ошибки
    for p in cadence(n_h)[-6:]:
        if p + H <= n_h:
            requests.append(full[max(0, p - CTX):p]); owners.append((si, "hist", p))
    for p in cadence(len(z_o)):
        gp = n_h + p
        if gp + H <= len(full):
            requests.append(full[max(0, gp - CTX):gp]); owners.append((si, "online", p))
print(f"запросов: {len(requests)}", flush=True)

errors = np.zeros(len(requests))
widths = np.zeros(len(requests))
B = 128
with torch.no_grad():
    for k in range(0, len(requests), B):
        batch = [torch.tensor(r) for r in requests[k:k + B]]
        q, _ = pipe.predict_quantiles(batch, prediction_length=H,
                                      quantile_levels=[0.25, 0.5, 0.75])
        for j, (si, kind, p) in enumerate(owners[k:k + B]):
            z_h, z_o, _ = series_data[si]
            full = np.concatenate([z_h, z_o])
            gp = p if kind == "hist" else len(z_h) + p
            actual = full[gp:gp + H]
            qj = q[j] if isinstance(q, list) else q[j]
            qj = qj.squeeze()
            med = qj[:, 1].numpy()
            iqr = np.maximum((qj[:, 2] - qj[:, 0]).numpy(), 1e-3)
            errors[k + j] = float(np.mean(np.abs(actual - med)))
            widths[k + j] = float(np.mean(iqr))
        if (k // B) % 20 == 0:
            print(f"  {k}/{len(requests)} [{time.time()-t0:.0f}s]", flush=True)

# Канал на ряд: ошибка онлайн-точки, делённая на медианную ошибку истории ряда.
scores_rows, labels_rows, steps_rows = [], [], []
for si, (z_h, z_o, labels) in enumerate(series_data):
    hist_errs = [errors[i] for i, (s2, kind, _) in enumerate(owners) if s2 == si and kind == "hist"]
    hist_w = [widths[i] for i, (s2, kind, _) in enumerate(owners) if s2 == si and kind == "hist"]
    base = np.median(hist_errs) if hist_errs else 1.0
    base_w = np.median(hist_w) if hist_w else 1.0
    pts = [(p, errors[i], widths[i]) for i, (s2, kind, p) in enumerate(owners) if s2 == si and kind == "online"]
    pts.sort()
    err_curve = np.zeros(len(z_o)); wid_curve = np.zeros(len(z_o)); peak_curve = np.zeros(len(z_o))
    fe, fw, pk = 1.0, 1.0, 0.0
    idx = 0
    for step in range(len(z_o)):
        while idx < len(pts) and pts[idx][0] <= step:
            fe += 0.5 * (min(pts[idx][1] / (base + 1e-9), 8.0) - fe)
            fw += 0.5 * (min(pts[idx][2] / (base_w + 1e-9), 8.0) - fw)
            pk = max(pk, fe)
            idx += 1
        err_curve[step] = fe; wid_curve[step] = fw; peak_curve[step] = pk
    scores_rows.append(np.column_stack([err_curve, wid_curve, peak_curve]))
    labels_rows.append(labels.astype(int))
    steps_rows.append(np.arange(len(z_o)))
S = np.vstack(scores_rows); L = np.concatenate(labels_rows); T = np.concatenate(steps_rows)
np.save("chronos_fold0.npy", S)
print(f"соло сырая ошибка:    {ts_auc(S[:,0], L, T):.4f}", flush=True)
print(f"соло ширина интервала: {ts_auc(S[:,1], L, T):.4f}", flush=True)
print(f"соло пик ошибки:      {ts_auc(S[:,2], L, T):.4f}", flush=True)
print(f"всего {time.time()-t0:.0f}s", flush=True)
