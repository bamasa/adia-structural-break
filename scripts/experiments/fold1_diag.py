"""Диагностика слабого фолда 1: какие ряды там живут и где теряются пары."""
import sys
sys.path.insert(0, "repo/src")
import numpy as np
from structural_break.combiners import split_by_series, ts_auc

y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
oof = np.load("oof_rank.npy")
assignment = split_by_series(g, folds=5, seed=0)
for fold in range(5):
    m = assignment == fold
    yy, ss, gg, pp = y[m], s[m], g[m], oof[m]
    sids = np.unique(gg)
    starts = np.flatnonzero(np.concatenate([[True], gg[1:] != gg[:-1]]))
    bounds = np.append(starts, len(gg))
    n_broken = sum(1 for a in starts if yy[a:].max() and yy[np.arange(a, bounds[np.searchsorted(starts, a)+0+1] if False else len(gg))].max())
    # проще: по рядам
    lens, taus = [], []
    for a, b in zip(starts, bounds[1:]):
        lens.append(b - a)
        lab = yy[a:b]
        taus.append(int(lab.argmax()) if lab.max() else -1)
    lens = np.array(lens); taus = np.array(taus)
    br = taus >= 0
    print(f"фолд {fold}: рядов {len(lens)}, сломанных {br.mean():.0%}, "
          f"медианная длина {np.median(lens):.0f}, медианный слом на шаге {np.median(taus[br]):.0f}, "
          f"ранняя треть сломов {np.mean(taus[br] < np.median(lens)/3):.0%}, "
          f"TS-AUC {ts_auc(pp, yy, ss):.4f}")
