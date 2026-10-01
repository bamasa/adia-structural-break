"""Task 4 summary: do the 'invisible' breaks carry any signature? Groups: INV353 (fold-2 'no visible change' list),
VIS_f2 (fold-2 visible), NONE_full (full-data 'none' with post_len>=100), ALL (post_len>=100). For each statistic
(post - pre on matched windows) compare breaks vs their length-matched controls: IQR ratio, fraction beyond 2 robust
control sigmas, KS p-value, Mann-Whitney p-value, and a cross-validated logistic-regression AUC on all statistics."""
import os, sys, numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
OUT = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(OUT, "t4_stats.csv"))
f2 = pd.read_csv("/Users/organist/projects/adia-structural-break/fold2_break_kinds.csv")
inv353 = set(f2[f2.kind == "none evident"].sid); vis_f2 = set(f2[f2.kind != "none evident"].sid)
kinds = pd.read_csv(os.path.join(OUT, "t3_break_kinds_full.csv"), usecols=["id", "kind", "post_len"])
none_full = set(kinds[(kinds.kind == "none") & (kinds.post_len >= 100)].id); vis_full = set(kinds[(kinds.kind != "none") & (kinds.post_len >= 100)].id)
base = [c[5:] for c in df.columns if c.startswith("post_") and c != "post_len" and c != "post_n"]
for s in base: df["d_" + s] = df["post_" + s] - df["pre_" + s]
DS = ["d_" + s for s in base] + ["ks", "wass", "ks_win"] + ["post_" + s for s in ("sd", "mad", "mean", "tail2", "up90", "lo10", "up50", "res_var", "jumpfrac")]
out = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)
bins = [30, 60, 100, 200, 400, 1000]; df["lbin"] = pd.cut(df.post_len, bins, right=False)

def analyse(name, ids):
    b = df[(df.is_break == 1) & df.id.isin(ids)]; c = df[(df.is_break == 0) & df.target.isin(ids)]
    P(f"\n== group {name}: breaks {len(b)}, controls {len(c)}; post_len median {b.post_len.median():.0f} (q25 {b.post_len.quantile(.25):.0f}, q75 {b.post_len.quantile(.75):.0f})")
    P(f"{'stat':18s} {'n':>5s} {'iqr_ratio':>9s} {'brk>2s':>7s} {'ctl>2s':>7s} {'medshift':>9s} {'KS_p':>9s} {'MW_p':>9s}")
    rows = []
    for s in DS:
        zb_all, zc_all = [], []
        for lb, g in c.groupby("lbin", observed=True):
            v = g[s].to_numpy(float); v = v[np.isfinite(v)]
            if len(v) < 10: continue
            med = np.median(v); mad = 1.4826 * np.median(np.abs(v - med)) + 1e-12
            vb = b[b.lbin == lb][s].to_numpy(float); vb = vb[np.isfinite(vb)]
            zb_all.append((vb - med) / mad); zc_all.append((v - med) / mad)
        if not zb_all: continue
        zb = np.concatenate(zb_all); zc = np.concatenate(zc_all)
        if len(zb) < 10: continue
        iqr = lambda v: np.subtract(*np.percentile(v, [75, 25]))
        ksp = stats.ks_2samp(zb, zc).pvalue; mwp = stats.mannwhitneyu(zb, zc).pvalue
        rows.append((s, len(zb), iqr(zb) / (iqr(zc) + 1e-12), np.mean(np.abs(zb) > 2), np.mean(np.abs(zc) > 2), np.median(zb) - np.median(zc), ksp, mwp))
    rows.sort(key=lambda r: r[6])
    for r in rows: P(f"{r[0]:18s} {r[1]:5d} {r[2]:9.3f} {r[3]:7.3f} {r[4]:7.3f} {r[5]:+9.3f} {r[6]:9.2e} {r[7]:9.2e}")
    P(f"  stats with KS p<0.001: {sum(r[6] < 1e-3 for r in rows)} of {len(rows)}; with p<0.05: {sum(r[6] < 0.05 for r in rows)}")
    # composite: CV logistic AUC breaks vs controls using robust-z features (per-bin z computed above is per stat; here use raw d + post_len as covariate)
    X = pd.concat([b, c])[DS + ["post_len"]].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median()); y = np.r_[np.ones(len(b)), np.zeros(len(c))]
    # clip extreme values (heavy tails) at 1st/99th percentiles
    lo, hi = X.quantile(0.01), X.quantile(0.99); X = X.clip(lo, hi, axis=1)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000))
    pr = cross_val_predict(model, X.to_numpy(), y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    auc = roc_auc_score(y, pr)
    # permutation baseline (shuffle labels) for reference
    rng = np.random.default_rng(0); yp = rng.permutation(y)
    prp = cross_val_predict(model, X.to_numpy(), yp, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    P(f"  composite CV logistic AUC (breaks vs matched controls): {auc:.3f}   [label-permuted baseline {roc_auc_score(yp, prp):.3f}]")
    try:
        import lightgbm as lgb
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30, subsample=0.8, colsample_bytree=0.8, verbose=-1, n_jobs=4)
        pr2 = cross_val_predict(m, X.to_numpy(), y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
        P(f"  composite CV LightGBM AUC: {roc_auc_score(y, pr2):.3f}")
    except Exception as e: P("  lgbm failed", e)
    return rows

analyse("INV353 (fold-2 no-visible-change)", inv353)
analyse("VIS_f2 (fold-2 visible)", vis_f2)
analyse("NONE_full (post_len>=100, no |z|>3 on task-3 stats)", none_full)
analyse("VIS_full (post_len>=100, some |z|>3)", vis_full)
allb = set(df[(df.is_break == 1) & (df.post_len >= 100)].id)
analyse("ALL breaks post_len>=100", allb)
# post-break length comparison
b = df[df.is_break == 1]
P("\n== post-break length: INV353 vs VIS_f2")
a1 = b[b.id.isin(inv353)].post_len; a2 = b[b.id.isin(vis_f2)].post_len
P(f"  INV353: n={len(a1)} median {a1.median():.0f} q10 {a1.quantile(.1):.0f} q90 {a1.quantile(.9):.0f} | VIS_f2: n={len(a2)} median {a2.median():.0f} q10 {a2.quantile(.1):.0f} q90 {a2.quantile(.9):.0f} | MW p={stats.mannwhitneyu(a1, a2).pvalue:.3g}")
a1 = kinds[kinds.kind == "none"].post_len; a2 = kinds[kinds.kind != "none"].post_len
P(f"  full data: none n={len(a1)} median {a1.median():.0f} | visible n={len(a2)} median {a2.median():.0f} | MW p={stats.mannwhitneyu(a1, a2).pvalue:.3g}")
P("  fraction 'none' by post_len bin (full data):", kinds.groupby(pd.cut(kinds.post_len, [20, 50, 100, 200, 400, 1000], right=False), observed=True).kind.apply(lambda v: (v == "none").mean()).round(3).to_dict())
# family composition of INV353 vs VIS_f2
fam = df[df.is_break == 1].drop_duplicates("id").set_index("id")
P("\n== family composition: INV353 vs VIS_f2 (lin):", fam.loc[list(inv353 & set(fam.index))].lin.value_counts(normalize=True).round(2).to_dict(), "|", fam.loc[list(vis_f2 & set(fam.index))].lin.value_counts(normalize=True).round(2).to_dict())
P("   vol:", fam.loc[list(inv353 & set(fam.index))].vol.value_counts(normalize=True).round(2).to_dict(), "|", fam.loc[list(vis_f2 & set(fam.index))].vol.value_counts(normalize=True).round(2).to_dict())
P("   innov:", fam.loc[list(inv353 & set(fam.index))].innov.value_counts(normalize=True).round(2).to_dict(), "|", fam.loc[list(vis_f2 & set(fam.index))].innov.value_counts(normalize=True).round(2).to_dict())
open(os.path.join(OUT, "t4_summary.txt"), "w").write("\n".join(out))
