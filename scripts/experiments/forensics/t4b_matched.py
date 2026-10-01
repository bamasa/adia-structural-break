"""Task 4b: selection-matched test for 'invisible' breaks. Apply the SAME 300-window visibility rule
(|dmean|<=0.3, vratio in [0.85,1.18], |dar1|<=0.12, |dkurt|<=1.5) to breaks and to the length-matched controls of
t4_stats, then compare invisible breaks vs invisible controls (and visible vs visible) on the broad statistics,
with a CV composite AUC. Also: excess of post residual variance in the invisible group. -> t4b_summary.txt"""
import os, sys, numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.dirname(os.path.abspath(__file__))
vals = np.load(os.path.join(OUT, "values.npy"), mmap_mode="r"); off = np.load(os.path.join(OUT, "offsets.npy")); meta = pd.read_csv(os.path.join(OUT, "meta.csv"))
df = pd.read_csv(os.path.join(OUT, "t4_stats.csv"))
base = [c[5:] for c in df.columns if c.startswith("post_") and c not in ("post_len", "post_n")]
for s in base: df["d_" + s] = df["post_" + s] - df["pre_" + s]
def ar1(v): return np.corrcoef(v[:-1], v[1:])[0, 1]
rows = []
for k, cut in zip(df.id, df.cut):
    a = off[k]; h = int(meta.hist_len[k]); o = int(meta.onl_len[k]); x = np.asarray(vals[a:a + h + o], float)
    pre = x[: h + cut][-300:]; post = x[h + cut:][:300]
    rows.append(dict(w_dmean=abs(post.mean() - pre.mean()) / pre.std(), w_vratio=post.std() / pre.std(), w_dar1=abs(ar1(post) - ar1(pre)), w_dshape=abs(stats.kurtosis(post, bias=False) - stats.kurtosis(pre, bias=False))))
df = pd.concat([df, pd.DataFrame(rows)], axis=1)
df["invisible"] = (df.w_dmean <= 0.3) & (df.w_vratio >= 0.85) & (df.w_vratio <= 1.18) & (df.w_dar1 <= 0.12) & (df.w_dshape <= 1.5)
out = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)
P("invisible rule: breaks %.3f (n=%d), controls %.3f (n=%d)" % (df[df.is_break == 1].invisible.mean(), (df.is_break == 1).sum(), df[df.is_break == 0].invisible.mean(), (df.is_break == 0).sum()))
for lo, hi in ((30, 100), (100, 200), (200, 400), (400, 1000)):
    g = df[(df.post_len >= lo) & (df.post_len < hi)]
    P(f"  post_len [{lo},{hi}): invisible frac breaks {g[g.is_break == 1].invisible.mean():.3f}  controls {g[g.is_break == 0].invisible.mean():.3f}")
DS = ["d_" + s for s in base] + ["ks", "wass", "ks_win"] + ["post_" + s for s in ("sd", "mad", "mean", "tail2", "up90", "lo10", "up50", "res_var", "jumpfrac")]
bins = [30, 60, 100, 200, 400, 1000]; df["lbin"] = pd.cut(df.post_len, bins, right=False)

def compare(name, b, c):
    P(f"\n== {name}: breaks {len(b)}, controls {len(c)}; post_len medians {b.post_len.median():.0f} / {c.post_len.median():.0f}")
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
        rows.append((s, len(zb), iqr(zb) / (iqr(zc) + 1e-12), np.mean(np.abs(zb) > 2), np.mean(np.abs(zc) > 2), np.median(zb) - np.median(zc), stats.ks_2samp(zb, zc).pvalue, stats.mannwhitneyu(zb, zc).pvalue))
    rows.sort(key=lambda r: r[6])
    for r in rows[:16]: P(f"{r[0]:18s} {r[1]:5d} {r[2]:9.3f} {r[3]:7.3f} {r[4]:7.3f} {r[5]:+9.3f} {r[6]:9.2e} {r[7]:9.2e}")
    P(f"  ... stats with KS p<0.001: {sum(r[6] < 1e-3 for r in rows)} of {len(rows)}; p<0.01: {sum(r[6] < 1e-2 for r in rows)}; p<0.05: {sum(r[6] < 0.05 for r in rows)} (49 tests)")
    X = pd.concat([b, c])[DS + ["post_len"]].replace([np.inf, -np.inf], np.nan); X = X.fillna(X.median())
    lo, hi = X.quantile(0.01), X.quantile(0.99); X = X.clip(lo, hi, axis=1); y = np.r_[np.ones(len(b)), np.zeros(len(c))]
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000))
    pr = cross_val_predict(model, X.to_numpy(), y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=4)
    pr2 = cross_val_predict(m, X.to_numpy(), y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    rng = np.random.default_rng(0); yp = rng.permutation(y)
    prp = cross_val_predict(m, X.to_numpy(), yp, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    P(f"  composite CV AUC: logistic {roc_auc_score(y, pr):.3f}, LightGBM {roc_auc_score(y, pr2):.3f}  [permuted-label LightGBM {roc_auc_score(yp, prp):.3f}]")
    # bootstrap CI for lgbm auc
    idx = np.arange(len(y)); aucs = []
    for i in range(200):
        s = rng.choice(idx, len(idx));
        if len(np.unique(y[s])) < 2: continue
        aucs.append(roc_auc_score(y[s], pr2[s]))
    P(f"  LightGBM AUC bootstrap 95%% CI: [{np.percentile(aucs, 2.5):.3f}, {np.percentile(aucs, 97.5):.3f}]")
    # top LightGBM features by gain
    m.fit(X.to_numpy(), y); imp = pd.Series(m.booster_.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
    P("  top-8 LightGBM features (gain share):", ", ".join(f"{k}={v / imp.sum():.2f}" for k, v in imp.head(8).items()))

b_inv = df[(df.is_break == 1) & df.invisible]; c_inv = df[(df.is_break == 0) & df.invisible]
compare("INVISIBLE (rule) breaks vs INVISIBLE controls, post_len>=30", b_inv, c_inv)
compare("INVISIBLE breaks vs INVISIBLE controls, post_len>=200", b_inv[b_inv.post_len >= 200], c_inv[c_inv.post_len >= 200])
f2 = pd.read_csv("/Users/organist/projects/adia-structural-break/fold2_break_kinds.csv"); inv353 = set(f2[f2.kind == "none evident"].sid)
compare("INV353 breaks vs INVISIBLE controls (all)", df[(df.is_break == 1) & df.id.isin(inv353)], c_inv)
b_vis = df[(df.is_break == 1) & ~df.invisible]; c_vis = df[(df.is_break == 0) & ~df.invisible]
compare("VISIBLE (rule) breaks vs VISIBLE controls", b_vis, c_vis)
# residual variance excess in invisible group by post_len bin
P("\n== invisible group: log2 post residual variance (pre-fitted AR model) breaks vs controls, by post_len bin")
for lb, g in df[df.invisible].groupby("lbin", observed=True):
    vb = np.log2(g[g.is_break == 1].post_res_var); vc = np.log2(g[g.is_break == 0].post_res_var)
    P(f"  {str(lb):12s} n_b={len(vb):4d} n_c={len(vc):4d} median b {vb.median():+.3f} c {vc.median():+.3f} | frac>0.15: b {np.mean(vb > 0.15):.3f} c {np.mean(vc > 0.15):.3f} | frac<-0.15: b {np.mean(vb < -0.15):.3f} c {np.mean(vc < -0.15):.3f} | MW p {stats.mannwhitneyu(vb, vc).pvalue:.2e}")
P("\n== invisible group: d_acf1 |.|>0.05 fraction breaks vs controls (post_len>=200):")
g = df[df.invisible & (df.post_len >= 200)]
P("  ", np.mean(g[g.is_break == 1].d_acf1.abs() > 0.05).round(3), np.mean(g[g.is_break == 0].d_acf1.abs() > 0.05).round(3), " |d_acf1|>0.08:", np.mean(g[g.is_break == 1].d_acf1.abs() > 0.08).round(3), np.mean(g[g.is_break == 0].d_acf1.abs() > 0.08).round(3))
df[["id", "is_break", "target", "cut", "post_len", "invisible", "w_dmean", "w_vratio", "w_dar1", "w_dshape"]].to_csv(os.path.join(OUT, "t4b_flags.csv"), index=False)
open(os.path.join(OUT, "t4b_summary.txt"), "w").write("\n".join(out))
