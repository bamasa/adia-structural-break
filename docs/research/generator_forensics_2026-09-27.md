# Data forensics: ADIA Structural Break (Real-Time Edition) training set

All numbers below come from the scripts in this folder, run on `X_train.parquet` / `y_train_index.parquet`
(10 000 series, 35 036 464 rows, no NaN). Scripts: `build_cache.py` (compact cache), `t1_*` (marginals),
`t2_*` (dynamics, families), `t3_*` (breaks), `t4_*` (invisible breaks), `t5_*` + `synth.py` (generator).
Outputs are the `*_summary.txt`, `*.txt` and `*.csv` files next to the scripts.

## 0. Basic facts (build_cache.py, t1_summary.txt)

| fact | value |
|---|---|
| history length | 1000..5000, roughly uniform (quartiles 1977 / 2998 / 4022) |
| online length | 10..999, uniform, same for break and non-break series |
| breaks | 4967 / 10000 (49.7 %) |
| break position | `tau_index / onl_len` is uniform on [0,1) (10 bins: 520,536,536,498,483,486,472,462,503,471) |
| post-break length | min 1, q05 11, q25 74, median 199, q75 394, q95 712; 208 breaks have < 10 post points |
| standardisation | every series has history mean = 0 and history sd = 1 to 1e-9: the whole series was standardised **on the history only** (online part is in history-sd units) |
| history/online boundary | sd of the online part before tau: median 0.97 for both break and control series -> no change at the boundary |
| discreteness | none: unique fraction >= 0.974, no zeros, no repeats, no integers; float32 precision only |
| bounds | none (min -43, max +69 in sd units); no non-negative or [0,1] series |

## 1. Marginal structure of histories (t1_marginal.py, t1_summary.txt, t1_families.txt)

Excess kurtosis quantiles 5/25/50/75/95/99 %: -0.22 / 0.00 / 0.22 / 1.48 / 25.9 / 87.4.
Skew quantiles 5/25/50/75/95 %: -0.41 / -0.04 / 0.02 / 0.13 / 0.77. Tail asymmetry (q99-med)/(med-q01) outside [0.8,1.25]: 799 series (8 %).

| marginal family (raw standardised history) | count |
|---|---|
| gaussian-like (abs kurt <= 1, abs skew <= 0.3) | 6375 |
| heavy-tailed (1 < kurt <= 20) | 2415 |
| very heavy (kurt > 20; typically one to a few spikes, median max abs z = 13.8) | 637 |
| skewed (abs skew > 0.3, abs kurt <= 1; 158 positive / 166 negative) | 324 |
| light/bimodal (kurt < -0.5) | 249 |

The light/bimodal marginals are mostly an artefact of strong persistence (25 % have abs acf1 > 0.5, median 0.28); only ~120
series have genuinely light-tailed (uniform-like) innovations after AR/GARCH standardisation. Heavy marginals come from
heavy-tailed innovations and/or volatility clustering (section 2). Verdict: continuous families, no discrete menu in the marginal.

## 2. Dynamic structure of histories (t2_dynamics.py, t2b_extra.py, t2c_families.py, t2d_orders.py, t2e_checks.py)

Global checks:
* **No unit roots**: ADF t (const, 4 lags) < -2.9 for all but 1 series; variance ratio VR(20) median 0.05 (white noise = 0.05); only ~90 series have acf1 > 0.9.
* **No trend**: total linear drift over the history, median 0.008 sd, 99 % < 0.3 sd.
* **No seasonality**: Fisher-g on AR-prewhitened residuals significant (log10 p < -8) for 28 series only, all with periods 2..12 (stochastic cycles), no seasonal-period menu. (1555 series are "seasonal" on levels; all are low-frequency AR peaks removed by prewhitening.)
* **Histories are stationary except for isolated spikes**: for WN + constant-vol + Gaussian series (724), the max/min sd across thirds has 99 % quantile 1.38 vs 1.11 under iid; the excess is single-point outliers (sd profiles show one block at 3-4x, e.g. ids 891, 3083, 7841). Mean CUSUM is iid-like (0.6 % beyond the iid 99.9 % bound).

Linear dynamics, BIC among WN / AR(1) / AR(2) / AR(3..10) / MA(1) / ARMA(1,1) (OLS; MA and ARMA via conditional LS / Hannan-Rissanen):

| family | count | Hannan-Rissanen ARMA(p<=3,q<=3) orders (BIC) |
|---|---|---|
| WN | 2424 | (0,0): 3316 of all series |
| AR(1) | 866 | (1,0): 1068 |
| AR(2) | 752 | (2,0): 718 (190 complex roots, modulus 0.25-0.5) |
| AR(p>=3) | 3413 | (2,2): 766, (3,3): 863, (3,0): 536, (3,2): 415, (2,1): 324, (2,3): 224 ... |
| MA(1) | 1295 | (0,1): 297, (0,2): 159, (0,3): 100 |
| ARMA(1,1) | 1250 | (1,1): 763 |

p+q distribution (HR): 0: 3316, 1: 1365, 2: 1640, 3: 1110, 4: 1067, 5: 639, 6: 863. The AR(p>=3) group is not explained by
ARMA(<=3,<=3): AR(p_BIC) beats the best HR ARMA by a median 6.9 BIC. A third of the series are white noise.

Parameter values (menu or continuous?):
* AR(1) phi: **not uniform** (KS vs U(-0.95,0.95) p = 1e-124, 77 % positive). Three components: a **cluster at phi ~ 0.5** (276 of 866 AR(1) series in [0.425, 0.55]; the same bump is visible in acf1 of all series), small abs phi 0.05-0.2, and a diffuse remainder incl. negative values down to -0.8.
* MA(1) theta: mostly small (+-0.05..0.2), no clusters.
* ARMA(1,1): phi in (0.4, 0.95), theta ~ -phi (abs(phi+theta) median 0.10, corr -0.87): near-cancelling "AR(1) plus noise" processes.
* AR(2): c2 mode at 0.05-0.10, c1 broad around 0-0.3.
* Verdict: the model classes look like a menu (ARMA orders, GARCH on/off, innovation type), the coefficient values are continuous; the only discrete-looking feature is the phi ~ 0.5 cluster.

Volatility (rank-based Ljung-Box(10) on abs residuals, tail-robust; iid 99.9 % quantile = 27 by simulation, threshold 30):

| volatility family | count | notes |
|---|---|---|
| constant | 7527 | |
| GARCH-like | 2417 | GARCH(1,1) QML: alpha mostly 0.02-0.15, beta 0.85-0.98, alpha+beta > 0.96 in 56 % (1383/2473); 146 fits find alpha+beta ~ 0 (slow/regime-type variance) |
| regime-like (rolling-var acf1 > 0.45 and bimodal log rolling variance) | 56 | |

Volatility clustering is more common in WN (35 %) and MA(1) (43 %) histories than in AR families (~20 %).

Innovations (AR residuals, GARCH-standardised where applicable; quantile-based Student-t df; Gaussian calibration gives 7.5 % false "df < 30", 0 % false "df < 10"):

| innovation family | count |
|---|---|
| Gaussian (df >= 30, kurt < 0.4) | 5663 |
| t, df 10-30 (mild; ~750 of these expected to be Gaussian by calibration) | 3092 |
| t, df 5-10 | 1025 |
| t, df 3-5 | 76 |
| t, df < 3 | 3 |
| light-tailed (kurt < -0.35) | 141 |

df is continuous (log10 df spread evenly over 0.7-1.5); asymmetric innovations in 484 series (4.8 %).

## 3. Break characterisation (t3_breaks.py, t3_summary.py, t3b_excess.py)

Design: pre = full history + online part before tau, post = all post-break points (breaks with >= 20 post points: 4548);
controls = non-break series cut at 2 random online positions (9131). Robust z = (stat - control median) / control MAD within
post-length bins [20,50), [50,100), [100,200), [200,400), [400,1000).

Fraction beyond abs z > 3 (breaks vs controls; a statistic carries signal only if the first number exceeds the second):

| statistic | breaks | controls |
|---|---|---|
| innovation variance ratio (post/pre residual var under the history AR model) | **0.155** | 0.089 |
| sd ratio | **0.111** | 0.072 |
| Wasserstein / KS (post vs pre, standardised) | 0.113 / 0.084 | 0.084 / 0.058 |
| AR(2) / AR(3) coefficient change (Euclidean) | 0.066 / 0.066 | 0.027 / 0.026 |
| AR(1) coefficient change | 0.047 | 0.023 |
| acf(2) change | 0.046 | 0.018 |
| mean shift | 0.057 | 0.053 |
| residual kurtosis / skew change | 0.221 / 0.130 | 0.199 / 0.108 (unreliable: heavy tails) |
| acf of squared residuals (vol clustering) | 0.030 | 0.027 |
| Hill tail index ratio | 0.016 | 0.014 |
| spectral slope | 0.032 | 0.024 |

Break kinds with abs z > 3 on any statistic of a group (M mean, V variance, D dependence, S shape, C vol-clustering):
none 2614 (57 %), S 646 (mostly false positives: the same union rule fires on 35 % of controls), V 248, D 227, VS 208, M 108,
VD 89, DS 53, SC 52, C 51, other combinations < 35 each.
"none" fraction by post length: 69 % (20-50), 67 % (50-100), 61 % (100-200), 54 % (200-400), **48 % (400-1000)**.

Magnitudes:
* **Variance** (the dominant break type): a pure scale change of the innovations (sd ratio and innovation-sd ratio agree, corr 0.995, median abs difference 0.004). Among clearly detected variance breaks (abs z > 4, n = 331) the sd ratio has quantiles 1/5/25/50/75/95/99 % = 0.22 / 0.34 / 0.70 / 1.59 / 2.15 / 4.8 / 11.3; **continuous, no clusters** in log2 (bin-0.05 histogram is smooth). The excess over controls is almost entirely on the **increase** side: excess P(log2 sd ratio > c) = +8.9 / +5.6 / +3.7 / +1.8 pp at c = 0.2 / 0.3 / 0.5 / 1.0, while excess P(< -c) = -0.8 / -0.5 / -0.5 / +0.2 pp. **No minimum magnitude**: the break distribution of log2 sd ratio is wider than the control distribution all the way down (effect sd from excess variance: 0.10-0.14 pooled for post >= 100; by family WN 0.21, MA1 0.19, ARMA11 0.16, AR1 0.13, AR2 0.11, ARp 0.08), and section 4 shows small increases (10-40 %) inside the "invisible" group.
* **Dependence**: only in AR-type families (effect sd of d_ar1 for post >= 100: AR2 0.071, ARp 0.066, AR1 0.056, ARMA11 0.030; **WN and MA1: 0.000**). Continuous magnitudes; clear cases (abs z > 4, n = 100) have d_ar1 in +-(0.2..0.75), 43 % increase abs phi, symmetric sign; corr(pre phi, post phi) = 0.47 vs 0.93 in controls (partial redraw / perturbation). Independent of the variance change (corr of abs effects 0.10; controls 0.21).
* **Mean**: essentially absent (effect sd <= 0.02; 118 series beyond abs z > 4 vs ~64 expected false positives; shifts 0.2-1.3 sd when present).
* **Shape / tails / vol clustering / spectral / Hill**: no excess over controls.
* Per family, share of breaks with a clear variance change (abs z > 3): WN 20 %, MA1 18 %, AR1 9 %, ARMA11 9 %, AR2 7 %, ARp 4 %; by volatility: constant 7 %, GARCH-like 21 %.
* The mixture-deconvolution fit in `t3b_excess.py` is numerically unstable (KDE artefacts) and is not used for conclusions.

## 4. The "invisible" breaks (t4_invisible.py, t4_summary.py, t4b_matched.py)

49 statistics on length-matched windows (post = all post points, pre = last min(post,1000) pre points, both in full-pre units):
Hurst R/S, DFA, permutation entropy (3,4), sample entropy, spectral slope, jump fraction, runs test, two time-reversibility
moments, BDS-style correlation-integral proxy, ACF of abs x in bands 1-5 / 6-20 / 21-50, acf lags 1-5,10, quantile
crossing rates (q10, q50, q90, abs > 2), aggregation variance, zero crossings, KS / Wasserstein, residual variance /
kurtosis / skew / acf under the history AR model, sd, MAD, mean. 2 length-matched non-break controls per break.

* The fold-2 list of 353 "no visible change" breaks vs unselected controls gives composite CV AUC 0.61 (logistic) / 0.66 (LightGBM), but this is confounded: the 353 were selected as "quiet" series (e.g. fewer jumps than random controls).
* **Selection-matched test**: the same 300-window rule applied to breaks and controls marks 40.1 % of breaks (1750/4364) and 52.6 % of controls as "invisible". Invisible breaks vs invisible controls: the only statistics with a real signature are scale statistics: post/pre residual variance (KS p = 1e-31; 17.3 % of breaks beyond 2 control sigma vs 7.8 %), sd, MAD, Wasserstein, windowed KS, abs > 2 frequency, and their scale-dependent proxies (sample entropy with r in pre-sd units, BDS proxy). Hurst/DFA, permutation entropy, runs, time reversibility, spectral slope, ACF of abs x, quantile crossings, residual kurtosis/skew, acf changes: no signature (KS p > 0.01).
* Composite CV AUC (all 49 statistics, LightGBM): invisible breaks vs invisible controls **0.635 [0.62, 0.65]**; with post >= 200: **0.673 [0.66, 0.69]**; the 353 fold-2 series vs invisible controls: 0.61 [0.58, 0.65]; permuted labels 0.49. Top features: post residual variance, its change, post sd.
* Inside the invisible group the fraction with log2 post residual variance > 0.15 (a 10 % innovation-variance increase) is 28-38 % for breaks vs 10-24 % for controls (by post length: [60,100) 37.7 vs 18.2; [100,200) 27.9 vs 17.4; [200,400) 26.5 vs 14.3; [400,1000) 33.0 vs 10.3 %); no excess of decreases.
* Post-break length: the 353 invisible breaks have **longer** post segments than the visible fold-2 breaks (median 272 vs 191, MW p = 6e-6): they are not invisible because tau is late. (On the full data the z-based "none" class has shorter posts, 194 vs 266, because detection power grows with length; even with >= 400 post points 48 % show nothing.)

**Verdict**: no hidden break type exists among 49 statistics. The invisible breaks are the low-magnitude tail of the same two
break types: roughly 12-20 percentage points of them carry a small innovation-variance increase (5-40 %) visible only in
aggregate, and the remaining ~25-30 % of all breaks show no measurable change at all (parameter changes too small to
identify in <= 1000 points, or none).

## 5. Synthetic generator draft (t5_params.py, synth.py, t5_compare.py; t5_compare.txt, t5_compare_v1.txt)

Model: bootstrap one real series' fitted parameter set (HR ARMA(p<=3,q<=3) or BIC-AR(p) coefficients, GARCH(1,1) alpha/beta if
vol-clustering, innovation type with t df = min(quantile df, 4 + 6/kurt), uniform if light-tailed), simulate ARMA-GARCH with
history ~ U{1000..5000}, online ~ U{10..999}, break with p = 0.5 at uniform tau. Break menu: innovation sd x 2^E,
E ~ Exp(mean 0.5) with p = 0.20 (increase) / p = 0.04 (decrease); AR/MA coefficients + N(0, 0.1^2) with p = 0.35 (non-WN only);
mean shift +-U(0.2, 1) with p = 0.02. Series standardised on the history.

500 synthetic vs 500 random real series (quantiles 5/25/50/75/95, KS p):

| statistic | real | synthetic (v2) | KS p | verdict |
|---|---|---|---|---|
| acf1 | -0.36 -0.03 0.02 0.16 0.66 | -0.39 -0.04 0.01 0.16 0.55 | 0.20 | match |
| acf2 / acf10 | -0.23 .. 0.63 / -0.07 .. 0.18 | -0.35 .. 0.55 / -0.07 .. 0.18 | 0.33 / 0.13 | match |
| BIC AR order | 0 0 1 4 9 | 0 0 1 4 9 | 0.98 | match |
| spectral slope | -0.90 -0.23 0.00 0.10 0.34 | -0.83 -0.17 0.01 0.11 0.39 | 0.20 | match |
| kurtosis | -0.25 0.01 0.26 1.89 31.5 | -0.15 0.00 0.23 1.25 6.3 | 0.02 | upper tail too light (v1: 2.7, p 4e-9) |
| max abs z | 3.1 3.7 4.2 6.5 13.3 | 3.2 3.6 4.1 5.4 8.9 | 0.006 | spikes under-represented |
| skew | -0.33 -0.04 0.03 0.15 0.93 | -0.26 -0.05 0.00 0.05 0.18 | 8e-11 | no skewed innovations in the draft |
| rank-LB(10) of abs residuals | 4.9 8.9 14.4 31.1 323 | 4.2 7.4 10.4 16.5 270 | 6e-9 | vol clustering too weak |
| sqacf1 / rolling-var acf1 | 0.02 0.06 0.23 / 0.04 0.27 0.67 | 0.00 0.03 0.24 / 0.00 0.12 0.43 | 4e-9 / 3e-5 | vol clustering too weak |
| sd max/min over thirds | 1.02 1.05 1.09 1.19 1.50 | 1.02 1.03 1.06 1.10 1.34 | 3e-10 | history variance too stable |
| breaks: log2 sd ratio | -0.31 -0.13 0.01 0.14 0.62 | -0.26 -0.08 0.01 0.10 0.79 | 0.04 | roughly right; upper-tail shape off |
| breaks: d_acf1 / d_mean / KS | p 0.96 / 0.19 / 0.27 | | | match |
| excess P(log2 sd ratio > c), c = 0.2/0.3/0.5/1.0 | +8.8 +7.9 +2.8 +3.2 pp | +11.4 +9.7 +7.1 +1.2 pp | | real multiplier tail is heavier at 1.0 and lighter at 0.5: use a heavier-tailed multiplier (e.g. log-normal mixture) |
| excess P(abs d_acf1 > 0.1) | +7.1 pp | +6.8 pp | | match |

What matches: linear dependence structure, AR orders, spectrum, and the break-statistic distributions (variance and dependence
effects). What does not: extreme tails/spikes, skewed innovations, strength and persistence of volatility clustering, and the
within-history variance instability (linked to the same three). Next iteration: add an additive-outlier component,
skewed-t innovations, and a stochastic-volatility / regime component with higher persistence.

## 6. Modelling consequences for cross-sectional AUC at steps 100-700

1. **Score the innovation-variance increase, one-sided, whitened, robust.** The single strongest statistic is the ratio of
   post-tau residual variance to the history residual variance under the history's own AR model; increases carry ~all the
   excess, decreases almost none; the magnitude is continuous from 0, so the score should be a likelihood ratio
   (sequential GLR/CUSUM on log residual variance with a prior on the multiplier, roughly log2 multiplier ~ Exp(0.5) on ~20 %
   of breaks) rather than a thresholded detector, and the null distribution must be series-specific (heavy tails and GARCH
   histories inflate variance-ratio noise; use the history's own block bootstrap or a rank/quantile-based scale statistic).
2. **Dependence changes only in AR-type histories, small.** For AR(1)/AR(2)/AR(p) histories (~50 % of series) add a
   whitened-residual autocorrelation test at lags 1-3 (or a recursive AR refit with a forgetting factor) against the history
   model; effect sizes are 0.05-0.07 on phi, so the test needs many post points, i.e. it matters mainly at steps 300-700.
   Mean, kurtosis, skew, tail index, spectral slope, entropy, nonlinearity and vol-clustering features carry no excess
   signal and only add noise.
3. **Accept the ceiling and calibrate.** ~30 % of breaks are unidentifiable and ~40 % are invisible to 300-point windows;
   the cross-sectional ranking should be a calibrated posterior probability that a change occurred at some tau <= t
   (uniform prior on tau within the online part, continuous effect priors above), accumulated online, so that series with weak
   but consistent evidence rank above pure noise. With those priors the synthetic generator (after fixing tails and
   volatility clustering) gives unlimited training data for a learned sequential scorer; train on real + synthetic and
   validate on real only.

Also worth using: the online part is in history-sd units (standardisation is history-only), and the ~30 % share of exactly
white-noise histories where only a scale change can occur.
