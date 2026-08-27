# adia-structural-break

Solutions for the [ADIA Lab Structural Break Challenge: Real-Time
Edition](https://hub.crunchdao.com/competitions/structural-break-real-time)
(CrunchDAO, May–September 2026).

## The task

A univariate series arrives in two parts. The **historical segment** — 1,000 to
5,000 observations, guaranteed free of breaks — is given in full at the start.
The **online segment** — 10 to 1,000 observations — is then revealed **one
observation at a time**, and after each one the model must output a score in
[0, 1]: the confidence that a permanent structural break has *already* occurred.
Half the series contain a break at an unknown position; half contain none.

Scored by **Time-Stratified AUC**: at each online step an ordinary AUC across
every series alive at that step, averaged with weights equal to the number of
positive-negative pairs available there.

Two properties of that metric shape every design decision here:

- Only the **ordering between series** at a given step matters, never the
  absolute level of any score. A uniformly pessimistic detector loses nothing;
  a detector that ties scores together loses everything, because ties are what
  an AUC is made of.
- **Early steps weigh as much as late ones**, and early steps are where almost
  no evidence exists. A detector that is only right after two hundred
  observations scores poorly however certain it eventually becomes.

## Layout

    src/structural_break/    the library: normalisation, streaming detectors,
                             multi-scale channels, retrospective scans, the CNN,
                             combiners, evaluation
    submissions/             one directory per submission, exactly as sent —
                             each main.py assembled from the library by
                             scripts/assemble_submission.py, never edited by hand
    notebooks/               model inspection on the labelled hundred, generated
                             and executed by scripts/build_analysis_notebook.py
    docs/                    the experiment log: hypothesis, design, result and
                             kill condition per submission, failures included

Competition data is never committed — it is 200 MB of parquet fetched by
`crunch setup`, and the workspace that command creates is gitignored.

## Submissions

Grouped 5-fold CV is the number decisions are made on; the local column is the
organisers' labelled hundred series, which at that sample size separates little
(±0.05) and exists to catch fabricated validation rather than to rank models.
Full history with hypotheses and kill conditions: [`docs/experiments.md`](docs/experiments.md).

| # | Approach | CV (5-fold) | Local | Fate |
|---|---|---|---|---|
| 001 | Three classical detectors, maximum | — | 0.5385 | cloud: ~top-400 of ~1500 |
| 002 | First learned combiner | 0.7487 | 0.4990 | **not submitted**: the CV was fabricated by length-dependent subsampling; the defect and fix are logged |
| 003 | Logistic weighting over 9 channels | 0.5293 | 0.5232 | cloud |
| 004 | LightGBM over 9 channels | 0.5568 | 0.5146 | cloud (resubmitted once: first run shipped no requirements.txt and died importing lightgbm) |
| — | Retrospective scan channels | 0.5570 vs 0.5568 | — | **killed by pre-stated condition**: no gain over the streaming channels |
| 005 | LightGBM over 40 channels: + multi-scale receptive fields (6 EWMA windows, 5–200 obs) + retrospective confirm/cancel verdicts | 0.5648 | 0.5146 | cloud |
| 006 | + a learned dilated-convolution channel (721 parameters, dilations 1/4/16, inline weights, numpy forward) | **0.5682** | 0.5146 | cloud (resubmitted once: json.loads without import json, hidden locally by a stale __pycache__) |

Current standing: ~330 of ~1500 on the public leaderboard (with 004, before the
40-channel models were scored). Field tail ≈ 0.62, top ≈ 0.652.

| — | 007: window classifier on 313k augmented cuts | 0.5627 vs 0.5682 | **killed by pre-stated condition**: window AUC nearly doubled (0.5315 → 0.5687), yet as channels it dilutes the combiner — better in isolation, redundant in ensemble |
| 008 | + nine reverting channels: each detector's current statistic beside its peak, so a false alarm can be recanted | **0.5719** vs 0.5662, better on all 5 folds | cloud |
| — | 009: rank view (probit of each observation's midrank in the history) | 0.5707 vs 0.5719 | **killed by pre-stated condition**: helps over the old base (+0.002), redundant over 008 — both fixes target the same false alarms, and two fixes for one disease do not stack |
| — | 010: series context (autocorrelation, kurtosis, length, scale, step) as features | 0.5681 vs 0.5719 | **killed by pre-stated condition**: fold spread doubles — the channels are already calibrated per-series, so context arrives pre-consumed and becomes an overfitting surface |
| — | 011a: rolling median preprocessing (windows 3/5/9) | gate only | **killed at the quick gate**: false alarms grow with the window — the filter shrinks the fitted scale and manufactures serial dependence |
| — | 011b: asinh as replacement preprocessing | 0.5689 vs 0.5719 | **killed**: the quick-gate gain was score compression flattering an untrained model; retraining consumed it |
| 011 | every channel twice — raw and asinh-compressed pipelines side by side (100 channels) | **0.5746** vs 0.5719, ahead on 3 of 5, losses ≤0.001 | cloud |
| — | 012: explicit reversion-depth channels (now − peak, 42 diffs) | 0.5754 vs 0.5746 | **killed by pre-stated condition**: better on 2 folds of 5 — the trees already extract recession from the peak/now pairs |
| 013 | + a forecaster: LightGBM predicts each next deviation-from-trailing-mean, pretrained on all histories, finetuned per series; the standardised prediction error is the channel (104 total) | **0.5779** vs 0.5746, ahead on 4 of 5 | cloud |
| — | 014: the 50-channel pipeline on the deviation view | 0.5762 vs 0.5779 | **killed**: the forecaster already mines that representation |
| — | 015: forecaster horizons 1+5 and signed error | 0.5751 vs 0.5779 | **killed**: the compact four channels carry the signal, extensions dilute |
| — | 015b: magnitude forecaster (predict the size of the deviation) | 0.5782 vs 0.5779 | deferred: +0.0003 is noise; a candidate for the neural-ensemble stage |
| — | combiners: weighted sum 0.5608, plain maximum 0.5401 | — | boosted trees unchallenged |
| 016 | peak-hold on the output: fast attack, slow release (drain 0.1%/step); channels underneath keep fast reversion | **0.5799** vs 0.5779, ahead on 4 of 5 | cloud |
| — | 017: TCN pilot, 16.5k params, 8 epochs | fold-0 0.532 vs stack 0.587 | concept learns, not yet competitive — needs ranking loss, augmentation, capacity |
| 018 | + a per-prefix battery: 42 drift-free two-sample statistics (history vs prefix) recomputed on a geometric cadence, both views; peak-hold softened to 0.995 | **0.5814** (0.5807 raw) vs 0.5779, ahead on 4 of 5 | cloud |
| 020 | + battery v2: forty more two-sample statistics (CvM, Anderson–Darling, Levene, increments, windows 10/25/100, tail exceedance, signs, slope t-stats); hold sped to 0.99 | **0.5821** (0.5817 raw) vs 0.5814, ahead on 3 of 5 | cloud |
| 021 | hyperparameter resweep at full width: 63 leaves, colsample 0.5 — each tree sees a different half of the channels | **0.5841** vs 0.5821, ahead on 4 of 5 | cloud |
| — | 024: gain-importance pruning (top-150/120/90) | 0.5825/0.5813/0.5805 | **killed**: monotone degradation — colsample already mines the weak channels |
| 022 | the ranking objective: lambdarank, groups = per-step cross-sections, truncation 2000 | **0.5881** raw, ahead on all 5; fold 0 crosses 0.60 | shipping |
| — | 023: TCN v2 — ranking loss, boundary augmentation, 75k params, 24 epochs | fold-0 0.538; ensemble with the stack flat (+0.0002 at 10%) | concept survives, needs scale: ensemble value starts at ~0.55+ |
| — | 025: ranker truncation 500/8000, resweep params for the ranker | fold-0 0.5919/0.5986; 0.6006 unchanged | 2000 stays; the ranking loss regularises itself |
| 022b | shipped blend: 0.6·sigmoid(ranker) + 0.4·classifier, no hold — the ranker carries the memory | fold-0 **0.6035** (ranker 0.6006, classifier 0.5952) | cloud |
| — | 027 seed-ensembled rankers / 029 stacking over OOF scores | 0.6031 / 0.5991 | **killed**: sharpness beats smoothing; two correlated level-0 models give a meta-layer nothing |
| 028 | the ranker lengthened to 600 trees at lr 0.03 in the blend (#16) | fold-0 **0.6045** | cloud |
| — | 030: Chronos-2 error/interval channels (zero-shot, whitelisted) | solo 0.5597 (interval width); blend +0.0004 | **killed**: the finetuned forecaster and variance detectors already tell the trees everything it knows |
| — | 031: online-anchor view (fold-1 gallery diagnosis) | 0.6094 → 0.6047 after removing length leakage | **killed**: the apparent gain was length leakage entirely; the diagnosis stands, the cure does not |
| — | 032: interaction channels (external difference × silent internal scan) | solo 0.6016, blend 0.6042 | **killed**: the trees already extract it; the fold-1 disease is marked resistant |
| — | 014: the 50-channel pipeline on the deviation view | 0.5762 vs 0.5779 | **killed by pre-stated condition**: the forecaster already extracts this representation |
| — | 015: forecaster horizons 1+5 and signed error | 0.5751 vs 0.5779 | **killed by pre-stated condition**: the extensions dilute the compact four channels |
| 016 | peak-hold on the output: the score rides its running maximum, draining 0.1%/step | **0.5799** vs 0.5779, ahead on 4 of 5 | cloud |

The augmentation pipeline (`augment.py`) survives its first product; the next
target, per the inspection notebook, is cross-sectional discrimination — clean
series carrying high scores — not another within-series channel.

## Reproducing

    pip install crunch-cli
    crunch setup structural-break-real-time <name> --token <token>
    cp submissions/001-classical-detectors/main.py <workspace>/
    cd <workspace> && crunch test

## Relationship to the organisers' quickstarter

The EWMA baseline published by CrunchDAO is used as a reference point and is
reimplemented in `src/structural_break/evaluate.py` so that both can be scored
on the same rows. No other code is taken from it.
