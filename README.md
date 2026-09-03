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
| — | 034: channel-velocity features for the trees | blend 0.6045 (baseline) | **killed**: hand-picked derivatives add nothing |
| 033 | a channel-trajectory TCN (111k params, ranking loss) joins the blend at weight 0.40 (#18; #17 is the pair with the timeout fix) | fold-0 solo 0.5985, triple **0.6068** | cloud |
| — | 035: the channel net tripled (314k params, RF 511) | best 0.5940 vs 0.5985 | **killed**: data-bound, not capacity-bound |
| — | 036: seed-ensembled channel nets | 0.5933 vs best single 0.5939 | **killed**: the nets converge to the same solution; averaging is dead in this project |
| — | 037: the net fed channels plus both combiners' scores | peak 0.5915 vs 0.5985 | **killed**: the scores are functions of the channels — capacity spent rediscovering them |
| — | cloud calibration | #18 scored **0.5877** in 30 min | fold-0 − 0.019 ≈ cloud; the 0.65 target means fold-0 ≈ 0.67 |
| 038 | a fold-ensemble of four channel nets (diversity through withheld folds, not seeds) at a 0.50 blend weight | net solo 0.6004, triple 0.6090 | folded into #19 |
| 039 | fold-bagged rankers join: 0.35 bagged rankers + 0.15 classifier + 0.50 fold-ensembled nets (#19) | ranker bag solo 0.6033, ensemble 0.6099; **cloud 0.5877 — identical to #18**: fold-0 exhausted as a ruler, protocol amended | cloud, rank 215 |
| — | 043: long augmented training (48 epochs, noise, channel dropout) | peak 0.5993 vs 0.5985 | **killed**: overfits through the augmentations; the 3080 programme is many short nets on diverse subsamples |
| — | 040b: stacking rerun on clean members (after the leak) | meta 0.6080 vs fixed weights 0.6099 | **killed**: the meta-layer loses to weighted averaging, twice attempted |
| — | 042: TabPFN | OOM-killed on CPU ×3 | deferred to the GeForce-3080 kit (scripts/gpu/) |
| 044 | ablation shipped as #20: the four channel nets alone, no trees | fold-0 solo 0.6004, **cloud 0.5846** (vs 0.5877 for the full #19) | the net line transfers better than the trees |
| 045 | the twelve-net bag (random-60% subsamples, batched streaming forward) at weight 0.7 (#21) | fold-0 **0.6201**, fold-1 **0.5918** — both folds verified | cloud |
| — | 046: scaling the bag to 72 mixed members | 12→72: 0.6152→0.6076 (fold 0) | **the curve points down**: the bag saturates at twelve; member quality dominates count — the GPU day retargeted to stronger members |
| — | the #21 cloud read | fold-0 0.6201 / fold-1 0.5918 → **cloud 0.5810** | both local folds burnt by selection; the paradigm shifts to few large bets validated in the cloud |
| 047 | eight heavy members (88% of all data, 16 epochs, private holdouts) at #19's exact weights (#22) | **cloud 0.5876 vs 0.5877** | the family's ceiling measured at ≈0.588: count, weights, and member quality all exhausted — a new representation family is required |
| — | 048: the hybrid two-tower (channels + raw series) | holdouts 0.6035/0.5589/0.5740/0.5938 | **killed at laptop scale**: the raw tower adds variance, not signal; revisit only pretrained at 3080 capacity |
| — | 049: self-supervised pretraining of the channel backbone | finetuned members 0.5887/0.5943/0.5874 vs ≈0.60 from scratch | **killed**: the backbone clings to forecasting and resists ranking; the laptop era closes |
| — | 050: metric-aligned training weights (pair-proportional) | 0.5912 / 0.6003 vs 0.5952 / 0.6016 | **killed**: equal-step weighting regularises; the 48× mismatch was not the problem |
| — | 051: history-level calibration of the score | 0.5577 subtracting, +0.0007 adding | **killed with a reversal**: the history level predicts breaks *positively* (0.5297 solo) |
| 053 | the spectral family: 14 frequency-domain channels (#23) | fold-0 pair 0.6060; **cloud 0.5893 vs the 0.5877 ceiling** | the ceiling breaks — the first new modality since the forecaster transfers exactly |
| — | 054: spectral v2 (windows 32/128, increment spectra, band balance) | 0.6026 / 0.6053 vs v1's 0.6032 / 0.6060 | **killed**: v1's fourteen channels are the family's optimum |
| 055 | trees alone on the 200 spectral channels, no nets (#24) | **cloud 0.5853** vs #23's 0.5893 | the nets are worth +0.0040 in battle; nothing to simplify away |
| 056 | the spectral gain re-measured on fold 2, untouched by any selection | 0.5912 → **0.5967 (+0.0055)** | three independent readings agree; folds 3–4 held in reserve as unspent rulers |
| — | 057–060: wavelets, rank tests, matched filters | 0.5933 / 0.5929 / 0.5815 vs 0.5979 | **all killed**: at 200 channels hand-built families dilute |
| 062 | ranker resweep at 200 channels (63 leaves, colsample 0.5) | 0.5922 → **0.5956** | adopted |
| 065 | augmented training data: history boundary moved right, breaks land earlier | tree pair 0.5999 → **0.6043** | adopted — the largest single gain of the day |
| 066–069 | augmentation splits the combiners: rankers and nets train on augmented data (best ranker 0.6029, best aug net 0.6286), the classifier on clean | **fold-2 0.6106** vs 0.5979 for the configuration that scored 0.5893 | shipped |
| 071–072 | shipped: #26 (six selected nets) and #27 (augmented + clean rankers, clean classifier, eight nets) | fold-2 0.6106 | cloud |
| 073 | ten augmented nets finish; best member 0.6473 (project record), weighted averaging tested | fold-2 **0.6116** — +0.001 over the shipped eight | the net half is saturated at this recipe |
| 074 | third ranker breed (127 leaves, truncation 1000) + joint weight sweep | solo 0.5960, any share degrades the blend; sweep plateaus at 0.6104–0.6116 | killed — laptop-scale levers exhausted |
| 075 | six diff nets (600 inputs) trained on augmented data | fold-2 **0.6120** with all six added (12 plain + 6 diff) | diversity helps, member holdouts don't transfer; awaits a bigger gain to ship |
| 076 | six plain nets trained on the **triple** boundary augmentation (26,887 pseudo-series) | fold-2 **0.6163**; best single member 0.6102 solo, six nets alone 0.6108 | shipped as **#28** — **cloud 0.6004**, project best (was 0.5893) |
| 076b | six more aug3 members (holdouts 0.62–0.66) | twelve averaged: 0.6147 < 0.6163 for the first six; holdouts anti-correlate with fold 2 | #28 stands; holdouts unreliable for ranking members |
| 077 | 96-channel nets on the triple augmentation (capacity hypothesis) | solos 0.597–0.600, three together 0.6048 vs 0.6108 for #28's six | killed — capacity is not the constraint |
| — | #25: broken upload, do not run | — | a `channels[:186]` slice survived the spectral upgrade; the push followed the check with `;` instead of `&&` |
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
