# adia-structural-break

Solutions for the [ADIA Lab Structural Break Challenge: Real-Time
Edition](https://hub.crunchdao.com/competitions/structural-break-real-time)
(CrunchDAO, May–October 2026; the deadline was extended from 17 September to 1 October).

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

## Where it stands

Five weeks, a hundred logged experiments, thirty-three submissions. The
platform moved four times, each time for a different reason:

| Cloud | Submission | What moved it |
|---|---|---|
| 0.5877 | #18 | the channel–trajectory networks joined the trees |
| 0.5893 | #23 | a new channel family — fourteen spectral bands |
| 0.6004 | #28 | boundary augmentation, tripled: pseudo-series with early breaks |
| 0.6007 | #30 | a run-length posterior for the classifier, twelve networks |

Everything else measured on the untouched fold since #28 — more members,
more or other data, capacity, schedule, loss, input, verifiers, stacking
in the winners' form, the sibling competition's ten thousand series — sits
within ±0.001 of the recipe. The strongest single signal in the data, the
position of a series inside its own online part, is exactly what the
real-time protocol withholds. The record of what was tried, in what order,
and why each was kept or killed is the point of this repository as much as
the score.

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
| 078 | the #28 recipe with nets retrained on every series (fold 2 included) | no local number by construction | shipped as **#29** — cloud **0.5996**, no difference from #28 (0.6004); full data does not move the nets |
| 079 | nine boundary slices per series (80k pseudo-series, 8:1 to originals) | fold-2 solos 0.5824 / 0.5748 vs 0.598–0.610 for the triple | killed — the volume axis has an optimum; pseudo-series drown the originals |
| 079b | nine slices at the triple's volume per epoch (variety at constant ratio) | three members 0.6053 vs 0.6092 for the triple's first three; no gain on #28 | killed — the ratio was the thing, not the variety |
| 080 | ranking-loss batch 96 instead of 24 | 0.5939 / 0.5922 / 0.6004 at 10 epochs; 30 epochs memorise (holdout 0.7255, fold-2 0.5678) | killed — and the leaky holdout is found: pseudo-series of holdout originals were in training |
| 081 | clean holdout; per-epoch curve on fold 2 | fold 2 rises monotonically to the last epoch (0.6043); the clean holdout peaks at epoch 0 and falls | best-epoch selection is noise — take the last epoch; 082 tests 15 epochs |
| 082 | fifteen-epoch cosine, last epoch, no holdout | 0.5936 vs 0.6036 for ten epochs, same seed | killed — past ten epochs the updates memorise; 082b runs ten epochs on the six #28 seeds |
| 082b | last-epoch rule on the six #28 seeds | five of six members improve, ensemble 0.6157 vs 0.6163; twelve together 0.6163 | neutral — the recipe's ceiling is ≈0.616 fold-2; #28 stands |
| 083 | ranking loss with the metric's step weights (n_pos·n_neg) | 0.6049 / 0.6049 / 0.6031 vs pairs 0.6057 / 0.6065 / 0.6051 | killed — slightly worse every time |
| 084 | raw series as two extra input channels (z against history, asinh) | 0.5937 / 0.6036 / 0.6011 vs pairs 0.6057 / 0.6065 / 0.6051 | killed — worse on every seed |
| 085 | Bayesian online change-point channels (run-length posterior, hazard 1/50) | classifier +0.0046, ranker −0.0014, nets worse; blend **+0.0005** (0.6169) | kept for the classifier only (+0.0005 in the blend); not a submission on its own |
| 086 | GRU (recurrent) members on the same trajectories | no number — twelve hours on MPS did not finish one member | deferred to CUDA |
| 087 | early specialists — ranking loss on the first 200 steps only | early-step AUC 0.5489 vs 0.569–0.576 for ordinary members | killed — the early region is information-limited |
| 088 | #28 + BOCPD suffix for the classifier + twelve nets | fold-2 **0.6169** vs 0.6163 | shipped as **#30** — **cloud 0.6007**, project best (+0.0003 over #28) |
| 089 | #30 with the blend at 0.55 trees / 0.45 nets | fold-2 tie (0.6162 vs 0.6163) | shipped as **#31** — cloud **0.6000** vs 0.6007 for #30: the platform does not prefer the trees; blend stays 0.45/0.55 |
| 090 | the online-length prior: tau is uniform in the online part, (t+1)/L alone reads 0.6288 on fold 2 | unobservable — `x_online` is a length-less generator, and L is unpredictable from history (R² ≈ 0) | closed, and recorded |
| 091 | the 2025 edition's 9,948 series as extra data: mixed into nets, into the ranker, and as pretraining | nets 0.6020/0.5940 and 0.6047/0.6012 vs pairs 0.606; ranker 0.6043 → 0.5882 | killed in all three forms — a shifted domain, its volume buys nothing |
| 092 | entry/exit over the score trajectory: peak-hold, reset rules, learned verifiers (trees, linear) | all below the raw score: rules ≤ 0.6163, verifiers 0.579–0.610 vs 0.6167 | killed — the score already carries its memory |
| 093–095 | history-shape prior (AUC 0.511, closed); self-normalisation by the early baseline (+0.0005–0.0013) | fold-2 0.6172 | shipped as **#32** — cloud **0.5991** vs 0.6007 for #30: self-normalisation does not transfer |
| 096 | change-point filter with an AR(1) observation model + P(break occurred) | classifier −0.0040 / −0.0033 | killed — the posterior accumulates with time, not evidence |
| 097 | the 2025 winners' divergence block (JS, Hellinger, Wasserstein, entropy) as streaming channels | classifier −0.0056, ranker −0.0008 | killed — redundant with the two-sample battery on a growing prefix |
| 099 | stacking the winners' way: 11 block models, 7 model families, an OOF ranker, four kinds of meta | best meta in the blend 0.6161 vs 0.6167; as the whole tree half 0.6140; +0.0007 from averaging the classifier with wide LightGBM and ExtraTrees | closed — level-0 members read the same channels; the hand blend stands |
| 100 | #32 with the classifier slot as the standard + a wide LightGBM | fold-2 **0.6184** vs 0.6180 | shipped as **#33** — cloud **0.5991**, identical to #32: the wide classifier is a null |
| 101 | #33 with twenty-four networks (six full-data, six pool added) | fold-2 unchanged by construction | shipped as **#34** (resources079); carries #32's self-normalisation, so expected ≈0.599 — not yet run |
| 102 | positive rows weighted by post-break evidence, min(1, (t − tau + 1)/W) | classifier 0.5999 unweighted vs 0.5942 / 0.5999 / 0.5988 / 0.5968 for W = 20 / 50 / 100 / 200 | killed — no W beats the unweighted classifier |
| 104 | tree specialists per step range (t < 50, 50-200, 200+) | 0.5957 / 0.5991 vs 0.5999; the early specialist alone +0.012 on t < 50, hybrid blend 0.6168 vs 0.6167 | killed — the early gain dissolves in the blend |
| 103 | nets with the first 15 post-break steps masked out of the loss | 0.6061 / 0.6021 / 0.6005 vs paired 0.6057 / 0.6065 / 0.6051 | killed — one tie, two losses; the post-break rows are not label noise |
| 105 | mirrored series (x → −x) and a backward-shifted boundary as extra training data | mirrors 0.6030 / 0.6024 / 0.6013, backward 0.6059 / 0.6009 / 0.6042 vs paired 0.6057 / 0.6065 / 0.6051 | both killed — the augmentation axis is exhausted |
| 106 | #30 with twenty-four networks instead of twelve, nothing else changed | fold 2 cannot separate them | shipped as **#35** — the pool size measured on its own |
| 107 | a 511-step receptive field (8 blocks) instead of 127 | 0.5906 vs 0.6057 paired, killed after one member | reach hurts — 127 steps is what the channels make meaningful |
| 108 | a network on the raw signal alone (2047-step field, no channels) | solo 0.5329, Spearman 0.60 with the ensemble, every blend share negative | killed — a worse view of what the channels already compute |
| 109 | anatomy of the blindness: what each break changes vs the ensemble's per-series AUC | variance breaks 0.69, dependence 0.58, and 40% of breaks change nothing measurable — AUC 0.542 there; nineteen further statistics separate them no better than a random cut | the ceiling is the task's, not the recipe's |
| 110 | direct autocorrelation-vs-history channels (lags 1/2/5/10, two windows) | classifier 0.5954 vs 0.5999 | killed — the dependence gap is not a missing channel |
| 111 | the near-null history prior on the earliest steps | ensemble 0.5373 vs prior 0.5385 over steps 0-30; blending buys +0.0006 | not shipped — below the fold's trust threshold, but it dates the blindness |
| 112 | full GLR scan over every candidate break position (6 channels, 0.02 ms/step) | classifier 0.5974 vs 0.5999; alone 0.5569; blended +0.0005; argmax position reads 0.4994 | killed — works as a detector, adds nothing in the cross-section |
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

### Shipping a submission

    scripts/ship_submission.sh 078-two-classifiers interface_078.py meta_078.txt ../resources078 "message"

Assembles from the library, refuses a channel-width mismatch, verifies the
assembled monitor against the training matrices with the verifier that
lives next to the interface, profiles milliseconds per step, and only then
pushes. Nothing reaches the platform unverified.

### Tests

    PYTHONPATH=src python -m unittest discover -s tests -v

Standard library only. The tests guard the two places where a submission
can silently disagree with its training: the streaming change-point monitor
is checked against the batch filter that built the training channels, and
the assembler's channel check is exercised on fake artifacts — including
the exact mismatch that killed submission #25.

## Relationship to the organisers' quickstarter

The EWMA baseline published by CrunchDAO is used as a reference point and is
reimplemented in `src/structural_break/evaluate.py` so that both can be scored
on the same rows. No other code is taken from it.
