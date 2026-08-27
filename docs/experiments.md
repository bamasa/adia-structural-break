# Experiment log

Every submission, the hypothesis behind it, what it scored, and what was learned
— including the failures, which is where most of the learning is. Newest at the
bottom. The convention follows the trading project this grew out of: the kill
condition is written before the test, and a result only counts on data the
choice never saw.

The metric everywhere is Time-Stratified AUC (TS-AUC): at each online step, an
ordinary AUC across all series alive at that step, weight-averaged by the
number of positive-negative pairs. Two consequences drive every design choice:
only the *ordering between series at the same step* matters, and early steps —
where almost no evidence exists — weigh as much as late ones.

Reference points: random = 0.50, the organisers' EWMA baseline = 0.5170 locally,
public leaderboard top ≈ 0.652, tail of the ranked field ≈ 0.62–0.63.

---

## 001 — three classical detectors, combined by maximum

**Hypothesis.** CUSUM (level shifts), Page-Hinkley (small persistent shifts) and
a variance ratio (spread changes) cover complementary break types; their
maximum, properly normalised, beats the EWMA baseline without any learning.

**Design decisions that mattered.**
- Winsorising at 4 SE: without it the variance detector scored AUC 1.00 on a
  *spike that reverts* — perfectly detecting a non-break. With it, 0.60.
- Null normalisation per detector, with the growth law measured by simulation:
  CUSUM's running maximum grows like 0.84·log(n), Page-Hinkley's like 1.0·√n.
  One law for both leaves a length effect, and the metric compares series of
  different lengths directly.
- Logistic squashing centred at the null — never a clip, because ties are
  exactly the comparisons an AUC is made of.

**Result.** Local TS-AUC **0.5385** (baseline 0.5170). Cloud: accepted,
mid-table (~400 of ~1500 on first run).

**Learned.** The classical families work but plateau: their breaks — mean and
variance — are apparently the minority here. Shape and dependence changes need
either more channels or learning.

---

## 002 — learned combiner, first attempt: **not submitted, and the reason matters**

**Hypothesis.** Ten thousand labelled series can replace the assumed maximum
with a fitted weighting over nine channels (three detectors × raw / whitened /
absolute views).

**What happened.** Grouped 5-fold CV said **0.7487** — above the public
leaderboard's first place. The organisers' held-out 100 series said **0.4990**
— worse than coin flipping. Both numbers were real; the first was fabricated by
the pipeline.

**The defect.** Training rows were subsampled on a geometric grid whose spacing
depended on the series' length. At any given step, the training set therefore
contained a *length-dependent selection* of series rather than all of them —
and the metric compares series against each other at the same step, so the
model learned to rank on "who got sampled here", which the real evaluation
never reproduces.

Same defect class as the trading project's clustered-significance mistake:
validation constructed differently from evaluation. The subsampling is gone;
everything below trains on every step of every series.

**Also learned, and kept.** Context features describing the *series* (history
length, dispersion, kurtosis, autocorrelation) and the step index actively
destroy the combiner — 0.7487 → 0.5049 on the same (flawed) CV, reproduced on
the honest one. The metric compares series at the same step: ranking by series
properties is ranking on something orthogonal to whether a break occurred, and
the step index is constant within a comparison group. The combiner sees the
nine channels and nothing else.

---

## 003 / 004 — weighted and boosted combiners, honest validation

**Hypothesis.** With the sampling defect removed: (a) a fitted linear weighting
beats the maximum; (b) boosted trees, which can make one channel's weight
conditional on another, beat the linear form.

**Validation.** All 5,036,517 rows (no subsampling), grouped 5-fold by series,
rows weighted so each step index contributes comparably — the metric averages
per-step AUCs equally while a long series floods the late steps.

**Result (5-fold, per-fold spread):**

| combiner | mean | folds |
|---|---|---|
| maximum (001's rule) | 0.5091 | 0.500–0.524 |
| weighted (logistic on logits) | 0.5293 | 0.523–0.539 |
| **boosted (LightGBM)** | **0.5568** | 0.552–0.564 |

Ordering is consistent on every fold. Both learned combiners beat the maximum;
the trees beat the line, so the conditional structure is real, not variance.

**Local held-out check before submitting** (100 series, ±0.05 at that size):
weighted 0.5232, boosted 0.5146 — both consistent with their CV numbers, no
repeat of 002's fabrication.

**Status.** Submitted: 003 = submission #2, 004 = submission #3. Cloud scores
pending.

---

## 005 — retrospective scan channels

**Hypothesis.** The streaming detectors carry one number of state and cannot
change their mind. But the platform's constraint is one-directional: at step t
everything up to t may be re-read freely. A full-rescan detector — for every
candidate split of the seen segment, compare the two sides — buys two things
streaming cannot have: retrospective placement (a break at 100 noticed at 120
is *found at 100*) and recantation (a spike that looked like a break stops
being the best split once calm data follows).

**Channels.** Maximum over splits of: standardised mean gap (CUSUM-type scan);
log variance ratio; and Kolmogorov-Smirnov of the online segment against the
*historical* empirical CDF — the shape channel the classical families lack.
Prefix sums make a scan O(t); scans run on a geometric schedule with the score
carried between them, so the whole series stays O(t·scans).

**Calibration, simulated as always.** Mean-scan null grows as 1.02·√(log t)
(measured constant 1.06→0.99 across t = 30..1000); variance-scan null is flat
at 1.10 — the first guess of 2.6/√t + 0.35 sat far below it, saturating the
channel at 0.97 on *quiet* series, which would have made every comparison a
tie; KS null grows as 0.37·√(log t).

**Sanity checks passed.** Break at 100 seen from 120: scan_mean = 0.90. Lone
spike: all three scan channels at 0.27–0.33, *below* the quiet series — the
recantation property, which no streaming statistic has.

**Kill condition, stated before the run:** if the 12-channel boosted combiner
does not beat the 9-channel one on grouped CV, the scan channels carry nothing
the stream did not, and 005 is not submitted.

**Result: killed.** 12-channel boosted CV 0.5570 against 9-channel 0.5568 — a
gain of 0.0002 where the fold spread is ±0.006. The scans alone reach 0.52 as a
maximum, so they do detect; what they detect, the streaming channels already
carry. Not submitted. The class stays in the library: a convolutional model
over the scan surfaces is the natural next attempt, and it starts from these
channels.

---

## 005 / 006 — forty channels, and a convolution that earns its keep

**Hypothesis.** (a) Multi-scale receptive fields (six EWMA windows, 5–200
observations) and retrospective confirm/cancel verdicts carry break types the
nine streaming channels miss; (b) a small learned convolution adds shapes
nobody hand-named.

**The retrospective verdicts.** A best-split ladder at trailing depths
{10, 20, 50, 100}, plus two channels built to *remove* evidence: the stability
of where the best split lands (a real break pins it; noise wanders it —
measured 0.97 vs 0.01 on synthetic cases), and the calmness of the tail after
the split, which withdraws a spike-induced alarm once ordinary data follows.

**Result (grouped 5-fold CV):** 9 channels 0.5568 → 40 channels **0.5648**
(sweep of 12 hyperparameter configurations, selected by mean minus fold
spread) → 41 with the CNN channel **0.5682**, ahead on 4 folds of 5. The CNN
alone: 0.5315 — weak as a detector, useful as a channel. Both submitted
(#4, #6), plus the 004 cloud-crash fix (#5: requirements.txt with lightgbm,
INFER_PARALLELISM=8 — the first run had neither and died importing the model).

**Inspection notebook** (notebooks/model_inspection.ipynb, generated by
scripts/build_analysis_notebook.py): on the labelled hundred, median
within-series AUC on broken series is **0.983** — within a series the model
separates before from after almost perfectly. The loss is *cross-sectional*:
clean series carry a median final score of 0.317, so a quiet-but-jittery clean
series can outrank a genuinely broken one. That is the number to attack.

---

## 007 — planned: a break classifier on augmented windows

**Hypothesis.** The training set holds ~5,000 true breaks; a window-level
classifier can be fed hundreds of thousands of examples instead by cutting:
negatives from break-free stretches (histories, and online segments before
tau, with thinning for length variety); positives as windows straddling a true
tau at varying offsets (with resampling), so the break sits at a known,
varying position inside the window. Train a classifier on "does this window
contain a break", use its suffix-scan as a channel or standalone score.

**Why it might beat the current combiner.** Every current channel compares
"now" against "the history". The classifier sees the joint shape of both sides
at once, on vastly more examples than 10,000 — and the inspection notebook
says the weakness is discrimination between series, which more training
signal addresses directly.

**Kill condition, stated before the build:** grouped CV against the 41-channel
combiner; if adding (or replacing with) the classifier does not clear 0.5682 by
more than the fold spread (~0.006), it is not submitted.

**Result: killed — and the shape of the failure is the lesson.** The
augmentation worked exactly as intended at the window level: 313,119 windows
(controlled break offsets at 15–85% of the view, stride thinning at 1/2/4,
negatives from histories), and the classifier's held-out window AUC rose from
0.5315 to **0.5687** — nearly double the learnable signal of the 006 network.
As a standalone TS-AUC channel it reaches 0.5353, also better.

And the combiner got *worse*: 41 channels 0.5682, 47 channels 0.5627, behind on
every fold. Six new channels correlated with the forty already there diluted
the trees' signal rather than adding to it. A feature can be genuinely better
in isolation and still carry nothing the ensemble does not already have — the
same verdict the retrospective scans met, now measured on a channel that cost a
full augmentation pipeline to build. The pipeline stays (augment.py); the next
use of it should target what the inspection notebook says is actually missing —
cross-sectional discrimination — rather than another within-series channel.


---

## 008 — reverting channels: the current statistic beside every peak

**Hypothesis** (raised by looking at the score panel of the inspection
notebook: the purple line climbs and never comes back). Every streaming
channel reports its detector's *running peak* over the null — the right
semantics for "has a break already occurred", but a false alarm becomes
permanent: one bad stretch and a clean series outranks real breaks at every
later step. The metric is cross-sectional per step, so a score that cannot
come back down keeps paying for the same mistake until the series ends. Give
the combiner each detector's *current* statistic too — CUSUM's reflected sums
drain through the drift term, the variance EWMA forgets on its own — and the
trees can learn "peak high, current low, long since: discount".

**Design.** Nine new channels (`*_now`, one per view × family), the current
statistic under the same null normalisation and squash. No channel removed:
the peak stays correct for true breaks, the pair is what carries information.

**Kill condition, stated before the run:** grouped 5-fold CV, 41 vs 50
channels on the same folds; adopted only if better on most folds.

**Result: adopted — better on all five folds.** 41 channels 0.5662, 50
channels **0.5719** (+0.0057 mean; the widest fold gains +0.0120). First gain
since 006, and it came from the failure mode the notebook diagnosed: false
alarms that could not be recanted. Submitted as #8.

---

## 009 — planned: the rank view

**Hypothesis** (imported from the public 2nd-place solution of the first,
offline edition — segment tests and robust transformations were its core).
The z-view calibrates detectors against a unit Gaussian; a genuinely
heavy-tailed clean series lives in that view's tails and false-alarms
forever (measured: final raw CUSUM 0.919 on a calm t(2.5) stream). Map each
online observation to its midrank within the standardised history, then
through the probit: under the null the result is N(0,1) *by construction*,
whatever the noise distribution — the same calm t(2.5) stream ends at 0.653.
Six channels: three detector families on the rank view, peak and current.

**Kill condition, stated before the run:** grouped 5-fold CV over the
41-channel base on the same folds; adopted only if better on most folds (and
re-checked over the 50-channel base before entering a submission).

**Result: killed at the second gate — and the pair of numbers is the story.**
Over the 41-channel base the rank view helps: 0.5684 vs 0.5662, ahead on 4
folds of 5. Over the 50-channel base it does not: 0.5707 vs 0.5719, three
narrow wins (+0.002 and less) against one loss of −0.0074. The false alarms
the rank view was built to suppress are the same ones the reverting channels
of 008 already recant, and what remains of it is dilution. Same verdict as the
retro scans and 007, with a sharper mechanism this time: two fixes for one
disease do not stack. The module stays (rankview.py) — it is the better of the
two fixes for any future channel set that lacks 008's.


---

## 010 — killed: series context as features

**Hypothesis.** The combiner never sees what kind of series it is watching:
the history's autocorrelation, kurtosis, length and scale, and the step index
were features of the early 15-column set but never entered the 40-channel
build. Handing them to the trees is the cheap version of a mixture-of-experts:
a split on kurtosis *is* a per-type model choice.

**Kill condition, stated before the run:** grouped 5-fold CV, 50 vs 56
channels on the same folds; adopted only if better on most folds.

**Result: killed — worse on 3 folds of 5** (0.5681 vs 0.5719), and the fold
spread widened from ±0.005 to ±0.011 in both directions. The context lets the
model memorise which series types break in the training folds rather than how
breaking looks; the channels themselves are already conditioned on the history
(every threshold is calibrated per-series), so the context arrives
pre-consumed, and what is left of it is an overfitting surface. A true gated
mixture over series types would need to beat this bar first.


---

## 011 — preprocessing: rolling median killed, asinh promoted to a full run

**Hypothesis (raised looking at spike-driven score jumps).** The detectors
treat a lone spike as the start of a break; a causal rolling median (windows
3/5/9) should erase spikes before anything sees them. A log-like squash
(asinh, defined on negatives) is the smooth alternative.

**Quick gate, before any retraining** — the 008 model applied unchanged to
transformed input, over the labelled hundred (median within-series AUC on
broken / median final score on clean, lower better):

| variant   | AUC broken | false score |
|-----------|-----------|-------------|
| current   | 0.973     | 0.320       |
| median 3  | 0.977     | 0.417       |
| median 5  | 0.981     | 0.466       |
| median 9  | 0.967     | 0.516       |
| asinh     | **0.985** | **0.249**   |

**Median: killed at the gate.** False alarms grow monotonically with the
window (0.32 → 0.52). The filter smooths the *history* too, shrinking the
fitted scale, and its overlapping windows manufacture serial dependence that
the cumulative detectors read as drift. The spike it was built to erase is
already handled by the kurtosis-widened winsor. Figure:
`scripts/preprocessing_demo.py`.

**asinh: promoted.** Better on both axes simultaneously, on a model that
never saw transformed input. Full run: rebuild all 50 channels on
asinh-transformed series, retrain, grouped 5-fold CV against 0.5719 on the
same folds; adopted only if better on most folds.

**Full run, two verdicts.** As a *replacement* — killed: 0.5689 vs 0.5719,
better on 2 folds of 5 with wild spread. The quick gate's gain was score
compression flattering an untrained model, and retraining consumed it: a gate
run without retraining can only suggest, never adopt. As a *union* — adopted:
all 100 channels (raw fifty + asinh fifty), **0.5746 vs 0.5719**, ahead on 3
folds of 5 with the two losses at −0.001 or less. First addition to survive
the dilution that killed the retro scans, 007 and 009 — because the
compressed pipeline is not derived from the raw one: its normalisation,
trend, AR coefficient and thresholds are all fitted on the compressed series,
so it disagrees with the raw view exactly where heavy tails mislead one of
them. Submitted as #9.


---

## 012 — planned: explicit reversion-depth channels

**Hypothesis** (from the cancellation report, scripts/cancellation_report.py:
across the labelled hundred the median peak-to-final drop on clean series is
just 12%, and strict cancellations — score below half its peak — number three.
The recanting works, but timidly). Trees split on one feature at a time and
cannot subtract, so "how far has the evidence receded from its peak" must be
assembled indirectly. Hand it over directly: now-minus-peak differences for
all nine detectors and current-minus-peak for all twelve EWMA scales, in both
views — 42 channels, all computable from the existing matrices.

**Kill condition, stated before the run:** grouped 5-fold CV, 142 vs 100
channels on the same folds; adopted only if better on most folds.

**Result: killed — better on 2 folds of 5** (0.5754 vs 0.5746; +0.0065 and
+0.0028 against three small losses). The mean nudges up by less than 0.001:
whatever "distance from peak" carries, the trees were already extracting from
the peak/now pairs indirectly, and the explicit subtraction adds noise on as
many folds as it helps. The cancellation timidity the report measured is
real, but this was not the lever.

---

## 013 — planned: prediction-error channels (pretrained + per-series forecaster)

**Hypothesis.** Every current channel asks "does the stream look like its
history's statistics". A forecaster asks the stronger question: "is the next
value *predictable* from the recent past the way the history was". The series
is standardised, turned into deviations from a trailing mean (window 10) —
levels are discarded, as in the trading pipeline — and a small LightGBM
regressor predicts each next deviation from its lags: pretrained on all
histories, then finetuned on each series' own break-free history, where its
residual scale σ is measured. Online, |actual − predicted|/σ is the channel:
it rises while the series stops being predictable and falls back once it is
again — reversible by construction, an organic cancel/re-arm. Four channels:
fast EWMA, slow EWMA, running peak, instantaneous.

**Kill condition, stated before the run:** grouped 5-fold CV over the current
base on the same folds; adopted only if better on most folds.

**Result: adopted — better on 4 folds of 5.** Standalone the channels are the
strongest single signals the project has produced: the running peak of the
prediction error reaches TS-AUC 0.5586 alone — a whisker from the full
nine-channel model of 004 (0.5568), far above the CNN (0.5315) and the 007
classifier (0.5353). In ensemble: 100 channels 0.5746 → 104 channels
**0.5779** (+0.0070, −0.0018, +0.0022, +0.0035, +0.0057). Unlike 007 it
survives dilution because it asks a question no statistic channel asks:
whether the *dynamics* remain forecastable, not whether the *distribution*
matches. Submitted as #10, with the pretrained forecaster shipped in the
artifact and the per-series finetune running at inference time.


---

## 014–016 — planned: the combination sweep

**014 — the deviation view.** The full 50-channel pipeline (detectors with
their reverting counterparts, the EWMA bank, the retrospective verdicts, the
convolution) applied to a third representation: each point's deviation from
the trailing ten-point mean, levels discarded. The forecaster proved the
representation carries signal; this hands it to every algorithm we own.

**015 — the forecaster, extended.** Horizons 1 and 5 (a slow break invisible
one step ahead is visible five ahead), plus the *signed* error smoothed — a
forecaster that keeps missing on the same side has found a trend change.

**016 — how to combine.** Boosted trees against a weighted sum, a plain
maximum ("at least one fired"), and asymmetric post-processing of the
combiner's own score: fast attack with slow release (peak-hold with decay),
and release gated by a different algorithm — drain the alarm quickly when the
forecaster says the series is predictable again, hold it otherwise.

**Kill condition, stated before all runs:** grouped 5-fold CV against 0.5779
on the same folds; a variant is adopted only if better on most folds. The
best survivor is submitted.

**Results (scripts in scripts/experiments/).**

*Channel sets — all killed.* The extended forecaster (horizons 1+5, signed
error): 0.5751, worse on 4 folds — the compact four-channel version carries
the signal, the extensions dilute it. The deviation view: 0.5762, worse on 3
— the forecaster already mines that representation, and fifty detectors over
it are its paler copy. Everything at once (170 channels): 0.5736 — dilution
compounds. A magnitude forecaster (predict |deviation|, 015b): solo 0.5524,
combined 0.5782 — passes the letter of the condition (3 of 5) but the mean
gain is +0.0003, noise; deferred rather than adopted, a candidate for the
neural-ensemble stage.

*Combiners.* A weighted sum over the 104 channels: 0.5608 on fold 0. A plain
maximum: 0.5401. Boosted trees stay unchallenged.

*Post-processing — adopted.* Fast attack, slow release on the OUTPUT score:
ride the running maximum, drain 0.1% per step. **0.5799 vs 0.5779**, ahead
on 4 folds of 5 (α=0.995 wins all five at 0.5795; α=0.999 taken for the
higher mean). Release gated by the forecaster's calm signal also beats the
base (0.5789–0.5792) but loses to the plain hold. The pair with the
channels' fast reversion is the interesting part: recant *evidence* quickly,
release the *verdict* slowly. Submitted as #11.

*The TCN pilot (017).* A 16.5k-parameter causal TCN (receptive field 255,
dropout, weight decay, 8 epochs, BCE on online steps): fold-0 TS-AUC climbs
0.520 → 0.532 against the stack's 0.5874. The concept learns; it is nowhere
near competitive yet. What it lacks is known — a ranking loss, augmentation
(the 007 pipeline finally has its customer), capacity, epochs — and that is
a deliberate, larger build, not a channel experiment.


**Results.**

*014, the deviation view: killed.* Alone its fifty channels reach 0.5316 by
maximum; over the base, 0.5762 vs 0.5779, behind on 3 folds. The forecaster
already lives on this representation and had extracted what it holds — the
detectors on deviations are its paler copy.

*015, the extended forecaster: killed.* Horizon 5 is strong alone (0.5561,
nearly the horizon-1 peak), but 100+E8 lands at 0.5751, behind on 4 folds:
the extra horizons and the signed error dilute the compact four channels of
013. The everything-together 162-channel run (0.5756) confirms two losers do
not make a winner.

*016, how to combine: the peak-hold is adopted.* Weighted sum (0.5608 on the
first fold) and plain maximum (0.5401) lose to boosting outright. But holding
the *output* score at its running maximum with a 0.1%-per-step drain lifts
every number: **0.5799 vs 0.5779**, ahead on 4 folds of 5 (α=0.995 is ahead
on all five at 0.5795). Gating the release by the forecaster's own normality
also beats the base but loses to the plain hold. The instructive part: the
channels underneath revert fast (008), the output holds its peak — attack
and release live at different layers, and the metric pays for both. Submitted
as #11 — the models of #10 untouched, one line of inference added.


---

## 018 — planned: the per-prefix battery

**Hypothesis** (the strategic one). The streaming O(1)-per-step architecture
was a self-imposed constraint — the platform allows any amount of prefix
recomputation per step. The first edition's winners reached 0.90 offline with
a large battery of two-sample statistics between the segments; the real-time
metric asks the same question at every prefix. So: at a geometric cadence
(held between recomputes, like the retro scans and the CNN already do),
compute ~50 drift-free two-sample features "history vs online prefix" —
quantile differences, moment differences, KS, MAD ratio, autocorrelation
differences, gauss-rank moments, recent-window and half-split contrasts — on
both views (raw and asinh). Statistics are kept drift-free under the null
(no √t multipliers), so length is not evidence.

**Kill condition, stated before the run:** grouped 5-fold CV: the battery
alone, and 104 + battery, against 0.5779 on the same folds; adopted only if
better on most folds. This is the first candidate for a new *foundation*
rather than a new add-on: if the battery alone approaches the 104-channel
stack, the second stage is to rebuild on top of it, not to append it.

**Result: adopted — better on 4 folds of 5.** Solo the battery is weak
(0.5381 — classical-detector territory), so it is not yet a foundation. As an
addition it is the largest gain since the forecaster: 104 channels 0.5779 →
146 channels **0.5807** (−0.0029, +0.0031, +0.0028, +0.0078, +0.0032). The
strategic read stands: prefix-recomputed two-sample statistics carry
information the streaming summaries do not, and this was v1 with 21 features
per view out of a few hundred available. Next: measure with the peak-hold,
ship, then scale the battery (more tests, more transformations, more
windows) — the first add-on family whose obvious next iteration is *larger*,
not different.

**With the peak-hold, and the correction.** The channel count is 146, not 144
(42 battery features, mislabelled in the first write-up of this entry). On
the 146-channel OOF the hard hold (α=0.999) no longer helps — the battery
carries part of the memory the hold was supplying — but the soft hold
(α=0.995) still does: **0.5814**, ahead on 3 folds of 5 over the raw output.
Shipped as #12 with the forecaster artifact reused from 013 and scipy added
to requirements (erfinv for the gauss-ranks).


---

## 020 — planned: battery v2

**Hypothesis.** v1 adopted with 21 features per view; the family's obvious
next move is more of the same. v2 adds ~20 per view: outer quantiles (q10,
q90), Cramér–von Mises and a clipped Anderson–Darling-style CDF distance,
the Levene contrast (mean absolute deviation from the median), statistics on
*increments* (dispersion ratio, lag-1 autocorrelation shift), fixed recent
windows of 10/25/100 against the history, tail-exceedance fractions beyond
the history's 5th/95th percentiles, sign statistics, and OLS slope
t-statistics of the prefix and of its magnitudes (a drift-free trend-change
and heteroscedasticity-trend probe). All drift-free under the null, both
views, same cadence.

**Kill condition, stated before the run:** grouped 5-fold CV, 186 vs 146
channels on the same folds (raw output; the hold re-checked after); adopted
only if better on most folds.

**Result: adopted — better on 3 folds of 5, and the vein is thinning.** Raw
output 0.5817 vs 0.5807: two folds gain strongly (+0.005, +0.006), two lose
mildly (−0.003, −0.004). The hold re-check moved the optimum to a faster
drain: α=0.99 now wins (the battery carries part of the memory), final
**0.5821**. Shipped as #13. The read on the family: still paying, but v3
should be feature *selection* within the battery rather than another forty —
the mixed folds are the first sign of dilution inside the vein itself.


---

## 021 — planned: hyperparameter resweep and feature selection at 186 channels

**Hypothesis.** The combiner's hyperparameters were swept once, at 40
channels; five channel families later the model is fit with settings tuned
for a fifth of its present width. A fresh sweep (leaves, child samples,
learning rate, rounds, feature fraction) plus importance-based pruning of the
186 should recover whatever the stale settings are leaving on the table.

**Kill condition, stated before the run:** each candidate configuration on
grouped 3-fold CV first (same folds 0–2), the best confirmed on the full
five; adopted only if the confirmed run beats 0.5817 raw on most folds.

**Result: adopted — better on 4 folds of 5.** The winner is not depth alone
but decorrelation: 63 leaves with **colsample 0.5**, each tree seeing a
different half of the 186 channels — the direct antidote to the dilution
that killed five add-on families. Confirmed raw 0.5831 vs 0.5817; with the
peak-hold **0.5841** (fold 0 reaches 0.5962). Shipped as #14 — same
channels, same artifact structure, retrained combiner.


---

## 022 — planned: lambdarank

**Hypothesis.** The combiner still optimises classification log-loss; the
metric is a per-step ranking. LightGBM's ranking objective with groups = step
indices (rows sorted by step, every cross-section one group) optimises the
ordering directly. Truncation level raised so the objective looks deep into
each group, since AUC cares about the whole order, not just the top.

**Kill condition, stated before the run:** grouped 5-fold CV against the
0.5817 raw baseline on the same folds (or against 021's winner if it
confirms); adopted only if better on most folds.

**Result: adopted on all five folds — the project's largest single gain.**
**0.5881** against 0.5831 for the resweep classifier on the same folds
(+0.0054, +0.0034, +0.0020, +0.0048... fold 4 +0.0090); fold 0 crosses 0.60
for the first time (0.6006). The metric is a per-step ranking, and training
the trees to rank per step — groups are cross-sections, truncation deep at
2000 — pays more than every channel family added this week. And this ran on
the *pre-resweep* hyperparameters; the combination with 63 leaves and
colsample 0.5 is measured next, and every later combiner (the ensemble
включая) moves to the ranking objective.

---

## 023 — planned: the TCN, seriously this time

**Hypothesis.** The 017 pilot learned but was starved: BCE instead of the
metric, no augmentation, 16k parameters. v2: pairwise ranking loss computed
inside each batch at matched steps (the metric's own question), boundary
augmentation (random history crops, online truncations, and moving the split
point of clean series — the 007 machinery finally feeding a data-hungry
learner), roughly 100k parameters, receptive field 511, cosine schedule.
Judged on fold 0 (stack: 0.5874); the real prize is ensemble diversity, so
the second number that matters is the rank-average with the stack's OOF on
that fold.

**Kill condition, stated before the run:** the ensemble number must beat the
stack's fold-0 alone; the net ships only inside an ensemble, never alone.

**Result: not yet.** The v2 recipe trains — 0.512 → 0.538 over 24 epochs on
fold 0, sawtooth but climbing to the end, loss still falling — and the
ranking loss with boundary augmentation beats the pilot's ceiling within
three epochs. But at 0.538 against the stack's 0.5952 the ensemble is flat:
+10% of the net moves fold 0 by +0.0002 (noise), more of it hurts. The net
appears to have learned a subset of what the channels already encode. The
concept survives; ensemble value requires a net at ~0.55+, which means
scale — capacity, epochs, cloud GPU — a deliberate build with the owner in
the loop on architecture, per plan. Checkpoint kept (tcn_v2.pt), aligned
fold-0 scores kept (tcn_fold0_aligned.npy).


---

## 024 — planned: pruning the 186

**Hypothesis.** colsample 0.5 winning the resweep says the channel set
carries redundancy the trees pay for; explicit pruning by gain importance
(top-150/120/90 on 3 folds, best confirmed on 5) may pay again on top, and a
cleaner set feeds every later stage (lambdarank, the ensemble).

**Kill condition, stated before the run:** confirmed 5-fold CV must beat the
current best raw configuration on most folds; otherwise the full set stays.

**Result: killed — monotone degradation.** 186 → 150 → 120 → 90 channels:
0.5849 → 0.5825 → 0.5813 → 0.5805 on the same three folds. With colsample
0.5 the trees already mine the weak channels for what they carry; hard
removal only takes it away. The full set stays, and "prune then rebuild" is
off the roadmap.


---

## 025–026 — the fold-0 protocol era begins

By the owner's call, research and pre-ship validation now run on fold 0 only
(the fold-0 ordering of models has never disagreed with the five-fold
verdict here); full CV returns at 0.62.

**025 — truncation: 2000 is the optimum.** Fold-0: 500 → 0.5919, 2000 →
0.6006, 8000 → 0.5986. Shallow cuts the signal, deep dilutes the gradient.
The resweep hyperparameters do nothing for the ranker (0.6006 → 0.6006 on
fold 0; its loss regularises by itself) — the classifier keeps them, the
ranker keeps the defaults.

**026 — the shippable blend.** Per-step cross-sectional rank averaging is
not implementable at inference (series arrive alone), so the blend is score
space: 0.6·sigmoid(ranker) + 0.4·classifier. Fold-0: classifier 0.5952,
ranker 0.6006, blend **0.6035**. The peak-hold now *costs* on top of the
blend (0.6030) — the ranker carries the memory the hold used to supply, and
that closes the hold's arc: adopted at #11, softened at #12, sped at #13,
retired at #15. Shipped as #15 with both combiners in the artifact.

---

## 027–029 — the fold-0 sprint: seeds and stacking killed, the longer ranker adopted

**027 — seed ensembling of rankers: killed.** Seeds 1 and 2 land weaker than
seed 0 (0.5963/0.5968 vs 0.6006) and averaging drags the blend down (0.6031
vs 0.6035). The ranking loss rewards sharpness, not smoothing.

**028 — a longer ranker: adopted.** 600 trees at lr 0.03: solo 0.6016, blend
**0.6045** (over 0.6035). 900 at 0.02 overshoots (blend 0.6026) — the
optimum is 600. Shipped as #16 with the final ranker retrained at 600.

**029 — stacking over the OOF scores: killed.** Three meta-rankers over
[classifier, ranker] scores (+forecaster channels, +slope t-stats):
0.5936–0.5991, all below the fixed 60/40 blend. Brandão's stacking presumes
many diverse level-0 models; ours are two and correlated, so the meta-layer
only overfits. Revisit when a strong net joins.

**Fast screening, calibrated.** Half the training series, 150 trees, max_bin
63: a hypothesis in ~4 minutes. Within a family the ordering reproduces
exactly (rank-2000 > rank-500); across families it distorts (the classifier
suffers more than rankers) — so the fast mode screens within-family only,
and cross-family calls plus ship candidates run full.

**Fold-1 diagnosis, first pass.** Composition is unremarkable (same break
share, lengths, tau positions as other folds) — the 0.02 weakness lives in
the series content, not the metadata. Next: an eyes-on gallery of fold-1's
worst earners.

---

## 030–031 — the world-model's nerves, and the online anchor

**030 — Chronos-2 error channels (fold-0 screen).** The first cut scored a
coin flip (0.4978) for an instructive reason: the error was normalised by
Chronos's own predicted interval, and a good probabilistic forecaster widens
its intervals after a break — the model absorbed the break into its
uncertainty and the signal self-cancelled. Re-measured in raw units: error
0.5308, error peak 0.5381, and the **predicted interval width itself 0.5597**
— "the world model got nervous" is the best of the three, zero-shot, at a
coarse cadence, ~100 forecasts/s on the laptop's GPU. Full-train channels are
building; the verdict is the ensemble screen. The library is whitelisted on
the platform (GPU runners available).

**031 — the online anchor (from the fold-1 gallery).** The gallery of
fold-1's worst earners showed five of six to be *clean* series whose online
segment simply differs from the history (variance blown, level shifted) —
the label means a break *inside* the online part, not "online differs from
history", and nearly every channel we own asks the latter question. The fix:
the full 50-channel pipeline referenced to the online segment's own first 40
points. Fold-0 screen: ranker solo 0.6016 → 0.6063, blend 0.6045 → **0.6094**
— seemingly the largest single addition since the ranking objective.

**Result: killed — the gain was length leakage, all of it.** The first build
derived the anchor size from the online length (`len // 4`), which the model
may not know. Fixed at 40: blend 0.6051. Rebuilt causally as a *progressive*
anchor (refit at steps 10 and 40, no length anywhere): **0.6047 vs 0.6045**
— two ten-thousandths, noise. The honest residual of the anchor view is
zero: whatever "is the online segment internally broken" carries, the
battery's half-split contrasts and the retrospective scan already encode.
The fold-1 diagnosis stands — clean series whose online differs from the
history bleed pairs — but this cure does not work, and the disease goes back
on the board as an open question. A leak caught before shipping, again, by
the pre-stated-condition discipline.

**030, closed. Chronos in the ensemble: killed.** 186+Chronos blend 0.6049
vs 0.6045 (noise), ranker solo slightly worse; with the anchor stacked on,
worse still (0.6025). The zero-shot signal is real (interval width 0.5597
solo) and the lesson is kept — a strong probabilistic forecaster absorbs
breaks into its own widening uncertainty, so surprise must be measured in
raw units — but everything it knows, the finetuned forecaster and the
variance detectors already tell the trees. Cloud GPU and a 120M-parameter
dependency, for nothing the ensemble can use: not shipped.

---

## 032 — killed: interaction channels for the fold-1 disease

Ten hand-built interactions — "external difference × a silent internal
scan", external minus internal, composite evidence — from existing battery
and scan columns. Fold-0: solo 0.6016 (the baseline exactly), blend 0.6042.
The trees already extract whatever these interactions encode. The fold-1
disease (clean series whose online differs from the history) has now
survived two cures — the anchor view and explicit interactions — and goes
back on the board marked *resistant*: the next candidate is a learned
detector trained specifically on these cases, which belongs to the network
line. Meanwhile TCN v3 (297k parameters, 40 epochs) trains overnight.

---

## 033–034 — the trees' blind spot: trajectories

**The premise.** A boosted tree sees each step's 186 channels as an isolated
snapshot; how the evidence *moves* — ramp shapes, fronts, agreement — is
invisible to it. Two attacks, one per family.

**034 — channel velocities for the trees: killed.** Deltas of the top-12
channels over 10 and 30 steps: solo 0.6019 (+0.0003), blend 0.6045 — the
baseline exactly. Hand-picked derivatives add nothing.

**033 — a network over the channel trajectories: adopted, decisively.** A
111k-parameter causal TCN reading the 186-channel sequence (ranking loss,
receptive field 127): fold-0 solo climbs to **0.5985 by epoch 9** — near the
ranker, from a completely different mechanism — with an overfitting tail
after (final retrain stops at 10 epochs). The raw-series nets never came
close (0.5441 at 4x the size); the representation was the bottleneck, not
capacity. In the shippable score-space blend the net takes a **0.40 weight**:
0.36 ranker + 0.24 classifier + 0.40 net = **0.6068** on fold 0, against
0.6045 for the pair. The first network in the project to earn its seat.
Shipped as #18 with an incremental numpy forward (0.8 ms/step, verified to
2e-7 against torch). An alignment trap resurfaced on the way — validation
scores saved in length-sorted order scored 0.5023 against row-ordered labels
until re-aligned — same trap as the raw-series nets, now twice learned.

**The run #109121 timeout, root-caused.** Submission #16 died in the cloud
at 7 hours (exit 124) with only 21 quota-minutes consumed: the sklearn
wrappers spawn a thread pool on every one-row predict, and five million
spawns across eight workers thrashed the box. The fix — raw
`booster_.predict(..., num_threads=1)` — profiles at **0.77 ms/step** on the
longest series (~10 cloud-minutes for the full test). Both #17 (the pair,
fixed) and #18 (the triple) shipped through the new mandatory profiling
gate; #15 and #16 must not be re-run.

---

## 035 — killed: a bigger channel net

Tripled (314k parameters, receptive field 511, heavier dropout, best-epoch
checkpointing): best 0.5940 against the small net's 0.5985. The dataset
saturates around 100k parameters — the line is data-bound, not
capacity-bound, which redirects the effort to averaging and richer inputs
rather than size. 036 (a seed ensemble of the small architecture) and 037
(the net fed channels plus both combiners' scores) queued.

---

## 036 — killed: seed-ensembled channel nets

Three seeds of the small architecture: solos 0.5939/0.5929/0.5894, the
average 0.5933 — below the best single seed — and the triple 0.6067 against
0.6068. On channel trajectories the nets converge to nearly the same
solution; there is nothing to average. With 027 this closes the whole
"multiply and average" family for this project. Next: 037, the net fed the
channels plus both combiners' OOF scores — learning when to trust them.

---

## 037 — killed: the score-fed net

The channel net with both combiners' OOF scores appended as inputs (188
dims): peak 0.5915 against the plain net's 0.5985. The scores are functions
of the channels, and the net spends capacity rediscovering that. With 035 and
036 this closes the declared trio: bigger, averaged, and score-fed all lose
to the small plain net. **The line has found its plateau: fold-0 ≈ 0.607,
cloud ≈ 0.588 (first calibration point: #18 ran 0.5877 in 30 minutes).**
Breaking it is a different class of effort — a transformer over the channel
trajectories, real GPU training budget, the owner in the loop on
architecture — plus the one unclosed averaging idea, an ensemble across fold
splits rather than seeds.

---

## 038 — adopted: the fold-ensemble of channel nets

Four small nets, each trained with a different training fold withheld —
diversity through data where seeds gave none. Fold-0: solo **0.6004** against
0.5939 for the single net, and the triple at a 0.50 net weight reaches
**0.6090** against 0.6068. The strongest single member is the one that never
saw fold 1 (0.5991) — its clean-but-different series poison the net's
training, one more echo of the resistant disease. Shipping as #19 with the
four members retrained to include fold 0 (each sees four folds of five).

---

## 039 — adopted: fold-bagged rankers

The data-diversity principle applied to the other leg: four rankers, each
missing one training fold. Solo the bag reaches 0.6033 against 0.6016 for
the single ranker, and the full ensemble — bagged rankers 0.35, classifier
0.15, fold-ensembled nets 0.50 — lands at **0.6099** on fold 0. Both legs of
the ensemble now stand on the same principle: models diversified by withheld
data, averaged in sigmoid space. Shipping as #19: four rankers + one
classifier + four nets, all finals retrained with fold 0 included.