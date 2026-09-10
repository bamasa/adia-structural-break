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

---

## 040 — invalidated: the stacking meta-ranker leaked

The meta-features used the *final* bagged rankers (fold 0 included in their
training) to score fold-0 validation rows: 0.7731 "out of nowhere" was the
smell, and the leak was found in minutes. The clean bag members from the 039
screen were never saved as models — only their fold-0 scores — so an honest
rerun needs four retrained clean members. Queued at low priority: the fixed
sigmoid weights over clean arrays are already measured (0.6099), and the
meta-layer's headroom over them is the only open question.

---

## 043 — killed: the long augmented training

48 epochs, input noise, channel dropout, heavier weight decay, receptive
field 255, best-epoch checkpointing: the peak lands at **0.5993 vs 0.5985**
— noise — around epoch 15, and the tail overfits straight through the
augmentations (loss 0.60 by the end, validation sliding to 0.566). The
bottleneck is data diversity, not epochs or capacity — for the third time.
The GeForce-3080 programme is therefore *many short nets on diverse
subsamples*, not one long one.

## The #19 cloud read: fold-0 is exhausted as a ruler

#19 (nine models, fold-0 0.6099) scored **0.5877 — identical to #18**
(fold-0 0.6068). Rank 215. Dozens of selection decisions have overfitted the
fold-0 protocol; increments of +0.002–0.003 on it no longer transfer.
Protocol amended: fold-0 stays for screening and kills; adoption into a
submission now requires the gain visible on two folds (0 and 1) or ≥0.005 on
fold 0 alone. The transferable frontier: cloud 0.5877.

---

## 042 — deferred to the 3080: TabPFN

Whitelisted on the platform (with tabpfn-extensions, GPU recommended), and
the v3 client now needs a PriorLabs API token — so the open v2 it is. On the
laptop it trains in six seconds and then dies silently on predict: the
10k-row training context multiplies into memory on every batch, and the CPU
build gets OOM-killed at batch 4000, and again at 500. Three attempts is
enough flogging: the screen (30k cadence points of fold 0, battery vectors)
is written and waiting in scripts/experiments/tabpfn_screen.py — it runs in
minutes on CUDA. First item of the GeForce-3080 programme.

---

## 040b — killed, and the stacking line closes

The honest rerun: four clean members retrained (fold 0 and their own fold
withheld, models saved to resources040), the meta-ranker fed only
leak-free features. Verdict: the meta's tree leg 0.6015, with the nets
0.6080 — against 0.6099 for the fixed sigmoid weights. Twice attempted, one
leak caught in between: the meta-layer has nothing to add over weighted
averaging here. The line is closed; the laptop era ends with every local
computation finished and every verdict written.

---

## 044 — shipped: the nets-alone ablation, and the overnight bag

**#20** is the four fold-ensembled channel nets with the trees removed
(fold-0 solo 0.6004, 1.2 ms/step): after #18 and #19 scored identically in
the cloud, this is the measurement that decides how much the net line
transfers — before a GPU day is spent scaling it.

Meanwhile the Mac runs the 3080 programme in miniature overnight: 24 short
nets, each on a random 60% of the training series, folds 0 and 1 held out as
the two-fold validation the amended protocol requires. By morning the
member table says whether the bag scales past its 4-member 0.6004.

---

## 045 — adopted on both folds and shipped: the twelve-net bag

The Mac ran the 3080 programme in miniature overnight: 24 short nets on
random 60% subsamples, folds 0 and 1 never trained on. The bag alone:
fold-0 **0.6136**, and **0.5880 on fold 1** — the poisonous fold where the
whole tree stack lived at 0.57x. Member-count sweep says 12 is the plateau
(0.6152/0.5886). The full ensemble — 0.3·(0.7 bagged rankers + 0.3
classifier) + 0.7·bag-12 — lands at **fold-0 0.6201, fold-1 0.5918**: both
folds up ~0.01 over #19, the amended protocol satisfied with room. The cloud
agreed in advance: the #20 ablation (four nets alone) scored 0.5846 against
the full #19's 0.5877 — the net line transfers *better* than the trees.

Shipped as #21 with a batched streaming forward: the twelve members' weights
stacked into single einsums, 1.77 ms/step for the whole bag (verified
against the per-member forward to 3e-15), ~19 cloud-minutes. The GeForce-3080
programme now has its confirmed shape: more members, richer subsamples.

---

## 046 — the scaling law says no: the bag saturates at twelve

Forty-eight more members over six hours — three architectures, subsample
fractions 0.5/0.6/0.7 — and the growth curve points *down*: 12 → 24 → 48 →
72 members gives 0.6152 → 0.6136 → 0.6099 → 0.6076 on fold 0, the same shape
on fold 1. Averaging does not rescue weak members; it dilutes with them. The
twelve of #21 remain the optimum, and six Mac-hours just saved the GPU day
from its default mistake: a hundred members would have been worse than
twelve. The 3080 programme is rewritten: *stronger members, not more* —
larger fractions (0.75–0.85) across different splits, more epochs with
early stopping per member, richer member inputs — plus the TabPFN screen.
Member quality dominates count; the curve is the proof.

---

## The #21 cloud read: both folds are burnt, the paradigm shifts

#21 (the twelve-net bag at 0.7; fold-0 0.6201, fold-1 0.5918 — both folds
verified) scored **0.5810** in the cloud: below the nine-model 0.5877 and
below the nets-alone 0.5846. Two lessons, neither small.

First: fold 1 joined the protocol yesterday and was burnt within a day —
member selection and the 0.7 weight were tuned against it, and its +0.01
was selection noise, exactly as fold 0's was at #19. With ~10k training
series, local validation is exhausted as a way to certify +0.01 increments:
any fold we optimise against stops transferring within dozens of decisions.

Second: the bag members each saw ~36% of the data, and averaging does not
buy that back on fresh series — the #19/#20 nets that saw 80% each are
simply stronger in the cloud.

The operating mode changes: fewer, larger bets, validated where it counts —
five cloud runs a day are the only ruler that cannot be burnt. The GPU day
becomes one strong thing (members on 85–90% of the data, trained long),
A/B-tested in the cloud, not against folds. The cloud frontier to beat
remains #18/#19's 0.5877.

---

## 047 — the first bet of the cloud-validated era

Heavy members: each net sees 88% of ALL training series (a fresh random cut
per member), trains 16 epochs, and picks its best epoch on a private 8%
holdout of its own — the burnt folds play no part in anything. The Mac
pipeline is resumable (finished members are skipped) and will keep producing
members until the 3080 takes over with the same script.

Shipped as **#22** after the first eight members: the ensemble formula and
weights are exactly #19's (0.35 bagged rankers + 0.15 classifier + 0.50
nets), so the cloud score isolates a single variable — member quality (88%
data vs #19's 80%, 16 epochs vs 10, private-holdout early stopping). The
frontier to beat: 0.5877.

---

## The #22 verdict: the family's ceiling, measured

The clean A/B returned **0.5876 against #19's 0.5877** — member quality
(88% data, 16 epochs, private holdouts) moves the cloud by nothing, exactly
as member count (#21) and weights did not. The conclusion is now
overdetermined: the current family — 186 hand-built channels, TCNs over
their trajectories, bagged trees — has a cloud ceiling at **≈0.588**, and
we have hit it from every direction the family allows. The heavy-member
pipeline is stopped (14 members kept on disk).

What can move past a family ceiling is a new family: representations we do
not have — raw-series models at real scale, hybrid raw+channel towers,
TabPFN's tabular prior, channel types not yet invented (the resistant
fold-1 disease still marks one). That is the GPU's actual job.

---

## 048 — killed at laptop scale: the hybrid two-tower

The first genuinely new family the Mac could afford: a channel tower (the
proven TCN) fused per-step with a raw-series tower (normalised stream with a
256-point history tail, receptive field 255). Four members, private
holdouts: **0.6035 / 0.5589 / 0.5740 / 0.5938** — mean ≈0.583 with wild
variance, against ≈0.60 for the plain heavy channel nets on the same
protocol. The raw tower adds noise, not signal, at this scale; no cloud run
is spent on it (the new regime reserves those for promising bets). The idea
survives as a 3080 item in one specific form — a raw tower *pretrained* at
real capacity — but as a laptop line it is closed.

---

## 049 — killed, and the laptop era closes for real

Self-supervised pretraining of the channel backbone (predict the next
channel vector — huber falls 0.107 → 0.047 in twelve minutes, the dynamics
are learnable) followed by ranking-loss finetunes: **0.5887 / 0.5943 /
0.5874** against ≈0.60 for the same recipe trained from scratch. The
pretrained backbone clings to its forecasting task and slightly *resists*
the ranking one. The last laptop-scale idea is spent. Forty-nine
experiments; the cloud ceiling of the family stands at 0.588; everything
that remains lives on CUDA: TabPFN, a raw-series model at real capacity,
and whatever new family the two of us design next.

---

## 050–052 — three structural probes, one survivor

**050 — metric-aligned weights: killed.** The metric weights a step by its
pos×neg pair count; our training weighted every step equally — a mismatch of
up to 48× (step 0 carries 289k pairs, step 300 carries 9.9M). Retraining
with pair-proportional weights *hurt*: classifier 0.5952 → 0.5912, ranker
0.6016 → 0.6003. Equal-step weighting is acting as a regulariser, and the
misalignment was never the problem.

**051 — history-level calibration: killed, with a reversal.** The
inspection notebook's old diagnosis (clean-but-jittery series outrank broken
ones) suggested discounting each series by the score its own break-free
history earns. Subtracting it collapses the pair to 0.5577 — because the
history level *predicts breaks positively* (0.5297 standalone: series that
look restless in history do break more often). Adding it back gives +0.0007,
noise: the models already know.

**052 — the spectral family: adopted, shipped as #23.** Every channel this
project owned lived in the time domain; a break that rearranges periodicity
without touching level or variance passed them all unseen. Fourteen
frequency-domain channels — eight power bands read in the history's own
sigmas, plus entropy, peak-frequency, total-power shifts and spectral shape
distance — lift the ranker 0.6016 → 0.6032 and the pair 0.6045 → 0.6060.
Adding the step index on top cancels the gain (0.6017/0.6045), confirming
experiment 010's verdict on step. The trees are retrained on all 200
channels, the eight heavy nets of #22 read the first 186 unchanged, and the
ensemble weights are #22's exactly, so the cloud isolates one variable: new
information, in a modality the project had never touched.

---

## 054 — killed: spectral v2

Thirty-two more frequency channels — windows 32 and 128 beside the shipped
64, the spectrum of *increments* (a different question: has the correlation
structure moved), low-versus-high band balance — and the ranker drops to
0.6026 against v1's 0.6032, the pair to 0.6053 against 0.6060. The
fourteen-channel v1 is the family's optimum; #23 already carries exactly
those. Dilution inside a family, for the seventh time in this project: the
first good channels of a modality take almost everything it has.

---

## 055 — shipped as #24: trees alone, the untried cloud configuration

Every cloud submission this project has made carried networks, and every one
landed within 0.003 of 0.5877; the nets alone scored 0.5846. The one thing
never tried is the other half by itself. #24 is the spectral-augmented
ranker and classifier with no nets at all (0.74 ms/step, the fastest
submission yet). Together with #23 it forms today's pair of cloud
questions: does the new modality transfer (#23), and have the nets been
carrying anything at all (#24).

---

## #23 in the cloud: 0.5893 — the ceiling breaks

Five consecutive submissions sat at 0.5877 whatever we changed inside the
family; the spectral channels moved it to **0.5893**. The gain matches the
fold-0 screen exactly (+0.0016 there, +0.0016 here), which is the cleanest
transfer this project has recorded — and the reason is what the channels
are: the first *new information* since the forecaster, not another
arrangement of the same evidence.

The lesson generalises past this competition: when a family saturates, no
amount of model work moves it, and a single new modality does. Everything
that failed this week (bag size, member quality, stacking, weights,
pretraining, the hybrid) rearranged known information; the one thing that
worked added an unknown one.

New frontier: **0.5893**. Next: #24 answers whether the nets carry anything
in battle, and the modality question reopens — what else is the project
still blind to?

---

## #24 in the cloud: 0.5853 — the nets do earn their place

Trees alone on the spectral channels score 0.5853 against the full #23's
0.5893: the networks are worth **+0.0040** in battle, and the ensemble's
two halves are now both measured against a common baseline —

| configuration | cloud |
|---|---|
| trees alone (#24) | 0.5853 |
| nets alone (#20) | 0.5846 |
| trees + nets, no spectral (#19) | 0.5877 |
| trees + nets, with spectral (#23) | **0.5893** |

Neither half reaches the pair; the halves are close to each other and the
combination beats both — textbook complementarity, and the spectral
channels lift the whole. Nothing here is to be simplified away: the
frontier stands at 0.5893 with everything in place.

---

## 056 — the virgin-fold confirmation

Folds 2, 3 and 4 have never taken part in a single decision of this
project; fold 2 is therefore the one honest local ruler left. Trained on
everything else, the spectral channels move it **0.5912 → 0.5967
(+0.0055)** — larger than the burnt fold-0 screen suggested (+0.0016) and
in the same direction as the cloud (+0.0016). Three independent readings,
one sign: the modality is real.

It also quantifies what selection cost us: on a burnt fold, a genuine +0.005
was compressed to +0.002 — years of small "improvements" measured there were
probably the same effect running the other way. Folds 3 and 4 remain
unspent; they are the reserve rulers for the GPU era, to be used once each
and never for tuning.

---

## 057–066 — the hand-crafted era ends, the data era begins

Four more channel families were built and screened on fold 2 (untouched by
any earlier decision), and all four lost:

| family | fold-2 pair | verdict |
|---|---|---|
| baseline (200 channels) | 0.5979 | — |
| wavelets (16) | 0.5933 | killed |
| rank two-sample tests (14) | 0.5929 | killed |
| matched filter bank (24) | 0.5815 | killed |

At 200 channels every additional hand-built family dilutes: with colsample
0.8 each tree sees 160 columns, and weak newcomers crowd out strong
incumbents. The spectral family remains the exception that proved the rule —
it carried a modality nothing else had.

What did pay, in order of size:

* **Augmented training data (065): +0.0044 on the tree pair.** Move the
  history boundary right: history H + online O[:k] with k < tau, online
  O[k:], break at tau−k. The new history is break-free by construction and
  the break lands earlier — precisely where the metric is thin. Nine
  thousand pseudo-series doubled the rows and lifted the pair 0.5999 →
  0.6043.
* **Nets retrained on the 200 channels (058).** They had never seen the
  spectral family; the strongest member reached holdout 0.6349 against ~0.60
  typical for the 186-channel generation.
* **Ranker resweep at 200 channels (062): +0.0034.** 63 leaves with
  colsample 0.5 — the same decorrelation that helped the classifier a
  hundred channels ago. The classifier's own sweep found nothing: it is
  already optimal.
* **Member selection by private holdout: +0.0014.** Members vary from 0.588
  to 0.635; averaging the best half beats averaging all.
* **A second net variant (063): channels plus their differences**, 600
  inputs — the differences the trees rejected (034) help the nets, whose
  members reached 0.6178.

Together: **fold-2 0.5979 → 0.6091** (augmented trees + best half of both
net pools at weight 0.5), which calibrates to roughly 0.601 in the cloud
against the shipped 0.5893. Nothing is submitted yet — the target is
fold-2 0.617, and the next lever is the one that just proved itself:
three cuts per series instead of one.

---

## 067–068 — augmentation splits the two combiners

Three cuts per series instead of one produced 27k pseudo-series, and the
result is a split verdict: the **ranker reached 0.6029** — its best ever,
against 0.5956 clean — while the **classifier fell to 0.5874** from 0.5980.
Ranking learns from more, noisier orderings; the classifier's probabilities
degrade when the same series appears three times with correlated rows.

So the components stop sharing a training set. The best assembly to date
takes both augmented rankers (single and triple cuts), the classifier
trained on clean data only, and the six best members of the two net pools
selected by their private holdouts:

| stage | fold-2 |
|---|---|
| morning baseline | 0.5979 |
| + augmented trees | 0.6043 |
| + ranker resweep, net retrain | 0.6091 |
| + split training sets, two rankers | 0.6103 |
| + both net pools, six best members | **0.6106** |

That calibrates to roughly 0.604 in the cloud against the shipped 0.5893.
The target is fold-2 0.617; nothing is submitted until it is met.

---

## 069–072 — augmentation reaches the networks, and two submissions ship

Nets trained on the augmented pool (original series plus pseudo-series with
the boundary moved right) came out consistently strong: holdouts 0.6010,
0.6286, 0.6217 against ~0.59 typical for the clean-data generation. The
strongest members of every pool now come from augmented training, which
matches the trees' story and the project's oldest lesson — this line is
data-bound, and the boundary shift manufactures the data it lacks.

Two submissions went out:

* **#26** — the six best net members by private holdout on spectral trees,
  weights from the untouched fold 2.
* **#27** — the full shape: an augmented ranker (trained on 8.8M rows) and a
  clean one, a classifier on clean data only, and the eight best nets.
  Fold 2: **0.6106** against 0.5979 for the configuration that scored 0.5893
  in the cloud.

**#25 is broken and must not be run.** Its interface kept a `channels[:186]`
slice from the era when the nets predated the spectral family, so the
200-channel members received 186 inputs. The verification caught it — and
the push ran anyway, because the shell chained it after the check with `;`
instead of `&&`. The rule that follows: the push command must depend on the
verification's exit status, never merely follow it.

---

## 073 — the net half is saturated

Ten augmented members finished, and the pool now holds seventeen. The best
of them reached **0.6473** on its private holdout — a project record, the
previous being 0.6349 — yet the ensemble barely moved: ten members at equal
weights give fold-2 0.6116 against 0.6106 for the eight shipped in #27, and
softmax weighting by holdout quality adds exactly nothing (0.6116 again,
whatever the temperature). A single excellent member dissolves in the
average; the half is saturated at its current recipe.

The day's ledger on the untouched fold: **0.5979 → 0.6116**, of which
augmentation contributed the most (trees +0.0044, and the strongest members
of every net pool now come from augmented training), the ranker resweep
+0.0034, and holdout-based selection the rest. Four hand-crafted channel
families were built and killed in the same span.

What remains untried at laptop scale is thin: a third ranker breed, and the
GPU pool of twenty-four members where selection could actually bite. The
cloud numbers for #26 and #27 will say where the 0.6116 really landed.

---

## 074 — a third ranker breed, killed

Hypothesis: the tree half rests on two rankers of near-identical
configuration; a breed with different geometry (127 leaves, lr 0.02,
colsample 0.7, truncation 1000, its own seed) trained on the same augmented
set would add diversity. Kill condition: the three-ranker blend is no better
than two.

Killed cleanly. Solo 0.5960 against 0.6034 for the standard breed, and the
blend degrades monotonically with its share — 0.6107 at zero, 0.6093 at a
quarter. Tree geometry is not a source of diversity here: both breeds read
the same channels off the same rows and disagree only where neither is
right. A joint weight sweep the same day (rankers × classifier × nets on a
0.05 grid) plateaued at 0.6104–0.6116 — the current composition is fully
squeezed.

Laptop-scale levers are now exhausted. The fold-2 ledger stands at
**0.6116**; what remains is the cloud verdict on #26/#27 and the
twenty-four-member GPU pool.

---

## 075 — augmentation reaches the diff nets

The last untried combination at laptop scale: the diff architecture (600
inputs — channels plus lag-1 and lag-10 differences) had only ever trained
on clean data (best holdout 0.6178), while augmentation had lifted every
plain pool it touched. Six diff members trained overnight on the augmented
set; holdouts 0.5695–0.6458, two above the old diff record.

The ensemble verdict is mixed. Adding diff members helps monotonically —
the best mix is *all six*, including the weakest — and lifts fold-2 from
0.6116 to **0.6120**. So the gain is architectural diversity, not member
quality: the member with the record 0.6458 holdout scores a dismal 0.5863
on fold-2, the worst of the six. Diff-net holdouts do not transfer; plain
holdouts do. Selection within the diff pool should therefore be by fold-2,
or not at all.

+0.0004 does not justify a submission slot by itself. The mix (12 plain +
6 diff, net weight 0.5) is the new local ledger and goes into the next
assembly whenever a bigger gain — the GPU pool, or a new modality — pays
for the slot. Per-member fold-2 sigmoids are cached as
``fold2_sig_*.npy`` for instant recombination.

---

## 076 — the triple augmentation reaches the nets; submission #28

Hypothesis: the nets are data-bound, and every pool so far saw at most the
single boundary augmentation (7,241 pseudo-series). The triple version —
three random slices per original, 26,887 pseudo-series, 8.3 GB read through
a memory map — had only ever fed the ranker, where it did nothing. Kill
condition: members no stronger than the single-augmentation pools on the
untouched fold.

Not killed — the opposite. Six plain members, fold-2 solo scores
0.6048 / 0.6027 / 0.6047 / 0.5983 / 0.5981 / **0.6102**: every earlier
member in the project sat below 0.60, and the last one alone beats the
previous ten-member net ensemble (0.6016). The six together read 0.6108 as
the net half — above the *entire* #27 ensemble — and blended 0.55 to the
trees (augmented ranker 0.7, clean classifier 0.3) give fold-2 **0.6163**,
against 0.6106 for #27.

Three lessons on the way. A diff member trained the same way reached a
0.6543 holdout — the highest number the project has produced — and scored
0.5875 on fold 2; diff holdouts are noise, and the two remaining diff jobs
were cut from the queue. The old pools add nothing on top: twelve plain
members at 0.3 group weight are worth +0.0002, so the shipped configuration
is six nets, one ranker, one classifier, 1.2 ms/step. And the GPU kit was
found training on clean data — the recipe the laptop had outgrown — and
rewritten around AUG3 in the same session.

Shipped as **#28**, resources073. The assembler's new channel check ran on
this build: eleven counts, all 200.

**Cloud: 0.6004** (45 min 37 s). The best cloud score of the project, up
from 0.5893 for #23 — the first move above 0.59 after five submissions
pinned at 0.588. The fold-2 → cloud gap widened, though: 0.0159 here
against 0.0086 for #23. Part of that is honest variance; part is that the
composition and blend weights of #28 were read off fold 2, which has now
carried a handful of decisions. Calibration for the next step: 0.61 in
the cloud wants fold-2 ≈ 0.626, and the next composition should be
checked on fold 3 before it ships.


---

## 076b — six more of the same, and the pool gets worse

Six further plain members on the triple augmentation, same recipe, seeds
51006–51011. Holdouts 0.6233–0.6571 — the highest the plain breed has
posted. Fold-2 solos 0.5913–0.6019, the lowest of the aug3 pool. Twelve
members averaged: net half 0.6078 against 0.6108 for the first six, blend
**0.6147 against 0.6163**. The shipped #28 stands.

Two readings, both uncomfortable. Private holdouts do not rank plain
members either — the correlation with fold 2 across twelve members is
negative (0.6571 → 0.5913, 0.6533 → 0.5942, while 0.6167 → 0.6048). An 8%
holdout of eight hundred series is too small a yardstick, and a member that
fits its holdout may be one that fits its particular 92%. And the first six
were, in part, lucky: the spread of solos (0.591–0.610) is wider than the
gap between the two batches, so a fresh six would land anywhere in between.

What this rules out: selecting members by fold 2 (it would burn the last
honest fold), and expecting more members alone to move the number. What
it leaves: the cloud verdict on #28, and a GPU pool large enough that
averaging drowns the member variance — twenty-four members with the
spread seen here should sit near the pool mean, ≈0.6155 blended, not
above the lucky six.

---

## 077 — capacity, killed

Hypothesis: with three times the data, the laptop-era finding that small
nets beat big ones may have expired — adding members no longer moves the
pool (076b), so perhaps capacity binds. Three members with 96 channels
instead of 64 (~250k parameters), same recipe, same triple augmentation.
Kill condition: fold-2 solos not above 0.605.

Killed. Solos 0.5975 / 0.5974 / 0.5998 — below the 64-channel median, not
above it; the three together read 0.6048 as a net half against 0.6108 for
the six of #28, and folding them into #28 lowers it to 0.6156. The wider
net fits its holdout better (0.6175–0.6381) and transfers worse, which is
the same story the diff nets told: on this data, extra fit is extra noise.

The net half is bounded by the data even at three times the data. What
remains is not capacity, members, or architecture, but the cloud gap:
#28 reads 0.6163 here and 0.6004 there, and the next 0.01 has to come from
a member whose gain survives a fold it was not tuned on.

---

## 078 — the #28 recipe on every series; submission #29

#28's six nets had never seen fold 2: two thousand series and their 5,345
augmented pseudo-series, a fifth of the data, held out so the laptop could
measure them. For a data-bound model that is a real cost paid for a
yardstick, and the final model should not pay it — the trees have trained
on everything all along. Six members, same seeds and recipe as #28, trained
on all ten thousand series and all 26,887 pseudo-series (holdouts
0.6195–0.6545, recorded for the protocol, not consulted).

There is no local number for this configuration by construction, and none
is claimed. What is claimed is direction: single augmentation → triple
augmentation moved the members from below 0.60 to above it, so a further
quarter more data should not move them down. Shipped as **#29**,
resources074, everything else identical to #28 — the cloud reads the
difference directly, as a paired comparison against #28's 0.6004.

**Cloud: 0.5996** — against 0.6004 for #28. No difference within noise. A
fifth more data, fold 2 and its pseudo-series included, moved the net half
by nothing the cloud can see: the six-member average sits at 0.600 either
way. Two readings. The nets are not short of *this kind* of data any more
— the triple augmentation saturated the axis, and 079 showed that more of
it hurts. And the fold-2 → cloud gap (0.016) is not a data-quantity
artefact; it is what these members lose on the platform's series, which
the honest fold predicts only up to a constant. The remaining 0.01 will
not come from feeding the same members more of the same.


---

## 079 — nine slices per series, killed

The one axis that ever moved the nets was the volume of boundary
augmentation (single → triple took members from below 0.60 to above it),
so: six more random slices per series, built in eight shards (22.1M rows,
53,128 pseudo-series, 17.7 GB), added to the triple — 80k pseudo-series
against 10k originals. Stored as float16 to fit in memory; checked: no
overflow, median rounding error 1.7e-4.

Killed at two members: fold-2 solos **0.5824 / 0.5748**, far below the
0.598–0.610 of the triple. The volume axis has an optimum, and it is not
"more". At 8:1 the pseudo-series dominate and the model drifts toward
their distribution — longer histories, early breaks — while the metric is
scored on the originals. The private holdout, drawn from originals, picks
the least-bad epoch but cannot undo the drift.

**079b**, the follow-up — the triple's volume per epoch (21.5k
pseudo-series) but a fresh random third of the nine slices every epoch,
variety at constant ratio — is neutral at best: three members read
0.5975 / 0.6047 / 0.5993 solo, 0.6053 together against 0.6092 for the
first three members of the plain triple, and folding them into #28 gives
0.6161 against 0.6163. The ratio was the thing; which slices, and how many
different ones, is not. The augmentation axis is closed in every direction
it has: single → triple was the gain, and everything past it is flat or
worse.

---

## 080 — cross-sectional batch size, killed; and a leak found

The ranking loss sees pairs only inside a batch of 24 series, while the
metric ranks thousands per step. Batch 96, four times the pairs per step:
three members at ten epochs read 0.5939 / 0.5922 / 0.6004 on fold 2 —
slightly below the triple's 0.598–0.610. A quarter of the optimiser steps
is a confound, so **080b** ran batch 96 for thirty epochs. Its first member
posted a *0.7255* holdout — and 0.5678 on fold 2.

That pair of numbers exposes a leak that has been in every net pool since
augmentation began: the private holdout is drawn from the originals, but
the augmented pseudo-series of those same originals stay in the training
set. A long schedule memorises them, the holdout inflates, and best-epoch
selection — which every member relies on — has been biased toward
memorisation all along. It also explains why holdouts never ranked members.

080 is killed (batch size is not the lever); 080b is invalid rather than
killed. **081** now tests the fix directly: the #28 recipe with the
holdout's pseudo-series excluded from training, same seeds as p0–p2 for a
paired comparison.

---

## 081 — the clean holdout, and what the epoch curve actually looks like

Fixing the leak the obvious way — holding the pseudo-series of the holdout
originals out of training — made things *worse*: the first clean member
(seed 51000, paired with #28's p0 at 0.6048) read **0.5623** on fold 2. So
081d trained the same seed once more and scored every epoch on both the
clean holdout and fold 2:

| epoch | clean holdout | fold 2 |
|---|---|---|
| 0 | 0.5994 | 0.5623 |
| 1 | 0.5882 | 0.5834 |
| 2 | 0.5962 | 0.5863 |
| 3 | 0.5702 | 0.5877 |
| 4 | 0.5773 | 0.5953 |
| 5 | 0.5821 | 0.5973 |
| 6 | 0.5834 | 0.5999 |
| 7 | 0.5779 | 0.5998 |
| 8 | 0.5682 | **0.6043** |
| 9 | 0.5693 | 0.6036 |

Fold 2 climbs monotonically to the last epochs. The clean holdout — 640
series — peaks at epoch zero and drifts *down* while the model improves:
as a selection criterion it is noise with the wrong sign. The leaky holdout
of #28 picked late epochs only because memorising the pseudo-series pulled
it late, and so was right by accident. Best-epoch selection, the recipe's
one piece of validation machinery, has been either useless or harmful all
along.

The rule that follows is simpler: **train to the end and take the last
epoch**, put the holdout back into training, and — since the curve has not
turned down at ten — see whether fifteen epochs buy anything. 082 runs
that on the six #28 seeds, scoring fold 2 at epochs 9 and 14 only.

---

## 082 — fifteen epochs, killed at one member

Seed 51000 under a fifteen-epoch cosine, last epoch, no holdout: fold-2
**0.5936** at epoch 14, against 0.6036 for the same seed's last epoch under a
ten-epoch cosine. Consistent with 080b's thirty epochs (0.5678): past ten
epochs the extra updates go into memorising the pseudo-series. The schedule
has an optimum near ten, and it is not a soft one.

082b runs the direct candidate for #30: the six #28 seeds, ten epochs, last
epoch, every series in training. Fold 2 is scored once per member at the
end, and the six together against #28's 0.6163.

---

## 082b — the last-epoch rule, measured on the six #28 seeds

Ten epochs, last epoch, every series in training, seeds 51000–51005 — the
#28 members re-trained under the clean rule. Paired on fold 2:

| seed | #28 (leaky best-epoch) | last epoch |
|---|---|---|
| 51000 | 0.6048 | 0.6057 |
| 51001 | 0.6027 | 0.6065 |
| 51002 | 0.6047 | 0.6051 |
| 51003 | 0.5983 | 0.6023 |
| 51004 | 0.5981 | 0.6026 |
| 51005 | **0.6102** | 0.5954 |

Five of six improve, by 0.001–0.005; the sixth loses the lucky epoch that
made it #28's best member. Mean member 0.6031 → 0.6046. The ensemble does
not move: six nets 0.6108 → 0.6091 solo, blend 0.6163 → 0.6157, and the
twelve together read 0.6163 again. The rule is the right one — it removes
a selector that was noise and adds the holdout to training — but it is
neutral in expectation, and #28's edge over it was one member's luck.

The recipe has a ceiling: ≈0.616 on fold 2, ≈0.600 in the cloud, and every
lever inside it — members, data volume, data variety, capacity, schedule,
batch, epoch choice — is now measured flat or worse. #28 stands as
shipped. What remains is outside the recipe: the loss itself (083), and
what the networks see.

---

## 083 — the metric's step weights in the loss, killed

TS-AUC averages steps weighted by the number of positive–negative pairs at
each; the ranking loss averaged its 48 sampled steps uniformly. Weighting
each step's pairwise loss by n_pos·n_neg, everything else as 082b: three
members 0.6049 / 0.6049 / 0.6031 against their 082b pairs 0.6057 / 0.6065 /
0.6051. Slightly lower every time. The loss already ranks the right thing;
re-weighting where it ranks is not the missing piece.

---

## 084 — the raw series as input, killed

The channels are summaries; the networks had never seen the series itself.
Two raw channels — the point's z-score against the (extended) history,
clipped at ±20, and its asinh — appended to the 200, for originals and
pseudo-series alike. Three members under the 082b recipe: 0.5937 / 0.6036 /
0.6011 against their pairs 0.6057 / 0.6065 / 0.6051. Worse on every seed,
by 0.002–0.012. The raw trajectory adds noise the dilated convolutions
cannot filter better than the engineered channels already do.

---

## Where this leaves the project (6 September)

Since #23 (0.5893) the cloud has moved once — to #28's 0.6004 — and that
move came from a single lever, boundary augmentation at a 1:2.7 ratio,
which was found and then bracketed from both sides. Everything else
measured on the untouched fold since then is flat or negative: member
count, augmentation volume and variety, full-data training (#29: 0.5996),
capacity, schedule length, batch size, epoch selection, loss weighting,
raw input. Fourteen experiments, one gain.

The fold-2 → cloud gap sits at 0.016 and did not shrink with more data,
so it is a property of the platform's series, not of our sample size. To
reach 0.61 in the cloud the fold needs ≈0.626, a full 0.01 above the
recipe's ceiling — the size of the augmentation gain itself. That is a
new-modality-sized gap, and the laptop has no untried modality left that
costs less than a day. The GPU pool would tighten the variance, not lift
the mean.

---

## 085 — Bayesian online change-point channels (in progress)

A new modality for the trees, the only kind of thing that has ever moved
the cloud: the run-length posterior of Adams & MacKay's BOCPD — a
Normal-Gamma model of the point, prior fitted to the history, hazard
1/200, run lengths to 600. Six channels per step: P(r<5), P(r<20), P(r<60),
P(r<200), E[r]/(t+1), and the point's surprise (−log predictive density).
0.1 ms per step; built for all ten thousand series in a minute across
eight shards.

First screen, classifier on fold 2, same configuration with 200 and 206
channels: 0.5953 → 0.5963, **+0.0010**. Below the +0.002 acceptance line,
but two of the six — P(r<200) and E[r]/t — rank 31st and 33rd of 206 by
gain. The hazard sweep, classifier on fold 2 against the same 200-channel
baseline (0.5953): 1/200 → +0.0010; **1/50 → +0.0046** (E[r]/t sixth of
206 by gain); 1/20 → +0.0023 (E[r]/t first by gain, but the rest weaker);
both 1/50 and 1/20 together, twelve channels → +0.0018. One hazard, 1/50.

The ranker does not want them: same configuration as #28's, on originals
plus the single augmentation, 200 channels 0.6043 against 206 channels
0.6029, **−0.0014**. A cross-sectional ranker already has what an absolute
run-length posterior adds to a classifier — a scale that is comparable
across series — and the six extra columns only dilute its column sampling.

Blended with the six #28 nets (0.55): classifier at 206 and ranker at 200
lifts the ensemble from 0.6164 to **0.6169**, +0.0005. The modality is
real, and it lands on the one member that carries the least weight.

Nets with the six channels, three paired seeds: 0.5985 / 0.6076 / 0.6000
against 0.6057 / 0.6065 / 0.6051 — mean −0.003. The trajectory networks
read the run length off the channels they already have.

Verdict: kept as an ingredient — the classifier ships with 206 channels
in the next assembly, the ranker and the nets stay at 200 — but not worth
a submission slot on its own at +0.0005.

---

## A check on the cloud gap: train and test are the same population

Before spending more on the fold, the obvious alternative explanation for
the 0.016 fold-2 → cloud gap: the platform's series differ from ours. The
local test sample (100 series) says no. History length median 3014 vs
2998, online length 510 vs 502, share with a break 0.53 vs 0.50, break
position at 0.42 vs 0.48 of the online part, value scale identical. The
gap is fold-2's accumulated optimism plus sampling noise, not a shift —
which means a gain has to be large to be seen at all, and small ones
(the +0.0005 of 085) will not be.

---

## 086 — a recurrent family, not feasible on the laptop

A two-layer GRU (96 units, ~85k parameters) on the same 200-channel
trajectories, same data and recipe as 082b — the one model family the
project had not tried. On Apple's MPS backend the recurrence has no fused
kernel: twelve hours of wall-clock did not finish the first ten-epoch
member, where a convolutional member takes thirty minutes. Stopped
without a number. The script (`nets_gru.py`) runs unchanged on CUDA,
where cuDNN makes it a twenty-minute member; it belongs to the GPU box,
not here.

---

## Where the metric's weight is, and where the ensemble is blind

Fold 2, the #28 blend, by step of the online part (weight = share of
positive–negative pairs the metric counts there):

| steps | trees | nets | blend | weight |
|---|---|---|---|---|
| 0–25 | 0.536 | 0.550 | 0.541 | 1% |
| 25–50 | 0.556 | 0.558 | 0.559 | 3% |
| 50–100 | 0.567 | 0.563 | 0.568 | 8% |
| 100–200 | 0.573 | 0.577 | 0.580 | 20% |
| 200–400 | 0.616 | 0.619 | 0.626 | 36% |
| 400–800 | 0.637 | 0.644 | 0.652 | 30% |
| 800+ | 0.620 | 0.631 | 0.635 | 1% |

A step-dependent blend weight buys nothing (the best schedule is the
constant 0.55). But a third of the metric's weight sits below step 200,
where both halves score 0.54–0.58 — near chance — while past step 200
they read 0.62–0.65. The early region is where the boundary augmentation
paid, and it is the only region with room. 087 tests early specialists:
members whose ranking loss is taken on the first 200 steps only.

---

## 087 — early specialists, killed at one member

A member whose ranking loss is taken on the first 200 steps only, to
attack the region where the ensemble is near chance. It is worse *there*:
early-step AUC 0.5489 against 0.5692–0.5756 for the ordinary members, and
0.5790 over the whole fold. The late pairs are not a distraction from the
early ones — they are where the network learns what a break looks like,
and the early steps borrow that. The early region is information-limited,
not attention-limited: a few post-break points are a few post-break
points, whichever loss is looking at them.

---

## The local test sample is not a yardstick

The workspace ships a hundred labelled test series (`y_test.reduced`).
Scored through the assembled #28: **0.4958** — chance. The same pipeline
on a hundred fold-2 training series reads 0.7534, against 0.616 on the
full fold, so a hundred-series cross-section is far too small to read
anyway; and the reduced labels, though internally consistent with their
`tau_index`, may well be placeholders. Either way: not a third fold. The
cloud remains the only judge outside fold 2.

---

## 088 — submission #30: the run-length posterior ships, for the classifier

The two ingredients measured since #28, assembled: the classifier reads
206 channels (the BOCPD suffix, hazard 1/50, now a streaming module in the
library — `bocpd.py`, verified against the batch builder to 5e-7), the
ranker and the nets stay at 200, and twelve trajectory networks replace
six — #28's members and the six trained under the last-epoch rule. Fold-2
0.6169 against 0.6163. The assembler's channel check now reads each group
against its own width (nets 200, classifier 206, rankers 200: seventeen
counts). 1.7–2.4 ms per step. Shipped as **#30**, resources075.

**Cloud: 0.6007** (1 h 2 min) — the project's best, by 0.0003 over #28's
0.6004. Fold 2 had promised +0.0006; the cloud paid half of it, which is
the same coin as before: real ingredients, small ones, read through noise
of about ±0.001. Three cloud readings of this recipe (#28, #29, #30) now
sit at 0.5996–0.6007. That is the recipe's number, and the platform's
distance from 0.61 remains a new-modality-sized 0.009.

---

## 089 — submission #31: the blend turned toward the trees

Every cloud reading since the nets took over the blend has sat 0.016 below
fold 2, against 0.009 when the trees led (#23). Fold 2 cannot say whether
that is the nets overstating — at 0.55 trees / 0.45 nets it reads 0.6162,
at 0.45 / 0.55 it reads 0.6163. So the question goes to the cloud as a
paired test: #31 is #30 in every member and channel, with the blend at
0.55 trees / 0.45 nets. If the platform prefers the trees, the blend is
a lever the fold cannot see; if it does not, the gap is the fold's
optimism about everything equally, and the blend stays where it is.

**Cloud: 0.6000** against 0.6007 for #30. The platform does not prefer the
trees — if anything the reverse, by a hair inside the noise. The gap is
the fold's optimism about the whole ensemble, not about the nets, and
the blend stays at 0.45 / 0.55. Four cloud readings of the recipe now:
0.5996, 0.6000, 0.6004, 0.6007 — a spread of 0.001, which is also the
resolution below which the cloud cannot be asked anything.

---

## 090 — the prior that cannot be observed

Break positions are uniform in the online part: tau/L sits at 0.47–0.50 in
every quintile of L, and tau correlates 0.64 with L. So at step t, the
probability that a breaking series has already broken is t/L — and a
series close to its end is far more likely to be past its break than one
with a long way to go. As a score, (t+1)/L alone reads **0.6288** on fold 2
— above the entire ensemble — and blended 0.4 with #30 it reads 0.68.

It is not usable. A probe submission run through the platform's own
runner shows `x_online` arriving as a generator with no length: the
points are streamed over a connection one at a time, by design. Nor is L
recoverable from what *is* observed: the history length is uncorrelated
with it (0.005), and a regressor on eleven history statistics or on the
two hundred channels of the first step reaches R² ≈ 0 out of fold. The
prior exists in the data-generating process and is invisible at
inference — which also says the leaderboard's 0.65 is not built on it.

Recorded so nobody rediscovers it: the strongest single signal in the
dataset is the one the protocol withholds.

---

## 091 — the 2025 edition's series as training data (prepared)

The organisers' documentation admits the first edition's dataset — ten
thousand labelled series from the same family of generators ("changes in
mean, variance, distributional shape, dependence structure, and tail
behaviour") — as prior data, "not officially encouraged". For a data-bound
net half whose only real gain came from more series, that is the largest
untouched lever there is: it doubles the originals.

Two things had to be built before the data arrives. The first edition puts
its break exactly at the period boundary; to give it the real-time
edition's uniform tau, the boundary is moved *back* — the last k points of
the pre-break segment open the online part, so tau = k, with k uniform and
the history kept above a thousand points, then standardised on the history
as the platform does (`convert_first_edition.py`). And the channels are
built by the shipped monitor itself — the assembled #30's `TriMonitor`,
sharded eight ways — which reproduces the training matrices to 1e-6 and
covers ten thousand series in about twelve minutes
(`build_channels_fe.py`).

The data arrived by download instead of token: 10,001 series, 29% with a
break, pre-segments 1000–2499 and post-segments 250–999, unstandardised.
Converted: 9,948 series, online 252–999, tau/L median 0.27 (the backward
shift is bounded by the thousand-point history floor). The two editions
are relatives, not twins: the 2025 histories carry heavier tails (kurtosis
median 3.3 against 0.3) and weaker autocorrelation (95th percentile 0.46
against 0.59). Channels built in eleven minutes on eight shards, 7.77M
rows.

The honest cross-dataset reading: #30, trained on the real-time edition
only, scores **0.5668** on all 9,948 converted series — the ranker 0.5623,
the classifier 0.5607, the nets 0.5638. It transfers, with a loss of about
0.03 to the domain shift. Paired against 082b, nets with the 2025 series simply added to training
read **0.6020 / 0.5940** against 0.6057 / 0.6065 — worse on both seeds,
killed at two. Doubling the originals does not help when the doubling
comes from a shifted domain: the nets spend capacity on tails and
dependence structures the platform does not show them. The standard
remedy runs next as 091p — pretrain on the 2025 series, then fine-tune on
the real-time edition alone, so the shifted domain shapes the
initialisation rather than the final fit.

The ranker's verdict is harsher than the nets': with the 7.8M rows of the
2025 edition added to its cross-sections, 0.6043 → **0.5882**. A
cross-sectional ranker is hurt twice by a shifted domain — by the tails
it learns to rank on, and by a break rate of 29% against 50% that
rebalances every step's positives and negatives. The trees will not see
this data again. 091p (pretrain on the 2025 series, then fine-tune on the real-time
edition): **0.6047 / 0.6012** against 0.6057 / 0.6065. The pretrained
initialisation is worth nothing the real-time data does not already
teach, and on the second seed it costs 0.005.

Closed. Three ways of using ten thousand labelled series from the sibling
competition — mixed into the nets, mixed into the ranker, as a pretraining
stage — and none moved the untouched fold up. The platform's series are a
specific mixture (real-world and synthetic, standardised histories, breaks
uniform in the online part, half the series breaking), and the sibling's
mixture is different enough in tails, dependence and break rate that its
volume buys nothing here. The data lever, in every form the laptop can
pull it, is now measured flat.