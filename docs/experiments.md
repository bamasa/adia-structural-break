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
