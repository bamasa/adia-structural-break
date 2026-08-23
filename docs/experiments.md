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
