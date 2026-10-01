# Method: how the search was run

The research pipeline behind the solution — the loop one experiment goes
through, the rules it accumulated, the research layer that broke the plateau,
and the division of labour between the owner and the assistant — as the
repository records it. Every number comes from `docs/experiments.md` (the
journal) or `README.md`. Plain numbers (114, 140c) are journal entries, `#n` is
a platform submission, and a three-digit directory under `submissions/` is a
shipped unit (`088-whitened-nets` is #44).

## 1. The loop

Each stage was added after the failure it prevents; the journal names that
failure.

**Hypothesis, kill condition first.** The journal's header states the
convention: "the kill condition is written before the test, and a result only
counts on data the choice never saw". An entry opens with the hypothesis and
the condition — "adopted only if better on most folds" in the five-fold era,
later a member-level acceptance such as "+0.003 per member" (107).

**Cheap local measurement on the untouched fold.** Training rows — one per
online step of every series, 5,036,517 in the first honest build (003/004) —
sit in the workspace as one `.npy` block per channel family (`X40.npy` …
`WHITE90.npy`, with `Y40`/`G40`/`S40` for labels, series ids and steps). Folds
come from `split_by_series(g, folds=5, seed=0)` in
`src/structural_break/combiners.py`; fold 2 has been the ruler since 056
(`virgin_fold.py`), after folds 0 and 1 were burnt by selection (#19, #21);
folds 3–4 are reserves. The metric is `ts_auc` in the same module, which the
survey of 27 September found identical to the organisers' `scoring.py`. Every
member's fold-2 predictions are cached (`fold2_*.npy`, `fold2_sig_*.npy`), so
blends, gates, share sweeps, step-range tables and Spearman correlations with
the shipped ensemble are read from caches in seconds (136, 142, 143). A fast
tree screen (half the series, 150 trees, about four minutes) is trusted within
one family only (027–029); networks are compared paired on the three 082b
seeds. Each experiment is one script in `scripts/experiments/`.

**Threshold.** A reading counts only if it clears the bar in force (section 2)
on a plateau, and — for a member — alongside the two numbers that predict
transfer: strength alone and Spearman correlation with the ensemble.

**Streaming implementation, verified against the batch matrices.** A channel
family that earns its place is rewritten as a streaming class in
`src/structural_break/` (`bocpd.py`, `mass.py`, `freqdep.py`, `novelty.py`,
`depcusum.py`, `white.py`) and its rows are compared with the batch matrix the
model was trained on, to between 1e-7 and 3e-6 (088, 114, 133, 140c, 145, #44),
on series that include the shortest history. The unit tests in `tests/`
(standard library; `PYTHONPATH=src python -m unittest discover -s tests -v`)
guard the same places.

**Assembler with width checks.** `scripts/assemble_submission.py` concatenates
the library modules in dependency order, strips imports and docstrings,
prepends a header (`INFER_PARALLELISM = 8`) and appends the submission's
`interface_NNN.py`. Its `verify_channels` compares every width and offset the
interface declares (`NET_CHANNELS`, `CLF_CHANNELS`, `MASS_OFFSET`,
`WHITE_CLF_WIDTH`, …) with the artifact — `num_feature()` of every booster,
the input width of every network, the length of every normalisation vector —
and refuses on any disagreement. Eleven checks at #28, thirty-one at #38.

**Ship script.** `scripts/ship_submission.sh <dir> <interface> <meta>
<resources> "<message>"` runs under `set -euo pipefail`: assemble, copy
`requirements.txt`, run the verifier that must sit next to the interface
(`verify_NNN.py`; without one it refuses), copy into the crunch workspace,
`crunch push`. The verifier runs the assembled monitor on two training series
against the memory-mapped matrices and asserts offsets, widths and
milliseconds per step. A shipped unit such as `submissions/088-whitened-nets/`
holds `build_088.py` (the artifact composed from its parts), `interface_088.py`,
`meta_088.txt` (the `main.py` docstring), `verify_088.py`, the assembled
`main.py` and `requirements.txt`. `main.py` is never edited by hand.

**One change per cloud run.** #29 is #28 with the networks retrained on every
series; #31 is #30 with the blend turned toward the trees; #35 is #30 with
twenty-four networks; #37 is #36 with twenty-four; #24 is #23 without the
networks. #34 changed two things at once, and "its number could never have
answered the question".

**Journal entry.** What was built, the readings (alone, correlation, blend gain
by share, gain by step range), the verdict and the reason; then the README row.

## 2. The rules, and the experiment that taught each

| Rule | Taught by |
|---|---|
| Validation must be built like evaluation; a number that looks too good is a leak until shown otherwise | 002: CV 0.7487 against 0.4990 on the labelled hundred — length-dependent subsampling. 031: `len // 4` leaked the online length. 080: a 0.7255 holdout beside 0.5678 on fold 2 — the holdout's pseudo-series were in training |
| A fold that is optimised against stops transferring; keep one untouched and spend it slowly | #19 scored 0.5877, identical to #18, after dozens of fold-0 decisions; #21 scored 0.5810 a day after fold 1 joined; 056 designated fold 2, with folds 3–4 "to be used once each and never for tuning"; after #32/#33, gains under 0.002 "are no longer evidence" |
| A member pays when it is built on its own inputs and trained on its own; independence is necessary, not sufficient | 114: the same ninety channels appended read 0.5965, as their own classifier 0.6201 (Spearman 0.68 against 0.84); #36 paid +0.0039 in the cloud for +0.0038 on the fold. 115–123: every rebuild of the paying member lands at 0.89–0.95 correlation, and the raw window is independent (0.32) but weak (0.5319) |
| Weak members transfer at a sixth, strong ones one-for-one; a member ships only when strong alone (0.57+) with a fold-2 gain above 0.003 | #36 (0.5830 alone): +0.0038 → +0.0039. #39's two weaker members (0.5400, 0.5670): +0.0046 → +0.0008. #41 (0.6150 alone): +0.0118 → +0.0130 |
| Select on a plateau, never at the peak of a grid | 094/#32: the best of forty settings promised +0.0005 and paid −0.0016; 118 refused +0.0009 chosen from nine options at 0.935 correlation; 140c shipped at +0.0019 "flat across T, with a mechanism behind it rather than a sweep" |
| Train to the end, take the last epoch, keep the holdout in training | 081: fold 2 rose monotonically to epoch 8 while the clean holdout peaked at epoch 0; 082: fifteen epochs 0.5936 against 0.6036; 082b: five of six members improved; 076b: holdouts anti-correlated with fold 2 |
| The streaming code must reproduce the batch matrix, and the push must depend on the check | #25 died in the cloud on a `channels[:186]` slice left from before the spectral family: the verifier caught it and the push ran anyway, chained with `;`. 140c: writing the streaming novelty module exposed a batch channel that had been constant all along (a NaN inside a median). Both are now tests and script conditions |
| Profile before shipping | #16 ran seven hours to a timeout with 21 quota-minutes used (sklearn thread pools per one-row predict); the fix, `booster_.predict(num_threads=1)` at 0.77 ms/step, became a gate; 107 measured its cost with random weights before training |
| One variable per cloud run | #34 carried two changes; #35 and #37 asked the pool question alone (+0.0002) |
| Memory discipline: one heavy job at a time, memory-map the matrices, never stack them in float64 | 152: a 20 GB pool cancelled with the machine in swap (load average 116); 154: a third ranker tipped it into swap and was killed; #44: the verifier's 21 GB float64 stack swapped the machine for twenty minutes and was rewritten to `mmap_mode="r"`; 079 stored 17.7 GB as float16 |
| Never hand the model the base rate | 002/010: series context and the step index destroy the combiner; 052: the step index cancels the spectral gain; 096: an accumulating posterior teaches a time dependence the cross-section does not reward |
| When a family saturates, model work does nothing and a new modality moves it | #18–#22 sat at 0.5877 through count, weights, quality and stacking; #23 moved to 0.5893 with fourteen spectral channels; 114 and 145 repeated the pattern |

## 3. The research layer

**The anatomy of the ceiling (109, 109b, 111).** Before any external source was
read, the assistant classified every breaking series in fold 2 by what the break
changed in 300-point windows and measured the ensemble's per-series AUC:
variance-and-shape 0.705, variance only 0.686, shape only 0.649, dependence only
0.583, and "none of the above" — 353 series, 40% — at 0.542. 109b tried
nineteen further statistics on those and found nothing: the largest ratio of
break-spread to control-spread was 1.11, and the picture was the same at 600
points. 111 added that on steps 0–30 the whole ensemble (0.5373) ranks no
better than a classifier that has seen only the history's shape (0.5385). 112
then corrected the journal's own conclusion: fifty leaderboard entries between
0.640 and 0.664 meant the ceiling was not the task's.

**Survey one — public research** (`docs/research/public_research_2026-09-27.md`).
Sources: organiser pages and FAQ, the runner and scorer source, forum threads,
the 2025 winners' write-ups and code, the leaderboards; every fact tagged quote,
inferred or participant claim. It asked what the organisers say about the data,
what the runner and scorer do, what the leaders build that this project does
not, and what to try in what order. Findings: the board's top was 0.680, rank
50 0.642, rank 200 0.621, the project at about rank 300 of 1,688; `x_online`
is a true generator with no length, predictions are cast to float32, the
scorer is exactly `ts_auc`, the quota is 15 hours a week. The strongest
documented real-time approach (0.6263 public) rested on whitening the stream
properly — AR(p ≤ 12) by BIC, a conditional scale, the empirical CDF of the
innovations — its author measuring raw statistics as "about 2.5x worse"; then
per-series null calibration; then Bayes factors integrated over the change
time. Participant-measured dead ends matched the journal's (networks over
trees +0.0007 at most, features +0.0003 per doubling). The project's stack
whitened with AR(1) only. That was the gap.

**Survey two — generator forensics** (`docs/research/generator_forensics_2026-09-27.md`,
scripts in `scripts/experiments/forensics/`). The training series taken apart
by the assistant's own scripts, no external source: every history standardised
on itself; stationary ARMA-type processes (AR(p ≥ 3) 3413, white noise 2424,
MA(1) 1295, ARMA(1,1) 1250, AR(1) 866, AR(2) 752), a quarter GARCH-like,
innovations Gaussian in 5663 series and Student-t in 4200. Against
length-matched controls only two kinds of change carry excess signal: an
innovation-variance change, almost entirely increases, continuous with no
minimum; and a shift of the AR coefficients in AR-type histories (effect sd
0.06–0.07). Mean, shape, tails and volatility clustering show no excess. The
40% that 109 called invisible are the small-magnitude tail of the same two
kinds; about thirty percent of breaks show no measurable change in any of
forty-nine statistics. A synthetic generator built from the fitted
distributions matches the dependence structure but not the tails or the
volatility clustering; it is not a training source yet.

**What changed.** The survey's first idea and the forensics' first two
consequences became 145, built and measured in the local loop: the whitened
stream with a ninety-channel battery as its own member — 0.6150 alone, blend
0.6369 against 0.6251 (+0.0118), gaining on every step range. Two details came
from the loop, not the sources: the first draft ran everything on the
conditional stream and could not see a variance break, so the battery moved to
the unconditional one; and unclipping the step index gained the member 0.006.
Shipped as #41, it scored 0.6186, +0.0130, rank 222. The third consequence
became 147, Shiryaev–Roberts odds (member 0.6172 against 0.6150, +0.0010 in the
blend), folded into #42 with a per-step ranker (148: 0.6216 alone, blend
0.6405). The survey's second idea, per-series null calibration, lost in 149
(0.6102 against 0.6186: "the whitening is the calibration"); its fifth, the
recent tail of the history as reference, lost in 146 (+0.0004); its sixth,
history-family context, gave +0.0012 in 153 and shipped in #44. The float32
hygiene item is not recorded as tested.

## 4. Division of labour

The journal is written by the assistant and records the owner's decisions where
they set the course. The owner set the targets the journal works toward (the
0.65 target recorded in the cloud-calibration row of docs/experiment_table.md; "the target is fold-2 0.617;
nothing is submitted until it is met", 067–068), ran the platform side — the
cloud runs and their budget (15 hours a week; "five cloud runs a day are the
only ruler that cannot be burnt"; at #39, "room for two or three more runs
before the deadline"), the Selected mark, the GPU box — and pushed the
direction at the turning points the journal attributes explicitly: the fold-0
protocol ("by the owner's call", 025), the network builds ("the owner in the
loop on architecture", 023, 037), the new family after the laptop era
("whatever new family the two of us design next", 049). Ties went to the
owner: whether a tie is sent to the cloud as a paired run (#31, #35), which of
two identical uploads is run (#43 or #44), and the final call on what to ship.
The assistant built the matrices, members, streaming modules, assembler, ship
script and verifiers, ran and measured the experiments, and wrote the tests,
the journal, the README tables and the two surveys.

## 5. Timeline of the score

The journal gives no cloud TS-AUC for #1–#17, only ranks (about 400 of 1,500
for #1; about 330 with 004). From #18 on, every move of the project's best:

| Cloud | Submission | Lever |
|---|---|---|
| 0.5877 | #18 | a 111k-parameter TCN over the 186 channel trajectories joins the trees at 0.40 (033) |
| 0.5893 | #23 | fourteen spectral channels — the first new modality since the forecaster (052) |
| 0.6004 | #28 | six networks on the triple boundary augmentation, 26,887 pseudo-series (076) |
| 0.6007 | #30 | the BOCPD run-length suffix for the classifier, twelve networks (088) |
| 0.6046 | #36 | ninety independently built statistics as their own member (114) |
| 0.6048 | #37 | the network pool doubled to twenty-four (120) |
| 0.6056 | #39 | the frequency/dependence member and the gated frequency+novelty union (131b, 140c) |
| 0.6186 | #41 | the whitened stream as a member (145), from the two surveys |
| 0.6277 | #44 | the whitened member rebuilt: history context, a bag of two rankers, three networks over the whitened channels (152, 153) |
| 0.6299 | #45 | three networks reading the core's 200 channels and the whitened 111 together (152a) — the final selection |

Readings that did not move it, each answering one question: #19 0.5877 (nine
models, fold-0 tuned), #20 0.5846 (nets alone), #21 0.5810 (twelve-net bag),
#22 0.5876 (heavy members), #24 0.5853 (trees alone), #29 0.5996 (nets on all
data), #31 0.6000 (blend toward trees), #32 0.5991 (self-normalisation), #33
0.5991 (wide classifier), #35 0.6009 (a variant of #30). #34, #38, #40 and
#42 were never run; #43 is an identical copy of #44. The final selection is
#45 at 0.6299; the organisers' out-of-sample evaluation follows at the end
of October 2026.

## 6. What did not work

| Group | Experiments | The journal's verdict |
|---|---|---|
| Channel families appended to one model | 005 retro scans, 007 window classifier, 009 rank view, 012 reversion depth, 014 deviation view, 015 extended forecaster, 024 pruning, 030 Chronos-2, 032 interactions, 034 velocities, 054 spectral v2, 057–060 wavelets / rank tests / matched filters, 096 AR change-point filter, 097 divergences, 110 autocorrelation, 112 GLR scan | "better in isolation, redundant in ensemble" (007); "two fixes for one disease do not stack" (009); "this ensemble is not short of detectors" (112) |
| Preprocessing | 011a rolling median, 011b asinh as replacement | the filter "manufactures serial dependence"; a gate without retraining "can only suggest, never adopt" |
| Network scale and architecture | 035 tripled, 036 seeds, 037 score-fed, 043 long training, 046 bag to 72, 047 heavy members, 048 two-tower, 049 pretraining, 077 capacity, 080 batch 96, 082 fifteen epochs, 084 raw input, 086 GRU, 087 early specialists, 107 511-step field, 108 raw-signal net, 121 nets on mass, 128 nets on spectra | "data-bound, not capacity-bound" (035); "averaging is dead in this project" (036); "member quality dominates count" (046); "reach hurts" (107); "the channels are the ceiling" (108) |
| More or other data | 079 nine slices, 079b, 091 the 2025 edition three ways, 105a mirrors, 105b backward boundary, 119 augmentation for the mass member | "the ratio was the thing, not the variety"; "a shifted domain, its volume buys nothing"; "the augmentation axis is exhausted" |
| Loss and labels | 050 pair-proportional weights, 083 step weights, 102 evidence-weighted positives, 103 masked positives, 104 step specialists | "equal-step weighting regularises"; "the post-break rows are not label noise"; "the early gain dissolves in the blend" |
| Learned combination | 027 seed ensembles, 029 / 040b / 099 stacking, 074 third ranker breed, 113 rank space, 125 weak members combined, 138 monotone meta | "the fourth learned meta to lose to a hand blend"; "redundancy does not average away" |
| Output transforms | 051 history-level calibration, 092 entry/exit, 094–095 self-normalisation (#32: −0.0016), 098 calibration, 143 memory, 149 per-series null | "the score already carries its memory"; "self-normalisation does not transfer"; "the whitening is the calibration" |
| Members outside the paying zone | 115 rejected detectors (0.80), 116 raw window (0.32, weak), 117 rank battery, 118 wider battery (0.935), 122 whitened battery (0.939), 123 online reference (0.952), 124/126 spectra (0.12, weak), 129 autocorrelation (0.35, weak), 134, 135, 137, 139 tails (0.86), 141, 146, 150 (in reserve) | "independence is necessary, not sufficient"; "on the independent side strength is bought with correlation"; "a model on the union is the stronger set's model" |
| Closed by the data | 090 online-length prior (0.6288 alone, unobservable), 093 history-shape prior (0.511), 111 near-null prior, 109/109b | "the strongest single signal in the dataset is the one the protocol withholds" |
| Infrastructure | 004 no requirements, 006 stale `__pycache__`, #16 timeout, #25 broken upload, #43 duplicate push, 042 OOM, 154 swap | each became a script condition, a test or a rule |

## 7. Reusing the pipeline on another problem

The minimal set, in the order it is needed:

1. **The scorer, reimplemented exactly** (`ts_auc`), checked against the
   organisers' code and used for every local number.
2. **Row-level matrices built once per feature family**, grouped folds fixed
   by seed, one fold designated untouched before the first decision, and every
   member's out-of-fold predictions cached on it. The leak checks (validation
   built like evaluation, no series-level context, no step index) come before
   the first model.
3. **A library with batch builders and streaming twins**, and a tolerance
   assertion between them in unit tests. A channel that cannot be streamed to
   1e-6 of its batch version does not ship.
4. **An assembler** that produces the one-file submission from the library and
   refuses on any width mismatch between interface and artifact.
5. **A ship script** that chains assemble, verify (channels, widths, ms/step)
   and push under `set -e`, and refuses without a verifier.
6. **The journal**, one entry per experiment with the kill condition written
   first, and the README table updated on every push.
7. **One change per cloud run**, and a running calibration of the local-to-
   cloud offset (here 0.016–0.020) so a local gain is converted into an
   expected platform gain before the run is spent.
8. **The research layer, when the loop plateaus**: a survey of what the leaders
   build and what the protocol is, and forensics of the data itself — every
   fact tagged by source, every idea sent back through the loop, not adopted.
