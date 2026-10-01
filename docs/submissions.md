# Submission map

Two numbering schemes run through this repository: the directories under
`submissions/NNN-name/` carry the experiment number of the journal
(`docs/experiments.md`), which counts every hypothesis whether or not it
shipped, while the platform numbers (`#1`–`#45`) count only the uploads the
CrunchDAO platform received, in upload order. The table below joins them,
with the fold-2 reading and the cloud score where the journal states one; a
cell reads "—" when the journal gives no number.

| directory | platform # | what changed (short) | fold-2 | cloud | note |
|---|---|---|---|---|---|
| 001-classical-detectors | #1 [a] | CUSUM, Page–Hinkley and a variance ratio, combined by maximum | — | — | 5-fold CV 0.5385; cloud rank ≈400 of ≈1,500, no score recorded |
| 002-learned-combiner | — | logistic combiner over nine channels, first attempt | — | — | not submitted: CV 0.7487 was a sampling leak (held-out 0.4990) |
| 003-weighted-combiner | #2 | weighted (logistic) combiner, honest grouped validation | — | — | CV 0.5293; cloud score never recorded |
| 004-boosted-combiner | #3, #5 | LightGBM boosted combiner over the nine channels | — | — | CV 0.5568; #3 died importing the model (no requirements.txt), re-uploaded as #5; rank ≈330 [b] |
| 005-forty-channels | #4 | forty channels: multi-scale EWMAs and retrospective verdicts | — | — | CV 0.5648 |
| 006-with-cnn | #6 | a learned dilated-convolution channel joins (41 channels) | — | — | CV 0.5682 |
| 008-reverting-channels | #8 | nine reverting "current statistic" channels beside the peaks (50) | — | — | CV 0.5719 |
| 011-dual-view | #9 | every channel twice, raw and asinh views (100) | — | — | CV 0.5746 |
| 013-forecaster | #10 | forecaster prediction-error channels (104) | — | — | CV 0.5779 |
| 016-peak-hold | #11 | peak-hold on the output: fast attack, slow release | — | — | CV 0.5799 |
| 019-battery | #12 | per-prefix two-sample battery, 42 channels (146) | — | — | CV 0.5814; journal entry numbered 018 [c] |
| 020-battery-v2 | #13 | battery v2, forty more two-sample statistics (186) | — | — | CV 0.5821 |
| 021-resweep | #14 | hyperparameter resweep: 63 leaves, colsample 0.5 | — | — | CV 0.5841 |
| 022-rank-blend | #15 | lambdarank ranker beside the classifier, 0.6/0.4 blend, no hold | — | — | fold-0 0.6035; do not re-run (thread-pool timeout, see #16) [d] |
| 033-channel-net | #18 | channel-trajectory TCN joins the blend at weight 0.40 | — | 0.5877 | fold-0 0.6068; first recorded cloud score; #17 is the pair with the timeout fix [d] |
| 039-fold-ensemble | #19 | four fold-bagged rankers, four fold-ensembled nets, classifier | — | 0.5877 | fold-0 0.6099; rank 215; identical to #18 in the cloud |
| 044-nets-ablation | #20 | the four channel nets alone, trees removed | — | 0.5846 | fold-0 solo 0.6004; ablation |
| 045-net-bag | #21 | twelve-net bag on random 60% subsamples at weight 0.7 | — | 0.5810 | fold-0 0.6201, fold-1 0.5918; both folds burnt by selection |
| 047-heavy-members | #22 | eight heavy nets (88% of data, 16 epochs) at #19's weights | — | 0.5876 | no local number (private holdouts); paired A/B against #19 |
| 053-spectral | #23 | fourteen spectral channels (200), trees retrained | 0.5979 [e] | 0.5893 | fold-0 pair 0.6060; journal entry numbered 052 [c] |
| 055-trees-only | #24 | spectral ranker and classifier, no nets | — | 0.5853 | ablation: the nets are worth +0.0040 |
| 071-selected-nets | #26 | six best net members by private holdout on spectral trees | — | — | cloud reading never recorded [f] |
| 072-augmented-ranker | #27 | augmented and clean rankers, clean classifier, eight nets | 0.6106 | — | cloud reading never recorded [f] |
| 073-aug3-nets | #28 | six nets on the triple boundary augmentation, 0.45/0.55 blend | 0.6163 | 0.6004 | resources073; meta file says 0.6152 [g] |
| 074-aug3-full | #29 | #28 with the nets trained on every series, fold 2 included | — | 0.5996 | resources074; no local number by construction |
| 075-bocpd-clf | #30 | BOCPD run-length suffix for the classifier, twelve nets | 0.6169 | 0.6007 | resources075; later tables quote 0.6167; Selected until #36 |
| 076-tree-weight | #31 | #30 with the blend turned to 0.55 trees / 0.45 nets | 0.6162 | 0.6000 | paired cloud test; blend stays 0.45/0.55 |
| 077-selfnorm | #32 | #30 plus self-normalisation by the first ten steps | 0.6172 | 0.5991 | did not transfer (−0.0016 in the cloud) |
| 078-two-classifiers | #33 | #32 with a wide LightGBM averaged into the classifier slot | 0.6184 | 0.5991 | resources078; a null against #32 |
| 079-final-pool | #34 | #33 with twenty-four networks instead of twelve | — | — | resources079; never run; fold 2 blind to the pool |
| 080-pool24 | #35 | #30 with twenty-four networks, nothing else changed | — | 0.6009 | resources080; fold 2 blind to the pool; read 19 September [h] |
| 081-mass-member | #36 | ninety-channel mass battery as its own member at 0.25 | 0.6205 | 0.6046 | resources081; the gain transferred in full |
| 082-mass-pool24 | #37 | #36 with the network pool doubled to twenty-four | 0.6205 | 0.6048 | resources082; Selected for a while |
| 083-freqdep-member | #38 | frequency/dependence member at 0.20 under a slow classifier | 0.6232 | — | resources083; never run |
| 084-novelty-union | #39 | frequency+novelty union replaces the frequency member from step 100 | 0.6251 | 0.6056 | Selected for a while; the weak members transferred at a sixth |
| 085-dependence-cusum | #40 | dependence-CUSUM member at a fifth from step 300 | 0.6265 | — | never run |
| 086-whitened-member | #41 | whitened stream (AR(p) by BIC, conditional scale, ECDF) as a member at 0.30 | 0.6369 | 0.6186 | rank 222; meta file says 0.6358 [g] |
| 087-whitened-ranker | #42 | whitened member rebuilt: Shiryaev–Roberts odds, per-step ranker, 0.40 | 0.6405 | — | never run |
| 088-whitened-nets | #43, #44 | history context, a bag of two rankers, three whitened-channel networks | 0.6439 | 0.6277 | #43 is an identical copy of #44 (duplicate push); #44 was the one run |
| 089-joint-nets | #45 | pool of three networks over the core's 200 and the whitened 111 channels | 0.6452 | 0.6299 | **final selection**; meta file says 0.6453 [g] |

Platform numbers with no directory of their own: **#7** (never mentioned in
the journal), **#16** (journal entry 028: the #15 blend with the ranker
lengthened to 600 trees; timed out after seven hours, must not be re-run),
**#17** (the #16 pair with the thread-pool fix, shipped beside #18), and
**#25** (a broken upload: a `channels[:186]` slice left from before the
spectral family; must not be run).

The fold-2 column is empty before #27 because fold 2 entered the protocol
only at experiment 056; earlier submissions were validated on grouped
5-fold CV (#1–#14) or on fold 0 (#15–#22), and that number is given in the
note column.

## Footnotes

- **[a]** The journal never writes "#1" for directory 001. It states
  "003 = submission #2, 004 = submission #3", and `docs/method.md` (section
  5) gives "#1" the rank the 001 entry reports (about 400 of 1,500), so the
  mapping is taken from those two statements together.
- **[b]** The rank "about 330 with 004" is from `docs/method.md`, not the
  journal; the journal itself records no cloud result for #3 or #5.
- **[c]** Two directory numbers do not match their journal entry: the
  battery shipped as #12 is journal entry 018 but lives in
  `submissions/019-battery/` (the journal has no entry 019; 017 was the
  TCN pilot), and the spectral family shipped as #23 is journal entry 052
  but lives in `submissions/053-spectral/`. The README ledger follows the
  journal for the first (row "018") and the directory for the second (row
  "053"). The `main.py` docstrings confirm the contents match the entries
  named here.
- **[d]** #15 and #16 both shipped the sklearn-wrapper predict that spawned a
  thread pool per row; #16 ran seven hours to a timeout, and the journal
  rules that neither is to be re-run. #17 is the same pair with the fix and
  #18 the triple; only #18 has a cloud reading.
- **[e]** The fold-2 number for #23 is retrospective: the configuration
  "that scored 0.5893 in the cloud" was scored on fold 2 at 0.5979 in
  entries 057–072 (the 056 pair-only reading was 0.5967). Fold 2 did not
  exist as a ruler when #23 shipped.
- **[f]** The journal says "the cloud numbers for #26 and #27 will say where
  the 0.6116 really landed" and never records them. #28 is then described as
  the first move above 0.5893, so whatever #26 and #27 read, if they were
  run at all, did not exceed it.
- **[g]** Three `meta_NNN.txt` files in the directories disagree with the
  journal by a few ten-thousandths: 073 says fold-2 0.6152 (journal 0.6163),
  086 says 0.6358 in the blend and 0.6088 alone (journal 0.6369 and 0.6150,
  after the step index was unclipped), 089 says 0.6453 (journal 0.6452). The
  table follows the journal.
- **[h]** `docs/method.md` (section 5) says no cloud number is recorded for
  #35; the journal's "Notes on the record (29 September)" gives 0.6009, read
  on 19 September. The table follows the journal.
- **Agreement with the external list of cloud scores.** Every cloud score
  the journal states (#18, #19, #20, #21, #22, #23, #24, #28–#33, #35–#37,
  #39, #41, #44, #45) matches the figures supplied for this map; no
  disagreement was found. The journal quotes #30's fold-2 reading as both
  0.6169 (entry 088, the #39 transfer table) and 0.6167 (entries 142 and
  144); the table keeps the shipping entry's 0.6169.
- **Unmapped directories.** Only `002-learned-combiner`, which was never
  submitted (its 0.7487 CV was a length-dependent sampling leak); it has no
  platform number by design. Every other directory maps to at least one
  platform number.
