# Reproducing the workspace and one fold-2 number

This is the honest version of "how to rebuild it": what the committed scripts
can regenerate, in which order, with the commands the scripts themselves
document, and what they cannot. Where a matrix or a cache was produced by code
that never reached the repository, the guide says so instead of inventing a
command. Every timing below is read from the workspace logs or the journal
(`docs/experiments.md`); the machine was a 32 GB laptop with an Apple GPU (the
networks train on `torch.device("mps")`).

The end of the guide is one command sequence that reads the fold-2 caches and
prints the TS-AUC of the shipped blend: **0.6452** for #45 (`submissions/089-joint-nets`),
0.6439 for #44, 0.6251 for #39, 0.6167 for the #30 core — the numbers the
journal records for those submissions.

## 1. Layout

The repository is one directory inside a larger **workspace**; every
experiment script is run from the workspace root and finds the repository as
`repo/`:

    <workspace>/
      repo/                                the git repository (this file is repo/docs/reproduce.md)
      .venv/                               the Python environment
      structural-break-real-time-test/     the crunch workspace: data/, resources/, main.py
      X40.npy ... HISTCTX8.npy             training matrices, one row per online step (5,036,517 rows)
      Y40.npy  G40.npy  S40.npy            label, series id and step index per row
      AUG3_*.npy                           the triple boundary augmentation (11,159,663 rows)
      fold2_*.npy  nets_*/fold2_logits_*   fold-2 predictions of the members (1,006,223 rows)
      resourcesNNN/model.joblib            the shipped artifacts
      *_parts/                             shard outputs of the builders, merged into the matrices

Scripts open matrices by bare name (`np.load("X40.npy")`), import the library
with `sys.path.insert(0, "repo/src")` and read the data from
`structural-break-real-time-test/data/`. So:

    cd <workspace>
    .venv/bin/python repo/scripts/experiments/<script>.py ...

## 2. Environment

The venv actually used, read from `importlib.metadata` on 1 October 2026
(the venv has no `pip`, so `pip list` is not available there):

| package | version | used by |
|---|---|---|
| Python | 3.12.13 | everything |
| numpy | 2.5.2 | everything |
| scipy | 1.18.1 | the library (`ndtri`, `gammaln`, `lfilter`), the battery |
| lightgbm | 4.7.0 | trees: classifiers, rankers, the forecaster |
| scikit-learn | 1.9.0 | `ExtraTreesClassifier` in 077, the forensics |
| torch | 2.13.0 | the trajectory networks (MPS backend) |
| pandas | 2.3.3 | reading the parquet data |
| pyarrow | 25.0.1 | the parquet engine behind `pandas.read_parquet` |
| joblib | 1.5.3 | the artifacts |
| matplotlib | 3.11.1 | `scripts/make_figures.py` |
| pillow | 12.3.0 | the figures |
| crunch-cli | 11.11.1 | `crunch setup`, `crunch test`, `crunch push` |

The root `requirements.txt` pins exactly these. The platform side needs only
`submissions/requirements.txt` (lightgbm, scikit-learn, scipy), which
`scripts/ship_submission.sh` copies into every submission.

    python3.12 -m venv .venv
    .venv/bin/pip install -r repo/requirements.txt crunch-cli==11.11.1

The networks were trained on the laptop's GPU through MPS; the scripts hard-code
`DEVICE = torch.device("mps")`. On a CUDA machine the core pool has its own
trainer, `scripts/gpu/train_nets_cuda.py` (see `scripts/gpu/README.md`); the
whitened and joint pools have no CUDA variant.

## 3. Data

The competition data is fetched by the crunch CLI into the crunch workspace
(from `README.md`, "Reproducing"):

    pip install crunch-cli
    crunch setup structural-break-real-time <name> --token <token>

The directory it creates must sit in the workspace as
`structural-break-real-time-test/`. Its `data/` holds `X_train.parquet`
(218 MB: MultiIndex `id, time`; columns `value`, `period` with 1 = history,
2 = online), `y_train.parquet` (per-step `target`), `y_train_index.parquet`
(`tau_index`, `tau` per series), and the reduced hundred-series test sample
(`X_test.reduced.parquet`, `y_test.reduced.parquet`, `y_test_index.reduced.parquet`).
Ten thousand training series, 5,036,517 online steps in all.

## 4. Conventions every builder shares

* **Row alignment.** A matrix has one row per online step of every training
  series, in the order of `G40.npy` (series id) and `S40.npy` (step). The sharded
  builders write `<name>_parts/part_<shard>.npz` per shard and their `merge`
  step places each series' block at the rows `G40`/`S40` dictate — so `G40.npy`
  and `S40.npy` must exist before any merge. For the pseudo-series the same
  role is played by `AUG3_G.npy` / `AUG3_S.npy`.
* **Shard / merge CLI.** `python build_<x>.py <shard> <n>` runs one shard
  (series with `id % n == shard`), `python build_<x>.py merge <n>` assembles the
  matrix. Eight shards were used throughout. Several builders also have a
  `test` mode on synthetic series.
* **Memory.** Never stack the matrices in float64 (21 GB). Load with
  `mmap_mode="r"` and gather the rows you need (`CLAUDE.md`, memory limits).
* **Folds.** `split_by_series(G40, folds=5, seed=0)` from
  `structural_break.combiners`; fold 2 is the evaluation fold. `ts_auc` from the
  same module is the platform metric line for line.

## 5. Build order

Each step names the script, the CLI the script documents, what it needs, and
the time read from the logs. Steps 2 to 5 depend only on the data and on
`G40`/`S40`; step 6 depends on the core (step 1) and on a shipped artifact.

### 5.1 The core: labels, ids, steps and the 200 channels — not rebuildable from committed scripts

`X40.npy` (40), `C_cnn.npy` (1), `N9.npy` (9), `X50a.npy` (50), `E4.npy` (4),
`B40.npy` (42 — the name predates the count), `B2.npy` (40), `SPEC14.npy` (14)
are read, in that order, as the 200 core channels; `Y40.npy`, `G40.npy`,
`S40.npy` are the label, series id and step per row. They were built in
August 2026, one family at a time, as experiments 005–013, 018, 020 and 052
adopted them:

| matrix | what it is (journal entry) | builder in the repository |
|---|---|---|
| X40 | the forty streaming channels (005) | **none** — never committed (the git history has no deleted builder either) |
| C_cnn | the dilated-convolution channel (006) | **none** |
| N9 | the nine reverting "current statistic" channels (008) | **none** |
| X50a | the fifty channels on the asinh view (011) | **none** |
| E4 | the forecaster's prediction-error channels (013); needs the pretrained forecaster | **none** |
| Y40, G40, S40 | label, series id, step per row | **none** |
| B40 | the per-prefix two-sample battery (018) | `scripts/experiments/build_battery.py` — single process, no shards: `python build_battery.py`; 486 s |
| B2 | battery v2 (020) | `scripts/experiments/build_battery2.py` — single process; 493 s |
| SPEC14 | the spectral family (052) | `scripts/experiments/build_spectral.py` — single process; 556 s |

What *is* in the repository is the streaming code that produces all 200 at
once: the `TriMonitor` of any assembled submission from 053 on emits the 200
channels (plus the later suffixes) step by step, and the batch matrices were
verified against it to 1e-4 before every push. Two committed scripts use
exactly that route to build 206-wide rows for other series sets:

    # pattern: scripts/experiments/build_channels_fe.py / build_channels_any.py (091)
    #   TriMonitor of repo/submissions/075-bocpd-clf/main.py + the forecaster from resources075/model.joblib
    SERIES_PREFIX=<prefix> python repo/scripts/experiments/build_channels_any.py <shard> 8   # x8
    SERIES_PREFIX=<prefix> python repo/scripts/experiments/build_channels_any.py merge 8
    # -> <prefix>_X206.npy (200 core + 6 BOCPD, hazard 1/50), <prefix>_Y.npy, <prefix>_G.npy, <prefix>_S.npy

They read `<prefix>_series.parquet` / `<prefix>_index.parquet`, not
`X_train.parquet`; pointing them at the training data is a small edit that is
not in the repository, so it is described here and not claimed. Expect hours
in a single process: `build_aug3.py`, which walks 11.2 M steps through the
053 `TriMonitor`, took 10,905 s. Two caveats stand even then:

* The forecaster behind `E4` is a LightGBM regressor pretrained on every
  history (`train()` in any `interface_NNN.py` fits it as `base`, saved as
  `model["forecaster"]`). It was trained once, for 013, and carried forward
  unchanged: the model string in `resources013/model.joblib` is identical
  (same SHA-1) to the one in `resources053`, `resources072`, `resources075`
  and `resources089`. A rebuild through a `TriMonitor` that loads the
  forecaster from any of these artifacts therefore uses exactly the one
  `E4.npy` was built with; retraining it would change `E4` slightly.
* The verifiers (`submissions/*/verify_*.py`) compare the streaming monitor
  with the stored matrices on two series and assert agreement below 1e-4
  (1e-2 for the whitened block) — the only recorded check that the stored
  matrices and the streaming code agree.

### 5.2 The run-length posterior for the classifier

    BOCPD_H=50 BOCPD_OUT=BOCPD6_H50 python repo/scripts/experiments/build_bocpd.py <shard> 8   # x8, 94 s per shard
    BOCPD_H=50 BOCPD_OUT=BOCPD6_H50 python repo/scripts/experiments/build_bocpd.py merge 8

`build_bocpd.py` (085) defaults to hazard 1/200 and `BOCPD6`; the shipped
suffix is hazard 1/50. Depends on the data and `G40`/`S40`. The journal: "built
for all ten thousand series in a minute across eight shards".

### 5.3 The mass battery (member of #36)

    python repo/scripts/experiments/build_mass.py <shard> 8      # x8, 23 s per shard
    python repo/scripts/experiments/build_mass.py merge 8        # -> MASS90.npy

`MASS_OUT` renames the output (default `MASS90`). 0.028 ms per step.

### 5.4 The frequency / dependence / novelty channels (members of #38 and #39)

    python repo/scripts/experiments/build_mspec.py  <shard> 8 ; ... merge 8   # MSPEC32.npy, 93 s per shard
    python repo/scripts/experiments/build_mspec2.py <shard> 8 ; ... merge 8   # MSPEC60.npy, 260 s per shard
    python repo/scripts/experiments/build_acf.py    <shard> 8 ; ... merge 8   # ACF8.npy, 15 s per shard
    python repo/scripts/experiments/build_knn.py    <shard> 8 ; ... merge 8   # KNN20.npy, 483 s per shard

`build_mspec2.py`'s docstring still describes `build_mspec.py`; its own
constants (windows 64–1024, ten bands, `MSPEC_OUT` default `MSPEC60`) are what
runs. `KNN20.npy` is the matrix that ships; `build_knn2.py` → `KNN20B.npy` is
the rebuild with the median-distance channel alive (141), which was measured
and not adopted.

### 5.5 The whitened stream, the odds and the history context (the #41–#45 member)

    python repo/scripts/experiments/build_white.py   <shard> 8 ; ... merge 8   # WHITE90.npy, 31 s per shard
    python repo/scripts/experiments/build_sr.py      <shard> 8 ; ... merge 8   # SR22.npy (21 channels), 25 s per shard
    python repo/scripts/experiments/build_histctx.py                           # HISTCTX8.npy, no shards, 41 s

`build_sr.py` and `build_histctx.py` import `fit_history` /
`normal_scores_online` from `build_white.py` in the same directory. `SR22`
holds 21 channels; the name was kept. The streaming counterpart is
`src/structural_break/white.py` (`WhiteMonitor(history, odds=True, context=True)`),
verified against these three matrices to 3e-6.

### 5.6 The triple boundary augmentation (the networks' training data)

    python repo/scripts/experiments/build_aug3.py          # AUG3_X/Y/G/S.npy; single process, 10,905 s (3 h)
    python repo/scripts/experiments/build_raw2.py          # RAW2.npy and AUG3_RAW2.npy (also the alignment reference for the check below)
    BOCPD_H=50 BOCPD_OUT=AUG3_BOCPD6_H50 python repo/scripts/experiments/build_bocpd.py <shard> 8 ; ... merge 8   # 136 s per shard
    MASS_OUT=AUG3_MASS90 python repo/scripts/experiments/build_mass_aug.py <shard> 8 ; ... merge 8
    python repo/scripts/experiments/build_mspec_aug.py <shard> 8 ; ... merge 8   # AUG3_MSPEC32.npy
    python repo/scripts/experiments/build_white_aug.py check                     # alignment of the regenerated pseudo-series
    python repo/scripts/experiments/build_white_aug.py <shard> 8 ; ... merge 8   # AUG3_WHITE111.npy, 95 s per shard
    python repo/scripts/experiments/build_wstream.py <shard> 4 ; ... merge 4     # WSTREAM5.npy (4 shards, 17 s each)
    python repo/scripts/experiments/build_wstream.py <shard> 8 aug ; ... merge 8 aug   # AUG3_WSTREAM5.npy

`build_aug3.py` loads `repo/submissions/053-spectral/main.py` and the
forecaster from `resources053/model.joblib`: the pseudo-series (history +
`online[:k]`, three random cuts per qualifying series, `rng` seed 1) are walked
through that submission's `TriMonitor`, so this step needs a built artifact.
The `AUG3_*` builders regenerate the same pseudo-series from the same seed and
recover `k` as `len(online) - len(pseudo-series)`; `build_white_aug.py check`
confirms group ids, lengths, labels and the raw z-scores against the stored
`AUG3_*` files. Only `AUG3_X`, `AUG3_Y/G/S` and `AUG3_WHITE111` are needed for
the shipped networks; `AUG3_BOCPD6_H50` is read by `core_white_ranker.py`
(156), `AUG3_MASS90` / `AUG3_MSPEC32` by killed experiments (119, 128),
`AUG3_WSTREAM5` by 158.

The *single* augmentation `AUG_X/Y/G/S.npy` (065, one cut per series) has no
builder in the repository; it is read by `screen_rank_bocpd.py` and
`final_aug_ranker.py` (below).

### 5.7 Derived inputs for the networks

* `X200.npy` (the 200 core channels as one float32 matrix) and `W111.npy`
  (`WHITE90` + `SR22`) are written by `nets_white_lean.py` on its first run,
  memory-mapped, 500,000 rows at a time.
* `mu_w.npy` / `sd_w.npy` (111) by `nets_white.py`; `mu_a.npy` / `sd_a.npy`
  (311) by `nets_white_lean.py`: training-row means and standard deviations.
* `mu200.npy` / `sd200.npy` (the core networks' normalisers) have **no local
  writer in the repository**; `scripts/gpu/train_nets_cuda.py` writes them with
  the same definition into its `--out-dir`.

## 6. Fold-2 caches

Every cache is a float vector over the 1,006,223 fold-2 rows in row order
(`te = split_by_series(G40, 5, 0) == 2`). The blend scripts expect them in the
workspace root by these names.

| cache | written by | trained on / needs |
|---|---|---|
| `fold2_rank_bocpd_200.npy` | `screen_rank_bocpd.py` (085r) | 200 core channels, folds ≠ 2 plus the single augmentation `AUG_*` and `AUG_BOCPD6_H50` — `AUG_*` has no builder here |
| `fold2_clf_bocpdh50_206.npy` | **not committed**: `screen_bocpd.py` writes `fold2_clf_bocpd_200/206.npy` from `BOCPD6.npy`; the shipped `_h50_` cache came from its hazard-1/50 variant (`BOCPD6_H50.npy` in place of `BOCPD6.npy`), same recipe (600 trees, lr 0.03, 63 leaves) | 206 channels |
| `fold2_sig_nets_aug3_member_p{0..5}.pt.npy` | `fold2_aug3_probe.py` over `nets_aug3/member_p*.pt` | the pool trained by `nets_aug3.py` (12 members in 12,120 s); needs `mu200`/`sd200` |
| `fold2_sig_nets_aug3_last_member_z{0..5}.pt.npy` | `fold2_last_probe.py` over `nets_aug3_last/member_z*.pt` | `nets_aug3_last.py` (082b: six members, 11,665 s) |
| `fold2_base30.npy` | `entry_exit_rules.py` (092): `0.45·(0.7·sigmoid(rank_bocpd_200) + 0.3·clf_bocpdh50_206) + 0.55·mean(12 net sigmoids)` | the four lines above; reproduces the stored file to 0.0 |
| `fold2_clf_mass_90.npy` | `screen_mass.py` (114) | `MASS90`, folds ≠ 2 |
| `fold2_clf_freqdep_1500.npy` | `freqdep_strengthen.py` (131b, tag `freqdep_1500`) | `MSPEC32`+`MSPEC60`+`ACF8`, 1500 trees lr 0.015 |
| `fold2_clf_freqdep_knn.npy` | **not committed** (the 140 script); recipe = the slow learner of `eval_141.py` on `MSPEC32`+`MSPEC60`+`ACF8`+`KNN20` (`eval_141.py` runs it with `KNN20B` and writes `fold2_clf_freqdep_knnB.npy`) | the journal records Spearman 1.0000 between this cache and the final union classifier |
| `fold2_clf_white_slow.npy` | `white_member_slow.py` | `WHITE90` (needed by `white_learners.py` for its print-out only) |
| `fold2_clf_white_sr_slower.npy`, `fold2_rank_white_sr.npy` | `white_learners.py` (148) | `WHITE90`+`SR22`; the ranker is the `r0` of the blend |
| `fold2_clf_white_ctx.npy` | `histctx_member.py` (153) | `WHITE90`+`SR22`+`HISTCTX8`; 314 s |
| `fold2_rank_white_ctx.npy` | `histctx_ranker.py` (153b) | the same 119 channels |
| `nets_white_w/fold2_logits_w{0,1,2}.npy` | `nets_white.py` (152; `NET_KIND=w`, `NET_MEMBERS=0,1,2`) | `WHITE90`+`SR22`, `AUG3_WHITE111`; three members in 9,284 s; also writes `member_w*.pt`, `mu_w`, `sd_w` |
| `nets_white_a/fold2_logits_a{0,1,2}.npy` | `nets_white_lean.py` (152a) **then** `export_logits.py a nets_white_a 0,1,2` | `X200`+`W111`, `AUG3_X`+`AUG3_WHITE111`; three members in 7,656 s. The lean script's own export took one point per series; the shipped logits are the recomputation |

The scripts that read these caches run from the workspace root and import the
metric from `repo/src`; the two chain scripts (`seed_chain.sh`,
`fold_ens_chain.sh`) belong to the fold-0 era and need companion scripts that
are not all in the repository (`tcn_chan_seed2.py`, `tcn_folddrop*.py`).

## 7. Artifacts

`resources089/model.joblib` is built as a chain: 075 → 078/081 → 082 → 083 →
084 (#39), and then 086, 087 and 088 each start from `resources084` and add
their own member (088 also takes the ranker trained for 087); 089 starts from
`resources088`:

| artifact | built by | adds |
|---|---|---|
| `resources075` | `submissions/075-bocpd-clf/{train_clf206,convert_075,build_075}.py` | the 206-channel classifier, the augmented ranker (`scripts/experiments/final_aug_ranker.py`, needs `AUG_*`), the forecaster from `resources072` (no build script for 072 in the repository), twelve nets |
| `resources081` | `submissions/081-mass-member/{train_mass_final,build_081}.py` | the mass classifier |
| `resources082` | `submissions/082-mass-pool24/build_082.py` + `074/convert_074.py`, `079/convert_079.py` | twelve more nets (`nets_aug3_full`, `nets_pool24`) |
| `resources083` | `scripts/experiments/train_freqdep_final.py`, `build_083.py` | the frequency member |
| `resources084` (#39) | `train_union_final.py`, `build_084.py` | the gated union |
| `resources087` | `train_white2_final.py`, `build_087.py` | the first whitened ranker (52 min on all series) |
| `resources088` (#44) | `train_white3_clf.py`, `train_white3_rank_ctx.py`, `convert_white_nets.py`, `build_088.py` | the context classifier, the second ranker, the three whitened nets |
| `resources089` (#45) | `submissions/089-joint-nets/build_089.py` | the three joint nets from `resources089_joint_nets.npz` — the converter that wrote that archive from `nets_white_a/member_a*.pt` is **not committed**; `convert_white_nets.py` is the same operation for `nets_white_w` |

Shipping is `scripts/ship_submission.sh` only (assemble, verify, copy, push);
the verifier `submissions/089-joint-nets/verify_089.py` memory-maps the
seventeen matrices and checks the assembled `TriMonitor` against them on
series 7 and 4242.

## 8. The fold-2 number

With the caches of section 6 in place, this prints the fold-2 TS-AUC of the
shipped blends. The formula is `infer()` of `submissions/089-joint-nets/interface_089.py`
applied to the cached member predictions — the same arithmetic
`core_white_ranker.py` and the 30 September screen brief use for #44 — with
`UNION_GATE = 100`, `WHITE_SHARE = 0.40`, `WHITE_RANK = 0.70`,
`NET_SHARE = 0.10`, `JOINT_SHARE = 0.10`, `NET_GATE = 0`:

    cd <workspace>
    .venv/bin/python - <<'EOF'
    import sys, glob, numpy as np
    sys.path.insert(0, "repo/src")
    from structural_break.combiners import split_by_series, ts_auc
    y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
    te = split_by_series(g, folds=5, seed=0) == 2; yf, sf = y[te], s[te]
    sig = lambda a: 1.0 / (1.0 + np.exp(-a.astype("float64")))
    base = np.load("fold2_base30.npy"); mass = np.load(glob.glob("fold2_clf_mass_90*.npy")[0])
    fq = np.load("fold2_clf_freqdep_1500.npy"); un = np.load("fold2_clf_freqdep_knn.npy")
    r0 = np.load("fold2_rank_white_sr.npy"); r1 = np.load("fold2_rank_white_ctx.npy"); c1 = np.load("fold2_clf_white_ctx.npy")
    pool_w = sum(sig(np.load(f"nets_white_w/fold2_logits_w{i}.npy")) for i in range(3)) / 3
    pool_a = sum(sig(np.load(f"nets_white_a/fold2_logits_a{i}.npy")) for i in range(3)) / 3
    # #39: the core, the mass member and, before step 100, the frequency member; from step 100 the union
    s39 = np.where(sf < 100, 0.55 * base + 0.25 * mass + 0.20 * fq, 0.50 * base + 0.25 * mass + 0.25 * un)
    # the whitened tree member: a bag of two per-step rankers and the classifier with context
    m3 = 0.7 * 0.5 * (r0 + r1) + 0.3 * c1
    tree = 0.6 * s39 + 0.4 * m3
    print(f"#30 core           {ts_auc(base, yf, sf):.4f}")
    print(f"#39                {ts_auc(s39, yf, sf):.4f}")
    print(f"#44 (0.85 tree + 0.15 whitened nets)              {ts_auc(0.85 * tree + 0.15 * pool_w, yf, sf):.4f}")
    print(f"#45 (0.80 tree + 0.10 whitened + 0.10 joint nets) {ts_auc(0.80 * tree + 0.10 * pool_w + 0.10 * pool_a, yf, sf):.4f}")
    EOF

Output on the workspace of 1 October 2026 (1,006,223 fold-2 rows; about a
minute, two threads):

    #30 core           0.6167
    #39                0.6251
    #44 (0.85 tree + 0.15 whitened nets)              0.6439
    #45 (0.80 tree + 0.10 whitened + 0.10 joint nets) 0.6452

The journal's readings for the same submissions are 0.6167–0.6169 (#30),
0.6251 (#39), 0.6439 (#44) and 0.6452 (#45; `meta_089.txt` says 0.6453).
The cloud read 0.6007, 0.6056, 0.6277 and 0.6299.

## 9. What cannot be reconstructed from the repository

* The builders of the 200-channel core and of `Y40`/`G40`/`S40` (`X40`,
  `C_cnn`, `N9`, `X50a`, `E4`): never committed; the route that exists is an
  assembled submission's `TriMonitor` over the training set (section 5.1),
  which reproduces the stored matrices to the verifiers' tolerance but is not
  itself a committed command.
* The single augmentation `AUG_X/Y/G/S.npy` (065) and `AUG_BOCPD6_H50.npy`'s
  parent set; `mu200.npy` / `sd200.npy` as locally produced.
* `fold2_clf_bocpdh50_206.npy` and `fold2_clf_freqdep_knn.npy`: the exact
  scripts were variants of `screen_bocpd.py` and of the 140 experiment that
  were not kept; the recipes are recorded above and in the journal.
* `resources072/model.joblib` (the forecaster the whole chain carries) and the
  converter behind `resources089_joint_nets.npz`.
* The `scratchpad` copies the chain scripts and the logs refer to
  (a temporary working folder outside the repository): the committed `scripts/experiments/`
  files are those scripts, moved into the repository after the runs.

Everything else in sections 5–8 is a committed script with the command its
docstring states, and the stored matrices, caches and artifacts in the
workspace are the ones those commands produced.
