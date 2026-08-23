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
