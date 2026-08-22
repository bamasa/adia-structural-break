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

    src/structural_break/    the library: detectors, calibration, evaluation
    submissions/             one directory per submission, exactly as sent
    tests/                   what the library is asserted to do
    docs/                    what was tried, what it scored, and why

Competition data is never committed — it is 200 MB of parquet fetched by
`crunch setup`, and the workspace that command creates is gitignored.

## Submissions

| # | Approach | Local TS-AUC | Notes |
|---|---|---|---|
| 001 | Three classical detectors combined by maximum | 0.5385 | Organisers' EWMA baseline: 0.5170 |

## Reproducing

    pip install crunch-cli
    crunch setup structural-break-real-time <name> --token <token>
    cp submissions/001-classical-detectors/main.py <workspace>/
    cd <workspace> && crunch test

## Relationship to the organisers' quickstarter

The EWMA baseline published by CrunchDAO is used as a reference point and is
reimplemented in `src/structural_break/evaluate.py` so that both can be scored
on the same rows. No other code is taken from it.
