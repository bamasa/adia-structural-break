# The GPU job

The ensemble has two halves: gradient-boosted trees over 200 engineered
channels, and small causal networks reading those channels as trajectories.
The trees train in minutes on a laptop. The networks are the half that wants
a GPU — they are also the half that transfers best to the leaderboard, so
this directory exists to train them properly and bring them home.

Everything else — building the channels, training the trees, assembling and
profiling the submission — stays on the laptop and is fast there.

## What you need

* A CUDA GPU (developed against a 3080, 16 GB; less will do with a smaller
  `--batch`).
* Python 3.12, `numpy`, and a CUDA build of `torch`. Nothing else.
* Eleven matrices from the laptop workspace root, about 2.5 GB:

      X40.npy  C_cnn.npy  N9.npy  X50a.npy  E4.npy  B40.npy
      B2.npy   SPEC14.npy  Y40.npy  G40.npy  S40.npy

  They are the per-step channels for all ten thousand training series, plus
  labels, series ids and step indices. They are not in git — copy them with
  `rsync` or a stick.

## Run it

    git clone <this repository> && cd adia-structural-break
    pip install numpy torch --index-url https://download.pytorch.org/whl/cu121
    ./scripts/gpu/run.sh /path/to/matrices 24

The runner checks the box first (CUDA present, all matrices consistent,
measured speed per member), prints what it found, and only then trains. Each
member reports its private-holdout TS-AUC as it finishes:

    plain member 0: holdout 0.6041  [143s]
    diff  member 1: holdout 0.6178  [291s]

Interrupting is safe: finished members are skipped on the next run, so you
can stop, reboot, and continue.

## What is being trained

Twenty-four small dilated causal convolutional networks (~110k parameters,
receptive field 127 steps), alternating between two input views:

* **plain** — the 200 channels as they are;
* **diff** — 600 inputs: the channels plus their first- and tenth-order
  differences. Explicit motion helps the networks (their best member reached
  a 0.6178 holdout) even though the same differences were useless to the
  trees.

Each member trains on its own random 92% of the series and keeps the epoch
that scores best on its private 8% holdout. Diversity comes from the data
split and the input view — not from random seeds, which were measured and
found to converge.

## Bringing it home

Copy `nets_cuda/` back to the laptop workspace root. There the members are
selected by their holdout scores, averaged in sigmoid space, blended with
the trees at the weight the untouched validation fold prefers, verified
against the training matrices, profiled for milliseconds per step, and only
then submitted.

## House rules, learned the hard way

* **Never validate on folds 0 or 1.** Both were burnt by selection: gains
  measured there stopped transferring to the leaderboard. Fold 2 is the
  current ruler, folds 3–4 are held in reserve, and the cloud is the final
  word — five runs a day, so they are spent on real bets only.
* **Short schedules.** Past roughly twelve epochs these networks overfit
  through any augmentation; the best-epoch checkpoint is what ships.
* **More members beat bigger members.** Tripling capacity lost to the small
  architecture; the data, not the parameter count, is the binding
  constraint.
* **Profile before shipping.** One cloud run died at a seven-hour timeout
  because a scikit-learn wrapper spawned a thread pool on every single-row
  predict. Raw `booster_.predict(..., num_threads=1)` fixed it; a
  milliseconds-per-step measurement is now mandatory before any submission.
