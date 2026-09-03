# The GPU job

The ensemble has two halves: gradient-boosted trees over 200 engineered
channels, and small causal networks reading those channels as trajectories.
The trees train in minutes on a laptop. The networks are the half that wants
a GPU — they are also the half that transfers best to the leaderboard, so
this directory exists to train them properly and bring them home.

Everything else — building the channels, training the trees, assembling and
profiling the submission — stays on the laptop and is fast there.

## Quick start

    git clone <this repository> && cd adia-structural-break
    pip install numpy torch --index-url https://download.pytorch.org/whl/cu121
    ./scripts/gpu/run.sh /path/to/matrices 24

That is the whole job. The rest of this page explains what it needs, what it
trains, and why it is shaped this way.

## What you need

* A CUDA GPU (developed against a 3080, 16 GB; less will do with a smaller
  `--batch`).
* Python 3.12, `numpy`, and a CUDA build of `torch`. Nothing else.
* Fourteen matrices from the laptop workspace root, about 11 GB:

      X40.npy  C_cnn.npy  N9.npy  X50a.npy  E4.npy  B40.npy
      B2.npy   SPEC14.npy  Y40.npy  G40.npy  S40.npy
      AUG3_X.npy  AUG3_Y.npy  AUG3_G.npy

  The first eleven are the per-step channels for all ten thousand training
  series, plus labels, series ids and step indices (2.5 GB). The last three
  are the *triple boundary augmentation* — 26,887 pseudo-series, 8.3 GB —
  and they are the point of the job: the networks are data-bound, and this
  file is where the gains came from. None are in git — copy them with
  `rsync` or a stick.

## Run it

The runner checks the box first (CUDA present, all matrices consistent,
measured speed per member), prints what it found, and only then trains. Each
member reports its private-holdout TS-AUC as it finishes:

    plain member 0: holdout 0.6167  [1460s]
    plain member 1: holdout 0.6302  [2910s]

A smoke test — three hundred series, one epoch, two members — checks the
plumbing in about a minute before you commit the GPU for a day:

    python scripts/gpu/train_nets_cuda.py --data-dir /path/to/matrices --smoke

Interrupting is safe: finished members are skipped on the next run, so you
can stop, reboot, and continue.

## What is being trained

Twenty-four small dilated causal convolutional networks (~110k parameters,
receptive field 127 steps) reading the 200 channels as trajectories, each
trained on the original series plus the triple boundary augmentation.

The augmentation is what makes the job worth a GPU. A pseudo-series takes an
original's history, extends it with the first *k* online steps (before the
true break), and treats that as the new history: the break now lands early,
in the region where the metric is hardest. On the laptop, single
augmentation lifted the best member from 0.635 to 0.647 on its holdout, and
the triple version produced the strongest single member the project has on
its untouched fold — 0.6048, better than the previous ten-member ensemble.

A second input view — 600 inputs with explicit first- and tenth-order
differences — is kept behind `--variant diff` but not trained by default:
its members reach spectacular holdouts (0.654) that do not transfer to the
untouched fold (0.588). Plain members transfer; diff members do not.

Each member trains on its own random 92% of the original series (fold 2 of
the series split is held out entirely — it is the laptop's yardstick) and
keeps the epoch that scores best on its private 8% holdout. Diversity comes
from the data split — not from random seeds, which were measured and found
to converge.

## Bringing it home

Copy `nets_cuda/` back to the laptop workspace root. There the members are
measured on fold 2, averaged in sigmoid space, blended with
the trees at the weight the untouched validation fold prefers, verified
against the training matrices, profiled for milliseconds per step, and only
then submitted.

## House rules, learned the hard way

* **Never validate on folds 0 or 1.** Both were burnt by selection: gains
  measured there stopped transferring to the leaderboard. Fold 2 is the
  current ruler, folds 3–4 are held in reserve, and the cloud is the final
  word — five runs a day, so they are spent on real bets only.
* **Short schedules.** Ten epochs over the augmented set; the best-epoch
  checkpoint is what ships.
* **Holdouts rank plain members, not diff members.** A diff member's private
  holdout says nothing about its fold-2 score; only the untouched fold does.
* **More members beat bigger members.** Tripling capacity lost to the small
  architecture; the data, not the parameter count, is the binding
  constraint.
* **Profile before shipping.** One cloud run died at a seven-hour timeout
  because a scikit-learn wrapper spawned a thread pool on every single-row
  predict. Raw `booster_.predict(..., num_threads=1)` fixed it; a
  milliseconds-per-step measurement is now mandatory before any submission.
