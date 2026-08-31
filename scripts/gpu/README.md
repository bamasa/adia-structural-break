# The GPU job

One command trains the half of the ensemble that the laptop cannot scale:
the channel-trajectory networks. Everything else (channel building, trees,
submission assembly) stays on the laptop and is fast there.

## Transfer

1. `git clone` this repository on the Linux box.
2. Copy these matrices from the laptop's workspace root into one directory
   (~2.5 GB total):

       X40.npy  C_cnn.npy  N9.npy  X50a.npy  E4.npy
       B40.npy  B2.npy  SPEC14.npy  Y40.npy  G40.npy  S40.npy

3. Environment: Python 3.12, `pip install numpy torch` (CUDA build from
   pytorch.org). Nothing else is needed for this job.

## Run

    python scripts/gpu/train_nets_cuda.py --data-dir /data --out-dir nets_cuda \
        --members 24 --epochs 14

Each member prints its private-holdout TS-AUC. On a 3080 expect roughly two
to three minutes per member; twenty-four members is about an hour.

Members alternate between two input variants:

* **plain** — the 200 channels as they are;
* **diff** — 600 inputs: the channels plus their first- and tenth-order
  differences. On the laptop this variant produced the strongest single
  members (holdout 0.6178 against ~0.60 typical), so the GPU trains both and
  the assembler picks by holdout.

## Return

Copy `nets_cuda/` back to the laptop workspace root. The submission is then
assembled there — the members are selected by their holdout scores, averaged
in sigmoid space, and blended with the trees at the weight the untouched
fold-2 prefers.

## Rules that survived the laptop era

- **Never validate on folds 0 or 1.** Both are burnt by selection; a gain
  measured there does not transfer. Fold 2 is the current ruler, folds 3–4
  are held in reserve, and the cloud is the final word (five runs a day).
- **Short schedules.** Past ~12 epochs the nets overfit through any
  augmentation; the best-epoch checkpoint is what ships.
- **Diversity through data, not seeds.** Seed ensembling was measured and
  killed; subsample and input-variant diversity both pay.
- **Profile ms/step before any submission.** One run died at a 7-hour
  timeout because sklearn wrappers spawn threads per predict; raw
  `booster_.predict(..., num_threads=1)` fixed it.
