# The GeForce-3080 programme

Everything the laptop era learned, ready to scale on CUDA.

## Transfer

1. Clone this repository on the Linux box.
2. Copy the training matrices from the laptop workspace root
   (`~/projects/adia-structural-break/`) into a data directory (~2 GB):
   `X40.npy C_cnn.npy N9.npy X50a.npy E4.npy B40.npy B2.npy Y40.npy G40.npy S40.npy`
3. Environment: python 3.12, `pip install torch numpy lightgbm scipy tabpfn==2.0.9`
   (torch with CUDA per pytorch.org).

## Part 1 — the net bag

    python scripts/gpu/train_net_bag.py --data-dir /data --members 16

Sixteen short channel-trajectory nets, each on a random 60% of the training
series; folds 0 and 1 are never trained on — they are the two-fold validation
the adoption protocol now requires. Prints per-member fold-0/fold-1 scores;
members average in sigmoid space. Laptop baselines to beat: single net
fold-0 0.5985, the 4-member fold-ensemble 0.6004.

## Part 2 — TabPFN

    python scripts/experiments/tabpfn_screen.py

Runs in minutes on CUDA (set device="cuda" in the script); OOM-killed on CPU.

## Rules that survived the week

- Short schedules: past ~10 epochs the nets overfit through any augmentation.
- Diversity through data; seeds converge.
- Adoption: visible on folds 0 AND 1, or ≥0.005 on fold 0 alone.
- Profile ms/step before any submission; raw `booster_.predict(num_threads=1)`.
