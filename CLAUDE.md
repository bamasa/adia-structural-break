# CLAUDE.md — working protocol for this repository

Solution and research log for the ADIA Lab Structural Break Challenge, Real-Time
Edition (CrunchDAO). Before doing anything: read `README.md` ("Where it stands"),
the last entries of `docs/experiments.md`, and `docs/method.md`.

## Where things live
- `src/structural_break/` — the library, the only place code is edited. Each channel family is one module with a batch builder and a streaming `*Monitor` class.
- `scripts/experiments/` — one script per experiment (`build_*.py` matrices, `*_member.py` members, `nets_*.py` network pools, `fold2_*_probe.py` recipe probes); `scripts/experiments/forensics/` — the generator forensics.
- `scripts/assemble_submission.py` and `scripts/ship_submission.sh` — the only way a submission is built and pushed.
- `submissions/NNN-name/` — one directory per shipped unit: `build_NNN.py`, `interface_NNN.py`, `meta_NNN.txt`, `verify_NNN.py`, the assembled `main.py`, `requirements.txt`. `main.py` is assembled, never edited by hand. Directory numbers (001–088) are not platform numbers (#1–#44); the journal entry gives the mapping.
- `docs/experiments.md` — the journal; `docs/research/` — the two surveys of 27 September; `tests/` — standard-library unit tests.
- The workspace is the repository's parent directory: training matrices (`X40.npy` … `WHITE90.npy`, `Y40/G40/S40.npy`), fold-2 caches (`fold2_*.npy`, `fold2_sig_*.npy`), artifacts (`resourcesNNN/model.joblib`), the venv (`.venv`), the crunch workspace (`structural-break-real-time-test/`). Experiment scripts run from the workspace with `sys.path.insert(0, "repo/src")`. Competition data is never committed.

## Evaluation protocol
- Folds: `split_by_series(g, folds=5, seed=0)` from `structural_break.combiners`. Fold 2 is the ruler: never trained on, never used for selection beyond a blend share read on a plateau. Folds 0 and 1 are burnt. Folds 3 and 4 are reserves, to be used once each and never for tuning.
- Metric: `ts_auc` from `structural_break.combiners` — identical to the platform scorer (per-step AUC weighted by n_pos·n_neg). No other local number counts. The local 100-series test sample is not a yardstick.
- Use the fold-2 caches for blends, gates, shares, step ranges and Spearman correlations. For any candidate member report: alone on fold 2, Spearman with the shipped ensemble, blend gain by share, gain by step range.
- Networks: the 082b recipe (ten epochs, last epoch, no holdout, triple augmentation), paired on seeds 51000–51002. Trees: full runs for ship candidates; the fast screen only within one family.
- Local-to-cloud offset is 0.016–0.020 and the platform resolves about 0.001. Nothing derived from the online length may enter a model (it is withheld by the protocol); the step index and the history's own context are legitimate inputs (145, 153 — they hurt the core trees in 010/052 and help the whitened member).

## Shipping bar
- A member ships only if it is strong alone (0.57 or better on fold 2) and its fold-2 gain over the shipped ensemble clears 0.003, read on a plateau of shares and gates, not at the peak of a grid. Below that: record it, do not ship.
- One change per cloud run. Cloud runs are the owner's to schedule (15 hours a week of quota); the owner decides what is run and what is marked Selected.
- Never run #25 (broken upload). #43 and #44 are identical; only one is to be run.

## Mandatory verification before any push
1. The streaming module reproduces the batch matrix row by row (1e-6 typical; the verifier asserts below 1e-4, the whitened block below 1e-2) on at least two training series, one with a short history.
2. `PYTHONPATH=src python -m unittest discover -s tests -v` passes; a new channel family gets its own test.
3. The assembler's width check passes: every `*_CHANNELS`, `*_WIDTH` and `*_OFFSET` in the interface against the artifact.
4. `verify_NNN.py` next to the interface: channels against the memory-mapped matrices, offsets and widths, milliseconds per step (assert below 10; a cloud run of #44 is expected to take about four hours at 8 ms per step).
5. Ship only through `scripts/ship_submission.sh`. Never `crunch push` by hand. Never chain a push after a check with `;` — the push must depend on the verification's exit status.

## Memory limits (32 GB machine)
- One heavy LightGBM job at a time; do not start a second training or a network pool while one runs (154 tipped the machine into swap at load average 116).
- Never stack the training matrices in float64 (21 GB). Load with `np.load(path, mmap_mode="r")` and gather only the rows needed.
- Pseudo-series matrices are read memory-mapped, per series or per row block; a pool that needs 20 GB in memory does not fit beside the matrices.

## Conventions
- Commits: English messages naming the experiment number, the reading and the verdict. The README experiment table and "Where it stands" are updated on every push. No co-author lines and no assistant mentions in git history.
- Journal: one entry per experiment — what was built, the readings, the verdict, the reason. The kill condition is written before the run; failures are logged with the same care as gains; newest at the bottom; numbers exactly as measured, never rounded up.
- Journal, README, docs, commit messages and new code comments in English. No emoji.
