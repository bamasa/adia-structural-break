"""140c rerun: exact blend weights after the gate and a check of the p_med channels in KNN20."""
import sys, glob, numpy as np
sys.path.insert(0, "repo/src")
from structural_break.combiners import split_by_series, ts_auc
y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
f = split_by_series(g, folds=5, seed=0); te = f == 2; yf, sf = y[te], s[te]
base = np.load("fold2_base30.npy"); mass = np.load(glob.glob("fold2_clf_mass_90*.npy")[0])
fq = np.load("fold2_clf_freqdep_1500.npy"); un = np.load("fold2_clf_freqdep_knn.npy")
ref = 0.55 * base + 0.25 * mass + 0.20 * fq
print(f"#38: {ts_auc(ref, yf, sf):.4f}")
for M in (0.20, 0.25):
    for U in (0.20, 0.25, 0.30):
        late = (1 - M - U) * base + M * mass + U * un
        print(f"core {1-M-U:.2f} mass {M:.2f} union {U:.2f}: flat {ts_auc(late, yf, sf):.4f} | gate<100 {ts_auc(np.where(sf < 100, ref, late), yf, sf):.4f}")
for M, U in ((0.25, 0.20), (0.25, 0.25), (0.20, 0.25)):
    late = 0.55 * base + M * mass + U * un
    print(f"core 0.55 mass {M:.2f} union {U:.2f}: flat {ts_auc(late, yf, sf):.4f} | gate<100 {ts_auc(np.where(sf < 100, ref, late), yf, sf):.4f}")
K = np.load("KNN20.npy")
print("p_med (channels 2,7,12,17) unique values:", [len(np.unique(K[:, 2 + 5 * i])) for i in range(4)], "min/max:", float(K[:, 2::5].min()), float(K[:, 2::5].max()))
print("p_nn unique values:", [len(np.unique(K[:, 1 + 5 * i])) for i in range(4)])
