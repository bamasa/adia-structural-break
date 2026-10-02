"""Build a compact cache of X_train: values float32 (concatenated), offsets, meta table.

Memory: reads parquet row-group by row-group; never materialises float64 matrices.
Outputs (in this folder): values.npy (float32), offsets.npy (int64), meta.csv
"""
import os, sys, time
import numpy as np, pandas as pd, pyarrow.parquet as pq

D = "structural-break-real-time-test/data/"   # relative to the workspace root, where the scripts run
OUT = os.path.dirname(os.path.abspath(__file__))

t0 = time.time()
pf = pq.ParquetFile(D + "X_train.parquet")
n = pf.metadata.num_rows
vals = np.empty(n, np.float32); ids = np.empty(n, np.int32); tms = np.empty(n, np.int32); per = np.empty(n, np.int8)
pos = 0
for rg in range(pf.num_row_groups):
    t = pf.read_row_group(rg)
    m = t.num_rows
    vals[pos:pos+m] = t.column("value").to_numpy(zero_copy_only=False).astype(np.float32)
    ids[pos:pos+m] = t.column("id").to_numpy().astype(np.int32)
    tms[pos:pos+m] = t.column("time").to_numpy().astype(np.int32)
    per[pos:pos+m] = t.column("period").to_numpy().astype(np.int8)
    pos += m
    del t
assert pos == n
print("read", n, "rows in", round(time.time()-t0, 1), "s")

# ordering checks
d_id = np.diff(ids)
print("id non-decreasing:", bool((d_id >= 0).all()))
same = d_id == 0
d_t = np.diff(tms)
print("time strictly increasing within id:", bool((d_t[same] > 0).all()), " unit steps:", bool((d_t[same] == 1).all()))
starts = np.flatnonzero(np.concatenate([[True], d_id != 0]))
offsets = np.append(starts, n).astype(np.int64)
uid = ids[starts]
print("n series", len(uid), "ids unique:", len(np.unique(uid)) == len(uid), "ids 0..N-1 in order:", bool((uid == np.arange(len(uid))).all()))
# period structure: 1s then 2s
hist_len = np.zeros(len(uid), np.int32); onl_len = np.zeros(len(uid), np.int32); t_first = np.zeros(len(uid), np.int32); t_onl0 = np.zeros(len(uid), np.int32)
nan_cnt = np.zeros(len(uid), np.int32)
ok = True
for k in range(len(uid)):
    a, b = offsets[k], offsets[k+1]
    p = per[a:b]
    h = int((p == 1).sum()); o = int((p == 2).sum())
    if not ((p[:h] == 1).all() and (p[h:] == 2).all()): ok = False
    hist_len[k] = h; onl_len[k] = o; t_first[k] = tms[a]; t_onl0[k] = tms[a+h] if o > 0 else -1
    nan_cnt[k] = int(np.isnan(vals[a:b]).sum())
print("period = 1s then 2s in every series:", ok, " total NaN:", int(nan_cnt.sum()))
yi = pd.read_parquet(D + "y_train_index.parquet")
yi = yi.reindex(uid)
meta = pd.DataFrame(dict(id=uid, hist_len=hist_len, onl_len=onl_len, t_first=t_first, t_onl0=t_onl0,
                        tau_index=yi.tau_index.to_numpy(), tau=yi.tau.to_numpy(), nan_cnt=nan_cnt))
# consistency of tau vs tau_index
b = meta.tau_index >= 0
print("breaks:", int(b.sum()), "/", len(meta))
print("tau == t_onl0 + tau_index for breaks:", bool((meta.tau[b] == meta.t_onl0[b] + meta.tau_index[b]).all()))
print("t_first==0 all:", bool((meta.t_first == 0).all()), " t_onl0==hist_len all:", bool((meta.t_onl0 == meta.hist_len).all()))
print(meta.describe().T.to_string())
np.save(os.path.join(OUT, "values.npy"), vals); np.save(os.path.join(OUT, "offsets.npy"), offsets)
meta.to_csv(os.path.join(OUT, "meta.csv"), index=False)
print("saved; total", round(time.time()-t0, 1), "s")
