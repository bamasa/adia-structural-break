"""resources089: #44 (resources088) plus the pool of joint-input trajectory networks (200 + 111 channels)."""
import os, joblib, numpy as np
m = joblib.load("resources088/model.joblib")
assert "white_nets" in m and "joint_nets" not in m
arc = np.load("resources089_joint_nets.npz"); n = 1 + max(int(k.split("|")[0]) for k in arc.files)
m["joint_nets"] = [{k.split("|", 1)[1]: arc[k] for k in arc.files if int(k.split("|")[0]) == i} for i in range(n)]
m["joint_net_mu"] = np.load("mu_a.npy"); m["joint_net_sd"] = np.load("sd_a.npy")
assert all(s["inp.weight"].shape[1] == 311 for s in m["joint_nets"]) and len(m["joint_net_mu"]) == 311
os.makedirs("resources089", exist_ok=True)
joblib.dump(m, "resources089/model.joblib", compress=3)
print(f"resources089: nets {len(m['nets'])}, white nets {len(m['white_nets'])}, joint nets {len(m['joint_nets'])}, {os.path.getsize('resources089/model.joblib')/1e6:.1f} MB")
