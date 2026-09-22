"""resources082: массовый член (#36) плюс полный пул из 24 сетей (#35).

Два независимо обоснованных изменения к #30 в одном артефакте: массовая батарея
как отдельный член (114, +0.004 на фолде-2) и пул сетей, удвоенный с 12 до 24
(проверялся отдельно как #35). Фолд-2 не различает 12 и 24 сети, поэтому от
объединения ожидается прирост массы; пул — ставка на снижение дисперсии в облаке.
"""
import os, joblib, numpy as np
m = joblib.load("resources081/model.joblib")          # #36: 12 сетей + массовый член
assert "mass_classifier" in m and len(m["nets"]) == 12
def unpack(path):
    arc = np.load(path); n = 1 + max(int(k.split("|")[0]) for k in arc.files)
    return [{k.split("|", 1)[1]: arc[k] for k in arc.files if int(k.split("|")[0]) == i} for i in range(n)]
extra = unpack("resources074/nets_aug3.npz") + unpack("resources079/nets_pool6.npz")
assert len(extra) == 12 and all(s["inp.weight"].shape[1] == 200 for s in extra)
m["nets"] = list(m["nets"]) + extra
os.makedirs("resources082", exist_ok=True)
joblib.dump(m, "resources082/model.joblib", compress=3)
print(f"resources082: nets {len(m['nets'])}, clf {m['booster'].booster_.num_feature()}, "
      f"mass {m['mass_classifier'].booster_.num_feature()}, {os.path.getsize('resources082/model.joblib')/1e6:.1f} MB")
