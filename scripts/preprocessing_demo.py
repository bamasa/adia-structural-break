"""Что даёт скользящая медиана и asinh перед моделью 008 — картинка."""
import sys
sys.path.insert(0, "structural-break-real-time-test")
sys.path.insert(0, ".")
import importlib.util
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

spec = importlib.util.spec_from_file_location("sub", "repo/submissions/008-reverting-channels/main.py")
sub = importlib.util.module_from_spec(spec)
sys.modules["sub"] = sub
spec.loader.exec_module(sub)
model = joblib.load("resources008/model.joblib")["booster"]

x = pd.read_parquet("structural-break-real-time-test/data/X_test.reduced.parquet")
y = pd.read_parquet("structural-break-real-time-test/data/y_test.reduced.parquet")

def series(sid):
    part = x.loc[sid]
    return (part.loc[part.period == 1, "value"].to_numpy(),
            part.loc[part.period == 2, "value"].to_numpy(),
            y.loc[sid, "target"].to_numpy())

# Отбор примеров: чистый ряд с самым злым одиночным выбросом в онлайне
# (относительно СКО истории) и ряд с настоящим сломом и заметным выбросом.
cands = []
for sid, part in x.groupby(level="id"):
    hist = part.loc[part.period == 1, "value"].to_numpy()
    online = part.loc[part.period == 2, "value"].to_numpy()
    if len(online) < 50 or len(hist) < 50:
        continue
    sd = np.std(np.diff(hist)) + 1e-9
    spike = np.max(np.abs(online - np.median(online))) / sd
    tau = y.loc[sid, "target"].to_numpy()
    cands.append((int(sid), tau.max() > 0, spike))
cands = pd.DataFrame(cands, columns=["sid", "broken", "spike"])
clean_id = int(cands[~cands.broken].sort_values("spike").iloc[-1].sid)
brok_id = int(cands[cands.broken].sort_values("spike").iloc[-1].sid)

def causal_median(v, w):
    out = np.empty(len(v))
    for i in range(len(v)):
        out[i] = np.median(v[max(0, i - w + 1): i + 1])
    return out

VARIANTS = [
    ("как сейчас", lambda h, o: (h, o), "#5f6368"),
    ("медиана, окно 3", lambda h, o: (causal_median(h, 3), causal_median(o, 3)), "#1a73e8"),
    ("медиана, окно 5", lambda h, o: (causal_median(h, 5), causal_median(o, 5)), "#188038"),
    ("медиана, окно 9", lambda h, o: (causal_median(h, 9), causal_median(o, 9)), "#f9ab00"),
    ("asinh (аналог логарифма)", lambda h, o: (np.arcsinh(h), np.arcsinh(o)), "#d93025"),
]

def score(hist, online):
    m = sub.Monitor(hist)
    ch = np.asarray([m.update(float(v)) for v in online])
    return model.predict_proba(ch)[:, 1]

fig, axes = plt.subplots(4, 1, figsize=(11, 10), gridspec_kw=dict(height_ratios=[1.2, 1.5, 1.2, 1.5]))
for row, (sid, kind) in enumerate([(clean_id, "слома НЕТ — счёт должен молчать"),
                                   (brok_id, "слом ЕСТЬ — счёт должен подняться")]):
    hist, online, labels = series(sid)
    tau = int(labels.argmax()) if labels.max() > 0 else None
    t = np.arange(len(online))
    ax_raw, ax_sc = axes[row * 2], axes[row * 2 + 1]
    ax_raw.plot(t, online, lw=0.7, color="#9aa0a6", label="сырой онлайн-ряд")
    ax_raw.plot(t, causal_median(online, 5), lw=1.1, color="#188038", label="после медианы (окно 5)")
    ax_raw.set_ylabel("ряд")
    ax_raw.set_title(f"ряд {sid}: {kind}", fontsize=10, loc="left")
    for name, fn, colour in VARIANTS:
        h2, o2 = fn(hist, online)
        ax_sc.plot(t, score(h2, o2), lw=1.2, color=colour, label=name, alpha=0.9)
    ax_sc.set_ylim(-0.02, 1.02)
    ax_sc.set_ylabel("счёт модели 008")
    for ax in (ax_raw, ax_sc):
        if tau is not None:
            ax.axvline(tau, color="#d93025", lw=1.2, ls="--",
                       label="ИСТИННЫЙ слом" if ax is ax_raw else None)
        ax.legend(loc="upper left", fontsize=7, ncols=3, framealpha=0.85)
axes[-1].set_xlabel("шаг онлайн-части")
fig.suptitle("Скользящая медиана и asinh перед моделью: гасят ли выбросы, не глушат ли слом", y=0.995)
plt.tight_layout()
plt.savefig("/tmp/median_demo.png", dpi=130)
print("saved", clean_id, brok_id)
