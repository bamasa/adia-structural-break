"""123: массовая батарея с ОНЛАЙН-эталоном.

Те же 6 представлений x 6 окон, но эталон — не история, а накопленное среднее и
дисперсия самой онлайн-части до начала окна (всё, что было до t-W). История задаёт
только нормировку z. Каналы расходятся с массовыми ровно там, где онлайн-часть уже
ушла от истории: другая точка отсчёта, а не другое преобразование того же сравнения.

Исходное описание 114: массовая батарея — много простых статистик на многих окнах и представлениях.

Не новая «умная» модальность (четыре такие подряд дали ноль), а объём: решение
2-го места 2025 года имело 2408 признаков против наших 200. Строим то же дёшево:
6 представлений ряда x 6 скользящих окон x несколько сравнений с историей.
Всё через скользящие суммы, O(представлений x окон) на шаг.

Представления: z, |z|, z^2, приращение, отклонение накопленной суммы, знак.
Окна: 10, 25, 50, 100, 250, 500.
На каждое: (mean_W - mu_hist)/se, log(var_W/var_hist) -> 72 канала.
Плюс доля превышений исторических квантилей q75/q95/q99 на окнах -> 18.
Итого 90. python build_mass.py <shard> <n> | merge <n> | test
"""
import sys, time, os, numpy as np
WINS = (10, 25, 50, 100, 250, 500); NREP = 6
OUT = os.environ.get("MASS_OUT", "OMASS90"); PARTS = f"{OUT.lower()}_parts"
NCH = NREP * len(WINS) * 2 + len(WINS) * 3

def mass_channels(hist, online):
    h = np.asarray(hist, float); mu_h, sd_h = h.mean(), h.std() + 1e-12
    zh = (h - mu_h) / sd_h
    reps_h = [zh, np.abs(zh), zh * zh, np.diff(zh, prepend=zh[0]), np.zeros_like(zh), np.sign(zh)]
    hm = np.array([r.mean() for r in reps_h]); hv = np.array([r.var() + 1e-9 for r in reps_h])
    q = np.quantile(np.abs(zh), [0.75, 0.95, 0.99])
    n = len(online); out = np.empty((n, NCH), dtype="float32")
    W = np.array(WINS, float)
    s1 = np.zeros((NREP, len(WINS))); s2 = np.zeros((NREP, len(WINS)))   # скользящие EWMA-суммы
    qs = np.zeros((3, len(WINS)))
    prev = zh[-1]; cum = 0.0
    lag_buf = [None] * 600                                   # кольцевой буфер значений для отставания на W
    ref_s1 = np.tile(hm[:, None], (1, len(WINS))); ref_s2 = np.tile((hv + hm * hm)[:, None], (1, len(WINS)))
    aref = 1.0 / 500.0
    for t, x in enumerate(online):
        z = (x - mu_h) / sd_h
        cum += z
        vals = np.array([z, abs(z), z * z, z - prev, cum / np.sqrt(t + 1), np.sign(z)])
        a = 1.0 / W                                   # затухание = 1/окно
        s1 = (1 - a) * s1 + a * vals[:, None]
        s2 = (1 - a) * s2 + a * (vals * vals)[:, None]
        var = np.maximum(s2 - s1 * s1, 1e-9)
        # 123: эталон — самая длинная EWMA (окно 500) ОТСТАЮЩАЯ на W шагов: это «онлайн-часть до окна».
        # Реализуется как EWMA с окном 500, обновляемая значениями с задержкой W (кольцевой буфер).
        ref_m = ref_s1 if t >= 25 else hm[:, None] * np.ones((1, len(WINS)))
        ref_v = np.maximum(ref_s2 - ref_s1 * ref_s1, 1e-9) if t >= 25 else hv[:, None] * np.ones((1, len(WINS)))
        se = np.sqrt(ref_v / W[None, :])
        out[t, :NREP * len(WINS)] = ((s1 - ref_m) / se).ravel()
        out[t, NREP * len(WINS):NREP * len(WINS) * 2] = np.log(var / ref_v).ravel()
        exc = (np.abs(z) > q[:, None]).astype(float)
        qs = (1 - a[None, :]) * qs + a[None, :] * exc
        out[t, NREP * len(WINS) * 2:] = (qs - np.array([0.25, 0.05, 0.01])[:, None]).ravel()
        lag_buf[t % 600] = vals
        for i, w in enumerate(WINS):                          # отстающий эталон: значение, пришедшее W шагов назад
            j = t - int(w)
            if j >= 0:
                v_old = lag_buf[j % 600]
                ref_s1[:, i] = (1 - aref) * ref_s1[:, i] + aref * v_old
                ref_s2[:, i] = (1 - aref) * ref_s2[:, i] + aref * v_old * v_old
        prev = z
    return np.clip(np.nan_to_num(out, nan=0.0, posinf=20.0, neginf=-20.0), -20, 20)

if __name__ == "__main__":
    if sys.argv[1] == "test":
        rng = np.random.default_rng(0); hist = rng.normal(0, 1, 2000)
        online = np.concatenate([rng.normal(0, 1, 300), rng.normal(0.3, 1.2, 300)])
        t0 = time.time(); o = mass_channels(hist, online); dt = (time.time() - t0) / len(online) * 1000
        print(f"каналов {o.shape[1]}, {dt:.3f} мс/шаг; до/после слома: "
              f"среднее-окно100 {o[250:300,3].mean():+.2f}->{o[450:600,3].mean():+.2f}, "
              f"дисперсия-окно100 {o[250:300,NREP*len(WINS)+3].mean():+.2f}->{o[450:600,NREP*len(WINS)+3].mean():+.2f}")
        sys.exit(0)
    if sys.argv[1] == "merge":
        n = int(sys.argv[2]); parts = [np.load(f"{PARTS}/part_{i}.npz", allow_pickle=True) for i in range(n)]
        g = np.load("G40.npy"); s = np.load("S40.npy")
        st = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]])); bd = np.append(st, len(g))
        out = np.empty((len(g), NCH), dtype="float32"); block = {}
        for p in parts:
            for sid, arr in zip(p["sids"], p["arrs"]): block[int(sid)] = arr
        for a, b in zip(st, bd[1:]): out[a:b] = block[int(g[a])][s[a:b]]
        np.save(f"{OUT}.npy", out); print(f"{OUT}: {out.shape}, nan {np.isnan(out).sum()}", flush=True); sys.exit(0)
    shard, n_shards = int(sys.argv[1]), int(sys.argv[2])
    import pandas as pd
    t0 = time.time(); x = pd.read_parquet("structural-break-real-time-test/data/X_train.parquet")
    ids = x.index.get_level_values("id"); x = x[(ids % n_shards) == shard]; sids, arrs = [], []
    for i, (sid, part) in enumerate(x.groupby(level="id")):
        hist = part.loc[part.period == 1, "value"].to_numpy("float64"); online = part.loc[part.period == 2, "value"].to_numpy("float64")
        sids.append(int(sid)); arrs.append(mass_channels(hist, online))
        if (i + 1) % 300 == 0: print(f"шард {shard}: {i+1} рядов, {time.time()-t0:.0f}s", flush=True)
    os.makedirs(PARTS, exist_ok=True)
    np.savez(f"{PARTS}/part_{shard}.npz", sids=np.array(sids), arrs=np.array(arrs, dtype=object), allow_pickle=True)
    print(f"шард {shard} готов {time.time()-t0:.0f}s: {len(sids)} рядов", flush=True)
