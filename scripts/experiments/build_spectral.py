"""052: спектральные каналы — спектр префикса против спектра истории."""
import sys, time
sys.path.insert(0, "repo/src")
import numpy as np
import pandas as pd
from structural_break.features import Normalisation
from structural_break.stream import iter_series

t0 = time.time()
x = pd.read_parquet("structural-break-real-time-test/data/X_train.parquet")
y = pd.read_parquet("structural-break-real-time-test/data/y_train.parquet")
print(f"загрузка {time.time()-t0:.0f}s", flush=True)

NFFT = 64          # окно спектра
BANDS = 8          # полос

def spectrum(v):
    """Нормированная мощность по 8 полосам + спектральная энтропия + пик."""
    if len(v) < NFFT:
        v = np.concatenate([np.zeros(NFFT - len(v)), v])
    seg = v[-NFFT:] * np.hanning(NFFT)
    p = np.abs(np.fft.rfft(seg)) ** 2
    p = p[1:]                                   # без постоянной составляющей
    tot = p.sum() + 1e-12
    pn = p / tot
    band = np.add.reduceat(pn, np.linspace(0, len(pn), BANDS + 1)[:-1].astype(int))
    ent = float(-(pn * np.log(pn + 1e-12)).sum() / np.log(len(pn)))
    peak = float(np.argmax(pn)) / len(pn)
    return band, ent, peak, float(np.log(tot))

def hist_profile(z):
    """Средний спектр истории по скользящим окнам."""
    bands, ents, peaks, tots = [], [], [], []
    step = max(NFFT // 2, 1)
    for i in range(NFFT, len(z) + 1, step):
        b, e, pk, lt = spectrum(z[i - NFFT:i])
        bands.append(b); ents.append(e); peaks.append(pk); tots.append(lt)
    if not bands:
        b, e, pk, lt = spectrum(z)
        return b, e, pk, lt, np.ones(BANDS) * 0.1, 0.1
    B = np.stack(bands)
    return (B.mean(0), float(np.mean(ents)), float(np.mean(peaks)), float(np.mean(tots)),
            B.std(0) + 1e-3, float(np.std(ents)) + 1e-3)

rows, count = [], 0
for sid, hist, online, labels in iter_series(x, y):
    norm = Normalisation.fit(hist)
    n = len(hist)
    z_h = np.asarray([norm.clip(norm.standardise(float(v), i - n)) for i, v in enumerate(hist)])
    z_o = np.asarray([norm.clip(norm.standardise(float(v), i)) for i, v in enumerate(online)])
    hb, he, hp, ht, hb_sd, he_sd = hist_profile(z_h)
    buf = list(z_h[-NFFT:])
    cur = [0.0] * 14
    next_scan, step_i = 1, 0
    for v in z_o:
        buf.append(float(v))
        if len(buf) > NFFT:
            buf.pop(0)
        step_i += 1
        if step_i >= next_scan:
            next_scan = max(next_scan + 1, int(next_scan * 1.12))
            b, e, pk, lt = spectrum(np.asarray(buf))
            diff = (b - hb) / hb_sd                       # 8 полос в сигмах истории
            cur = list(diff) + [
                (e - he) / he_sd,                          # сдвиг энтропии
                pk - hp,                                   # сдвиг пиковой частоты
                lt - ht,                                   # сдвиг полной мощности
                float(np.abs(diff).max()),                 # худшая полоса
                float(np.abs(b - hb).sum()),               # суммарное отличие формы
                float(np.dot(b, hb) / (np.linalg.norm(b) * np.linalg.norm(hb) + 1e-12)),  # косинус
            ]
        rows.append(list(cur))
    count += 1
    if count % 2000 == 0:
        print(f"  {count} рядов, {time.time()-t0:.0f}s", flush=True)

S = np.asarray(rows, dtype="float32")
np.save("SPEC14.npy", S)
print(f"готово {time.time()-t0:.0f}s: {S.shape}", flush=True)
