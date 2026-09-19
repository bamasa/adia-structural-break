"""Стоимость инференса при 8 блоках вместо 6: потоковый батч-класс из посылки #30 на случайных весах."""
import importlib.util, sys, time, numpy as np
spec = importlib.util.spec_from_file_location("sub", "repo/submissions/075-bocpd-clf/main.py")
sub = importlib.util.module_from_spec(spec); sys.modules["sub"] = sub; spec.loader.exec_module(sub)
rng = np.random.default_rng(0)
def fake(dils, ch=64, n_in=200):
    w = {"inp.weight": rng.normal(0, .05, (ch, n_in, 1)), "inp.bias": np.zeros(ch),
         "head.weight": rng.normal(0, .05, (1, ch, 1)), "head.bias": np.zeros(1)}
    for li in range(len(dils)):
        w[f"blocks.{li}.conv.weight"] = rng.normal(0, .05, (ch, ch, 3)); w[f"blocks.{li}.conv.bias"] = np.zeros(ch)
        w[f"blocks.{li}.mix.weight"] = rng.normal(0, .05, (ch, ch, 1)); w[f"blocks.{li}.mix.bias"] = np.zeros(ch)
    return w
mu, sd = np.zeros(200), np.ones(200)
for name, dils, n_nets in (("6 блоков (как в #30), 12 сетей", (1,2,4,8,16,32), 12),
                           ("6 блоков, 24 сети (как в #35)", (1,2,4,8,16,32), 24),
                           ("8 блоков (107), 12 сетей", (1,2,4,8,16,32,64,128), 12),
                           ("8 блоков (107), 24 сети", (1,2,4,8,16,32,64,128), 24)):
    sub.TCN_DILS = dils
    bag = sub.BatchedStreamingChanTCN([fake(dils) for _ in range(n_nets)], mu, sd)
    x = rng.normal(0, 1, 200)
    for _ in range(50): bag.update(x)                      # прогрев
    t0 = time.time(); n = 700
    for _ in range(n): bag.update(rng.normal(0, 1, 200))
    dt = (time.time() - t0) / n * 1000
    print(f"{name:34s}: {dt:5.2f} мс/шаг только сети", flush=True)
