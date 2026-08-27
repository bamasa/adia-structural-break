"""Потоковый numpy-инференс ChanTCN + сверка с torch."""
import sys, math, time
sys.path.insert(0, "repo/src")
import numpy as np

DILS = (1, 2, 4, 8, 16, 32)

def gelu(x):
    from scipy.special import erf
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))

class StreamingChanTCN:
    """Каузальная TCN по шагам: на каждый шаг O(ch^2 * k * слои)."""

    def __init__(self, weights: dict, mu, sd):
        self.w = {k: np.asarray(v, dtype="float64") for k, v in weights.items()}
        self.mu = np.asarray(mu, dtype="float64")
        self.sd = np.asarray(sd, dtype="float64")
        self.hist = [[] for _ in range(len(DILS) + 1)]  # выходы: inp, затем каждый блок

    def update(self, chan_vec) -> float:
        x = (np.asarray(chan_vec, dtype="float64") - self.mu) / self.sd
        h = self.w["inp.weight"][:, :, 0] @ x + self.w["inp.bias"]
        self.hist[0].append(h)
        for li, d in enumerate(DILS):
            layer_in = self.hist[li]
            t = len(layer_in) - 1
            W = self.w[f"blocks.{li}.conv.weight"]; b = self.w[f"blocks.{li}.conv.bias"]
            acc = b.copy()
            for j, off in enumerate((2 * d, d, 0)):
                idx = t - off
                if idx >= 0:
                    acc += W[:, :, j] @ layer_in[idx]
            z = gelu(acc)
            out = layer_in[t] + self.w[f"blocks.{li}.mix.weight"][:, :, 0] @ z + self.w[f"blocks.{li}.mix.bias"]
            self.hist[li + 1].append(out)
        top = self.hist[-1][-1]
        return float(self.w["head.weight"][0, :, 0] @ top + self.w["head.bias"][0])

if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Block(nn.Module):
        def __init__(self, ch, dil):
            super().__init__()
            self.conv = nn.Conv1d(ch, ch, 3, dilation=dil)
            self.mix = nn.Conv1d(ch, ch, 1)
            self.drop = nn.Dropout(0.1)
            self.dil = dil
        def forward(self, h):
            r = h
            h = F.pad(h, (2 * self.dil, 0))
            h = self.drop(F.gelu(self.conv(h)))
            return r + self.mix(h)

    class ChanTCN(nn.Module):
        def __init__(self, ch=64):
            super().__init__()
            self.inp = nn.Conv1d(186, ch, 1)
            self.blocks = nn.ModuleList([Block(ch, d) for d in DILS])
            self.head = nn.Conv1d(ch, 1, 1)
        def forward(self, x):
            h = self.inp(x)
            for b in self.blocks:
                h = b(h)
            return self.head(h).squeeze(1)

    model = ChanTCN()
    model.load_state_dict(torch.load("tcn_chan10.pt", map_location="cpu"))
    model.eval()
    weights = {k: v.numpy() for k, v in model.state_dict().items()}
    mu, sd = np.load("chan_mu.npy") if False else (None, None)
    # сверка на случайной последовательности каналов
    rng = np.random.default_rng(0)
    T = 300
    seq = rng.random((T, 186)).astype("float32")
    with torch.no_grad():
        ref = model(torch.from_numpy(seq.T).unsqueeze(0).float()).numpy()[0]
    stream = StreamingChanTCN(weights, np.zeros(186), np.ones(186))
    t0 = time.time()
    got = np.array([stream.update(seq[t]) for t in range(T)])
    dt = (time.time() - t0) / T * 1000
    diff = np.abs(got - ref).max()
    print(f"расхождение поток/torch: {diff:.2e}, скорость {dt:.2f} мс/шаг")
    assert diff < 1e-4
    print("OK")
