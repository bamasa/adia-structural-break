"""Figures for the README: the platform score timeline, the pipeline and research-loop diagrams (SVG),
and an animation of the whitened monitor reading a break. Run from the repository root:
    PYTHONPATH=src python scripts/make_figures.py
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import io

OUT = Path("docs/figures"); OUT.mkdir(parents=True, exist_ok=True)
INK, MUTED, ACCENT, SOFT = "#1f2937", "#6b7280", "#0f766e", "#99f6e4"

# ---------------------------------------------------------------- score timeline
cloud = [(18, .5877), (22, .5876), (23, .5893), (24, .5853), (28, .6004), (29, .5996), (30, .6007), (31, .6000), (32, .5991),
         (33, .5991), (35, .6009), (36, .6046), (37, .6048), (39, .6056), (41, .6186), (44, .6277), (45, .6299)]
levers = {18: "trajectory networks\njoin the trees", 23: "spectral bands", 28: "boundary\naugmentation", 36: "the first independent\nmember (mass battery)",
          41: "the whitened stream\nas a member", 45: "final: joint-input\nnetworks"}
fig, ax = plt.subplots(figsize=(10, 4.6), dpi=160)
x, y = zip(*cloud)
best = np.maximum.accumulate(y)
ax.step(x, best, where="post", color=SOFT, lw=6, alpha=.9, zorder=1, label="best so far")
ax.plot(x, y, "o-", color=ACCENT, lw=1.6, ms=5.5, zorder=3, label="platform score (TS-AUC)")
for sub, text in levers.items():
    yy = dict(cloud)[sub]
    dx, dy = {23: (0, -40), 45: (18, -46), 44: (-6, 26)}.get(sub, (0, 26))
    ax.annotate(text, (sub, yy), xytext=(dx, dy), textcoords="offset points", ha="center", fontsize=8, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=.8))
ax.axhline(.680, color=MUTED, lw=.8, ls="--"); ax.text(46.3, .6815, "leader 0.680", fontsize=8, color=MUTED, va="bottom", ha="right")
ax.axhline(.642, color=MUTED, lw=.8, ls=":"); ax.text(17.0, .6435, "rank 50: 0.642", fontsize=8, color=MUTED, va="bottom", ha="left")
ax.set_xlabel("platform submission number", color=INK); ax.set_ylabel("TS-AUC on the platform", color=INK)
ax.set_ylim(.58, .69); ax.set_xlim(16.5, 46.5)
ax.set_title("Five weeks on the platform: 0.5877 → 0.6299 (about rank 160 of 1,716)", color=INK, fontsize=11, loc="left")
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.grid(axis="y", color="#e5e7eb", lw=.6); ax.legend(frameon=False, loc="lower right", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "score_timeline.png"); plt.close(fig)

# ---------------------------------------------------------------- diagrams (SVG by hand)
def box(x, y, w, h, title, body="", fill="#f0fdfa", stroke=ACCENT):
    lines = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>',
             f'<text x="{x + w / 2}" y="{y + 24}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="14" font-weight="600" fill="{INK}">{title}</text>']
    for i, b in enumerate(body.split("\n")):
        if b: lines.append(f'<text x="{x + w / 2}" y="{y + 44 + 16 * i}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="11.5" fill="{MUTED}">{b}</text>')
    return "\n".join(lines)
def arrow(x1, y1, x2, y2):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{MUTED}" stroke-width="1.6" marker-end="url(#a)"/>'
head = '<defs><marker id="a" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#6b7280"/></marker></defs>'

svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1180" height="420" viewBox="0 0 1180 420" font-family="Helvetica, Arial, sans-serif">', head,
       f'<rect width="1180" height="420" fill="white"/>',
       f'<text x="20" y="34" font-size="18" font-weight="700" fill="{INK}">The shipped ensemble (#45): one score per step, from six readers of the same stream</text>']
svg.append(box(20, 70, 180, 110, "The series", "history 1,000–5,000 points\nonline part, one point at a time\nz-scored on the history", fill="#f9fafb", stroke=MUTED))
svg.append(box(250, 60, 200, 130, "206 streaming channels", "CUSUM, Page–Hinkley, variance ratio\nmulti-scale, retrospective scans\nforecaster error, spectra, BOCPD", ))
svg.append(box(250, 230, 200, 130, "The whitened stream", "AR(p) by BIC, conditional scale,\ninnovation ECDF → normal scores\n90 statistics + 21 SR odds + context"))
svg.append(box(500, 40, 200, 90, "Core trees", "LightGBM ranker (augmented)\nand classifier, per step"))
svg.append(box(500, 150, 200, 90, "Core networks", "24 dilated causal TCNs\nover channel trajectories"))
svg.append(box(500, 260, 200, 110, "Whitened member", "two per-step rankers + classifier\n(0.63 alone on the fold)\n3 TCNs over whitened channels"))
svg.append(box(750, 150, 200, 110, "Independent members", "mass battery (90 statistics)\nfrequency/dependence + novelty\njoint TCNs over 200 + 111 channels\n(a reader of both streams)"))
svg.append(box(1000, 150, 160, 110, "Hand blend", "shares read on an\nuntouched fold\n→ score in [0, 1]", fill="#ecfeff", stroke=INK))
svg += [arrow(200, 125, 250, 125), arrow(200, 125, 250, 295), arrow(450, 110, 500, 90), arrow(450, 130, 500, 195), arrow(450, 295, 500, 310),
        arrow(450, 140, 750, 200), arrow(450, 290, 750, 215), arrow(700, 85, 1000, 190), arrow(700, 195, 1000, 200), arrow(700, 315, 1000, 215), arrow(950, 205, 1000, 205)]
svg.append('</svg>'); (OUT / "pipeline.svg").write_text("\n".join(svg))

svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1180" height="300" viewBox="0 0 1180 300" font-family="Helvetica, Arial, sans-serif">', head,
       f'<rect width="1180" height="300" fill="white"/>',
       f'<text x="20" y="34" font-size="18" font-weight="700" fill="{INK}">The research loop: 158 experiments, one change per cloud run, every verdict written down</text>']
steps = [("Hypothesis", "from the journal, the data,\nor an agent survey"), ("Cheap measurement", "fold 2, cached members,\nseconds per idea"),
         ("The bar", "strong alone (0.57+)\n+0.003 on a plateau"), ("Streaming build", "library module verified\nagainst the batch matrix"),
         ("Ship", "assembler width checks\nverifier, one script"), ("Cloud, then journal", "one change per run\nverdict and reason")]
for i, (t, b) in enumerate(steps):
    svg.append(box(20 + i * 195, 70, 170, 100, t, b, fill="#f0fdfa" if i % 2 == 0 else "#f9fafb", stroke=ACCENT if i % 2 == 0 else MUTED))
    if i < 5: svg.append(arrow(190 + i * 195, 120, 215 + i * 195, 120))
svg.append(f'<path d="M 1105 170 C 1105 240, 105 240, 105 172" fill="none" stroke="{MUTED}" stroke-width="1.6" stroke-dasharray="6 5" marker-end="url(#a)"/>')
svg.append(f'<text x="590" y="262" text-anchor="middle" font-size="12" fill="{MUTED}">what transferred to the platform, and what did not, rewrites the bar for the next hypothesis</text>')
svg.append('</svg>'); (OUT / "research_loop.svg").write_text("\n".join(svg))

# ---------------------------------------------------------------- detection animation
import sys; sys.path.insert(0, "src")
from structural_break.white import WhiteMonitor
rng = np.random.default_rng(7)
def ar(phi, n, sd=1.0):
    x = np.zeros(n); e = rng.normal(0, sd, n)
    for i in range(1, n): x[i] = phi * x[i - 1] + e[i]
    return x
hist = ar(0.5, 2000); tau = 260
online = np.concatenate([ar(0.5, tau), ar(0.5, 340, sd=1.45)])
wm = WhiteMonitor(hist, odds=True); rows = np.array([wm.update(float(v)) for v in online])
odds = rows[:, 90 + 17]          # the variance-up family mixture (log odds)
cusum = rows[:, 4]               # the scale CUSUM peak
frames = []
for t in range(10, len(online) + 1, 6):
    fig, axes = plt.subplots(2, 1, figsize=(8, 4.4), dpi=110, sharex=True, gridspec_kw=dict(height_ratios=[1.3, 1]))
    axes[0].plot(np.arange(t), online[:t], color=INK, lw=.8); axes[0].axvline(tau, color="#ef4444", lw=1, ls="--", alpha=.8 if t > tau else 0)
    axes[0].set_xlim(0, len(online)); axes[0].set_ylim(-6, 6); axes[0].set_ylabel("online series", color=INK)
    axes[0].set_title(f"step {t}: the whitened monitor reads a variance break (true break at step {tau})", loc="left", fontsize=10, color=INK)
    axes[1].plot(np.arange(t), odds[:t], color=ACCENT, lw=1.6, label="Shiryaev–Roberts log-odds, variance up")
    axes[1].plot(np.arange(t), cusum[:t] / 10, color=MUTED, lw=1, label="scale CUSUM peak (÷10)")
    axes[1].set_xlim(0, len(online)); axes[1].set_ylim(-8, max(12, odds.max() * 1.05)); axes[1].set_xlabel("online step"); axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    for ax in axes:
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0); frames.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))
frames[0].save(OUT / "detection.gif", save_all=True, append_images=frames[1:], duration=70, loop=0)
print("figures:", sorted(p.name for p in OUT.iterdir()))
