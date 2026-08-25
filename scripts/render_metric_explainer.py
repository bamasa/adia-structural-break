"""Render the competition-metric explainer: a standalone HTML page and a GIF.

The metric is easiest to misread as "find the break point in a series". It is
not: what is graded is the *sorting of series* at every step — already-broken
series must rank above still-clean ones. This script renders that idea on five
toy series whose fates cover the interesting cases (fast reaction, late
reaction, late break, calm clean, false spike), in two artefacts:

* ``docs/metric_explainer.html`` — self-contained page with a step slider and
  a toggle for the alarm-cancellation ability; opens from disk, no network.
* ``docs/metric_explainer.gif`` — the same story as an animation: one pass
  with cancellation, one pass without, so the difference in the running
  metric is visible frame by frame.

Both are in Russian: they are the project owner's reading material, like the
inspection notebook.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

N = 20

def _traj(fn) -> np.ndarray:
    return np.array([fn(t) for t in range(1, N + 1)])

SPIKE_STICKY = _traj(lambda t: 0.06 if t < 4 else 0.55)
SPIKE_CANCEL = _traj(lambda t: 0.06 if t < 4 else (0.55 if t < 7 else max(0.10, 0.55 - 0.09 * (t - 6))))

#: name, break step (None = clean), colour, score trajectory
SERIES = [
    ("А — слом на 5, быстрая реакция", 5, "#D85A30",
     _traj(lambda t: 0.05 if t < 5 else min(0.78, 0.10 + 0.09 * (t - 4)))),
    ("Д — слом на 5, опоздавшая реакция", 5, "#D4537E",
     _traj(lambda t: 0.05 if t < 11 else min(0.70, 0.10 + 0.09 * (t - 10)))),
    ("Г — поздний слом на 12", 12, "#BA7517",
     _traj(lambda t: 0.06 if t < 12 else min(0.72, 0.12 + 0.10 * (t - 11)))),
    ("Б — чистый спокойный", None, "#0F6E56",
     _traj(lambda t: 0.05 + 0.01 * (t % 2))),
    ("В — чистый, ложный всплеск на 4", None, "#185FA5", SPIKE_CANCEL),
]


def step_score(t: int, series) -> float | None:
    broken = [s[t - 1] for _, tau, _, s in series if tau is not None and t >= tau]
    clean = [s[t - 1] for _, tau, _, s in series if not (tau is not None and t >= tau)]
    if not broken or not clean:
        return None
    ok = sum(1.0 if b > c else (0.5 if b == c else 0.0) for b in broken for c in clean)
    return ok / (len(broken) * len(clean))


def total_score(series) -> float:
    vals = [v for t in range(1, N + 1) if (v := step_score(t, series)) is not None]
    return float(np.mean(vals))


def variant(cancel: bool):
    out = []
    for name, tau, colour, s in SERIES:
        if name.startswith("В"):
            s = SPIKE_CANCEL if cancel else SPIKE_STICKY
        out.append((name, tau, colour, s))
    return out


def render_gif(path: Path) -> None:
    frames = [(t, True) for t in range(1, N + 1)] + [(t, False) for t in range(1, N + 1)]
    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(11, 4.6), gridspec_kw=dict(width_ratios=[1.6, 1]))

    def draw(frame):
        t, cancel = frame
        series = variant(cancel)
        ax_l.clear(); ax_r.clear()
        for name, tau, colour, s in series:
            ax_l.plot(range(1, N + 1), s, color=colour, lw=2, label=name)
            if tau:
                ax_l.axvline(tau, color=colour, lw=1, ls="--", alpha=0.5)
        ax_l.axvline(t, color="#444441", lw=1.6)
        ax_l.set_ylim(0, 0.9); ax_l.set_xlim(1, N)
        ax_l.set_xlabel("шаг"); ax_l.set_ylabel("счёт модели")
        mode = "модель УМЕЕТ отменять тревогу" if cancel else "отмены НЕТ — всплеск застревает"
        ax_l.set_title(f"{mode}   |   итог за 20 шагов: {total_score(series):.3f}", fontsize=10)
        ax_l.legend(loc="upper left", fontsize=7)

        state = sorted(
            ((name.split(" ")[0], tau is not None and t >= tau, s[t - 1])
             for name, tau, _, s in series),
            key=lambda r: r[2])
        colours = ["#F0997B" if broken else "#9FE1CB" for _, broken, _ in state]
        ax_r.barh(range(len(state)), [v for _, _, v in state], color=colours)
        for i, (short, broken, v) in enumerate(state):
            ax_r.text(0.01, i, f"{short} — {'слом уже был' if broken else 'чистый'}",
                      va="center", fontsize=8,
                      color="#712B13" if broken else "#085041")
        ax_r.set_yticks([]); ax_r.set_xlim(0, 0.9)
        v = step_score(t, series)
        ax_r.set_title("сортировка на шаге "
                       f"{t}: оценка {'—' if v is None else f'{v:.2f}'}", fontsize=10)
        ax_r.set_xlabel("оранжевые должны быть выше зелёных")
        fig.tight_layout()

    anim = FuncAnimation(fig, draw, frames=frames, interval=700)
    anim.save(path, writer=PillowWriter(fps=1.6), dpi=110)
    plt.close(fig)


HTML_HEAD = """<!DOCTYPE html><html lang="ru"><head><meta charset="utf-8">
<title>Метрика соревнования на пальцах</title>
<style>
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:760px;margin:2rem auto;padding:0 1rem;color:#202124;}
h1{font-size:22px;} .muted{color:#5f6368;font-size:14px;line-height:1.6;}
.cards{display:flex;gap:12px;flex-wrap:wrap;margin:1rem 0;}
.card{background:#f1f3f4;border-radius:8px;padding:10px 16px;flex:1;min-width:150px;}
.card .l{font-size:13px;color:#5f6368;} .card .v{font-size:20px;font-weight:600;}
.rowbar{display:flex;align-items:center;gap:10px;margin:5px 0;}
.rowbar .track{flex:1;background:#f1f3f4;border-radius:4px;height:24px;position:relative;}
.rowbar .fill{height:100%;border-radius:4px;}
.rowbar .tag{position:absolute;left:8px;top:3px;font-size:12px;}
input[type=range]{width:100%;}
</style></head><body>
<h1>Метрика соревнования на пальцах</h1>
<p class="muted">Мы сдаём числа, но оценивается фактически <b>сортировка рядов на каждом шаге</b>:
ряды, где слом уже был (оранжевые), должны стоять выше рядов без слома (зелёные).
Оценка шага — доля правильных пар «сломанный выше чистого»; итог — среднее по шагам.
Подвигайте шаг и снимите галочку отмены — увидите, как ложный всплеск без отмены
застревает наверху и съедает итоговую метрику.</p>
"""


def render_html(path: Path) -> None:
    import json
    data = dict(
        n=N,
        sticky=[round(v, 3) for v in SPIKE_STICKY],
        cancel=[round(v, 3) for v in SPIKE_CANCEL],
        series=[dict(name=n_, tau=tau, color=c, s=[round(v, 3) for v in s])
                for n_, tau, c, s in SERIES],
    )
    body = """
<label style="font-size:14px"><input type="checkbox" id="cancel" checked> модель умеет отменять тревогу</label>
<div style="display:flex;align-items:center;gap:10px;margin:0.6rem 0;">
  <span class="muted">Шаг</span><input type="range" min="1" max="20" step="1" value="6" id="st">
  <b id="stout">6</b></div>
<svg id="chart" viewBox="0 0 720 260" width="100%"></svg>
<div id="rows"></div>
<div class="cards">
 <div class="card"><div class="l">Оценка этого шага</div><div class="v" id="sc">—</div></div>
 <div class="card"><div class="l">Итог за шаги 1–20</div><div class="v" id="tot">—</div></div>
 <div class="card"><div class="l">Итог, если отмены нет</div><div class="v" id="tot0">—</div></div>
</div>
<p class="muted" id="expl"></p>
<script>
const D=__DATA__;
const st=document.getElementById("st"),stout=document.getElementById("stout"),
cb=document.getElementById("cancel"),chart=document.getElementById("chart"),
rows=document.getElementById("rows"),scEl=document.getElementById("sc"),
totEl=document.getElementById("tot"),tot0El=document.getElementById("tot0"),
expl=document.getElementById("expl");
function seriesSet(cancel){return D.series.map(r=>r.name.startsWith("В")?{...r,s:cancel?D.cancel:D.sticky}:r);}
function stepScore(t,ss){const b=[],c=[];
 for(const r of ss){(r.tau!==null&&t>=r.tau?b:c).push(r.s[t-1]);}
 if(!b.length||!c.length)return null;let ok=0,n=0;
 for(const x of b)for(const y of c){n++;ok+=x>y?1:(x===y?0.5:0);}return ok/n;}
function total(ss){let s=0,n=0;for(let t=1;t<=D.n;t++){const v=stepScore(t,ss);if(v!==null){s+=v;n++;}}return s/n;}
function render(){
 const t=+st.value;stout.textContent=t;
 const ss=seriesSet(cb.checked);
 const L=40,B=232,W=712;
 const xs=i=>L+8+(W-L-16)*i/(D.n-1),ys=v=>B-(B-18)*v;
 let svg='<line x1="'+L+'" y1="10" x2="'+L+'" y2="'+B+'" stroke="#888"/>'
 +'<line x1="'+L+'" y1="'+B+'" x2="'+W+'" y2="'+B+'" stroke="#888"/>';
 for(const r of ss){
  svg+='<polyline fill="none" stroke="'+r.color+'" stroke-width="2" points="'
   +r.s.map((v,i)=>xs(i)+","+ys(v)).join(" ")+'"/>';
  if(r.tau)svg+='<line x1="'+xs(r.tau-1)+'" y1="14" x2="'+xs(r.tau-1)+'" y2="'+B+'" stroke="'+r.color+'" stroke-dasharray="4 3" opacity="0.5"/>';
  svg+='<text x="'+(xs(D.n-1)-8)+'" y="'+(ys(r.s[D.n-1])-5)+'" font-size="11" fill="'+r.color+'">'+r.name[0]+'</text>';}
 svg+='<line x1="'+xs(t-1)+'" y1="10" x2="'+xs(t-1)+'" y2="'+B+'" stroke="#444" stroke-width="1.5"/>'
 +'<text x="'+(xs(t-1)+4)+'" y="24" font-size="11" fill="#444">шаг '+t+'</text>';
 chart.innerHTML=svg;
 const state=ss.map(r=>({short:r.name[0],color:r.color,score:r.s[t-1],broken:r.tau!==null&&t>=r.tau}))
   .sort((a,b)=>b.score-a.score);
 rows.innerHTML=state.map(r=>{
  const fill=r.broken?"#F0997B":"#9FE1CB",txt=r.broken?"#712B13":"#085041";
  return '<div class="rowbar"><b style="width:18px;color:'+r.color+'">'+r.short+'</b>'
  +'<div class="track"><div class="fill" style="width:'+Math.round(r.score*100)+'%;background:'+fill+'"></div>'
  +'<span class="tag" style="color:'+txt+'">'+(r.broken?"слом уже был":"чистый")+'</span></div>'
  +'<b style="width:44px;text-align:right">'+r.score.toFixed(2)+'</b></div>';}).join("");
 const v=stepScore(t,ss);
 scEl.textContent=v===null?"пар нет":v.toFixed(2);
 totEl.textContent=total(ss).toFixed(3);
 tot0El.textContent=total(seriesSet(false)).toFixed(3);
 const bad=[];const b=state.filter(r=>r.broken),c=state.filter(r=>!r.broken);
 for(const x of b)for(const y of c)if(y.score>x.score)bad.push("«"+y.short+"» выше сломанного «"+x.short+"»");
 expl.textContent=v===null?"Сломов ещё нет — судья пропускает шаг."
   :(bad.length?"Потерянные пары: "+bad.join("; ")+".":"Все сломанные выше всех чистых — шаг идеальный.");}
st.addEventListener("input",render);cb.addEventListener("change",render);render();
</script></body></html>"""
    path.write_text(HTML_HEAD + body.replace("__DATA__", json.dumps(data, ensure_ascii=False)))


if __name__ == "__main__":
    DOCS.mkdir(exist_ok=True)
    render_html(DOCS / "metric_explainer.html")
    print("html ->", DOCS / "metric_explainer.html")
    render_gif(DOCS / "metric_explainer.gif")
    print("gif ->", DOCS / "metric_explainer.gif")
