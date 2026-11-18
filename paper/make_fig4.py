"""Fig. 4 del artículo: curva de FC alrededor de estallidos de ruido (configuración primaria, protocolo fijado)."""
import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h1_ruido_fc as H
from decode_e4 import listar_sesiones
from pathlib import Path
plt.rcParams.update({"font.size": 6.5, "axes.labelsize": 6.5, "legend.fontsize": 6, "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.spines.top": False, "axes.spines.right": False})
S = []
for persona, sdir, e4 in listar_sesiones(H.RAIZ / "DATAE4"):
    if e4.stat().st_size == 0: continue
    s = H.preparar_sesion(sdir, persona, e4)
    if s: S.append(s)
lags = np.arange(-10, 31); rng = np.random.default_rng(0); curvas = {"noise bursts": [], "control instants": []}
for s in S:
    ev, ctl, _ = H.estadistico(s, s["ruido"]); base = H.media_ventana(s["hr"], *H.PRE)
    for nombre, ids in (("noise bursts", ev), ("control instants", ctl[::5])):
        if len(ids) < 3: continue
        M = np.array([[s["hr"][i + l] - base[i] if 0 <= i + l < len(s["hr"]) else np.nan for l in lags] for i in ids])
        curvas[nombre].append(np.nanmean(M, axis=0))
fig, ax = plt.subplots(figsize=(4.8, 2.3))
for (nombre, col) in (("noise bursts", "#c53030"), ("control instants", "#718096")):
    C = np.array(curvas[nombre]); mu = np.nanmean(C, axis=0)
    bs = np.array([np.nanmean(C[rng.integers(0, len(C), len(C))], axis=0) for _ in range(2000)])
    ax.plot(lags, mu, color=col, lw=1.1, label=f"{nombre} ({len(C)} sessions)"); ax.fill_between(lags, *np.nanpercentile(bs, [2.5, 97.5], axis=0), color=col, alpha=.2)
ax.axvline(0, color="k", lw=.6); ax.axhline(0, color="k", lw=.4); ax.set_xlabel("seconds relative to burst onset"); ax.set_ylabel("HR change vs. [−10, −2] s baseline (bpm)"); ax.legend(loc="upper right")
plt.tight_layout(); plt.savefig(Path(__file__).parent / "figs" / "fig4_h1.pdf"); print("sesiones:", len(S), {k: len(v) for k, v in curvas.items()})
