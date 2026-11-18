#!/usr/bin/env python3
"""H1: respuesta cardíaca a estallidos de ruido (análisis de eventos, dentro de cada sesión).
Ver salida/novedoso/PROTOCOLO_novedoso.md. Desviación documentada: en vez de 20 controles aleatorios por evento se usan TODOS los
instantes elegibles a > 30 s de cualquier evento (mismo estimando, menor varianza)."""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.signal import butter, filtfilt
from scipy.stats import spearmanr

from build_ventanas import RECORTE, cargar_contexto, tramos_bus
from decode_e4 import listar_sesiones

warnings.filterwarnings("ignore")
RAIZ = Path(__file__).parent
SAL, OUT = RAIZ / "salida", RAIZ / "salida" / "novedoso"
SEED = 0
FS = 64
PRE, POST = (-10, -2), (3, 15)
UMBRAL_DB, SEP, MIN_SQI = 6.0, 30, 0.30
QUANT = 0.9


# ------------------------------------------------------------------ FC a 1 Hz
def hr_1hz(t, v, grid):
    tg = np.arange(t[0], t[-1], 1 / FS)
    y = np.interp(tg, t, v)
    idx = np.clip(np.searchsorted(t, tg), 1, len(t) - 1)
    hueco = np.minimum(np.abs(t[idx] - tg), np.abs(t[idx - 1] - tg)) > 0.25
    b, a = butter(3, [0.5, 4], btype="band", fs=FS)
    y = filtfilt(b, a, y - y.mean())
    L = 10 * FS
    hr, sqi = np.full(len(grid), np.nan), np.full(len(grid), np.nan)
    pos = np.round((grid - tg[0]) * FS).astype(int)
    ok = (pos - L // 2 >= 0) & (pos + L // 2 < len(y))
    N = 8192
    f = np.fft.rfftfreq(N, 1 / FS)
    banda = (f >= 0.7) & (f <= 3.0)
    tot = (f >= 0.5) & (f <= 4)
    win = np.hanning(L)
    ids = np.where(ok)[0]
    for ch in np.array_split(ids, max(1, len(ids) // 400)):
        seg = y[pos[ch][:, None] + np.arange(-L // 2, L // 2)] * win
        bad = np.array([hueco[p - L // 2:p + L // 2].any() for p in pos[ch]])
        P = np.abs(np.fft.rfft(seg, N, axis=1)) ** 2
        k = np.argmax(np.where(banda, P, -1), axis=1)
        f0 = f[k]
        for j, (i, f0j) in enumerate(zip(ch, f0)):
            cerca = P[j][(np.abs(f - f0j) <= 0.15) | (np.abs(f - 2 * f0j) <= 0.25)].sum()
            sqi[i] = cerca / P[j][tot].sum()
            hr[i] = 60 * f0j if not bad[j] else np.nan
    hr[(sqi < MIN_SQI) | (hr < 40) | (hr > 180)] = np.nan
    return hr


def acc_1hz(acc, grid):
    mag = np.sqrt((acc[["x_g", "y_g", "z_g"]] ** 2).sum(axis=1))
    s = pd.Series(mag.values, index=np.floor(acc["t"].values).astype(int)).groupby(level=0).std()
    out = s.reindex(grid.astype(int)).to_numpy()
    return np.log(out + 1e-4)


# ------------------------------------------------------------------ eventos
def media_ventana(x, a, b, minfrac=0.6):
    """media de x[i+a..i+b] (incluidos) con al menos minfrac de valores válidos."""
    n = len(x)
    v = np.isfinite(x)
    cs = np.concatenate([[0.0], np.cumsum(np.where(v, x, 0.0))])
    cv = np.concatenate([[0.0], np.cumsum(v)])
    i = np.arange(n)
    lo, hi = i + a, i + b + 1
    ok = (lo >= 0) & (hi <= n)
    lo, hi = np.clip(lo, 0, n), np.clip(hi, 0, n)
    num, cnt = cs[hi] - cs[lo], cv[hi] - cv[lo]
    return np.where(ok & (cnt >= minfrac * (b - a + 1)), num / np.maximum(cnt, 1), np.nan)


def detectar(noise):
    s = pd.Series(noise)
    med = s.rolling(61, center=True, min_periods=30).median()
    delta = (s - med).to_numpy()
    q90 = s.quantile(QUANT)
    cand = np.where((delta >= UMBRAL_DB) & (noise >= q90))[0]
    cand = cand[np.argsort(-delta[cand])]
    elegidos = []
    for i in cand:
        if all(abs(i - j) >= SEP for j in elegidos):
            elegidos.append(i)
    return np.array(sorted(elegidos), dtype=int), delta


def preparar_sesion(sdir, persona, e4_ruta):
    ctx, meta = cargar_contexto(sdir, persona)
    sid = meta["sid"]
    if sid in ("P01_2026-09-26_08-41-16", "P02_2026-09-28_18-57-04") or sdir.name.startswith("sesion_") or ctx["ruido"].notna().sum() < 600:
        return None
    ppg = pd.read_parquet(SAL / "decodificado" / f"{sid}__ppg.parquet")
    acc = pd.read_parquet(SAL / "decodificado" / f"{sid}__acc.parquet")
    t0, t1 = int(np.ceil(max(ctx["t"].iloc[0], ppg["t"].iloc[0]))), int(np.floor(min(ctx["t"].iloc[-1], ppg["t"].iloc[-1])))
    grid = np.arange(t0, t1)
    tc = ctx["t"].to_numpy()
    j = np.clip(np.searchsorted(tc, grid), 1, len(tc) - 1)
    cerca = np.minimum(np.abs(tc[j] - grid), np.abs(tc[j - 1] - grid)) <= 2
    ruido = np.where(cerca, np.interp(grid, tc, ctx["ruido"].fillna(-120).to_numpy()), np.nan)
    ruido[ruido <= -119] = np.nan
    # elegibilidad: [i-10, i+15] dentro de un tramo de bus recortado
    elig = np.zeros(len(grid), bool)
    for i0, i1 in tramos_bus(ctx):
        a, b = tc[i0] + RECORTE, tc[i1] - RECORTE
        elig |= (grid - 10 >= a) & (grid + 15 <= b)
    hr = hr_1hz(ppg["t"].to_numpy(), ppg["ppg"].to_numpy(), grid.astype(float))
    lacc = acc_1hz(acc, grid)
    dHR = media_ventana(hr, *POST) - media_ventana(hr, *PRE)
    dACC = media_ventana(lacc, *POST) - media_ventana(lacc, *PRE)
    return dict(sid=sid, part=meta["part"], grid=grid, ruido=ruido, elig=elig, hr=hr, lacc=lacc, dHR=dHR, dACC=dACC)


def estadistico(s, ruido):
    ev, delta = detectar(ruido)
    ev_ok = ev[s["elig"][ev] & np.isfinite(s["dHR"][ev])]
    prohibido = np.zeros(len(ruido), bool)
    for i in ev:
        prohibido[max(0, i - SEP):i + SEP + 1] = True
    ctl = np.where(s["elig"] & np.isfinite(s["dHR"]) & ~prohibido)[0]
    return ev_ok, ctl, delta


def global_contraste(res):
    n = np.array([r[0] for r in res]); c = np.array([r[1] for r in res])
    ok = n >= 3
    return np.average(c[ok], weights=n[ok]) if ok.any() else np.nan


def main():
    sesiones = []
    for persona, sdir, e4 in listar_sesiones(RAIZ / "DATAE4"):
        if e4.stat().st_size == 0:
            continue
        s = preparar_sesion(sdir, persona, e4)
        if s:
            sesiones.append(s)
            print(f"{s['sid']}: {len(s['grid'])} s, FC válida {np.isfinite(s['hr']).mean():.0%}, ruido válido {np.isfinite(s['ruido']).mean():.0%}", flush=True)
    # observado
    obs_res, filas, epocas = [], [], []
    for s in sesiones:
        ev, ctl, delta = estadistico(s, s["ruido"])
        mE, mC = s["dHR"][ev].mean() if len(ev) else np.nan, s["dHR"][ctl].mean() if len(ctl) else np.nan
        obs_res.append((len(ev), mE - mC))
        filas.append(dict(sesion=s["sid"], participante=s["part"], n_eventos=len(ev), n_controles=len(ctl), dHR_eventos=mE, dHR_controles=mC, contraste=mE - mC))
        for i in ev:
            epocas.append(dict(sesion=s["sid"], evento=1, dHR=s["dHR"][i], dACC=s["dACC"][i], delta_db=delta[i]))
        for i in ctl[::5]:
            epocas.append(dict(sesion=s["sid"], evento=0, dHR=s["dHR"][i], dACC=s["dACC"][i], delta_db=np.nan))
    tab = pd.DataFrame(filas)
    print("\n", tab.round(2).to_string(index=False))
    obs = global_contraste(obs_res)
    # bootstrap por sesiones
    rng = np.random.default_rng(SEED)
    valid = [r for r in obs_res if r[0] >= 3]
    bs = [global_contraste([valid[i] for i in rng.integers(0, len(valid), len(valid))]) for _ in range(5000)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    # permutación: desplazamiento circular del ruido dentro de cada sesión
    nulos = []
    for _ in range(500):
        res = []
        for s in sesiones:
            L = len(s["ruido"])
            sh = rng.integers(120, L - 120)
            ev, ctl, _ = estadistico(s, np.roll(s["ruido"], sh))
            res.append((len(ev), (s["dHR"][ev].mean() - s["dHR"][ctl].mean()) if len(ev) and len(ctl) else np.nan))
        nulos.append(global_contraste(res))
    nulos = np.array(nulos)
    p = (np.sum(np.abs(nulos) >= abs(obs)) + 1) / (len(nulos) + 1)
    n_ev = int(sum(r[0] for r in obs_res))
    print(f"\nH1  eventos={n_ev} en {len(valid)} sesiones | contraste ΔFC (eventos - controles) = {obs:+.2f} lpm  IC95%(sesiones)=[{lo:+.2f}, {hi:+.2f}]  p(permutación)={p:.4f}  nulo sd={np.nanstd(nulos):.2f}")
    # sensibilidad: ajustar por ΔACC
    ep = pd.DataFrame(epocas).dropna(subset=["dHR", "dACC"])
    m = smf.ols("dHR ~ evento + dACC + C(sesion)", ep).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(ep["sesion"])[0]})
    print(f"Ajustado por ΔACC y sesión: efecto del evento = {m.params['evento']:+.2f} lpm (IC95% {m.conf_int().loc['evento', 0]:+.2f} a {m.conf_int().loc['evento', 1]:+.2f}, p={m.pvalues['evento']:.3f}); efecto ΔACC = {m.params['dACC']:+.2f}")
    e = ep[ep.evento == 1].dropna(subset=["delta_db"])
    rho, pr = spearmanr(e["delta_db"], e["dHR"])
    print(f"Dosis-respuesta (exploratorio): Spearman(ΔdB del evento, ΔFC) = {rho:+.3f}, p={pr:.3f}, n={len(e)}")
    # por participante
    print("Por participante (contraste ponderado):", {p_: round(global_contraste([(r[0], r[1]) for r, s in zip(obs_res, sesiones) if s['part'] == p_]), 2) for p_ in sorted({s['part'] for s in sesiones})})
    # curva promediada
    lags = np.arange(-10, 31)
    curvas = {"eventos": [], "controles": []}
    for s in sesiones:
        ev, ctl, _ = estadistico(s, s["ruido"])
        base = media_ventana(s["hr"], *PRE)
        for nombre, ids in (("eventos", ev), ("controles", ctl[::5])):
            if len(ids) < 3:
                curvas[nombre].append(None); continue
            M = np.array([[s["hr"][i + l] - base[i] if 0 <= i + l < len(s["hr"]) else np.nan for l in lags] for i in ids])
            curvas[nombre].append(np.nanmean(M, axis=0))
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for nombre, col in (("eventos", "C3"), ("controles", "gray")):
        C = np.array([c for c in curvas[nombre] if c is not None])
        mu = np.nanmean(C, axis=0)
        bs_c = np.array([np.nanmean(C[rng.integers(0, len(C), len(C))], axis=0) for _ in range(2000)])
        ax.plot(lags, mu, color=col, label=f"{nombre}"); ax.fill_between(lags, *np.nanpercentile(bs_c, [2.5, 97.5], axis=0), color=col, alpha=.2)
    ax.axvline(0, color="k", lw=.7); ax.axhline(0, color="k", lw=.5); ax.set_xlabel("segundos respecto al estallido de ruido"); ax.set_ylabel("ΔFC (lpm) vs. línea base −10..−2 s"); ax.legend()
    plt.tight_layout(); plt.savefig(OUT / "fig_h1_curva.png", dpi=130); plt.close()
    tab.round(3).to_csv(OUT / "h1_por_sesion.csv", index=False)


if __name__ == "__main__":
    main()
