#!/usr/bin/env python3
"""Construye la tabla de ventanas (E4 + contexto del celular) solo con tramos de bus.

Criterio (acordado): se usa SOLO bus.
  - P01 (P1): todo es bus (el label 'walking' es un error de la app).
  - P02 (P2): sesiones 09-28_08-21, 09-28_13-24 y las dos sesion_2026-09-18_* son bus completas;
    en las demás solo el label 'bus'.
  - P03 (P3) y P09 (P4): solo el label 'bus'.
  - Se recortan RECORTE s al inicio y al final de cada tramo continuo de bus (subida/bajada).
  - Excluidas: P01_..._08-41-16 (E4 solo 20 s) y P02_..._18-57-04 (sin E4).
Ventanas de VENTANA s con paso PASO s (50 % de traslape).
Salida: salida/ventanas.csv y salida/resumen_ventanas.txt
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

from decode_e4 import decodificar_e4, listar_sesiones

RAIZ = Path(__file__).parent
DATOS, SALIDA = RAIZ / "DATAE4", RAIZ / "salida"

VENTANA, PASO, RECORTE = 30, 15, 60
COB_MIN = {"ppg": 0.8, "acc": 0.8, "temp": 0.5, "eda": 1 / 3}  # fracción mínima de muestras esperadas
PAD = 5  # segundos extra a cada lado al filtrar el PPG
UMBRAL_HR = 10  # bpm máx. de diferencia entre FC por picos y FC espectral

EXCLUIR = {
    "P01_2026-09-26_08-41-16": "E4 solo 20 s de 16 min",
    "P02_2026-09-28_18-57-04": "sin datos E4 (not_connected)",
}
PARTICIPANTES_TODO_BUS = {"P01"}
SESIONES_TODO_BUS = {
    "P02_2026-09-28_08-21-45", "P02_2026-09-28_13-24-09",
    "sesion_2026-09-18_09-57-46", "sesion_2026-09-18_13-22-02",
}
PULSERA = {"P01": "A", "P09": "A", "P02": "B", "P03": "C"}  # P01 y P09 compartieron pulsera
NOMBRES = {"P01": "P1", "P02": "P2", "P03": "P3", "P09": "P4"}


# ---------------------------------------------------------------- contexto (celular)
def _epoch(s):
    return (pd.to_datetime(s, utc=True) - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()


def cargar_contexto(sdir, persona):
    """Devuelve (ctx por segundo, meta de sesión) o None si no hay datos de contexto."""
    if sdir.name.startswith("NoiseGeo_"):
        meta = json.load(open(sdir / "metadatos.json"))
        df = pd.read_csv(sdir / "mediciones.csv")
        ctx = pd.DataFrame({
            "t": _epoch(df["timestamp"]), "lat": df["latitude"], "lon": df["longitude"],
            "gps_acc": df["gps_accuracy"], "vel": df["speed_kmh"], "ruido": df["noise_dbfs"],
            "ruido_etq": df["noise_label"], "evento": df["event"], "modo": df["transport_mode"],
        })
        sid, part = meta["session_id"], meta["participant_id"]
        info = dict(trip_stress=meta.get("trip_stress"), stress_before=meta.get("stress_before"),
                    stress_after=meta.get("stress_after"), traffic_at_end=meta.get("traffic_at_end"),
                    fuente="NoiseGeo")
        todo_bus = part in PARTICIPANTES_TODO_BUS or sid in SESIONES_TODO_BUS
        ctx["bus"] = True if todo_bus else (ctx["modo"] == "bus")
    else:  # carpeta sesion_*: gps + ruido sin velocidad ni modo
        g = pd.read_json(next(sdir.glob("noise_gps_*.jsonl")), lines=True)
        ctx = pd.DataFrame({"t": _epoch(g["ts"]), "lat": g["lat"], "lon": g["lon"], "gps_acc": g["accuracy"]})
        ctx = ctx.drop_duplicates("t").reset_index(drop=True)
        dy = np.diff(ctx["lat"]) * 111320
        dx = np.diff(ctx["lon"]) * 111320 * math.cos(math.radians(ctx["lat"].mean()))
        dt = np.diff(ctx["t"])
        v = np.r_[np.nan, np.hypot(dx, dy) / np.where(dt > 0, dt, np.nan) * 3.6]
        ctx["vel"] = pd.Series(v).rolling(5, center=True, min_periods=2).median().clip(upper=120)
        ctx["ruido"] = np.nan  # el micrófono grabó -120 dBFS (sin señal)
        ctx["ruido_etq"], ctx["evento"], ctx["modo"], ctx["bus"] = None, None, "bus", True
        sid, part = sdir.name.replace("sesion_", "sesion_"), "P02"
        info = dict(trip_stress=None, stress_before=None, stress_after=None, traffic_at_end=None,
                    fuente="sesion_0918")
    ctx.loc[ctx["ruido"] <= -119, "ruido"] = np.nan  # -120 = sin señal
    return ctx.sort_values("t").reset_index(drop=True), dict(sid=sid, part=part, **info)


def tramos_bus(ctx):
    """Índices (inicio, fin) de tramos continuos de bus (se parte si hay saltos > 5 s)."""
    b, t = ctx["bus"].to_numpy(bool), ctx["t"].to_numpy()
    tramos, ini = [], None
    for i in range(len(b)):
        if b[i] and ini is None:
            ini = i
        corta = ini is not None and (not b[i] or (i + 1 < len(b) and t[i + 1] - t[i] > 5))
        if ini is not None and (corta or i == len(b) - 1):
            fin = i - 1 if not b[i] else i
            tramos.append((ini, fin))
            ini = None
    return tramos


# ---------------------------------------------------------------- features
def _nan_dict(*claves):
    return {c: np.nan for c in claves}


def f_contexto(c):
    out = dict(ctx_n=len(c))
    r = c["ruido"].dropna().to_numpy()
    if len(r) >= 24:
        out.update(ruido_leq_dbfs=10 * np.log10(np.mean(10 ** (r / 10))), ruido_max_dbfs=r.max(), ruido_std_dbfs=r.std())
        et = c["ruido_etq"].dropna()
        out["ruido_etq_moda"] = et.mode().iloc[0] if len(et) else None
    else:
        out.update(_nan_dict("ruido_leq_dbfs", "ruido_max_dbfs", "ruido_std_dbfs"), ruido_etq_moda=None)
    v = c["vel"].dropna().to_numpy()
    if len(v) >= 15:
        out.update(vel_media_kmh=v.mean(), vel_std_kmh=v.std(), frac_detenido=(v < 1).mean())
    else:
        out.update(_nan_dict("vel_media_kmh", "vel_std_kmh", "frac_detenido"))
    out["gps_acc_med_m"] = c["gps_acc"].median()
    out["lat"], out["lon"] = c["lat"].mean(), c["lon"].mean()
    ev = c["evento"].dropna()
    out["evento"] = ev.iloc[0] if len(ev) else None
    return out


def f_ppg(t, v, ws, we):
    claves = ("hr_picos_bpm", "hr_espec_bpm", "rmssd_ms", "sdnn_ms", "ppg_sqi", "n_ibi_validos")
    fs = 64.0
    i0, i1 = np.searchsorted(t, [ws - PAD, we + PAD])
    tt, vv = t[i0:i1], v[i0:i1]
    cob = ((tt >= ws) & (tt < we)).sum() / (fs * VENTANA)
    out = dict(ppg_cobertura=cob, **_nan_dict(*claves))
    if cob < COB_MIN["ppg"]:
        return out
    grid = np.arange(ws - PAD, we + PAD, 1 / fs)
    x = np.interp(grid, tt, vv)
    b, a = butter(3, [0.5, 8], btype="band", fs=fs)
    y = filtfilt(b, a, x - x.mean())
    core = y[(grid >= ws) & (grid < we)]
    core = (core - core.mean()) / (core.std() + 1e-9)
    # espectro
    N = len(core) * 4
    P = np.abs(np.fft.rfft(core * np.hanning(len(core)), N)) ** 2
    f = np.fft.rfftfreq(N, 1 / fs)
    m = (f >= 0.7) & (f <= 3.0)
    f0 = f[m][np.argmax(P[m])]
    tot = P[(f >= 0.5) & (f <= 8)].sum()
    cerca = P[(abs(f - f0) <= 0.2) | (abs(f - 2 * f0) <= 0.3)].sum()
    out["hr_espec_bpm"], out["ppg_sqi"] = 60 * f0, cerca / tot
    # picos: señal 0.5-4 Hz y separación mínima = 60 % del periodo espectral (evita contar la muesca dicrota)
    b2, a2 = butter(3, [0.5, 4], btype="band", fs=fs)
    y2 = filtfilt(b2, a2, x - x.mean())
    core2 = y2[(grid >= ws) & (grid < we)]
    core2 = (core2 - core2.mean()) / (core2.std() + 1e-9)
    dist = max(int(0.33 * fs), int(0.6 * fs / f0))
    mejor = None
    for pol in (1, -1):
        idx, _ = find_peaks(pol * core2, distance=dist, prominence=0.3)
        ibi = np.diff(idx) / fs
        plaus = (ibi >= 0.4) & (ibi <= 1.5)
        if plaus.sum() == 0:
            continue
        med = np.median(ibi[plaus])
        valido = plaus & (abs(ibi - med) <= 0.3 * med)
        if mejor is None or valido.sum() > mejor[0].sum():
            mejor = (valido, ibi)
    if mejor is not None and mejor[0].sum() >= 15:
        valido, ibi = mejor
        out["n_ibi_validos"] = int(valido.sum())
        out["hr_picos_bpm"] = 60 / ibi[valido].mean()
        out["sdnn_ms"] = 1000 * ibi[valido].std()
        par = valido[1:] & valido[:-1]
        d = np.diff(ibi)[par]
        if len(d) >= 10:
            out["rmssd_ms"] = 1000 * np.sqrt(np.mean(d ** 2))
    return out


def _rango(t, v, ws, we):
    i0, i1 = np.searchsorted(t, [ws, we])
    return t[i0:i1], v[i0:i1]


def f_acc(t, xyz, ws, we):
    claves = ("acc_mag_media_g", "acc_mag_std_g", "acc_dif_media_g")
    i0, i1 = np.searchsorted(t, [ws, we])
    cob = (i1 - i0) / (32 * VENTANA)
    out = dict(acc_cobertura=cob, **_nan_dict(*claves))
    if cob >= COB_MIN["acc"]:
        mag = np.sqrt((xyz[i0:i1] ** 2).sum(axis=1))
        out.update(acc_mag_media_g=mag.mean(), acc_mag_std_g=mag.std(), acc_dif_media_g=np.abs(np.diff(mag)).mean())
    return out


def f_temp(t, v, ws, we):
    tt, vv = _rango(t, v, ws, we)
    cob = len(tt) / (4 * VENTANA)
    out = dict(temp_cobertura=cob, temp_media_c=np.nan, temp_pend_c_min=np.nan)
    if cob >= COB_MIN["temp"]:
        out["temp_media_c"] = vv.mean()
        out["temp_pend_c_min"] = np.polyfit((tt - tt[0]) / 60, vv, 1)[0]
    return out


def f_eda(t, v, ws, we):
    tt, vv = _rango(t, v, ws, we)
    cob = len(tt) / (0.5 * VENTANA)
    out = dict(eda_cobertura=cob, **_nan_dict("eda_media_us", "eda_std_us", "eda_min_us", "eda_max_us", "eda_pend_us_min"))
    if cob >= COB_MIN["eda"]:
        out.update(eda_media_us=vv.mean(), eda_std_us=vv.std(), eda_min_us=vv.min(), eda_max_us=vv.max(),
                   eda_pend_us_min=np.polyfit((tt - tt[0]) / 60, vv, 1)[0])
    return out


# ---------------------------------------------------------------- principal
def procesar_sesion(persona, sdir, e4_ruta):
    ctx, meta = cargar_contexto(sdir, persona)
    sid, part = meta["sid"], meta["part"]
    if sid in EXCLUIR:
        return [], dict(sesion=sid, motivo=EXCLUIR[sid])
    series, _ = decodificar_e4(e4_ruta)
    if not series:
        return [], dict(sesion=sid, motivo="sin e4")
    ppg = series["ppg"]
    acc = series["acc"]
    ppg_t, ppg_v = ppg["t"].to_numpy(), ppg["ppg"].to_numpy()
    acc_t, acc_v = acc["t"].to_numpy(), acc[["x_g", "y_g", "z_g"]].to_numpy()
    tmp_t, tmp_v = series["temp"]["t"].to_numpy(), series["temp"]["temp_c"].to_numpy()
    eda_t, eda_v = (series["eda"]["t"].to_numpy(), series["eda"]["eda_us"].to_numpy()) if "eda" in series else (np.empty(0), np.empty(0))
    t_ctx = ctx["t"].to_numpy()
    filas, n_tramos = [], 0
    for ti, (i, j) in enumerate(tramos_bus(ctx)):
        t0, t1 = t_ctx[i] + RECORTE, t_ctx[j] - RECORTE
        if t1 - t0 < VENTANA:
            continue
        n_tramos += 1
        ws = t0
        while ws + VENTANA <= t1:
            we = ws + VENTANA
            c = ctx.iloc[np.searchsorted(t_ctx, ws):np.searchsorted(t_ctx, we)]
            hora = pd.to_datetime(ws, unit="s", utc=True).tz_convert("America/Lima")
            fila = dict(
                participante=part, sesion=sid, fuente=meta["fuente"], pulsera=PULSERA[part], tramo_bus=ti,
                t_ini=ws, ts_inicio=hora.strftime("%Y-%m-%d %H:%M:%S"), fecha=hora.strftime("%Y-%m-%d"),
                hora_local=hora.hour + hora.minute / 60 + hora.second / 3600,
                **f_contexto(c), **f_ppg(ppg_t, ppg_v, ws, we), **f_acc(acc_t, acc_v, ws, we),
                **f_temp(tmp_t, tmp_v, ws, we), **f_eda(eda_t, eda_v, ws, we),
                trip_stress=meta["trip_stress"], stress_before=meta["stress_before"],
                stress_after=meta["stress_after"], traffic_at_end=meta["traffic_at_end"],
            )
            filas.append(fila)
            ws += PASO
    return filas, dict(sesion=sid, motivo="ok", tramos=n_tramos)


def main():
    SALIDA.mkdir(exist_ok=True)
    todas, estado = [], []
    for persona, sdir, e4 in listar_sesiones(DATOS):
        if e4.stat().st_size == 0:
            estado.append(dict(sesion=sdir.name, motivo="e4 vacío"))
            continue
        filas, est = procesar_sesion(persona, sdir, e4)
        todas += filas
        estado.append(est)
        print(f"{est['sesion']:28} {est['motivo']:30} ventanas={len(filas)}")
    df = pd.DataFrame(todas)
    # calidad por señal
    df["ppg_ok"] = (df["ppg_cobertura"] >= COB_MIN["ppg"]) & df["hr_picos_bpm"].notna() & df["hr_espec_bpm"].notna() \
        & ((df["hr_picos_bpm"] - df["hr_espec_bpm"]).abs() <= UMBRAL_HR)
    df["acc_ok"] = df["acc_mag_media_g"].notna()
    df["temp_ok"] = df["temp_media_c"].notna()
    df["eda_ok"] = df["eda_media_us"].notna()
    df["completa"] = df[["ppg_ok", "acc_ok", "temp_ok", "eda_ok"]].all(axis=1)
    df["con_ruido"] = df["ruido_leq_dbfs"].notna()
    df.round(5).to_csv(SALIDA / "ventanas.csv", index=False)
    resumen(df, estado)


def resumen(df, estado):
    L = []
    L.append(f"Ventanas de {VENTANA} s, paso {PASO} s, recorte {RECORTE} s por tramo de bus")
    L.append(f"Total de ventanas candidatas (bus, recortado): {len(df)}")
    L.append("\n== Por persona ==")
    g = df.groupby("participante").agg(
        sesiones=("sesion", "nunique"), ventanas=("t_ini", "size"), ppg_ok=("ppg_ok", "sum"), acc_ok=("acc_ok", "sum"),
        temp_ok=("temp_ok", "sum"), eda_ok=("eda_ok", "sum"), completas=("completa", "sum"), con_ruido=("con_ruido", "sum"))
    g.index = [f"{i} ({NOMBRES[i]})" for i in g.index]
    g.loc["TOTAL"] = g.sum()
    g["horas_aprox"] = (g["ventanas"] * PASO / 3600).round(2)
    L.append(g.to_string())
    L.append("\n== Por sesión ==")
    s = df.groupby(["participante", "sesion"]).agg(ventanas=("t_ini", "size"), completas=("completa", "sum"),
                                                   ppg_ok=("ppg_ok", "sum"), eda_ok=("eda_ok", "sum"), con_ruido=("con_ruido", "sum"))
    L.append(s.to_string())
    L.append("\n== Columnas con nulos (cantidad y % de las ventanas) ==")
    nn = df.isna().sum()
    nn = nn[nn > 0].sort_values(ascending=False)
    L.append(pd.DataFrame({"nulos": nn, "pct": (100 * nn / len(df)).round(1)}).to_string())
    L.append("\n== Sesiones sin ventanas / excluidas ==")
    for e in estado:
        if e["motivo"] != "ok":
            L.append(f"  {e['sesion']}: {e['motivo']}")
    txt = "\n".join(L)
    (SALIDA / "resumen_ventanas.txt").write_text(txt)
    print("\n" + txt)


if __name__ == "__main__":
    sys.exit(main())
