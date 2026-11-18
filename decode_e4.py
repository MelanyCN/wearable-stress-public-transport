#!/usr/bin/env python3
"""Decodifica los paquetes BLE crudos de la Empatica E4 (e4.jsonl) a series con unidades reales.

No existe documentación pública del protocolo BLE: los formatos están INFERIDOS y validados con
criterios físicos (ver notas). Cada paquete trae 20 bytes:

  temp   8 x uint16 little-endian, °C = raw*0.02 - 273.15, 4 Hz. Contador en byte 19 (bytes 16-18 = 0).   [confianza alta]
  acc    6 x (x, y, z) int8, g = v/64, 32 Hz. Contador en byte 19 (byte 18 desconocido).           [confianza alta]
  ppg    11 muestras de 14 bits con signo (MSB primero) en los 154 primeros bits, 64 Hz.
         Byte 19 = 2 bits de dato + contador de 6 bits. Relleno "idle" = 8127.                   [media-alta, sin validar vs referencia]
  eda    gsr tipo B: bytes 16-17 == 00 01 -> 4 x float32 little-endian (µS), contador en byte 18,
         1 paquete cada ~2 s. Se guarda la media de los 4 floats.                                [media-alta]
  gsr tipo A (6 x 3 bytes) y bat: NO decodificados.

Los paquetes llegan con jitter, así que el tiempo de cada muestra se reconstruye con el contador
(paquetes perdidos = huecos; el contador va de 1 a 63 en ppg y de 1 a 255 en los demás, nunca vale 0) y un ajuste lineal índice-de-muestra vs hora de llegada (ts_wall).

Uso:  python decode_e4.py            # decodifica todas las sesiones de DATAE4/ a salida/decodificado/
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CANALES = {
    # n = muestras por paquete, fs = Hz nominal, mod = módulo del contador
    # El contador va de 1 a mod (el 0 nunca aparece): se usa (contador-1) con módulo = mod
    "ppg": dict(n=11, fs=64.0, mod=63),
    "acc": dict(n=6, fs=32.0, mod=255),
    "temp": dict(n=8, fs=4.0, mod=255),
    "eda": dict(n=1, fs=0.5, mod=255),
}
PPG_IDLE = 8127
_PESOS14 = 1 << np.arange(13, -1, -1)


def _leer_paquetes(ruta):
    filas = []
    with open(ruta) as f:
        for linea in f:
            r = json.loads(linea)
            filas.append((r["channel"], r["ts_wall"], bytes.fromhex(r["payload_hex"])))
    return filas


def _matriz(paquetes):
    if not paquetes:
        return np.empty((0, 20), np.uint8), np.empty(0)
    arr = np.frombuffer(b"".join(p[2] for p in paquetes), np.uint8).reshape(-1, 20)
    return arr, np.array([p[1] for p in paquetes], float)


def _desenrollar(c, t, mod, tasa):
    """Contador -> índice de paquete creciente, usando el tiempo para resolver vueltas del contador."""
    k = np.zeros(len(c), np.int64)
    for i in range(1, len(c)):
        d = (int(c[i]) - int(c[i - 1])) % mod
        est = (t[i] - t[i - 1]) * tasa
        m = max(0, round((est - d) / mod))
        k[i] = k[i - 1] + d + m * mod
    return k


def _tiempos(k, t, n, fs):
    """Ajusta t_llegada = a + b * (índice de la última muestra del paquete). Devuelve a, b, fs_efectiva."""
    s_fin = (k + 1) * n
    if len(k) < 3 or s_fin[-1] == s_fin[0]:
        b = 1 / fs
        return float(np.median(t - b * s_fin)), b, np.nan
    b, a = np.polyfit(s_fin, t, 1)
    for _ in range(2):  # refinamiento ignorando atípicos
        res = t - (a + b * s_fin)
        ok = np.abs(res) < max(1.0, 3 * res.std())
        b, a = np.polyfit(s_fin[ok], t[ok], 1)
    fs_ef = 1 / b
    if abs(fs_ef / fs - 1) > 0.02:  # ajuste poco creíble -> frecuencia nominal
        b = 1 / fs
        a = float(np.median(t - b * s_fin))
    return a, b, fs_ef


def _muestras(canal, arr):
    """Matriz (paquetes x muestras) por columnas de valores y vector de contadores."""
    if canal == "ppg":
        bits = np.unpackbits(arr, axis=1)[:, :154].reshape(-1, 11, 14)
        v = (bits * _PESOS14).sum(axis=2).astype(float)
        v[v >= 8192] -= 16384
        # idle: 4+ muestras iguales a 8127 en el paquete -> relleno del sensor sin lectura
        idle = (v == PPG_IDLE)
        v[idle & (idle.sum(axis=1, keepdims=True) >= 4)] = np.nan
        return {"ppg": v}, (arr[:, 19] & 0x3F)
    if canal == "acc":
        x = arr[:, :18].view(np.int8).reshape(-1, 6, 3) / 64.0
        return {"x_g": x[:, :, 0], "y_g": x[:, :, 1], "z_g": x[:, :, 2]}, arr[:, 19]
    if canal == "temp":
        raw = np.ascontiguousarray(arr[:, :16]).view("<u2").reshape(-1, 8).astype(float)
        return {"temp_c": raw * 0.02 - 273.15}, arr[:, 19]
    if canal == "eda":
        f = np.ascontiguousarray(arr[:, :16]).view("<f4").reshape(-1, 4).astype(float)
        return {"eda_us": f.mean(axis=1, keepdims=True)}, arr[:, 18]
    raise ValueError(canal)


def _es_eda_b(arr):
    if len(arr) == 0:
        return np.zeros(0, bool)
    f = np.ascontiguousarray(arr[:, :16]).view("<f4").reshape(-1, 4)
    plausible = np.isfinite(f).all(axis=1) & (f > 0).all(axis=1) & (f < 100).all(axis=1)
    return (arr[:, 16] == 0) & (arr[:, 17] == 1) & plausible


def decodificar_e4(ruta):
    """Devuelve ({canal: DataFrame(t, valores...)}, {canal: dict QC}) para un e4*.jsonl."""
    paq = _leer_paquetes(ruta)
    series, qc = {}, {}
    por_canal = {"ppg": [], "acc": [], "temp": [], "gsr": []}
    for p in paq:
        if p[0] in por_canal:
            por_canal[p[0]].append(p)
    entradas = {c: _matriz(por_canal[c]) for c in ("ppg", "acc", "temp", "gsr")}
    arr_gsr, t_gsr = entradas["gsr"]
    es_b = _es_eda_b(arr_gsr)
    qc["gsr_A_paquetes_sin_decodificar"] = int((~es_b).sum())
    entradas["eda"] = (arr_gsr[es_b], t_gsr[es_b])
    for canal, cfg in CANALES.items():
        arr, t = entradas[canal]
        if len(arr) == 0:
            continue
        vals, cont = _muestras(canal, arr)
        ok = cont > 0  # contador 0 no es válido
        arr, t, cont = arr[ok], t[ok], cont[ok].astype(np.int64) - 1
        vals = {nombre: v[ok] for nombre, v in vals.items()}
        k = _desenrollar(cont, t, cfg["mod"], cfg["fs"] / cfg["n"])
        # eliminar paquetes repetidos (mismo índice)
        keep = np.r_[True, np.diff(k) > 0]
        arr, t, k = arr[keep], t[keep], k[keep]
        vals = {nombre: v[keep] for nombre, v in vals.items()}
        a, b, fs_ef = _tiempos(k, t, cfg["n"], cfg["fs"])
        n = cfg["n"]
        s = (k[:, None] * n + np.arange(n)[None, :] + 1)
        tt = a + b * s
        df = pd.DataFrame({"t": tt.ravel(), **{nom: v.reshape(-1) if v.shape[1] == n else np.repeat(v[:, 0], n) for nom, v in vals.items()}})
        df = df.dropna().reset_index(drop=True)
        perdidos = int(np.clip(np.diff(k) - 1, 0, None).sum())
        series[canal] = df
        qc[canal] = dict(
            paquetes=int(len(k)), perdidos=perdidos,
            pct_perdidos=round(100 * perdidos / (len(k) + perdidos), 2),
            fs_efectiva=None if np.isnan(fs_ef) else round(float(fs_ef), 3),
            fs_nominal=cfg["fs"], t_ini=float(df["t"].iloc[0]) if len(df) else None,
            t_fin=float(df["t"].iloc[-1]) if len(df) else None,
        )
    return series, qc


def listar_sesiones(raiz):
    """(persona, carpeta_sesion, ruta_e4) para cada sesión con archivo e4*.jsonl."""
    for pdir in sorted(Path(raiz).iterdir()):
        if not pdir.is_dir():
            continue
        for sdir in sorted(pdir.iterdir()):
            e4 = sorted(sdir.glob("e4*.jsonl"))
            if sdir.is_dir() and e4:
                yield pdir.name, sdir, e4[0]


def main():
    raiz = Path(__file__).parent / "DATAE4"
    salida = Path(__file__).parent / "salida" / "decodificado"
    salida.mkdir(parents=True, exist_ok=True)
    filas = []
    for persona, sdir, e4 in listar_sesiones(raiz):
        sid = sdir.name.replace("NoiseGeo_", "").split("[")[0]
        if e4.stat().st_size == 0:
            print(f"{persona:8} {sid}: e4 vacío")
            continue
        series, qc = decodificar_e4(e4)
        for canal, df in series.items():
            df.to_parquet(salida / f"{sid}__{canal}.parquet", index=False)
            filas.append({"persona": persona, "sesion": sid, "canal": canal, **qc[canal]})
        print(f"{persona:8} {sid}: " + ", ".join(f"{c}={len(d)}" for c, d in series.items()))
    pd.DataFrame(filas).to_csv(salida.parent / "qc_e4.csv", index=False)
    print("QC guardado en salida/qc_e4.csv")


if __name__ == "__main__":
    sys.exit(main())
