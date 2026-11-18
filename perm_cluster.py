#!/usr/bin/env python3
"""NMI de los clusters vs. proxy de tráfico con permutación por bloques (desplazamiento circular dentro de sesión+tramo)."""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import normalized_mutual_info_score as nmi
import analisis_avanzado as A, clustering as C, analisis_revision as R

def main():
    d = A.residualizar(A.cargar()); dA = d.copy(); dB = d[d.eda_ok & (d.t_desde_eda >= 300)].copy()
    filas = []
    for an, dd, fs, cols in (("A", dA, "fis+mov", A.FEATSETS["fis+mov"]), ("A", dA, "fis", A.FEATSETS["fis"]), ("B", dB, "fis+eda", A.FEATSETS["fis"] + A.EDA_COLS), ("B", dB, "fis+mov+eda", A.FEATSETS["fis+mov"] + A.EDA_COLS)):
        dd = dd.sort_values(["sesion", "tramo_bus", "t_ini"]).reset_index(drop=True)
        X = A.matriz(dd, cols); lab, _ = C.ajustar("kmeans", 6, X)
        for tgt in ("gold_trafico", "gold_ruido"):
            m = dd[tgt].notna().to_numpy(); y = dd.loc[m, tgt].to_numpy(); l = lab[m]; bl = (dd.sesion + "_" + dd.tramo_bus.astype(str)).to_numpy()[m]
            obs = nmi(y, l); rng = np.random.default_rng(0)
            nulo = np.array([nmi(R.permutar_bloques(y, bl, rng), l) for _ in range(2000)])
            filas.append(dict(analisis=an, features=fs, k=6, objetivo=tgt, nmi=obs, nulo_media=nulo.mean(), nulo_p95=np.percentile(nulo, 95), p_bloques=(np.sum(nulo >= obs) + 1) / 2001))
    t = pd.DataFrame(filas); t.round(4).to_csv(R.OUT / "cluster_nmi_permutacion_bloques.csv", index=False); print(t.round(4).to_string(index=False))

if __name__ == "__main__":
    sys.exit(main())
