#!/usr/bin/env python3
"""Permutaciones por bloques con 1000 repeticiones para las 3 celdas 'fisiología + vibración' (familia exploratoria 2)."""
import sys, warnings, zlib; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import balanced_accuracy_score
import analisis_avanzado as A
import analisis_revision as R

B = 1000
def una(X, y, bl, part, seed):
    rng = np.random.default_rng(seed)
    return balanced_accuracy_score(y, R.lopo_pred(X, R.permutar_bloques(y, bl, rng), part))

def main():
    d = A.residualizar(A.cargar()); dA = d.copy(); dB = d[d.eda_ok & (d.t_desde_eda >= 300)].copy()
    celdas = [("A", dA, "fis+mov", A.FEATSETS["fis+mov"], "gold_trafico"), ("A", dA, "fis+mov", A.FEATSETS["fis+mov"], "gold_ruido"),
              ("B", dB, "fis+mov+eda", A.FEATSETS["fis+mov"] + A.EDA_COLS, "gold_trafico")]
    filas = []
    for an, dd, fs, cols, tgt in celdas:
        dd = dd[dd[tgt].notna()].sort_values(["sesion", "tramo_bus", "t_ini"]).reset_index(drop=True)
        X = A.matriz(dd, cols); y = dd[tgt].to_numpy(); bl = (dd.sesion + "_" + dd.tramo_bus.astype(str)).to_numpy(); part = dd.participante.to_numpy()
        obs = balanced_accuracy_score(y, R.lopo_pred(X, y, part))
        nulo = np.array(Parallel(n_jobs=12)(delayed(una)(X, y, bl, part, 1000 + i) for i in range(B)))
        p = (np.sum(nulo >= obs) + 1) / (B + 1)
        filas.append(dict(analisis=an, features=fs, objetivo=tgt, bal_acc=obs, nulo_media=nulo.mean(), nulo_p95=np.percentile(nulo, 95), p_bloques_1000=p))
        print(filas[-1], flush=True)
    t = pd.DataFrame(filas); t["p_holm_fam2"] = R.holm(t.p_bloques_1000.to_numpy()); t.round(4).to_csv(R.OUT / "familia_exploratoria_vibracion.csv", index=False); print(t.round(4).to_string(index=False))

if __name__ == "__main__":
    sys.exit(main())
