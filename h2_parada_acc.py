#!/usr/bin/env python3
"""H2: ¿puede el acelerómetro de la pulsera detectar si el bus está detenido o en marcha? (verdad = velocidad GPS)
Ver salida/novedoso/PROTOCOLO_novedoso.md"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut

warnings.filterwarnings("ignore")
SAL = Path(__file__).parent / "salida"
OUT = SAL / "novedoso"
BANDAS = [(0.3, 1), (1, 3), (3, 6), (6, 10), (10, 16)]
SEED = 0


def features_acc(t, xyz, ws, we):
    i0, i1 = np.searchsorted(t, [ws, we])
    if i1 - i0 < 32 * 30 * 0.8:
        return None
    a = xyz[i0:i1]
    mag = np.sqrt((a ** 2).sum(axis=1))
    f, P = welch(mag - mag.mean(), fs=32, nperseg=128)
    out = {f"banda_{lo}_{hi}": np.log(P[(f >= lo) & (f < hi)].sum() + 1e-9) for lo, hi in BANDAS}
    for j, ax in enumerate("xyz"):
        out[f"std_{ax}"] = np.log(a[:, j].std() + 1e-5)
    out["ln_std_mag"] = np.log(mag.std() + 1e-5)
    return out


def construir():
    df = pd.read_csv(SAL / "gold.csv")
    df = df[df["acc_ok"] & df["frac_detenido"].notna()].copy()
    filas = []
    for sid, g in df.groupby("sesion"):
        acc = pd.read_parquet(SAL / "decodificado" / f"{sid}__acc.parquet")
        t, xyz = acc["t"].to_numpy(), acc[["x_g", "y_g", "z_g"]].to_numpy()
        for _, r in g.iterrows():
            f = features_acc(t, xyz, r["t_ini"], r["t_ini"] + 30)
            if f:
                filas.append(dict(participante=r["participante"], sesion=sid, t_ini=r["t_ini"], frac_detenido=r["frac_detenido"],
                                  vel=r["vel_media_kmh"], gps_acc=r["gps_acc_med_m"], **f))
    d = pd.DataFrame(filas)
    cols = [c for c in d.columns if c.startswith(("banda_", "std_", "ln_std"))]
    d[cols] = d.groupby("participante")[cols].transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))
    return d, cols


def cv_auc(X, y, grupos, modelo):
    y = np.asarray(y)
    p = np.zeros(len(y))
    for tr, te in LeaveOneGroupOut().split(X, y, grupos):
        clf = RandomForestClassifier(300, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=SEED) if modelo == "rf" \
            else LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(X[tr], y[tr]); p[te] = clf.predict_proba(X[te])[:, 1]
    por = {}
    for g in np.unique(grupos):
        m = grupos == g
        if len(set(y[m])) == 2:
            por[g] = roc_auc_score(y[m], p[m])
    return roc_auc_score(y, p), por, p


def main():
    d, cols = construir()
    d = d[(d.frac_detenido >= 0.5) | (d.frac_detenido < 0.1)].reset_index(drop=True)
    d["detenido"] = (d.frac_detenido >= 0.5).astype(int)
    print(f"ventanas: {len(d)}  detenido={d.detenido.sum()}  en marcha={(1 - d.detenido).sum()}  por persona: {d.groupby('participante').size().to_dict()}")
    X = d[cols].to_numpy()
    filas = []
    for esq, gcol in (("LOPO", "participante"), ("LOSO", "sesion")):
        for modelo in ("logreg", "rf"):
            auc, por, p = cv_auc(X, d.detenido, d[gcol].to_numpy(), modelo)
            filas.append(dict(validacion=esq, modelo=modelo, auc=auc, **{f"auc_{k}": v for k, v in por.items()}))
            print(f"{esq:5} {modelo:6} AUC={auc:.3f}  por grupo: " + ", ".join(f"{k}={v:.2f}" for k, v in por.items()))
    # línea base trivial: solo ln_std_mag
    auc0, _, _ = cv_auc(d[["ln_std_mag"]].to_numpy(), d.detenido, d["participante"].to_numpy(), "logreg")
    print(f"Solo desvío de la magnitud (LOPO): AUC={auc0:.3f}")
    # permutación (logreg, LOPO)
    rng = np.random.default_rng(SEED)
    obs = cv_auc(X, d.detenido, d["participante"].to_numpy(), "logreg")[0]
    nulos = []
    for _ in range(200):
        yp = d.groupby("participante")["detenido"].transform(lambda s: rng.permutation(s.to_numpy())).to_numpy()
        nulos.append(cv_auc(X, yp, d["participante"].to_numpy(), "logreg")[0])
    nulos = np.array(nulos)
    print(f"Permutación (logreg LOPO): obs={obs:.3f}  nulo media={nulos.mean():.3f} p95={np.percentile(nulos, 95):.3f}  p={(np.sum(nulos >= obs) + 1) / 201:.4f}")
    # sensibilidad: solo ventanas con GPS bueno
    b = d[d.gps_acc <= 10].reset_index(drop=True)
    if b.detenido.nunique() == 2:
        a2 = cv_auc(b[cols].to_numpy(), b.detenido, b["participante"].to_numpy(), "logreg")[0]
        print(f"Sensibilidad (GPS <= 10 m, n={len(b)}): AUC LOPO logreg={a2:.3f}")
    # importancia de bandas (RF)
    rf = RandomForestClassifier(300, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=SEED).fit(X, d.detenido)
    print("Importancia (RF):", dict(sorted(zip(cols, rf.feature_importances_.round(3)), key=lambda x: -x[1])))
    pd.DataFrame(filas).round(4).to_csv(OUT / "h2_resultados.csv", index=False)
    d.to_csv(OUT / "h2_ventanas.csv", index=False)


if __name__ == "__main__":
    main()
