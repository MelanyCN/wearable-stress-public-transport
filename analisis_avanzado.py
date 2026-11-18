#!/usr/bin/env python3
"""Análisis avanzado según salida/PROTOCOLO_analisis_avanzado.md (fijado antes de correr).

 1. Modelos mixtos (cuánto afecta cada factor)      -> salida/avanzado/modelo_mixto.csv, fig_modelo_mixto.png
 2. Fisiología ajustada por movimiento              -> salida/avanzado/residuales_movimiento.csv
 3. EDA sin calentamiento del sensor (>=300 s)      -> sensibilidad_eda.csv
 4. Límite superior supervisado (RF/LogReg, LOPO/LOSO + permutación) -> clasificador.csv, permutaciones.csv, fig_clasificador.png
 +  Clustering v2 (misma parrilla, features ajustadas) -> clustering_v2.csv
"""
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import HuberRegressor, LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut

import clustering as C

warnings.filterwarnings("ignore")
SAL = Path(__file__).parent / "salida"
OUT = SAL / "avanzado"
OUT.mkdir(exist_ok=True)
SEED = 0
TARGETS = ["gold_trafico", "gold_ruido", "gold_exposicion"]


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ datos
def cargar():
    df = pd.read_csv(SAL / "gold.csv")
    qc = pd.read_csv(SAL / "qc_e4.csv")
    eda0 = qc[qc.canal == "eda"].set_index("sesion")["t_ini"]
    df["t_desde_eda"] = df["t_ini"] - df["sesion"].map(eda0)
    d = df[df[["ppg_ok", "acc_ok", "temp_ok"]].all(axis=1) & df["rmssd_ms"].notna() & df["sdnn_ms"].notna()].copy()
    d["lnrmssd"] = np.log(d["rmssd_ms"].clip(lower=1e-3))
    d["ln_acc"] = np.log(d["acc_mag_std_g"].clip(lower=1e-4))
    d["ln_accdif"] = np.log(d["acc_dif_media_g"].clip(lower=1e-4))
    d["ln_eda"] = np.log(d["eda_media_us"].clip(lower=1e-3))
    return d.reset_index(drop=True)


# ------------------------------------------------------------------ 2. residuales por movimiento
def residualizar(d):
    filas = []
    for y, nuevo in (("hr_picos_bpm", "hr_res"), ("lnrmssd", "lnrmssd_res")):
        d[nuevo] = np.nan
        for p, g in d.groupby("participante"):
            Z = np.column_stack([g["ln_acc"], g["ln_accdif"]])
            Z = (Z - Z.mean(0)) / Z.std(0)
            m = HuberRegressor().fit(Z, g[y])
            res = g[y] - m.predict(Z)
            d.loc[g.index, nuevo] = res
            filas.append(dict(variable=y, participante=p, n=len(g), r2_movimiento=round(1 - res.var() / g[y].var(), 3),
                              coef_ln_acc_std=round(m.coef_[0], 3), coef_ln_acc_dif=round(m.coef_[1], 3)))
    pd.DataFrame(filas).to_csv(OUT / "residuales_movimiento.csv", index=False)
    log("2) Parte de la varianza de FC / ln RMSSD explicada por movimiento (por participante):")
    log(pd.DataFrame(filas).pivot(index="participante", columns="variable", values="r2_movimiento").to_string())
    return d


# ------------------------------------------------------------------ 1. modelos mixtos
def preparar_mixto(d):
    m = d[d["ruido_leq_dbfs"].notna() & d["frac_detenido"].notna()].copy()
    m = m.sort_values(["sesion", "tramo_bus", "t_ini"])
    m = m[m.groupby(["sesion", "tramo_bus"]).cumcount() % 2 == 0].copy()  # ventanas sin traslape
    for c, n in (("ruido_leq_dbfs", "ruido_z"), ("frac_detenido", "cong_z"), ("ln_acc", "acc_z")):
        m[n] = m.groupby("participante")[c].transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))
    m["hora_z"] = (m["hora_local"] - m["hora_local"].mean()) / m["hora_local"].std()
    return m


def ajustar_mixto(m, y, etiqueta):
    mod = smf.mixedlm(f"{y} ~ ruido_z + cong_z + acc_z + hora_z", m, groups=m["sesion"]).fit(reml=True)
    ci = mod.conf_int()
    filas = []
    for t in ["ruido_z", "cong_z", "acc_z", "hora_z"]:
        filas.append(dict(modelo=etiqueta, resultado=y, termino=t, coef=mod.params[t], ic_inf=ci.loc[t, 0], ic_sup=ci.loc[t, 1],
                          p=mod.pvalues[t], n=int(mod.nobs), sesiones=m["sesion"].nunique(), sd_resultado=m[y].std(),
                          icc_sesion=float(mod.cov_re.iloc[0, 0] / (mod.cov_re.iloc[0, 0] + mod.scale))))
    return filas


def modelos_mixtos(d):
    filas = []
    m = preparar_mixto(d)
    log(f"1) Modelos mixtos: A n={len(m)} (sin traslape), sesiones={m['sesion'].nunique()}")
    for y in ("hr_picos_bpm", "lnrmssd", "temp_media_c", "hr_res"):
        try:
            filas += ajustar_mixto(m, y, "A")
        except Exception as e:
            log("   fallo", y, e)
    # EDA: sensibilidad al calentamiento (0 / 300 / 600 s)
    sens = []
    for corte in (0, 300, 600):
        mb = d[d["eda_ok"] & (d["t_desde_eda"] >= corte)]
        mb = preparar_mixto(mb)
        try:
            f = ajustar_mixto(mb, "ln_eda", f"B_eda_corte{corte}s")
            filas += f if corte == 300 else []
            sens += f
        except Exception as e:
            log("   fallo EDA", corte, e)
    pd.DataFrame(sens).round(4).to_csv(OUT / "sensibilidad_eda.csv", index=False)
    res = pd.DataFrame(filas)
    res.round(4).to_csv(OUT / "modelo_mixto.csv", index=False)
    log(res[["modelo", "resultado", "termino", "coef", "ic_inf", "ic_sup", "p", "n"]].round(3).to_string(index=False))
    # figura
    fig, ax = plt.subplots(1, len(res["resultado"].unique()), figsize=(18, 3.6), sharey=True)
    for a, (y, g) in zip(ax, res.groupby("resultado", sort=False)):
        yy = np.arange(len(g))
        a.errorbar(g["coef"], yy, xerr=[g["coef"] - g["ic_inf"], g["ic_sup"] - g["coef"]], fmt="o", capsize=3)
        a.axvline(0, color="gray", lw=.8); a.set_yticks(yy); a.set_yticklabels(g["termino"]); a.set_title(y, fontsize=9)
    plt.tight_layout(); plt.savefig(OUT / "fig_modelo_mixto.png", dpi=120); plt.close()
    return res


# ------------------------------------------------------------------ 4. clasificador
FEATSETS = {
    "fis": ["hr_picos_bpm", "lnrmssd", "sdnn_ms", "temp_media_c", "temp_pend_c_min"],
    "fis+mov": ["hr_picos_bpm", "lnrmssd", "sdnn_ms", "temp_media_c", "temp_pend_c_min", "ln_acc"],
    "res": ["hr_res", "lnrmssd_res", "temp_media_c", "temp_pend_c_min"],
}
EDA_COLS = ["ln_eda", "eda_pend_us_min"]


def matriz(d, cols):
    X = d[cols].copy()
    X = X.groupby(d["participante"]).transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))
    return X.clip(-5, 5).to_numpy()


def cv_clasificador(X, y, grupos, modelo, trees=300):
    y = np.asarray(y)
    pred = np.empty(len(y), dtype=object)
    pf = []
    for tr, te in LeaveOneGroupOut().split(X, y, grupos):
        if modelo == "rf":
            clf = RandomForestClassifier(trees, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=SEED)
        else:
            clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(X[tr], y[tr]); pred[te] = clf.predict(X[te])
        if len(set(y[te])) >= 2:
            pf.append(balanced_accuracy_score(y[te], pred[te]))
    return balanced_accuracy_score(y, pred), f1_score(y, pred, average="macro"), float(np.mean(pf))


def clasificadores(dA, dB):
    filas = []
    for an, d, eda in (("A_sin_EDA", dA, False), ("B_con_EDA", dB, True)):
        d = d.reset_index(drop=True)
        for fs, base in FEATSETS.items():
            cols = base + (EDA_COLS if eda else [])
            X = matriz(d, cols)
            for tgt in TARGETS:
                m = d[tgt].notna().to_numpy()
                if d.loc[m, tgt].nunique() < 2:
                    continue
                for esq, gcol in (("LOPO", "participante"), ("LOSO", "sesion")):
                    for modelo in ("rf", "logreg"):
                        ba, f1, bam = cv_clasificador(X[m], d.loc[m, tgt], d.loc[m, gcol], modelo)
                        filas.append(dict(analisis=an, features=fs + ("+eda" if eda else ""), objetivo=tgt, validacion=esq,
                                          modelo=modelo, n=int(m.sum()), bal_acc=ba, f1_macro=f1, bal_acc_media_folds=bam))
            log(f"4) {an} {fs} listo")
            pd.DataFrame(filas).round(4).to_csv(OUT / "clasificador.csv", index=False)
    return pd.DataFrame(filas)


def permutaciones(dA, dB, reps=100):
    rng = np.random.default_rng(SEED)
    filas = []
    for an, d, eda, fs in (("A_sin_EDA", dA, False, "fis"), ("A_sin_EDA", dA, False, "fis+mov"), ("B_con_EDA", dB, True, "fis+mov")):
        d = d.reset_index(drop=True)
        cols = FEATSETS[fs] + (EDA_COLS if eda else [])
        X = matriz(d, cols)
        m = d["gold_trafico"].notna().to_numpy()
        Xm, dm = X[m], d.loc[m].reset_index(drop=True)
        obs = cv_clasificador(Xm, dm["gold_trafico"], dm["participante"], "rf", 100)[0]
        nulos = []
        for _ in range(reps):
            yp = dm.groupby("participante")["gold_trafico"].transform(lambda s: rng.permutation(s.to_numpy()))
            nulos.append(cv_clasificador(Xm, yp, dm["participante"], "rf", 100)[0])
        nulos = np.array(nulos)
        filas.append(dict(analisis=an, features=fs + ("+eda" if eda else ""), obs_bal_acc=obs, nulo_media=nulos.mean(),
                          nulo_p95=np.percentile(nulos, 95), p=(np.sum(nulos >= obs) + 1) / (reps + 1)))
        log(f"   permutación {an} {fs}: obs={obs:.3f} nulo p95={np.percentile(nulos, 95):.3f}")
        pd.DataFrame(filas).round(4).to_csv(OUT / "permutaciones.csv", index=False)
    return pd.DataFrame(filas)


# ------------------------------------------------------------------ clustering v2
def clustering_v2(dA, dB):
    filas = []
    sets = {"fis": FEATSETS["fis"], "fis+mov": FEATSETS["fis+mov"], "res": FEATSETS["res"]}
    for an, d, eda in (("A_sin_EDA", dA, False), ("B_con_EDA(>=300s)", dB, True)):
        d = d.reset_index(drop=True)
        bloques = (d["sesion"] + "_" + d["tramo_bus"].astype(str)).to_numpy()
        for fs, base in sets.items():
            cols = base + (EDA_COLS if eda else [])
            X = matriz(d, cols)
            for algo in ("kmeans", "gmm", "ward"):
                for k in range(2, 7):
                    lab, _ = C.ajustar(algo, k, X)
                    f = dict(analisis=an, features=fs + ("+eda" if eda else ""), algo=algo, k=k, n=len(d), **C.metricas_internas(X, lab))
                    for g in ("trafico", "ruido", "exposicion"):
                        ari, nmi, n = C.vs_gold(lab, d[f"gold_{g}"])
                        f[f"nmi_{g}"] = nmi
                    f["estabilidad_ari"] = C.estabilidad(algo, k, X, lab, bloques)
                    if algo != "ward":
                        f["lopo_nmi_trafico"] = C.lopo(algo, k, X, d, "gold_trafico")
                    filas.append(f)
        log(f"clustering v2 {an} listo")
    res = pd.DataFrame(filas)
    res.round(4).to_csv(OUT / "clustering_v2.csv", index=False)
    log("\nResumen clustering v2 (mediana y máx de NMI tráfico; máx LOPO):")
    log(res.groupby(["analisis", "features"]).agg(nmi_trafico_med=("nmi_trafico", "median"), nmi_trafico_max=("nmi_trafico", "max"),
                                                  lopo_max=("lopo_nmi_trafico", "max"), estab_med=("estabilidad_ari", "median"),
                                                  sil_med=("silhouette", "median")).round(3).to_string())
    # mejor config por criterio fijo (LOPO NMI tráfico) + permutación
    for an, d, eda in (("A_sin_EDA", dA, False), ("B_con_EDA(>=300s)", dB, True)):
        r = res[(res.analisis == an) & res.algo.isin(["kmeans", "gmm"])].sort_values("lopo_nmi_trafico", ascending=False).iloc[0]
        d = d.reset_index(drop=True)
        base = FEATSETS[r["features"].replace("+eda", "")]
        X = matriz(d, base + (EDA_COLS if eda else []))
        lab, _ = C.ajustar(r["algo"], int(r["k"]), X)
        obs, pv = C.prueba_permutacion(d, lab, "gold_trafico")
        log(f"[{an}] mejor por LOPO: {r['algo']} k={int(r['k'])} {r['features']} | NMI tráfico={obs:.3f} p={pv:.3f} | LOPO NMI={r['lopo_nmi_trafico']:.3f}")
    return res


def main():
    d = cargar()
    d = residualizar(d)
    dA = d.copy()
    dB = d[d["eda_ok"] & (d["t_desde_eda"] >= 300)].copy()
    log(f"A: {len(dA)} ventanas {dA.groupby('participante').size().to_dict()} | B(>=300 s): {len(dB)} {dB.groupby('participante').size().to_dict()}")
    modelos_mixtos(d)
    clustering_v2(dA, dB)
    cl = clasificadores(dA, dB)
    pm = permutaciones(dA, dB)
    # figura clasificador
    g = cl[(cl.modelo == "rf") & (cl.validacion == "LOPO")]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for a, (an, gg) in zip(ax, g.groupby("analisis")):
        pv = gg.pivot_table(index="objetivo", columns="features", values="bal_acc")
        pv.plot.bar(ax=a, rot=0); a.axhline(1 / 3, color="k", ls="--", lw=.8, label="azar"); a.set_title(f"{an}: RF, dejando un participante fuera"); a.set_ylabel("exactitud balanceada")
    plt.tight_layout(); plt.savefig(OUT / "fig_clasificador.png", dpi=120); plt.close()
    log("\n4) Límite superior (RF, LOPO):"); log(g.pivot_table(index=["analisis", "features"], columns="objetivo", values="bal_acc").round(3).to_string())
    log("\n   LOSO (RF):"); log(cl[(cl.modelo == "rf") & (cl.validacion == "LOSO")].pivot_table(index=["analisis", "features"], columns="objetivo", values="bal_acc").round(3).to_string())
    log("\n   Permutación (tráfico):"); log(pm.round(3).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
