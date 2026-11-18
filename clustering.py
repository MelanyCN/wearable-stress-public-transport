#!/usr/bin/env python3
"""GOLD por reglas de contexto + parrilla de clustering sobre salida/ventanas.csv.

GOLD (independiente de la fisiología, definido solo con contexto del celular):
  trafico    fluido / moderado / congestionado   por frac_detenido (<0.1 / entre / >=0.5)
  ruido      bajo / medio / alto                 terciles de ruido_leq_dbfs POR PARTICIPANTE (cada celular tiene otro micrófono)
  exposicion baja / media / alta                 rango_trafico(0-2) + rango_ruido(0-2): 0-1 baja, 2 media, 3-4 alta

Análisis A: sin EDA, 4 participantes.  Análisis B: con EDA (tipo B), 3 participantes.
Evaluación: métricas internas, ARI/NMI contra el GOLD, estabilidad (bootstrap por bloques de sesión+tramo)
y validación dejando un participante fuera (LOPO) para KMeans y GMM.
Salidas en salida/: gold.csv, resultados_clustering.csv, mejores_configs.csv, perfil_clusters.csv,
gold_muestra_validacion.csv y figuras fig_*.png
"""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             normalized_mutual_info_score, silhouette_score)
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore")
SAL = Path(__file__).parent / "salida"
SEED = 0
FEATS = {
    "fis": ["hr_picos_bpm", "rmssd_ms", "sdnn_ms", "temp_media_c", "temp_pend_c_min"],
    "fis+mov": ["hr_picos_bpm", "rmssd_ms", "sdnn_ms", "temp_media_c", "temp_pend_c_min", "acc_mag_std_g"],
}
EDA = ["eda_media_us", "eda_pend_us_min"]
LOGS = {"rmssd_ms", "sdnn_ms", "acc_mag_std_g", "eda_media_us"}
GOLDS = ["trafico", "ruido", "exposicion"]


# ------------------------------------------------------------------ GOLD
def construir_gold(df):
    df = df.copy()
    f = df["frac_detenido"]
    df["gold_trafico"] = np.where(f.isna(), None, np.where(f < 0.1, "fluido", np.where(f >= 0.5, "congestionado", "moderado")))
    r_t = df["gold_trafico"].map({"fluido": 0, "moderado": 1, "congestionado": 2})
    r_r = df.groupby("participante")["ruido_leq_dbfs"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 3, labels=False) if s.notna().sum() >= 9 else np.nan)
    df["gold_ruido"] = r_r.map({0: "bajo", 1: "medio", 2: "alto"})
    s = r_t + r_r
    df["gold_exposicion"] = np.where(s.isna(), None, np.where(s <= 1, "baja", np.where(s == 2, "media", "alta")))
    for g in GOLDS:
        df[f"gold_{g}"] = df[f"gold_{g}"].where(df[f"gold_{g}"].notna(), None)
    return df


# ------------------------------------------------------------------ datos y preprocesamiento
def preparar(df, cols, norm):
    X = df[cols].copy()
    for c in cols:
        if c in LOGS:
            X[c] = np.log(X[c].clip(lower=1e-3))
    if norm == "por_participante":
        X = X.groupby(df["participante"]).transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))
    else:
        X = (X - X.mean()) / (X.std() + 1e-9)
    return X.clip(-5, 5).to_numpy()


def ajustar(algo, p, X):
    if algo == "kmeans":
        m = KMeans(p, n_init=10, random_state=SEED).fit(X); return m.labels_, m
    if algo == "gmm":
        m = GaussianMixture(p, covariance_type="full", n_init=3, random_state=SEED).fit(X); return m.predict(X), m
    if algo == "ward":
        return AgglomerativeClustering(p, linkage="ward").fit_predict(X), None
    if algo == "dbscan":
        eps, ms = p
        return DBSCAN(eps=eps, min_samples=ms).fit_predict(X), None


def metricas_internas(X, lab):
    ok = lab >= 0
    ncl = len(set(lab[ok]))
    out = dict(n_clusters=ncl, frac_ruido=round(1 - ok.mean(), 3), silhouette=np.nan, davies_bouldin=np.nan, calinski=np.nan)
    if ncl >= 2 and ok.sum() > ncl:
        out.update(silhouette=silhouette_score(X[ok], lab[ok]), davies_bouldin=davies_bouldin_score(X[ok], lab[ok]),
                   calinski=calinski_harabasz_score(X[ok], lab[ok]))
    return out


def vs_gold(lab, gold):
    m = gold.notna().to_numpy()
    if m.sum() < 30 or gold[m].nunique() < 2:
        return np.nan, np.nan, int(m.sum())
    return adjusted_rand_score(gold[m], lab[m]), normalized_mutual_info_score(gold[m], lab[m]), int(m.sum())


def estabilidad(algo, p, X, lab_full, bloques, reps=10):
    rng = np.random.default_rng(SEED)
    ub = np.unique(bloques)
    out = []
    for _ in range(reps):
        sel = rng.choice(ub, size=max(2, int(0.8 * len(ub))), replace=False)
        idx = np.isin(bloques, sel)
        lab, _ = ajustar(algo, p, X[idx])
        out.append(adjusted_rand_score(lab_full[idx], lab))
    return float(np.mean(out))


def lopo(algo, p, X, df, gold_col):
    nmis = []
    for part in df["participante"].unique():
        te = (df["participante"] == part).to_numpy()
        _, m = ajustar(algo, p, X[~te])
        lab = m.predict(X[te])
        g = df.loc[te, gold_col]
        _, nmi, n = vs_gold(lab, g.reset_index(drop=True))
        if not np.isnan(nmi):
            nmis.append(nmi)
    return float(np.mean(nmis)) if nmis else np.nan


# ------------------------------------------------------------------ parrilla
def correr(df, nombre, usar_eda):
    cols_ok = ["ppg_ok", "acc_ok", "temp_ok"] + (["eda_ok"] if usar_eda else [])
    d = df[df[cols_ok].all(axis=1) & df["rmssd_ms"].notna() & df["sdnn_ms"].notna()].reset_index(drop=True)
    bloques = (d["sesion"] + "_" + d["tramo_bus"].astype(str)).to_numpy()
    print(f"[{nombre}] ventanas: {len(d)}  por persona: {d.groupby('participante').size().to_dict()}")
    filas = []
    for fs_name, base in FEATS.items():
        cols = base + (EDA if usar_eda else [])
        for norm in ("por_participante", "global"):
            X = preparar(d, cols, norm)
            configs = [("kmeans", k) for k in range(2, 7)] + [("gmm", k) for k in range(2, 7)] + \
                      [("ward", k) for k in range(2, 7)] + [("dbscan", (e, s)) for e in (0.8, 1.0, 1.3, 1.6) for s in (10, 20)]
            for algo, p in configs:
                lab, _ = ajustar(algo, p, X)
                fila = dict(analisis=nombre, features=fs_name + ("+eda" if usar_eda else ""), norm=norm, algo=algo,
                            param=str(p), n=len(d), **metricas_internas(X, lab))
                for g in GOLDS:
                    ari, nmi, n = vs_gold(lab, d[f"gold_{g}"])
                    fila.update({f"ari_{g}": ari, f"nmi_{g}": nmi, f"n_{g}": n})
                if algo != "dbscan":
                    fila["estabilidad_ari"] = estabilidad(algo, p, X, lab, bloques)
                if algo in ("kmeans", "gmm"):
                    fila["lopo_nmi_trafico"] = lopo(algo, p, X, d, "gold_trafico")
                    fila["lopo_nmi_exposicion"] = lopo(algo, p, X, d, "gold_exposicion")
                filas.append(fila)
    return d, pd.DataFrame(filas)


def prueba_permutacion(d, lab, gold_col, reps=500):
    g = d[gold_col]
    m = g.notna().to_numpy()
    obs = normalized_mutual_info_score(g[m], lab[m])
    rng = np.random.default_rng(SEED)
    sub = d.loc[m].reset_index(drop=True)
    l = lab[m]
    mayores = 0
    for _ in range(reps):
        perm = sub.groupby("participante")[gold_col].transform(lambda s: rng.permutation(s.to_numpy()))
        mayores += normalized_mutual_info_score(perm, l) >= obs
    return obs, (mayores + 1) / (reps + 1)


def main():
    df = pd.read_csv(SAL / "ventanas.csv")
    df = construir_gold(df)
    df.to_csv(SAL / "gold.csv", index=False)
    print("GOLD (ventanas):")
    for g in GOLDS:
        print(f"  {g}: {df[f'gold_{g}'].value_counts().to_dict()}  (nulos {df[f'gold_{g}'].isna().sum()})")

    dA, rA = correr(df, "A_sin_EDA", False)
    dB, rB = correr(df, "B_con_EDA", True)
    res = pd.concat([rA, rB], ignore_index=True)
    res.round(4).to_csv(SAL / "resultados_clustering.csv", index=False)

    # mejores configuraciones: por estabilidad+NMI propio y por LOPO (más honesto)
    best = []
    for an, g in res.groupby("analisis"):
        g = g[g["algo"].isin(["kmeans", "gmm", "ward"])]
        best.append(g.sort_values("lopo_nmi_trafico", ascending=False).head(5).assign(criterio="LOPO NMI tráfico"))
        best.append(g.sort_values("nmi_exposicion", ascending=False).head(5).assign(criterio="NMI exposición (en muestra)"))
        best.append(g.sort_values("silhouette", ascending=False).head(5).assign(criterio="silhouette"))
    pd.concat(best).round(4).to_csv(SAL / "mejores_configs.csv", index=False)

    # prueba de permutación y perfil para la mejor config por LOPO tráfico en A y B
    perfiles = []
    for nombre, d, usar_eda in (("A_sin_EDA", dA, False), ("B_con_EDA", dB, True)):
        r = res[(res.analisis == nombre) & res.algo.isin(["kmeans", "gmm"])].sort_values("lopo_nmi_trafico", ascending=False).iloc[0]
        fs_base = "fis+mov" if "mov" in r["features"] else "fis"
        cols = FEATS[fs_base] + (EDA if usar_eda else [])
        X = preparar(d, cols, r["norm"])
        p = int(float(r["param"]))
        lab, _ = ajustar(r["algo"], p, X)
        d = d.assign(cluster=lab)
        for g in ("trafico", "exposicion"):
            obs, pv = prueba_permutacion(d, lab, f"gold_{g}")
            print(f"[{nombre}] mejor={r['algo']} k={p} feats={r['features']} norm={r['norm']} | NMI vs {g} = {obs:.3f}  p(permutación)={pv:.3f}")
        extra = [c for c in ["acc_mag_std_g", "ruido_leq_dbfs", "vel_media_kmh", "frac_detenido", "hora_local"] if c not in cols]
        perf = d.groupby("cluster")[cols + extra].mean().round(2)
        perf.insert(0, "n", d.groupby("cluster").size())
        for part in sorted(d["participante"].unique()):
            perf[f"%{part}"] = (d.assign(x=d["participante"] == part).groupby("cluster")["x"].mean() * 100).round(0)
        perf.insert(0, "analisis", nombre)
        perfiles.append(perf.reset_index())
        # figura PCA
        Z = PCA(2, random_state=SEED).fit_transform(X)
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
        for a, (col, tit) in zip(ax, [(d["cluster"], "cluster"), (d["participante"], "participante"), (d["gold_trafico"].fillna("sin dato"), "GOLD tráfico")]):
            for v in sorted(pd.Series(col).unique(), key=str):
                m = (pd.Series(col) == v).to_numpy()
                a.scatter(Z[m, 0], Z[m, 1], s=8, alpha=.6, label=str(v))
            a.set_title(f"{nombre}: {tit}"); a.legend(fontsize=7, markerscale=2)
        plt.tight_layout(); plt.savefig(SAL / f"fig_pca_{nombre}.png", dpi=120); plt.close()
    pd.concat(perfiles).to_csv(SAL / "perfil_clusters.csv", index=False)

    # figura: NMI contra GOLD por configuración
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for a, (an, g) in zip(ax, res[res.algo.isin(["kmeans", "gmm", "ward"])].groupby("analisis")):
        g = g.assign(cfg=g["algo"] + "-" + g["param"], col=g["features"] + "/" + g["norm"].str[:4])
        pv = g.pivot_table(index="cfg", columns="col", values="nmi_trafico")
        im = a.imshow(pv.values, aspect="auto", cmap="viridis"); a.set_xticks(range(pv.shape[1])); a.set_xticklabels(pv.columns, rotation=30, ha="right", fontsize=7)
        a.set_yticks(range(pv.shape[0])); a.set_yticklabels(pv.index, fontsize=7); a.set_title(f"{an}: NMI vs GOLD tráfico"); plt.colorbar(im, ax=a)
    plt.tight_layout(); plt.savefig(SAL / "fig_nmi_parrilla.png", dpi=120); plt.close()

    # muestra para validación manual (2 evaluadores), estratificada por participante y GOLD exposición
    base = df[df[["ppg_ok", "acc_ok", "temp_ok"]].all(axis=1)].copy()
    base["gold_estrato"] = base["gold_exposicion"].fillna("sin_ruido")
    muestra = base.groupby(["participante", "gold_estrato"], group_keys=False).apply(
        lambda g: g.sample(min(len(g), 10), random_state=SEED))
    cols_m = ["participante", "sesion", "ts_inicio", "lat", "lon", "hora_local", "ruido_leq_dbfs", "vel_media_kmh", "frac_detenido",
              "gold_trafico", "gold_ruido", "gold_exposicion"]
    muestra = muestra[cols_m].assign(etiqueta_rater1="", etiqueta_rater2="")
    muestra.round(3).to_csv(SAL / "gold_muestra_validacion.csv", index=False)
    print(f"muestra de validación manual: {len(muestra)} ventanas")


if __name__ == "__main__":
    main()
