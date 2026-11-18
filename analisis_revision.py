#!/usr/bin/env python3
"""Análisis adicionales pedidos en la revisión del artículo. Salidas en salida/revision/.

Cambios metodológicos respecto a versiones previas:
 * Las pruebas de permutación desplazan circularmente las etiquetas DENTRO de cada bloque temporal (sesión+tramo de bus),
   preservando su autocorrelación (las ventanas se traslapan 50 %); antes se mezclaban ventana a ventana.
 * Random forest con 100 árboles (n_jobs=1 por celda; las celdas se paralelizan) para que el valor observado y el nulo usen el mismo modelo.
 * Corrección de Holm sobre la familia confirmatoria (H1, H2 y las 6 celdas "solo fisiología").
Semilla fija (0).
"""
import sys
import warnings
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut

import analisis_avanzado as A

warnings.filterwarnings("ignore")
SAL = Path(__file__).parent / "salida"
OUT = SAL / "revision"
OUT.mkdir(exist_ok=True)
SEED, B_CLF, B_H2, B_BOOT = 0, 200, 2000, 2000
TARGETS = ["gold_trafico", "gold_ruido", "gold_exposicion"]


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ utilidades
def permutar_bloques(y, bloques, rng):
    yp = y.copy()
    for b in np.unique(bloques):
        ix = np.where(bloques == b)[0]
        if len(ix) > 1:
            yp[ix] = np.roll(y[ix], rng.integers(1, len(ix)))
    return yp


def lopo_pred(X, y, grupo):
    pred = np.empty(len(y), dtype=object)
    for tr, te in LeaveOneGroupOut().split(X, y, grupo):
        clf = RandomForestClassifier(100, min_samples_leaf=5, class_weight="balanced", n_jobs=1, random_state=SEED).fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])
    return pred


def boot_sesiones(y, pred, ses, fn, reps=B_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    u = np.unique(ses)
    idx = {s: np.where(ses == s)[0] for s in u}
    vals = []
    for _ in range(reps):
        sel = rng.choice(u, len(u), replace=True)
        ii = np.concatenate([idx[s] for s in sel])
        try:
            vals.append(fn(y[ii], pred[ii]))
        except Exception:
            pass
    return np.percentile(vals, [2.5, 97.5])


# ------------------------------------------------------------------ celdas del clasificador
def correr_celda(an, fs, cols, tgt, d):
    d = d[d[tgt].notna()].sort_values(["sesion", "tramo_bus", "t_ini"]).reset_index(drop=True)
    X = A.matriz(d, cols)
    y = d[tgt].to_numpy()
    bl = (d["sesion"] + "_" + d["tramo_bus"].astype(str)).to_numpy()
    part, ses = d["participante"].to_numpy(), d["sesion"].to_numpy()
    pred = lopo_pred(X, y, part)
    ba = balanced_accuracy_score(y, pred)
    lo, hi = boot_sesiones(y, pred, ses, balanced_accuracy_score)
    rng = np.random.default_rng(SEED + zlib.crc32(f"{an}|{fs}|{tgt}".encode()) % 100000)
    nulo = np.array([balanced_accuracy_score(y, lopo_pred(X, permutar_bloques(y, bl, rng), part)) for _ in range(B_CLF)])
    clases = sorted(set(y))
    cm = pd.DataFrame(confusion_matrix(y, pred, labels=clases, normalize="true"), index=clases, columns=clases).round(3)
    cm.insert(0, "real", cm.index)
    cm["analisis"], cm["features"], cm["objetivo"] = an, fs, tgt
    porp = pd.DataFrame([dict(analisis=an, features=fs, objetivo=tgt, participante=p, n=int((part == p).sum()),
                              bal_acc=balanced_accuracy_score(y[part == p], pred[part == p])) for p in sorted(set(part)) if len(set(y[part == p])) >= 2])
    pors = pd.DataFrame([dict(analisis=an, features=fs, objetivo=tgt, sesion=s, n=int((ses == s).sum()),
                              bal_acc=balanced_accuracy_score(y[ses == s], pred[ses == s])) for s in sorted(set(ses)) if len(set(y[ses == s])) >= 2])
    fila = dict(analisis=an, features=fs, objetivo=tgt, n=len(y), bal_acc=ba, ic_inf=lo, ic_sup=hi, f1_macro=f1_score(y, pred, average="macro"),
                nulo_media=nulo.mean(), nulo_p95=np.percentile(nulo, 95), p_bloques=(np.sum(nulo >= ba) + 1) / (B_CLF + 1))
    return fila, cm, porp, pors


def clasificador(dA, dB):
    celdas = []
    for an, d, eda in (("A", dA, False), ("B", dB, True)):
        d = d.reset_index(drop=True)
        for fs, base in A.FEATSETS.items():
            cols = base + (A.EDA_COLS if eda else [])
            for tgt in TARGETS:
                if d[tgt].notna().sum() > 30:
                    celdas.append((an, fs + ("+eda" if eda else ""), cols, tgt, d))
    log(f"clasificador: {len(celdas)} celdas x {B_CLF} permutaciones por bloques (en paralelo)")
    res = Parallel(n_jobs=12)(delayed(correr_celda)(*c) for c in celdas)
    tab = pd.DataFrame([r[0] for r in res])
    pd.concat([r[1] for r in res]).to_csv(OUT / "matrices_confusion.csv", index=False)
    pd.concat([r[2] for r in res]).round(4).to_csv(OUT / "bal_acc_por_participante.csv", index=False)
    pd.concat([r[3] for r in res]).round(4).to_csv(OUT / "bal_acc_por_sesion.csv", index=False)
    return tab


# ------------------------------------------------------------------ H2
def h2():
    h = pd.read_csv(SAL / "novedoso" / "h2_ventanas.csv").sort_values(["sesion", "t_ini"]).reset_index(drop=True)
    cols = [c for c in h.columns if c.startswith(("banda_", "std_", "ln_std"))]
    y = h["detenido"].to_numpy()
    part, ses = h["participante"].to_numpy(), h["sesion"].to_numpy()

    def lopo_auc(X, y_, grupo, ret=False):
        p = np.zeros(len(y_))
        for tr, te in LeaveOneGroupOut().split(X, y_, grupo):
            p[te] = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X[tr], y_[tr]).predict_proba(X[te])[:, 1]
        return (roc_auc_score(y_, p), p) if ret else roc_auc_score(y_, p)

    X = h[cols].to_numpy()
    obs, p = lopo_auc(X, y, part, True)
    rng = np.random.default_rng(SEED)
    nulo = np.array([lopo_auc(X, permutar_bloques(y, ses, rng), part) for _ in range(B_H2)])
    p_bl = (np.sum(nulo >= obs) + 1) / (B_H2 + 1)
    lo, hi = boot_sesiones(y, p, ses, roc_auc_score)
    log(f"H2: AUC LOPO logreg={obs:.3f} IC95 sesiones [{lo:.3f},{hi:.3f}]; nulo por bloques media={nulo.mean():.3f} p95={np.percentile(nulo, 95):.3f}; p={p_bl:.4f}")
    # ablación
    grupos = {"bandas 0.3-1 Hz": ["banda_0.3_1"], "1-3 Hz": ["banda_1_3"], "3-6 Hz": ["banda_3_6"], "6-10 Hz": ["banda_6_10"], "10-16 Hz": ["banda_10_16"],
              "desvío por eje (x,y,z)": ["std_x", "std_y", "std_z"], "desvío de la magnitud": ["ln_std_mag"]}
    fil = []
    for g, cs in grupos.items():
        fil.append(dict(grupo=g, auc_solo=lopo_auc(h[cs].to_numpy(), y, part), auc_sin=lopo_auc(h[[c for c in cols if c not in cs]].to_numpy(), y, part)))
    ab = pd.DataFrame(fil); ab.loc[len(ab)] = ["todas las features", obs, np.nan]
    ab.round(3).to_csv(OUT / "h2_ablacion.csv", index=False); log(ab.round(3).to_string(index=False))
    # errores
    h["prob"] = p; h["pred"] = (p >= 0.5).astype(int); h["error"] = (h["pred"] != y).astype(int)
    log("H2 errores: tasa global %.3f | FN (detenido→marcha) %.3f | FP (marcha→detenido) %.3f" % (h.error.mean(), h[h.detenido == 1].error.mean(), h[h.detenido == 0].error.mean()))
    h["bin_det"] = pd.cut(h["frac_detenido"], [-.01, .1, .5, .7, .9, 1.01], labels=["<0.1", "0.1-0.5", "0.5-0.7", "0.7-0.9", ">0.9"])
    h["bin_gps"] = pd.cut(h["gps_acc"], [0, 5, 10, 1000], labels=["<=5 m", "5-10 m", ">10 m"])
    e1 = h.groupby("bin_det", observed=True).agg(n=("error", "size"), tasa_error=("error", "mean"))
    e2 = h.groupby("bin_gps", observed=True).agg(n=("error", "size"), tasa_error=("error", "mean"))
    e3 = h.groupby("sesion").agg(n=("error", "size"), tasa_error=("error", "mean"), auc=("prob", lambda s: roc_auc_score(h.loc[s.index, "detenido"], s) if h.loc[s.index, "detenido"].nunique() == 2 else np.nan)).sort_values("tasa_error", ascending=False)
    e1.round(3).to_csv(OUT / "h2_error_por_frac_detenido.csv"); e2.round(3).to_csv(OUT / "h2_error_por_gps.csv"); e3.round(3).to_csv(OUT / "h2_error_por_sesion.csv")
    log("error por frac_detenido:\n", e1.round(3).to_string(), "\nerror por precisión GPS:\n", e2.round(3).to_string(), "\nsesiones con más error:\n", e3.head(4).round(3).to_string())
    cm = confusion_matrix(y, h["pred"]); pd.DataFrame(cm, index=["real en marcha", "real detenido"], columns=["pred en marcha", "pred detenido"]).to_csv(OUT / "h2_matriz_confusion.csv")
    log("H2 matriz de confusión (umbral 0.5):\n", cm)
    # sensibilidad: solo sesiones con etiqueta 'bus' tal como se registró (sin relabelar)
    verificadas = h["sesion"].isin(["P02_2026-09-26_08-35-05", "P02_2026-09-26_09-27-48", "P02_2026-09-30_07-55-50"]) | h["participante"].isin(["P03", "P09"])
    hv = h[verificadas].reset_index(drop=True)
    av = lopo_auc(hv[cols].to_numpy(), hv["detenido"].to_numpy(), hv["participante"].to_numpy())
    log(f"H2 sensibilidad (solo ventanas con etiqueta 'bus' original, n={len(hv)}, participantes {sorted(hv.participante.unique())}): AUC LOPO={av:.3f}")
    # por pulsera
    pul = h["participante"].map({"P01": "A", "P09": "A", "P02": "B", "P03": "C"})
    por = {k: roc_auc_score(y[pul == k], p[pul == k]) for k in "ABC"}
    log("H2 AUC por pulsera (predicciones LOPO):", {k: round(v, 3) for k, v in por.items()})
    return dict(auc=obs, ic=(lo, hi), p_bloques=p_bl, auc_verificadas=av, auc_pulsera=por, tasa_error=h.error.mean())


# ------------------------------------------------------------------ otros
def calidad_pulso(g):
    m = g[g.frac_detenido.notna() & g.acc_ok].copy()
    m["det"] = np.where(m.frac_detenido >= 0.5, 1, np.where(m.frac_detenido < 0.1, 0, np.nan))
    m = m.dropna(subset=["det"])
    rng = np.random.default_rng(SEED)

    def dif(df, excl=()):
        v = []
        for p, x in df.groupby("participante"):
            if p in excl or x.det.nunique() < 2:
                continue
            v.append(x[x.det == 1].ppg_ok.mean() - x[x.det == 0].ppg_ok.mean())
        return np.mean(v)

    out = {}
    for nombre, excl in (("todos", ()), ("sin P09", ("P09",))):
        obs = dif(m, excl)
        bs = []
        for _ in range(B_BOOT):
            partes = []
            for p, x in m.groupby("participante"):
                ss = x.sesion.unique(); sel = rng.choice(ss, len(ss), replace=True)
                partes.append(pd.concat([x[x.sesion == s] for s in sel]))
            bs.append(dif(pd.concat(partes), excl))
        out[nombre] = (obs, *np.percentile(bs, [2.5, 97.5]))
        log(f"calidad PPG detenido-marcha ({nombre}): {obs*100:+.1f} pp, IC95 sesiones [{np.percentile(bs, 2.5)*100:+.1f}, {np.percentile(bs, 97.5)*100:+.1f}]")
    return out


def bayes_poder():
    filas = []
    for nombre, est, lo, hi in (("H1 primaria", 1.71, -0.66, 3.56), ("H1 sens. 4 dB", -1.05, -5.42, 2.23), ("H1 sens. 3 dB", -1.87, -4.55, 0.86)):
        se = (hi - lo) / (2 * 1.96)
        for tau in (1.0, 2.0, 4.0):
            bf10 = norm.pdf(est, 0, np.sqrt(se ** 2 + tau ** 2)) / norm.pdf(est, 0, se)
            post_var = 1 / (1 / se ** 2 + 1 / tau ** 2); post_m = post_var * est / se ** 2
            filas.append(dict(analisis=nombre, estimado=est, se=round(se, 3), mde_80=round(2.8 * se, 2), prior_sd=tau, BF01=round(1 / bf10, 2),
                              post_media=round(post_m, 2), post_ic_inf=round(post_m - 1.96 * np.sqrt(post_var), 2), post_ic_sup=round(post_m + 1.96 * np.sqrt(post_var), 2)))
    t = pd.DataFrame(filas); t.to_csv(OUT / "h1_bayes_poder.csv", index=False); log(t.to_string(index=False))
    mm = pd.read_csv(SAL / "avanzado" / "modelo_mixto.csv")
    mm["se"] = (mm.ic_sup - mm.ic_inf) / (2 * 1.96); mm["mde_80"] = 2.8 * mm.se
    mm[["modelo", "resultado", "termino", "coef", "se", "mde_80", "icc_sesion", "n", "sesiones"]].round(3).to_csv(OUT / "mixto_icc_mde.csv", index=False)
    log(mm[mm.termino.isin(["ruido_z", "cong_z"])][["resultado", "termino", "coef", "mde_80", "icc_sesion"]].round(3).to_string(index=False))


def correlaciones(dA):
    cols = ["hr_picos_bpm", "lnrmssd", "sdnn_ms", "temp_media_c", "temp_pend_c_min", "ln_acc"]
    nombres = ["HR", "lnRMSSD", "SDNN", "T mean", "T slope", "ln ACC sd"]
    c = dA[cols].corr(method="spearman"); c.index = c.columns = nombres; c.round(2).to_csv(OUT / "correlacion_features.csv")
    m = c.where(~np.eye(len(c), dtype=bool)).abs().stack().sort_values(ascending=False)
    log("correlación Spearman entre features (A): máx |rho| =", m.index[0], round(m.iloc[0], 2), "; segundo", m.index[2], round(m.iloc[2], 2))
    d = dA.dropna(subset=["frac_detenido"])
    r, p = spearmanr(d.frac_detenido, d.ln_acc)
    intra = {q: round(spearmanr(x.frac_detenido, x.ln_acc)[0], 2) for q, x in d.groupby("participante")}
    log(f"proxy de tráfico vs vibración: Spearman(frac_detenido, ln ACC sd) = {r:.2f} (n={len(d)}); por participante {intra}")
    return r, intra


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for k, i in enumerate(o):
        run = max(run, (m - k) * p[i]); adj[i] = min(1, run)
    return adj


def main():
    d = A.residualizar(A.cargar())
    dA, dB = d.copy(), d[d["eda_ok"] & (d["t_desde_eda"] >= 300)].copy()
    log(f"A: {len(dA)} ventanas | B(>=300 s): {len(dB)}")
    r2 = h2()
    pulso = calidad_pulso(pd.read_csv(SAL / "gold.csv"))
    bayes_poder()
    corr = correlaciones(dA)
    tab = clasificador(dA, dB)
    # familia confirmatoria: H1 primaria + H2 + 6 celdas solo fisiología (A y B, 3 objetivos)
    fis = tab[tab.features.isin(["fis", "fis+eda"])].copy()
    p_fam = [0.3373, r2["p_bloques"]] + fis["p_bloques"].tolist()
    nom = ["H1 (primaria)", "H2"] + [f"{a}-{o}" for a, o in zip(fis.analisis, fis.objetivo)]
    adj = holm(p_fam)
    fam = pd.DataFrame(dict(prueba=nom, p_cruda=np.round(p_fam, 4), p_holm=np.round(adj, 4)))
    fam.to_csv(OUT / "familia_confirmatoria_holm.csv", index=False)
    tab["p_holm"] = np.nan
    for (i, r), a in zip(fis.iterrows(), adj[2:]):
        tab.loc[i, "p_holm"] = a
    tab.round(4).to_csv(OUT / "clasificador_tabla3.csv", index=False)
    log("\n=== Tabla 3 (RF100, LOPO, IC95 por bootstrap de sesiones, p por desplazamiento circular por bloques)")
    log(tab[["analisis", "features", "objetivo", "n", "bal_acc", "ic_inf", "ic_sup", "nulo_p95", "p_bloques", "p_holm"]].round(3).to_string(index=False))
    log("\n=== Familia confirmatoria con Holm"); log(fam.to_string(index=False))
    import platform, sklearn, scipy, statsmodels, matplotlib, pyarrow
    log("\nVersiones:", platform.python_version(), "numpy", np.__version__, "pandas", pd.__version__, "scipy", scipy.__version__, "sklearn", sklearn.__version__,
        "statsmodels", statsmodels.__version__, "matplotlib", matplotlib.__version__, "pyarrow", pyarrow.__version__)


if __name__ == "__main__":
    sys.exit(main())
