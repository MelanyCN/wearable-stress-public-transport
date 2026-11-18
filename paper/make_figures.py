"""Figuras del artículo (todas desde los CSV/parquet de salida/)."""
import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1])); sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from labels import PART, label
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut

ROOT = Path(__file__).resolve().parents[1]; SAL = ROOT / "salida"; FIG = ROOT / "paper" / "figs"
plt.rcParams.update({"font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 6.5, "legend.fontsize": 5.8, "xtick.labelsize": 6, "ytick.labelsize": 6, "figure.dpi": 150,
                     "axes.spines.top": False, "axes.spines.right": False})
C = {"blue": "#2b6cb0", "red": "#c53030", "gray": "#718096", "green": "#2f855a", "orange": "#dd6b20"}

# ---------------- Fig 1: pipeline
def fig1():
    fig, ax = plt.subplots(figsize=(4.8, 2.35)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 40)
    def box(x, y, w, h, txt, col):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=col, ec="#2d3748", lw=.8))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=4.9)
    def arr(x0, y0, x1, y1): ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", lw=.9, color="#2d3748"))
    box(1, 27, 20, 10, "Wristband (BLE)\nraw 20-byte packets\nppg · acc · temp · gsr", "#ebf8ff")
    box(1, 3, 20, 10, "Phone app (1 Hz)\nGPS · speed · mic level\n(dBFS) · labels", "#f0fff4")
    box(27, 27, 20, 10, "Packet decoding\n+ clock reconstruction\n(counter + linear fit)", "#fefcbf")
    box(27, 3, 20, 10, "Bus-only filter\n1-min trimming of\neach bus segment", "#fefcbf")
    box(53, 15, 20, 10, "30-s windows (50 % overlap)\nHR/HRV · temp · ACC · EDA\nnoise · speed · stops", "#fefcbf")
    box(79, 29, 20, 8, "Unsupervised\nclustering vs. proxies", "#fed7d7")
    box(79, 19.5, 20, 8, "Supervised bound +\nmixed models", "#fed7d7")
    box(79, 10, 20, 8, "Vehicle-state detector\nfrom wrist ACC (H2)", "#c6f6d5")
    box(79, 0.5, 20, 8, "Event-locked analysis\nnoise bursts → HR (H1)", "#fed7d7")
    arr(21, 32, 27, 32); arr(21, 8, 27, 8); arr(37, 27, 55, 25); arr(37, 13, 60, 15); arr(73, 22, 79, 33); arr(73, 20, 79, 23.5); arr(73, 18, 79, 14); arr(73, 16, 79, 4.5)
    plt.tight_layout(pad=.2); plt.savefig(FIG / "fig1_pipeline.pdf"); plt.close()

# ---------------- Fig 2: decoded signals
def fig2():
    sid = "P09_2026-09-30_16-02-43"; D = SAL / "decodificado"
    ppg, acc, tmp, eda = [pd.read_parquet(D / f"{sid}__{c}.parquet") for c in ("ppg", "acc", "temp", "eda")]
    t0 = ppg["t"].iloc[0] + 600
    fig, ax = plt.subplots(2, 2, figsize=(4.8, 3.1))
    m = (ppg["t"] >= t0) & (ppg["t"] < t0 + 12); tt = ppg["t"][m].to_numpy() - t0; v = ppg["ppg"][m].to_numpy()
    b, a = butter(3, [0.5, 4], btype="band", fs=64); y = filtfilt(b, a, ppg["ppg"].to_numpy()[ppg["t"].between(t0 - 20, t0 + 30).to_numpy()])
    tw = ppg["t"][ppg["t"].between(t0 - 20, t0 + 30)].to_numpy() - t0; ys = y[(tw >= 0) & (tw < 12)]
    pk, _ = find_peaks(ys, distance=int(0.45 * 64), prominence=0.4 * ys.std())
    ax[0, 0].plot(tt, (v - v.mean()) / v.std(), color="#a0aec0", lw=.6, label="decoded (raw)"); ax[0, 0].plot(tt, ys / ys.std(), color=C["blue"], lw=1, label="0.5–4 Hz")
    ax[0, 0].plot(tt[pk], ys[pk] / ys.std(), "v", color=C["red"], ms=4, label="beats"); ax[0, 0].set_title("(a) PPG, 12 s (64 Hz)"); ax[0, 0].set_xlabel("s"); ax[0, 0].legend(loc="lower right", ncol=3)
    ma = (acc["t"] >= t0) & (acc["t"] < t0 + 12); mag = np.sqrt((acc[["x_g", "y_g", "z_g"]][ma] ** 2).sum(axis=1))
    ax[0, 1].plot(acc["t"][ma] - t0, mag, color=C["green"], lw=.8); ax[0, 1].axhline(1, color=C["gray"], ls="--", lw=.7); ax[0, 1].set_title("(b) accelerometer magnitude (32 Hz)"); ax[0, 1].set_xlabel("s"); ax[0, 1].set_ylabel("g")
    ax[1, 0].plot((tmp["t"] - tmp["t"].iloc[0]) / 60, tmp["temp_c"], color=C["orange"], lw=.8); ax[1, 0].set_title("(c) skin temperature (4 Hz)"); ax[1, 0].set_xlabel("min"); ax[1, 0].set_ylabel("°C")
    ax[1, 1].plot((eda["t"] - eda["t"].iloc[0]) / 60, eda["eda_us"], color=C["blue"], lw=.8); ax[1, 1].set_title("(d) EDA, float packets (0.5 Hz)"); ax[1, 1].set_xlabel("min"); ax[1, 1].set_ylabel("µS")
    plt.tight_layout(); plt.savefig(FIG / "fig2_signals.pdf"); plt.close()

# ---------------- Fig 3: H2
def fig3():
    h = pd.read_csv(SAL / "novedoso" / "h2_ventanas.csv")
    cols = [c for c in h.columns if c.startswith(("banda_", "std_", "ln_std"))]; X = h[cols].to_numpy(); y = h["detenido"].to_numpy()
    def loo(g):
        p = np.zeros(len(y))
        for tr, te in LeaveOneGroupOut().split(X, y, g):
            p[te] = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        return p
    gp, gs = h["participante"].to_numpy(), h["sesion"].to_numpy(); pp, ps = loo(gp), loo(gs)
    fig, ax = plt.subplots(1, 2, figsize=(4.8, 2.3), gridspec_kw={"width_ratios": [1, 1.25]})
    f, t, _ = roc_curve(y, pp); ax[0].plot(f, t, color="k", lw=1.8, label=f"pooled (AUC {roc_auc_score(y, pp):.2f})")
    for i, p in enumerate(sorted(set(gp))):
        mk = gp == p
        if len(set(y[mk])) == 2:
            f, t, _ = roc_curve(y[mk], pp[mk]); ax[0].plot(f, t, lw=.9, label=f"{PART[p]} ({roc_auc_score(y[mk], pp[mk]):.2f})")
    ax[0].plot([0, 1], [0, 1], ":", color=C["gray"]); ax[0].set_xlabel("false-positive rate"); ax[0].set_ylabel("true-positive rate"); ax[0].set_title("(a) leave-one-participant-out"); ax[0].legend(loc="lower right")
    rows = [(s, roc_auc_score(y[gs == s], ps[gs == s])) for s in sorted(set(gs)) if len(set(y[gs == s])) == 2]
    nom = {s_: label(s_) for s_ in set(gs)}
    rows = sorted(rows, key=lambda r: r[1]); lab = [nom[r[0]] for r in rows]
    ax[1].barh(range(len(rows)), [r[1] for r in rows], color=[C["red"] if r[1] < .7 else C["blue"] for r in rows]); ax[1].axvline(.5, color="k", ls=":", lw=.8)
    ax[1].set_yticks(range(len(rows))); ax[1].set_yticklabels(lab, fontsize=5.8); ax[1].set_xlim(.4, 1.02); ax[1].set_xlabel("AUC held-out session"); ax[1].set_title("(b) leave-one-session-out")
    plt.tight_layout(); plt.savefig(FIG / "fig3_h2.pdf"); plt.close()

# ---------------- Fig 5: mixed models + classifier
def fig5():
    mm = pd.read_csv(SAL / "avanzado" / "modelo_mixto.csv"); cl = pd.read_csv(SAL / "avanzado" / "clasificador.csv"); pm = pd.read_csv(SAL / "avanzado" / "permutaciones.csv")
    fig, ax = plt.subplots(1, 2, figsize=(4.8, 2.5), gridspec_kw={"width_ratios": [1.15, 1]})
    sel = mm[(mm["resultado"].isin(["hr_picos_bpm", "lnrmssd", "ln_eda"])) & (mm["termino"].isin(["ruido_z", "cong_z", "acc_z"]))].copy()
    nm = {"hr_picos_bpm": "HR (bpm)", "lnrmssd": "ln RMSSD", "ln_eda": "ln EDA"}; tn = {"ruido_z": "noise", "cong_z": "congestion", "acc_z": "movement"}
    sel["y"] = sel["resultado"].map(nm) + " ~ " + sel["termino"].map(tn)
    sel["esc"] = sel["coef"] / sel["sd_resultado"]; sel["lo"] = sel["ic_inf"] / sel["sd_resultado"]; sel["hi"] = sel["ic_sup"] / sel["sd_resultado"]
    sel = sel.iloc[::-1]
    cols = [C["red"] if t == "acc_z" else C["blue"] for t in sel["termino"]]
    ax[0].errorbar(sel["esc"], range(len(sel)), xerr=[sel["esc"] - sel["lo"], sel["hi"] - sel["esc"]], fmt="none", ecolor=C["gray"], capsize=2)
    ax[0].scatter(sel["esc"], range(len(sel)), c=cols, zorder=3, s=14); ax[0].axvline(0, color="k", lw=.7)
    ax[0].set_yticks(range(len(sel))); ax[0].set_yticklabels(sel["y"], fontsize=5.8); ax[0].set_xlabel("effect per 1 SD of predictor (in outcome SDs), 95 % CI"); ax[0].set_title("(a) mixed models")
    t3 = pd.read_csv(SAL / "revision" / "clasificador_tabla3.csv")
    t3 = t3[(t3.analisis == "A") & t3.features.isin(["fis", "fis+mov"]) & t3.objetivo.isin(["gold_trafico", "gold_ruido"])]
    x = np.arange(2); w = .35
    for k, (fs, lab, col) in enumerate([("fis", "physiology only", C["blue"]), ("fis+mov", "physiology + vibration", C["red"])]):
        r = [t3[(t3.features == fs) & (t3.objetivo == t)].iloc[0] for t in ("gold_trafico", "gold_ruido")]
        v = np.array([q.bal_acc for q in r]); lo = np.array([q.ic_inf for q in r]); hi = np.array([q.ic_sup for q in r])
        ax[1].bar(x + (k - .5) * w, v, w, color=col, label=lab, yerr=[v - lo, hi - v], capsize=2, error_kw=dict(lw=.8, ecolor="k"))
        for xi, q in zip(x + (k - .5) * w, r):
            ax[1].plot([xi - w / 2, xi + w / 2], [q.nulo_p95] * 2, color="k", lw=1.2, ls=":")
    ax[1].axhline(1 / 3, color="k", ls="--", lw=.8, label="chance (0.33)"); ax[1].plot([], [], color="k", lw=1.2, ls=":", label="null 95th percentile")
    ax[1].set_xticks(x); ax[1].set_xticklabels(["traffic proxy", "noise proxy"]); ax[1].set_ylim(.2, .68); ax[1].set_ylabel("balanced accuracy"); ax[1].legend(loc="upper left"); ax[1].set_title("(b) cross-person RF (LOPO)")
    plt.tight_layout(); plt.savefig(FIG / "fig5_models.pdf"); plt.close()

if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig5): f(); print(f.__name__, "ok", flush=True)
