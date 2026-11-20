"""Figures of the README, drawn from results/*.json and calc/ outputs.

Journal style of the author's other repositories: 14 cm wide, 300 dpi, white background, serif text (Liberation
Serif, or the metrically identical Times New Roman) with STIX math, ticks inward on all four sides, framed legends;
tick and legend text 9.5 pt, axis titles 10.5 pt. One colour per kind of tool in every figure and in the page.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

from . import paths  # noqa: E402
from . import voltage as vt  # noqa: E402

W_CM = 14.0
FS, FS_LAB = 9.5, 10.5
INK, GRAY, LIGHT = "#000000", "#5a5a5a", "#bdbdbd"
_INSTALLED = {f.name for f in font_manager.fontManager.ttflist}
SERIF = [f for f in ("Liberation Serif", "Times New Roman") if f in _INSTALLED] + ["DejaVu Serif"]
plt.rcParams.update(
    {
        "font.family": SERIF,
        "font.size": FS,
        "mathtext.fontset": "stix",
        "axes.labelsize": FS_LAB,
        "axes.titlesize": FS,
        "xtick.labelsize": FS,
        "ytick.labelsize": FS,
        "legend.fontsize": FS - 0.5,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "legend.frameon": True,
        "legend.fancybox": False,
        "legend.edgecolor": INK,
        "legend.framealpha": 1,
        "legend.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.unicode_minus": True,
        "axes.edgecolor": INK,
    }
)
COLORS = {
    "li": "#2a78d6",
    "na": "#d03b3b",
    "dft": "#eb6834",
    "chgnet": "#1baf7a",
    "others": "#6250d6",
    "lit": "#8c8c8c",
    "exp": "#141412",
}


def newfig(aspect, ncols=1, **kw):
    """A figure W_CM wide whose height is `aspect` times its width."""
    w = W_CM / 2.54
    return plt.subplots(1, ncols, figsize=(w, w * aspect), layout="constrained", **kw)


def save(fig, name):
    paths.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths.FIGURES / name, dpi=300)
    plt.close(fig)


def _json(name):
    return json.loads(paths.result(name).read_text(encoding="utf-8"))


def panel(ax, letter, title):
    ax.set_title(f"({letter}) {title}", loc="left")


def li_to_na():
    """(a) Out-of-sample predictions against the database voltage; (b) the Na - Li shift of matched hosts."""
    L, A = _json("learning.json"), np.load(paths.result("learning_arrays.npz"))
    P = pd.read_csv(paths.result("li_na_pairs.csv"))
    fig, (a, b) = newfig(0.47, ncols=2)
    lim = (0, 6.2)
    a.scatter(A["yl"], A["cv_pred_group"], s=4, lw=0, alpha=0.45, color=COLORS["li"], label="Li, unseen hosts")
    a.scatter(A["y_na"], A["p_na"], s=7, lw=0, alpha=0.8, color=COLORS["na"], label="Na")
    a.plot(lim, lim, color=INK, lw=0.8)
    a.set(xlim=lim, ylim=lim, aspect="equal", xlabel="database voltage (V)", ylabel="predicted voltage (V)")
    a.legend(loc="upper left", markerscale=2.5, handletextpad=0.2, borderpad=0.3)
    a.text(
        6.0, 0.2, f"MAE\nLi {L['li_cv']['group_mae']:.2f} V\nNa {L['li_to_na']['mae']:.2f} V", ha="right", va="bottom"
    )
    panel(a, "a", "Model trained on Li only")
    bins = np.linspace(-2.0, 1.5, 36)
    b.hist(P.v_na - P.v_li, bins=bins, color=INK, alpha=0.55, label="database")
    b.hist(P.pred_na - P.v_li, bins=bins, histtype="step", lw=1.4, color=COLORS["na"], label="Li-trained model")
    b.axvline(L["pairs"]["data_mean_vna_minus_vli"], color=INK, ls="--", lw=1)
    b.axvline(L["pairs"]["pred_na_minus_data_li"], color=COLORS["na"], ls="--", lw=1)
    b.set(xlabel=r"$V_\mathrm{Na} - V_\mathrm{Li}$, same host (V)", ylabel="hosts")
    b.set_ylim(0, b.get_ylim()[1] * 1.45)
    b.legend(loc="upper left")
    panel(b, "b", f"{L['pairs']['n']} matched hosts")
    save(fig, "li_to_na.png")


def licoo2_methods():
    """LiCoO2 average voltage (0 <= x <= 1) by method."""
    V = vt.dft_voltages(vt.load_dft())
    C = vt.load_chgnet()
    lit = [("GGA [lit.]", 3.38), ("GGA+U [lit.]", 3.85), ("HSE06 [lit.]", 4.51)]
    rows = [
        ("PBE, GPAW", V["PBE|O1|base"], COLORS["dft"]),
        ("PBE+U, GPAW", V["PBE+U|O1|base"], COLORS["dft"]),
        ("PBE+U, GPAW, O3 CoO$_2$", V["PBE+U|O3|base"], COLORS["dft"]),
        ("CHGNet, fixed lattice", C["fixed_lattice"]["V_Li"], COLORS["chgnet"]),
        ("CHGNet, free cell", C["cell_relaxed"]["V_Li"], COLORS["chgnet"]),
    ]
    rows = [(n, v, COLORS["lit"]) for n, v in lit] + rows
    fig, ax = newfig(0.5)
    y = np.arange(len(rows))[::-1]
    for yi, (_name, v, c) in zip(y, rows, strict=True):
        ax.barh(yi, v, color=c, height=0.62)
        ax.text(v + 0.03, yi, f"{v:.2f}", va="center")
    ax.axvline(4.1, color=COLORS["exp"], ls="--", lw=1)
    ax.text(4.12, y[0] + 0.45, "experiment", va="top")
    ax.set_yticks(y, [r[0] for r in rows])
    ax.tick_params(axis="y", which="both", left=False, right=False)
    ax.set(xlim=(3.0, 4.85), xlabel=r"average voltage of Li$_x$CoO$_2$, $0 \leq x \leq 1$ (V)")
    save(fig, "licoo2_voltage.png")


def learning_curve():
    LC = _json("learning.json")["learning_curve"]
    Lcv = _json("learning.json")["li_cv"]["group_mae"]
    fig, ax = newfig(0.5)
    ax.errorbar(
        LC["k"],
        LC["mae_mean"],
        yerr=LC["mae_std"],
        fmt="o-",
        color=COLORS["na"],
        capsize=3,
        ms=4,
        label="Li + $k$ Na reactions",
    )
    ax.axhline(
        LC["others_mae_mean"],
        color=COLORS["others"],
        ls="--",
        lw=1.2,
        label=f"all other ions, no Na ({_json('learning.json')['others_to_na']['n_train']:,} reactions)",
    )
    ax.axhline(
        LC["others_size_matched_mae_mean"],
        color=COLORS["others"],
        ls=":",
        lw=1.4,
        label=f"other ions, subsampled to {LC['size_matched_n_train']:,} reactions",
    )
    ax.axhline(Lcv, color=COLORS["li"], ls="-.", lw=1, label="Li model on unseen Li hosts")
    ax.set(xlabel="Na reactions added to the 2,043 Li reactions, $k$", ylabel="MAE on held-out Na hosts (V)")
    ax.set_ylim(0.40, 0.75)
    ax.legend(loc="upper right")
    save(fig, "learning_curve.png")


def distance():
    D, A = _json("distance.json"), np.load(paths.result("distance_arrays.npz"))
    fig, (a, b) = newfig(0.45, ncols=2)
    bins = np.logspace(0, 2, 41)
    a.hist(A["cv_dist"], bins=bins, color=COLORS["li"], alpha=0.65, label="Li, unseen hosts")
    a.hist(A["d_na"], bins=bins, color=COLORS["na"], alpha=0.55, label="Na")
    a.axvline(D["threshold"], color=INK, ls="--", lw=1)
    a.set(xscale="log", xlabel="distance to the 5 nearest Li reactions", ylabel="reactions")
    a.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a.legend(loc="upper right")
    panel(a, "a", "Distance to the training data")
    x = np.arange(len(D["bins"]))
    b.plot(x, [q["mae"] for q in D["bins"]], "o-", color=INK, ms=4, label="Li and Na pooled")
    b.plot(x, [q["mae_li"] for q in D["bins"]], "s--", color=COLORS["li"], ms=3.5, label="Li only")
    b.set_ylim(0.22, 0.74)
    for xi, q in zip(x, D["bins"], strict=True):
        b.text(xi, 0.245, f"{100 * q['frac_na']:.0f} %", ha="center", va="bottom", color=COLORS["na"])
    b.set_xticks(x, [f"{q['d_lo']:.1f}–{q['d_hi']:.1f}" for q in D["bins"]], rotation=35, ha="right")
    b.set(xlabel="distance bin (equal counts)", ylabel="mean absolute error (V)", xlim=(-0.7, len(x) - 0.4))
    b.legend(loc="upper left")
    panel(b, "b", "Error against distance")
    save(fig, "distance.png")


def diffusion():
    R = _json("diffusion.json")
    pooled = R["all_runs"]
    temps = pooled["temperatures"]
    shades = {500: "#f2b48a", 600: "#eb8a4f", 700: "#d0581f", 800: "#8c3510"}
    fig, (a, b) = newfig(0.45, ncols=2)
    for t in temps:
        curves = [json.loads((paths.CALC / "md" / "runs" / n / "analysis.json").read_text()) for n in t["runs"]]
        m = np.mean([c["msd_A2"] for c in curves], axis=0)
        a.plot(curves[0]["t_ps"], m, color=shades[int(t["T"])], label=f"{int(t['T'])} K")
    a.set(xlabel="time lag (ps)", ylabel=r"Li mean-squared displacement (Å$^2$)", xlim=(0, None), ylim=(0, None))
    a.legend(loc="upper left")
    panel(a, "a", "MSD, mean of the runs")
    arr = pooled["arrhenius"]
    for t in temps:
        x = 1000 / t["T"]
        b.plot([x] * t["n_runs"], np.array(t["D_each_cm2_s"]) * 1e5, "o", ms=3, mfc="none", color=shades[int(t["T"])])
        b.errorbar(
            x,
            t["D_cm2_s"] * 1e5,
            yerr=(t["D_sd_cm2_s"] or 0) * 1e5,
            fmt="s",
            ms=5,
            capsize=3,
            color=shades[int(t["T"])],
            mec=INK,
            mew=0.6,
        )
    xx = np.linspace(1000 / 820, 1000 / 480, 50)
    b.plot(xx, np.exp(arr["lnD0"] - arr["Ea_eV"] / (8.617333262e-5 * 1000 / xx)) * 1e5, color=INK, lw=1, ls="--")
    b.set(yscale="log", xlabel="1000 / T (K$^{-1}$)", ylabel=r"$D$ (10$^{-5}$ cm$^2$ s$^{-1}$)", ylim=(0.45, 3.4))
    b.yaxis.set_major_locator(FixedLocator([0.5, 1, 2, 3]))
    b.yaxis.set_minor_locator(NullLocator())
    b.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    b.text(
        0.97,
        0.95,
        f"$E_a$ = {arr['Ea_eV']:.3f} ± {arr['Ea_stderr_eV']:.3f} eV",
        transform=b.transAxes,
        ha="right",
        va="top",
    )
    panel(b, "b", "Arrhenius")
    save(fig, "li3n_diffusion.png")


ALL = [li_to_na, licoo2_methods, learning_curve, distance, diffusion]
