"""Checks behind VALIDATION.md: each row is (category, check, value, expected, tolerance, result, how).

A check passes when |value - expected| <= tolerance; INFO rows are reported without a pass/fail criterion.
run_all() reads results/*.json and calc/ outputs only, so it runs in seconds and in CI.
"""

import json

import numpy as np
import pandas as pd

from . import data as dt
from . import features as ft
from . import paths
from . import summary as sm
from . import voltage as vt

# experimental LiCoO2 (Akimoto et al. 1998): c = 14.0542 A, z(O) = 0.2396 -> plane spacings along c
C_HEX, Z_O = 14.0542, 0.2396
CO_O_EXP, LI_O_EXP = (Z_O + 1 / 3 - 0.5) * C_HEX, (1 / 3 - Z_O) * C_HEX


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def row(cat, check, value, expected, tol, how, info=False):
    if info:
        res = "INFO"
    else:
        res = "PASS" if abs(value - expected) <= tol + 1e-12 else "FAIL"
    return {
        "category": cat,
        "check": check,
        "value": value,
        "expected": expected,
        "tolerance": tol,
        "result": res,
        "how": how,
    }


def data_checks():
    log = _json(paths.result("data_log.json"))
    d, clean = dt.load(), dt.load_clean()
    counts = [log[k] for k in ("raw_rows", "after_voltage_filter", "after_stability_filter", "final_rows")]
    keys = {
        (ft.swap_ion(a, i, i), ft.swap_ion(b, i, i), i)
        for a, b, i in zip(clean.formula_charge, clean.formula_discharge, clean.working_ion, strict=True)
    }
    return [
        row(
            "Data",
            "rows of the snapshot and after each filter: 5,133 / 4,965 / 4,297 / 3,430",
            float(sum(int(a == b) for a, b in zip(counts, (5133, 4965, 4297, 3430), strict=True))),
            4,
            0,
            "data_log.json against the counts quoted in the README (number of matching counts)",
        ),
        row(
            "Data",
            "rows of the feature matrix with all 266 features finite",
            float(np.isfinite(d["X"]).all(axis=1).sum()),
            float(len(clean)),
            0,
            "data/features.npz against data/clean.csv",
        ),
        row(
            "Data",
            "duplicate reactions after reducing the formulas",
            float(len(clean) - len(keys)),
            0,
            0,
            "(charged, discharged, ion) with reduced formulas; polymorphs were merged by median",
        ),
    ]


def dft_checks():
    recs = vt.load_dft()
    V = vt.dft_voltages(recs)
    Vt = vt.dft_voltages(vt.load_dft(paths.CALC / "dft" / "first_run"))
    co_o, li_o = sm.dft_geometry()
    o1u = recs[("CoO2-O1", vt.U_CO, 520, 0)]
    conv = max(abs(V["PBE|O1|converged"] - V["PBE|O1|base"]), abs(V["PBE+U|O1|converged"] - V["PBE+U|O1|base"]))
    worst_first = max(abs(V[k] - Vt[k]) for k in Vt)
    return [
        row(
            "DFT",
            "relaxations converged to 0.03 eV/A within 60 BFGS steps",
            float(sum(r["relax_converged"] for r in recs.values())),
            float(len(recs)),
            0,
            "all 11 GPAW runs (4 structures, PBE and PBE+U, two settings)",
        ),
        row(
            "DFT",
            "650 eV and denser k points vs 520 eV: change of V, worst of PBE and PBE+U (mV)",
            1000 * conv,
            0,
            20,
            "same structures and relaxation, cutoff and k meshes raised",
        ),
        row(
            "DFT",
            "relaxed PBE+U LiCoO2: Co-O plane spacing along c (A)",
            co_o,
            CO_O_EXP,
            0.05,
            "against the single-crystal structure (Akimoto et al. 1998)",
        ),
        row(
            "DFT",
            "relaxed PBE+U LiCoO2: Li-O plane spacing along c (A)",
            li_o,
            LI_O_EXP,
            0.05,
            "against the single-crystal structure (Akimoto et al. 1998)",
        ),
        row("DFT", "magnetic moment of O1 CoO2, PBE+U (muB per Co)", o1u["magmom"], 1.0, 0.05, "low-spin Co4+ (d5)"),
        row(
            "DFT",
            "V(PBE), O1 CoO2, against GGA in the literature (V)",
            V["PBE|O1|base"],
            sm.LITERATURE["GGA"],
            0.05,
            "Seo, Urban & Ceder (2015)",
        ),
        row(
            "DFT",
            "V(PBE+U), O1 CoO2, against GGA+U in the literature (V)",
            V["PBE+U|O1|base"],
            sm.LITERATURE["GGA+U"],
            0.05,
            "Seo, Urban & Ceder (2015), U(Co) = 3.32 eV",
        ),
        row(
            "DFT",
            "same voltages from the first run, started with Li and Co exchanged in LiCoO2 (worst, mV)",
            1000 * worst_first,
            0,
            20,
            "calc/dft/first_run: relaxing the one free coordinate of O restores the structure",
        ),
    ]


def chgnet_checks():
    C = vt.load_chgnet()
    fr, fx, sw = C["cell_relaxed"], C["fixed_lattice"], C["swapped_start"]
    o1 = fr["energies"]["CoO2-O1"]
    return [
        row(
            "CHGNet",
            "V_Li, free cell minus fixed lattice (mV)",
            1000 * (fr["V_Li"] - fx["V_Li"]),
            0,
            0,
            f"O1 CoO2 swells by {100 * (o1['volume_per_atom'] / o1['volume_per_atom_start'] - 1):.0f} % in"
            " volume when its cell is free",
            info=True,
        ),
        row(
            "CHGNet",
            "V_Na from the swapped start minus the right start, free cell (mV)",
            1000 * (sw["V_Na"] - fr["V_Na"]),
            0,
            0,
            "NaCoO2 has a second minimum with Co-O 2.04 A; LiCoO2 returns to the same structure from either start",
            info=True,
        ),
        row(
            "CHGNet",
            "V_Li, swapped start minus right start (mV)",
            1000 * abs(sw["V_Li"] - fr["V_Li"]),
            0,
            1,
            "LiCoO2 relaxes to the same structure from either start",
        ),
    ]


def md_checks():
    R = _json(paths.result("diffusion.json"))
    first = _json(paths.CALC / "md" / "first_set_md_result.json")
    ours = {r["name"]: r for r in R["runs"]}
    worst_t = max(abs(r["mean_T_after_eq_K"] - r["T"]) for r in R["runs"])
    rel = []
    for T, r in first["runs"].items():
        mine = sorted(ours[n]["D_cm2_s"] for n in ours if n.startswith(f"{T}K_laptop"))
        rel += [abs(a / b - 1) for a, b in zip(mine, sorted(r["D_each_cm2_s"]), strict=True)]
    temps = R["all_runs"]["temperatures"]
    return [
        row(
            "MD",
            f"mean temperature after 2.5 ps minus the thermostat setting, worst of {len(R['runs'])} runs (K)",
            worst_t,
            0,
            5,
            "Berendsen NVT, 25 ps per run",
        ),
        row(
            "MD",
            "re-analysis of the first set's 7 trajectories against its own D values (worst relative difference)",
            max(rel),
            0,
            1e-9,
            "calc/md/first_set_md_result.json, written by the first set's own analysis",
        ),
        row(
            "MD",
            "D rises with temperature (pooled means; number of increasing steps)",
            float(sum(b["D_cm2_s"] > a["D_cm2_s"] for a, b in zip(temps[:-1], temps[1:], strict=True))),
            float(len(temps) - 1),
            0,
            "mean of the runs at each of 500, 600, 700 and 800 K",
        ),
        row(
            "MD",
            "activation energy, all runs (eV)",
            R["all_runs"]["arrhenius"]["Ea_eV"],
            0,
            0,
            f"Arrhenius fit of the four means; standard error {R['all_runs']['arrhenius']['Ea_stderr_eV']:.3f} eV",
            info=True,
        ),
    ]


def ml_checks():
    L, D = _json(paths.result("learning.json")), _json(paths.result("distance.json"))
    P = pd.read_csv(paths.result("li_na_pairs.csv"))
    return [
        row(
            "ML",
            "group-split MAE below the guess-the-mean baseline (V)",
            float(L["li_cv"]["group_mae"] < L["li_cv"]["dummy_mae"]),
            1,
            0,
            "Li, 5-fold GroupKFold by host system; 1 = yes",
        ),
        row(
            "ML",
            "random split no worse than group split (leakage makes it optimistic)",
            float(L["li_cv"]["random_mae"] <= L["li_cv"]["group_mae"]),
            1,
            0,
            "same model, shuffled 5-fold; 1 = yes",
        ),
        row(
            "ML",
            "share of unseen Li hosts above the distance threshold",
            D["frac_li_flagged"],
            0.05,
            0.005,
            "95th percentile of the same distances, by construction",
        ),
        row(
            "ML",
            "matched pairs: one Li partner per Na reaction",
            float(len(P)),
            float(L["pairs"]["n"]),
            0,
            "results/li_na_pairs.csv against learning.json",
        ),
        row(
            "ML",
            "web check cases: Python prediction of the named examples recomputed (max |dV|, V)",
            _web_named_check(),
            0,
            1e-9,
            "results/web_check.json; the JavaScript port is checked in CI by Node",
        ),
    ]


def _web_named_check():
    from . import learning as ln

    cases = [c for c in _json(paths.result("web_check.json")) if c["named"]][:4]
    m = ln.fit_li()
    worst = max(
        abs(float(m.predict([ft.reaction_vector(c["discharged"], c["charged"], c["ion"])])[0]) - c["pred"])
        for c in cases
    )
    return round(worst, 12)  # rounding noise of another platform must not change VALIDATION.md


def readme_check():
    text = (paths.ROOT / "README.md").read_text(encoding="utf-8")
    N = sm.compute()
    missing = [k for k, v in N.items() if v not in text]
    return [
        row(
            "README",
            "numbers of summary.compute() missing from README.md",
            float(len(missing)),
            0,
            0,
            "verbatim string match" + (f"; missing: {', '.join(missing)}" if missing else ""),
        )
    ]


def run_all(readme=True):
    rows = data_checks() + dft_checks() + chgnet_checks() + md_checks() + ml_checks()
    if readme:
        rows += readme_check()
    return rows


def markdown(rows):
    npass = sum(r["result"] == "PASS" for r in rows)
    nchk = sum(r["result"] != "INFO" for r in rows)
    ninfo = sum(r["result"] == "INFO" for r in rows)
    out = [
        "# Validation results",
        "",
        f"{npass} / {nchk} checks passed, plus {ninfo} observations reported without a pass/fail criterion (INFO).",
        "Generated by `python scripts/validate.py --markdown` from `results/*.json` and the outputs in `calc/`;",
        "CI reruns",
        "every check on each push and fails if this file is out of date.",
        "",
        "| # | Category | Check | Value | Expected | Tolerance | Result | How it is checked |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(rows, 1):
        val = f"{r['value']:.4g}"
        exp = "–" if r["result"] == "INFO" else f"{r['expected']:.4g}"
        tol = "–" if r["result"] == "INFO" else f"{r['tolerance']:.4g}"
        out.append(f"| {i} | {r['category']} | {r['check']} | {val} | {exp} | {tol} | {r['result']} | {r['how']} |")
    return "\n".join(out) + "\n"
