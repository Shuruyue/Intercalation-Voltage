"""Every number quoted in the README and on the page, computed from results/*.json and calc/ outputs.

compute() returns {name: formatted string}; scripts/validate.py checks that each string appears in README.md.
"""

import json
import re

import numpy as np

from . import paths
from . import voltage as vt

LITERATURE = {"GGA": 3.38, "GGA+U": 3.85, "HSE06": 4.51, "experiment": 4.1}  # Seo, Urban & Ceder, PRB 92 (2015)


def _json(name):
    return json.loads(paths.result(name).read_text(encoding="utf-8"))


def f2(x):
    return f"{x:.2f}"


def pct(x):
    return f"{100 * x:.0f} %"


SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def sci(x, digits=2):
    """1.3e-06 -> '1.3 × 10⁻⁶'."""
    m, e = f"{x:.{digits - 1}e}".split("e")
    return f"{m} × 10" + str(int(e)).translate(SUPERSCRIPT)


def signed(x):
    return f"{'+' if x >= 0 else '−'}{abs(x):.2f}"


def html_formula(f):
    """'Ti3AlC2' -> 'Ti<sub>3</sub>AlC<sub>2</sub>'."""
    return re.sub(r"(\d+(?:\.\d+)?)", r"<sub>\1</sub>", f)


def dft_geometry():
    """Spacing (A) along c between the Co plane and the nearest O plane in relaxed PBE+U LiCoO2, and the same for Li.

    In the primitive cell written by GPAW the c axis is z and both O planes lie between the Co plane and its image,
    so no periodic image is needed.
    """
    d = vt.load_dft()[("LiCoO2", vt.U_CO, 520, 0)]
    z = {s: [] for s in ("Li", "Co", "O")}
    for s, p in zip(d["symbols"], d["positions_A"], strict=True):
        z[s].append(p[2])
    return min(abs(z["Co"][0] - o) for o in z["O"]), min(abs(z["Li"][0] - o) for o in z["O"])


def compute():
    L, D, LOG, DF = _json("learning.json"), _json("distance.json"), _json("data_log.json"), _json("diffusion.json")
    V = vt.dft_voltages(vt.load_dft())
    C = vt.load_chgnet()
    N = {}
    # data
    for k in ("raw_rows", "after_voltage_filter", "after_stability_filter", "final_rows"):
        N[k] = f"{LOG[k]:,}"
    N["n_li"], N["n_na"] = f"{L['n_li']:,}", f"{L['n_na']:,}"
    N["n_groups_li"], N["n_groups_na"] = str(L["n_groups_li"]), str(L["n_groups_na"])
    N["n_ions"] = str(len(L["ion_counts"]))
    # learning
    N["mae_li"], N["r2_li"] = f2(L["li_cv"]["group_mae"]), f2(L["li_cv"]["group_r2"])
    N["mae_li_random"], N["mae_li_dummy"] = f2(L["li_cv"]["random_mae"]), f2(L["li_cv"]["dummy_mae"])
    N["mae_na"], N["r2_na"], N["bias_na"] = (
        f2(L["li_to_na"]["mae"]),
        f2(L["li_to_na"]["r2"]),
        signed(L["li_to_na"]["bias"]),
    )
    N["mae_na_dummy"] = f2(L["li_to_na"]["dummy_mae"])
    N["gt1_li"], N["gt1_na"] = pct(L["li_cv"]["frac_err_gt_1V"]), pct(L["li_to_na"]["frac_err_gt_1V"])
    N["mae_others"], N["n_others"] = f2(L["others_to_na"]["mae"]), f"{L['others_to_na']['n_train']:,}"
    P = L["pairs"]
    N["pairs_n"] = str(P["n"])
    N["pairs_data_abs"], N["pairs_pred_abs"] = (
        f2(abs(P["data_mean_vna_minus_vli"])),
        f2(abs(P["pred_na_minus_data_li"])),
    )
    N["pairs_insample"], N["pairs_oof"] = (
        signed(P["pred_na_minus_pred_li_insample"]),
        signed(P["pred_na_minus_pred_li_oof"]),
    )
    N["pairs_na_lower"] = pct(P["frac_na_lower"])
    LC = L["learning_curve"]
    N["lc_k0"], N["lc_k100"] = f2(LC["mae_mean"][0]), f2(LC["mae_mean"][-1])
    N["lc_others"], N["lc_size_matched"] = f2(LC["others_mae_mean"]), f2(LC["others_size_matched_mae_mean"])
    N["lc_size_n"] = f"{LC['size_matched_n_train']:,}"
    N["toy_pred"] = f"{L['toy']['pred_at_2']:.1f}"
    N["top_feature"] = L["top_features_li_model"][0]["name"].replace("MagpieData ", "")
    # distance
    N["n_feat_dist"], N["n_const"] = str(D["n_features_used"]), str(D["n_constant_in_li"])
    N["threshold"] = f"{D['threshold']:.1f}"
    N["flag_na"], N["flag_li"] = f"{100 * D['frac_na_flagged']:.0f}", f"{100 * D['frac_li_flagged']:.0f}"
    N["mae_li_flagged"], N["mae_li_noflag"] = f2(D["mae_li_flagged"]), f2(D["mae_li_not_flagged"])
    N["mae_na_flagged"], N["mae_na_noflag"] = f2(D["mae_na_flagged"]), f2(D["mae_na_not_flagged"])
    N["rho_na"], N["rho_li"] = signed(D["spearman_dist_abs_err_na"]), signed(D["spearman_dist_abs_err_li"])
    # screening, error shares, one transition metal left out, range of the Li model
    S = _json("screening.json")
    sl, sn = S["li_unseen_hosts"], S["na_li_model"]
    N["scr_target"], N["scr_top_share"] = f"{S['target_V']:.1f}", pct(S["top_share"])
    N["scr_n_li"], N["scr_n_top_li"], N["scr_n_target_li"] = (
        f"{sl['n']:,}",
        str(sl["top"]["n_selected"]),
        str(sl["n_target"]),
    )
    N["scr_base_li"], N["scr_base_na"] = pct(sl["base_rate"]), pct(sn["base_rate"])
    N["scr_top_prec_li"], N["scr_top_rec_li"] = pct(sl["top"]["precision"]), pct(sl["top"]["recall"])
    N["scr_top_enrich_li"] = f"{sl['top']['enrichment']:.1f}"
    N["scr_pred_prec_li"], N["scr_pred_rec_li"] = (
        pct(sl["predicted_above"]["precision"]),
        pct(sl["predicted_above"]["recall"]),
    )
    N["scr_top_prec_na"], N["scr_top_rec_na"] = pct(sn["top"]["precision"]), pct(sn["top"]["recall"])
    E = S["errors"]
    N["err_median_li"], N["err_median_na"] = f2(E["li_unseen_hosts"]["median"]), f2(E["na_li_model"]["median"])
    for lvl, key in (("0.3", "03"), ("0.5", "05"), ("1.0", "10")):
        N[f"err_{key}_li"] = pct(E["li_unseen_hosts"][f"within_{lvl}"])
        N[f"err_{key}_na"] = pct(E["na_li_model"][f"within_{lvl}"])
    LO = S["leave_element_out"]
    N["lto_n_el"], N["lto_min_n"], N["lto_pairs"] = (
        str(len(LO["elements"])),
        str(LO["min_reactions"]),
        f"{LO['n_pairs']:,}",
    )
    N["lto_mae"], N["lto_group"] = f2(LO["mae_left_out"]), f2(LO["mae_group_split"])
    worst = sorted(LO["elements"], key=lambda r: -r["mae_left_out"])[:3]
    N["lto_worst"] = ", ".join(f"{r['element']} {r['mae_left_out']:.2f} V" for r in worst)
    R = S["li_model_range"]
    N["gb_label_min"], N["gb_label_max"] = f2(R["label_min"]), f2(R["label_max"])
    N["gb_pred_min"], N["gb_pred_max"] = signed(R["pred_min"]), f2(R["pred_max"])
    dis, chg, ion = R["pred_min_reaction"]
    N["gb_min_reaction"] = f"{html_formula(dis)} → {html_formula(chg)} with {ion} as the working ion"
    # DFT
    N["V_pbe"], N["V_pbeu"] = f"{V['PBE|O1|base']:.3f}", f"{V['PBE+U|O1|base']:.3f}"
    N["V_pbeu_o3"] = f"{V['PBE+U|O3|base']:.3f}"
    N["V_pbe_conv"], N["V_pbeu_conv"] = f"{V['PBE|O1|converged']:.3f}", f"{V['PBE+U|O1|converged']:.3f}"
    conv = max(abs(V["PBE|O1|converged"] - V["PBE|O1|base"]), abs(V["PBE+U|O1|converged"] - V["PBE+U|O1|base"]))
    N["conv_mV"] = f"{1000 * conv:.0f}"
    minutes = [r["minutes"] for r in vt.load_dft().values()]
    N["dft_minutes"] = f"{min(minutes):.2f}–{max(minutes):.1f} min"
    co_o, li_o = dft_geometry()
    N["co_o_plane"], N["li_o_plane"] = f2(co_o), f2(li_o)
    # CHGNet
    fx, fr, sw = C["fixed_lattice"], C["cell_relaxed"], C["swapped_start"]
    N["chg_li_fixed"], N["chg_na_fixed"] = f2(fx["V_Li"]), f2(fx["V_Na"])
    N["chg_li_free"], N["chg_na_free"] = f2(fr["V_Li"]), f2(fr["V_Na"])
    N["chg_na_swapped"] = f2(sw["V_Na"])
    o1 = fr["energies"]["CoO2-O1"]
    N["coo2_swell"] = f"{100 * (o1['volume_per_atom'] / o1['volume_per_atom_start'] - 1):.0f} %"
    N["naco_co_o_swapped"], N["naco_co_o"] = (
        f2(sw["energies"]["NaCoO2"]["Co_O_A"]),
        f2(fr["energies"]["NaCoO2"]["Co_O_A"]),
    )
    N["naco_dE_meV"] = f"{1000 * (fr['energies']['NaCoO2']['E_per_fu'] - sw['energies']['NaCoO2']['E_per_fu']):.0f}"
    # MD
    A, T = DF["all_runs"]["arrhenius"], DF["all_runs"]["temperatures"]
    N["md_runs"] = str(len(DF["runs"]))
    N["md_Ea"], N["md_Ea_se"] = f"{A['Ea_eV']:.3f}", f"{A['Ea_stderr_eV']:.3f}"
    N["md_Ea_all"], N["md_Ea_all_se"] = f"{A['Ea_all_runs_eV']:.3f}", f"{A['Ea_all_runs_stderr_eV']:.3f}"
    At = DF["first_runs"]["arrhenius"]
    N["md_Ea_first"], N["md_Ea_first_se"] = f"{At['Ea_eV']:.3f}", f"{At['Ea_stderr_eV']:.3f}"
    N["md_D300"] = sci(A["D_300K_extrapolated_cm2_s"])
    for t in T:
        N[f"md_D{int(t['T'])}"] = sci(t["D_cm2_s"])
        N[f"md_D{int(t['T'])}_sd"] = sci(t["D_sd_cm2_s"])
    hot = T[-1]
    N["md_aniso_800"] = f"{hot['D_ab_cm2_s'] / hot['D_c_cm2_s']:.1f}"
    # page check cases and cost
    W = _json("web_check.json")
    N["n_web_cases"] = str(len(W))
    na = next(c for c in W if c["named"] and c["discharged"] == "NaCoO2")
    N["nacoo2_pred"] = f2(na["pred"])
    N["nacoo2_db"] = f2(next(x["v"] for x in L["lookup"]["NaCoO2"] if x["charged"] == "CoO2"))
    td = json.loads((paths.CALC / "timing" / "timing_dft.json").read_text(encoding="utf-8"))["runs"]
    tc = json.loads((paths.CALC / "timing" / "timing_chgnet.json").read_text(encoding="utf-8"))["runs"]
    big = max(td, key=lambda r: r["n_atoms"])
    t_chg = float(np.interp(big["n_atoms"], [r["n_atoms"] for r in tc], [r["seconds"] for r in tc]))
    N["t_dft_min"], N["n_dft_min"] = f"{td[0]['seconds']:.0f}", str(td[0]["n_atoms"])
    N["t_dft_max"], N["n_dft_max"] = f"{big['seconds']:.0f}", str(big["n_atoms"])
    N["t_chg_max"], N["dft_chg_ratio"] = f"{t_chg:.2f}", f"{big['seconds'] / t_chg:.0f}"
    N["t_ml_ms"] = f"{1000 * L['speed']['seconds_per_reaction']:.1f}"
    N["timing_minutes"] = f"~{(sum(r['seconds'] for r in td) + 6 * sum(r['seconds'] for r in tc)) / 60:.0f} min"
    N["lit_gga"], N["lit_ggau"], N["lit_hse"] = (f2(LITERATURE[k]) for k in ("GGA", "GGA+U", "HSE06"))
    N["exp"] = f"{LITERATURE['experiment']:.1f}"
    return N
