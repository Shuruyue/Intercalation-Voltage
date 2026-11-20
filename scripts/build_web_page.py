"""Build web/intercalation-voltage.html from web/src/: inline core.js and the data block.

The data block holds the exported Li model (build/web_model.json, from scripts/export_web_model.py), every reaction of
the cleaned snapshot (for the database readout and the nearest neighbours), the distance histograms and the numbers
of the About tab (intercalation_voltage.summary).

Usage: python scripts/build_web_page.py
"""

import json

import _path  # noqa: F401
import numpy as np

from intercalation_voltage import data as dt
from intercalation_voltage import paths, summary

SRC = paths.WEB / "src"
PAGE = paths.WEB / "intercalation-voltage.html"
MODEL = paths.ROOT / "build" / "web_model.json"
REPO = "https://github.com/Shuruyue/Intercalation-Voltage"


def about(N, L):
    lc = L["learning_curve"]
    return f"""
<h2>What the model is</h2>
<p>Gradient boosting (scikit-learn; 400 trees of depth 4, learning rate 0.05, subsample 0.8) on 266 composition features:
Magpie statistics of 22 element properties of the discharged and of the charged composition, plus two terms for the
amount of working ion inserted. It is trained on the {N["n_li"]} Li reactions of the Materials Project insertion-electrode
snapshot of 2019-12-03, after removing voltages outside 0–6 V and end members more than 0.2 eV/atom above the convex hull,
and merging polymorphs. Every label is a GGA+U calculation, not a measurement.</p>
<h2>How well it does</h2>
<table>
<tr><th>Test</th><th class="num">MAE (V)</th><th>Note</th></tr>
<tr><td>Unseen Li host systems</td><td class="num">{N["mae_li"]}</td><td>5-fold split that keeps each host system on one side; R² = {N["r2_li"]}</td></tr>
<tr><td>Same, random split</td><td class="num">{N["mae_li_random"]}</td><td>similar hosts on both sides: optimistic</td></tr>
<tr><td>Always the training mean</td><td class="num">{N["mae_li_dummy"]}</td><td>baseline</td></tr>
<tr><td>Na, trained on Li only</td><td class="num">{N["mae_na"]}</td><td>all {N["n_na"]} Na reactions; mean error {N["bias_na"]} V</td></tr>
<tr><td>Na, trained on every other ion</td><td class="num">{N["mae_others"]}</td><td>{N["n_others"]} reactions of Li, Mg, Ca, Zn, K, …, no Na</td></tr>
</table>
<p>{N["pairs_n"]} hosts appear with both Li and Na. In the database the Na version is {N["pairs_data_abs"]} V lower on average;
the Li-trained model puts it {N["pairs_pred_abs"]} V lower. Adding Na reactions to the training data helps
({N["lc_k0"]} V with none, {N["lc_k100"]} V with {lc["k"][-1]}, on held-out Na host systems); so do other ions without any Na
({N["lc_others"]} V), and that gain is not just the larger training set: the other-ion data cut to the size of the Li set
give {N["lc_size_matched"]} V.</p>
<h2>The warning light</h2>
<p>Distance = mean Euclidean distance to the 5 nearest Li training reactions in standardized feature space ({N["n_feat_dist"]}
of the 266 features; {N["n_const"]} never vary in the Li data and cannot be standardized). The light turns red above
{N["threshold"]}, the 95th percentile of the distances of unseen Li hosts to their training folds. {N["flag_na"]} % of the Na
reactions are red. Among unseen Li hosts, red ones have a mean error of {N["mae_li_flagged"]} V, the others {N["mae_li_noflag"]} V.
Within Na the distance does not rank the errors (Spearman ρ = {N["rho_na"]}): the light says that a composition is new, not
how wrong the number is.</p>
<h2>Limits</h2>
<ul>
<li>The model learns GGA+U voltages. For LiCoO<sub>2</sub> over the full range, our own PBE+U calculation gives {N["V_pbeu"]} V,
experiment about {N["exp"]} V; the model cannot be closer to experiment than its labels except by chance.</li>
<li>Composition features cannot tell polymorphs apart; the snapshot's polymorphs were merged by their median voltage.</li>
<li>The 2019 snapshot is fixed so that every number can be reproduced; today's Materials Project has more and recomputed entries.</li>
</ul>
<p>Source code, data, calculations and checks: <a href="{REPO}">{REPO.replace("https://", "")}</a>.</p>
"""


def main():
    M = json.loads(MODEL.read_text(encoding="utf-8"))
    clean = dt.load_clean()
    rows = [
        [d, c, i, round(float(v), 4)]
        for d, c, i, v in zip(
            clean.formula_discharge, clean.formula_charge, clean.working_ion, clean.average_voltage, strict=True
        )
    ]
    A = np.load(paths.result("distance_arrays.npz"))
    top = max(100.0, float(max(A["cv_dist"].max(), A["d_na"].max())))
    edges = np.logspace(0, np.log10(top), 41)
    N = summary.compute()
    L = json.loads(paths.result("learning.json").read_text(encoding="utf-8"))
    keys = [
        "n_li",
        "n_na",
        "n_groups_li",
        "mae_li",
        "mae_na",
        "mae_li_flagged",
        "mae_li_noflag",
        "pairs_pred_abs",
        "pairs_data_abs",
        "flag_na",
    ]
    D = {
        "model": M,
        "rows": rows,
        "hist": {
            "edges": [float(e) for e in edges],
            "li": np.histogram(A["cv_dist"], edges)[0].tolist(),
            "na": np.histogram(A["d_na"], edges)[0].tolist(),
        },
        "summary": {k: N[k] for k in keys},
        "about_html": about(N, L),
    }
    page = (SRC / "intercalation-voltage.src.html").read_text(encoding="utf-8")
    core = (SRC / "core.js").read_text(encoding="utf-8")
    data = "window.VOLTAGE_DATA = " + json.dumps(D, separators=(",", ":")).replace("</", "<\\/") + ";"
    for key, val in (("/*__CORE__*/", core), ("/*__DATA__*/", data)):
        assert page.count(key) == 1, key
        page = page.replace(key, val)
    PAGE.write_text(page, encoding="utf-8", newline="\n")
    print(PAGE, f"{len(page) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
