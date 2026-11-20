"""The Li-trained model and the distance check in a form the web page runs in JavaScript, plus test cases.

The page (web/src/core.js) recomputes the 266 features from the formulas, walks the 400 trees and searches the 5
nearest Li training reactions. results/web_check.json holds Python's prediction and distance for 300 random
reactions of the snapshot and a few named ones; tests/web_model.test.js checks the
JavaScript against them.
"""

import base64
import json

import numpy as np
from pymatgen.core import Element

from . import data as dt
from . import distance as ds
from . import features as ft
from . import learning as ln
from . import paths

WEB_CHECK_JSON = paths.result("web_check.json")

NAMED = [
    ("LiCoO2", "CoO2", "Li"),
    ("LiFePO4", "FePO4", "Li"),
    ("LiMn2O4", "Mn2O4", "Li"),
    ("LiNiO2", "NiO2", "Li"),
    ("Li3V2(PO4)3", "V2(PO4)3", "Li"),
    ("LiFeSO4F", "FeSO4F", "Li"),
    ("NaCoO2", "CoO2", "Na"),
    ("NaFePO4", "FePO4", "Na"),
    ("Na3V2(PO4)3", "V2(PO4)3", "Na"),
    ("NaCrO2", "CrO2", "Na"),
    ("Li2(CoO2)3", "Li(CoO2)3", "Li"),
    ("KCoO2", "CoO2", "K"),
    ("MgMn2O4", "Mn2O4", "Mg"),
]
N_RANDOM, CHECK_SEED = 300, 0


def element_table():
    ep = ft.magpie()
    table = {}
    for z in range(1, 104):
        el = Element.from_Z(z)
        try:
            vals = [float(ep.data_source.get_elemental_property(el, p)) for p in ep.features]
        except Exception:  # noqa: BLE001 - matminer raises different errors for missing data
            continue
        if all(np.isfinite(vals)):
            table[el.symbol] = vals
    return table


def trees(m):
    out = []
    for est in m.estimators_[:, 0]:
        t = est.tree_
        out.append(
            {
                "f": t.feature.tolist(),
                "t": [float(v) for v in t.threshold],
                "l": t.children_left.tolist(),
                "r": t.children_right.tolist(),
                "v": [float(v) for v in t.value[:, 0, 0]],
            }
        )
    return out


def export(model_json=paths.ROOT / "build" / "web_model.json", check_json=WEB_CHECK_JSON):
    d, clean = dt.load(), dt.load_clean()
    li = d["ion"] == "Li"
    Xl, yl = d["X"][li], d["y"][li]
    m = ln.fit_li(d)
    keep, sc, _ = ds.fit_reference(Xl)
    z_train = sc.transform(Xl[:, keep]).astype(np.float32)
    thr = json.loads(paths.result("distance.json").read_text(encoding="utf-8"))["threshold"]
    ep = ft.magpie()
    M = {
        "props": ep.features,
        "stats": ep.stats,
        "elements": element_table(),
        "init": float(m.init_.constant_.ravel()[0]),
        "lr": float(m.learning_rate),
        "trees": trees(m),
        "keep": np.where(keep)[0].tolist(),
        "mean": sc.mean_.tolist(),
        "scale": sc.scale_.tolist(),
        "train_b64": base64.b64encode(z_train.tobytes()).decode(),
        "n_train": int(z_train.shape[0]),
        "n_keep": int(keep.sum()),
        "k": ds.K_NN,
        "threshold": thr,
        "train_v_range": [float(yl.min()), float(yl.max())],
    }
    model_json.parent.mkdir(exist_ok=True)
    model_json.write_text(json.dumps(M), encoding="utf-8", newline="\n")

    # the distance as the page computes it: standardized query against the float32 training matrix
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=ds.K_NN).fit(z_train.astype(np.float64))
    rows = np.random.default_rng(CHECK_SEED).choice(len(clean), size=N_RANDOM, replace=False)
    cases = [(clean.formula_discharge[i], clean.formula_charge[i], clean.working_ion[i]) for i in sorted(rows)]
    out = []
    for dis, chg, ion in cases + NAMED:
        x = ft.reaction_vector(dis, chg, ion)
        out.append(
            {
                "discharged": dis,
                "charged": chg,
                "ion": ion,
                "pred": float(m.predict([x])[0]),
                "dist": float(nn.kneighbors(sc.transform([x[keep]]))[0].mean()),
                "named": (dis, chg, ion) in NAMED,
            }
        )
    check_json.write_text(json.dumps(out, indent=0), encoding="utf-8", newline="\n")
    return M, out
