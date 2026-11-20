"""Using the Li model to find materials, and how far "unseen host system" is from "unseen chemistry".

1. Screening. The out-of-fold predictions of learning.group_cv are what a screen of host systems the model has not seen
   returns. Target: an average voltage of at least 4.0 V (a high-voltage cathode). For the 10 % of candidates ranked
   highest and for every candidate predicted at or above the target: precision (share truly at or above it), recall
   (share of all such reactions that is found) and enrichment (precision over the share in the whole pool). The same
   for Na with the Li-only model, and the gain curve: share of target reactions found against share of candidates
   checked in ranked order.
2. Accuracy: share of the out-of-fold errors within 0.3, 0.5 and 1 V, and the median error.
3. Leave one transition metal out: for each transition metal in at least 30 Li reactions, train on the Li reactions
   without it and test on those with it, against the group-split error on the same reactions. A reaction with two such
   metals is tested once for each.
4. Range of the Li model over the whole snapshot against the range of its labels. Boosting adds trees fitted to
   residuals, so the sum is not confined to the range of the labels, although it never follows a trend beyond them.
"""

import json
from collections import Counter

import numpy as np
from pymatgen.core import Composition

from . import data as dt
from . import learning as ln
from . import paths

SCREENING_JSON = paths.result("screening.json")
SCREENING_ARRAYS_NPZ = paths.result("screening_arrays.npz")

V_TARGET = 4.0  # V vs the metal of the working ion
TOP_SHARE = 0.10
ERR_LEVELS = (0.3, 0.5, 1.0)
MIN_REACTIONS = 30


def screen(y, p, target=V_TARGET, top_share=TOP_SHARE):
    """Precision, recall and enrichment of two selections: the top share by prediction, and prediction >= target.

    Precision and enrichment are None when a selection is empty.
    """
    hit = y >= target
    top = np.zeros(len(y), dtype=bool)
    top[np.argsort(-p, kind="stable")[: int(round(top_share * len(y)))]] = True
    out = {"n": int(len(y)), "n_target": int(hit.sum()), "base_rate": float(hit.mean())}
    for name, sel in (("top", top), ("predicted_above", p >= target)):
        tp, n_sel = int((sel & hit).sum()), int(sel.sum())
        precision = tp / n_sel if n_sel else None
        out[name] = {
            "n_selected": n_sel,
            "precision": precision,
            "recall": float(tp / hit.sum()),
            "enrichment": precision / float(hit.mean()) if n_sel else None,
        }
    return out


def gain_curve(y, p, target=V_TARGET):
    """Share of the target reactions found after checking 0, 1, 2, ... candidates in order of decreasing prediction."""
    hit = (y >= target)[np.argsort(-p, kind="stable")]
    return np.r_[0.0, np.cumsum(hit) / hit.sum()]


def error_shares(y, p):
    e = np.abs(np.asarray(p) - np.asarray(y))
    return {"median": float(np.median(e)), **{f"within_{t}": float(np.mean(e <= t)) for t in ERR_LEVELS}}


def transition_metals(formulas, min_n=MIN_REACTIONS):
    """Transition metals of each formula, and those present in at least min_n formulas (most frequent first)."""
    sets = [{str(e) for e in Composition(f).elements if e.is_transition_metal} for f in formulas]
    counts = Counter(e for s in sets for e in s)
    return sets, sorted((e for e, n in counts.items() if n >= min_n), key=lambda e: (-counts[e], e))


def leave_element_out(X, y, oof, formulas):
    sets, elements = transition_metals(formulas)
    rows, err_out, err_group = [], [], []
    for el in elements:
        has = np.array([el in s for s in sets])
        a = np.abs(ln.model().fit(X[~has], y[~has]).predict(X[has]) - y[has])
        b = np.abs(oof[has] - y[has])
        rows.append(
            {
                "element": el,
                "n": int(has.sum()),
                "mae_left_out": float(a.mean()),
                "mae_group_split": float(b.mean()),
                "frac_err_gt_1V": float(np.mean(a > 1.0)),
            }
        )
        err_out.append(a)
        err_group.append(b)
    a, b = np.concatenate(err_out), np.concatenate(err_group)
    return {
        "min_reactions": MIN_REACTIONS,
        "elements": rows,
        "n_pairs": int(len(a)),
        "mae_left_out": float(a.mean()),
        "mae_group_split": float(b.mean()),
    }


def run(out_json=SCREENING_JSON, arrays_npz=SCREENING_ARRAYS_NPZ, verbose=True):
    d, clean = dt.load(), dt.load_clean()
    li = d["ion"] == "Li"
    A = np.load(paths.result("learning_arrays.npz"))
    yl, oof, yn, pn = A["yl"], A["cv_pred_group"], A["y_na"], A["p_na"]
    if not np.array_equal(yl, d["y"][li]):
        raise RuntimeError("results/learning_arrays.npz does not belong to data/features.npz: run learning.run() first")

    out = {
        "target_V": V_TARGET,
        "top_share": TOP_SHARE,
        "li_unseen_hosts": screen(yl, oof),
        "na_li_model": screen(yn, pn),
        "errors": {"li_unseen_hosts": error_shares(yl, oof), "na_li_model": error_shares(yn, pn)},
    }
    formulas = clean.formula_discharge[li].tolist()
    out["leave_element_out"] = leave_element_out(d["X"][li], yl, oof, formulas)

    p_all = ln.fit_li(d).predict(d["X"])
    lo, hi = int(np.argmin(p_all)), int(np.argmax(p_all))

    def reaction(i):
        return [clean.formula_discharge[i], clean.formula_charge[i], clean.working_ion[i]]

    out["li_model_range"] = {
        "label_min": float(yl.min()),
        "label_max": float(yl.max()),
        "pred_min": float(p_all[lo]),
        "pred_max": float(p_all[hi]),
        "pred_min_reaction": reaction(lo),
        "pred_max_reaction": reaction(hi),
        "n_outside_label_range": int(np.sum((p_all < yl.min()) | (p_all > yl.max()))),
    }
    with open(out_json, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
    np.savez(arrays_npz, gain_li=gain_curve(yl, oof), gain_na=gain_curve(yn, pn))
    if verbose:
        print(json.dumps({k: v for k, v in out.items() if k != "leave_element_out"}, indent=1))
        for r in out["leave_element_out"]["elements"]:
            mae = f"left out {r['mae_left_out']:.3f} V  group {r['mae_group_split']:.3f} V"
            print(f"{r['element']:3s} n={r['n']:4d}  {mae}")
    return out
