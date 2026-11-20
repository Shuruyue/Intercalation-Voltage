"""Gradient-boosting experiments on the cleaned snapshot. Deterministic: every model has random_state 0.

1. Li inside its range: 5-fold GroupKFold by host system (no host in both training and test), a random 5-fold split
   for comparison, and a guess-the-training-mean baseline.
2. Li -> Na: trained on every Li reaction, tested on every Na reaction.
3. Other ions -> Na: trained on every reaction except Na (Li, Mg, Ca, Zn, K, ...).
4. Matched hosts: Na and Li reactions whose (charged, discharged) compositions coincide once Na is written as Li;
   the Na - Li voltage shift in the data against the shift the Li-trained model predicts.
5. Learning curve: 10 splits, each holding out 40 % of the Na host systems; Li plus k Na reactions from the rest.
   On the same held-out sets: the other-ions model, and a size control, the other-ions model trained on a random
   subset as large as the Li set, which separates "more kinds of ion" from "more data".
6. Speed of a prediction, and a toy fit showing that trees do not extrapolate.
"""

import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold, KFold

from . import data as dt
from . import features as ft
from . import paths

LEARNING_JSON = paths.result("learning.json")
LEARNING_ARRAYS_NPZ = paths.result("learning_arrays.npz")
LI_NA_PAIRS_CSV = paths.result("li_na_pairs.csv")

SEED = 0
PARAMS = {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 4, "subsample": 0.8, "random_state": SEED}
LC_K = [0, 10, 25, 50, 100]
LC_SPLITS = 10
LC_TEST_SHARE = 0.4


def model():
    return GradientBoostingRegressor(**PARAMS)


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def r2(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def fit_li(d=None):
    """The Li-only model used for the Li -> Na test and the web page."""
    d = d or dt.load()
    li = d["ion"] == "Li"
    return model().fit(d["X"][li], d["y"][li])


def group_cv(X, y, groups):
    """Out-of-fold predictions of 5-fold GroupKFold and the guess-the-training-mean baseline."""
    pred, dummy = np.zeros_like(y), np.zeros_like(y)
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        pred[te] = model().fit(X[tr], y[tr]).predict(X[te])
        dummy[te] = y[tr].mean()
    return pred, dummy


def lc_splits(group, na):
    """Held-out Na reactions and the pool of addable Na reactions for each learning-curve split."""
    na_idx = np.where(na)[0]
    na_groups = np.array(sorted(set(group[na])))
    out = []
    for s in range(LC_SPLITS):
        rng = np.random.default_rng(100 + s)
        perm = rng.permutation(na_groups)
        n_test_g = int(round(LC_TEST_SHARE * len(perm)))
        test_g, pool_g = set(perm[:n_test_g]), set(perm[n_test_g:])
        test = np.array([i for i in na_idx if group[i] in test_g])
        pool = rng.permutation(np.array([i for i in na_idx if group[i] in pool_g]))
        out.append((test, pool))
    return out


def matched_pairs(clean, p_in, p_oof_li):
    """Na reactions with a Li reaction on the same host; one row per Na reaction (median over duplicate Li rows)."""
    cl = clean.copy()
    cl["dis_as_li"] = [ft.swap_ion(f, i, "Li") for f, i in zip(cl.formula_discharge, cl.working_ion, strict=True)]
    cl["chg_as_li"] = [ft.swap_ion(f, i, "Li") for f, i in zip(cl.formula_charge, cl.working_ion, strict=True)]
    li = (cl.working_ion == "Li").values
    cl["p_in"] = p_in
    cl["p_oof"] = np.nan
    cl.loc[li, "p_oof"] = p_oof_li
    L = cl[li].set_index(["chg_as_li", "dis_as_li"])
    rows = []
    for r in cl[cl.working_ion == "Na"].itertuples():
        k = (r.chg_as_li, r.dis_as_li)
        if k in L.index:
            m = L.loc[[k]]
            rows.append(
                {
                    "na": r.formula_discharge,
                    "na_charged": r.formula_charge,
                    "li": r.dis_as_li,
                    "v_na": r.average_voltage,
                    "v_li": float(m["average_voltage"].median()),
                    "pred_na": r.p_in,
                    "pred_li_insample": float(m["p_in"].median()),
                    "pred_li_oof": float(m["p_oof"].median()),
                }
            )
    return pd.DataFrame(rows)


def run(out_json=LEARNING_JSON, arrays_npz=LEARNING_ARRAYS_NPZ, pairs_csv=LI_NA_PAIRS_CSV, verbose=True):
    d, clean = dt.load(), dt.load_clean()
    X, y, ion, group, names = d["X"], d["y"], d["ion"], d["group"], d["names"]
    li, na = ion == "Li", ion == "Na"
    Xl, yl, gl = X[li], y[li], group[li]
    out = {
        "params": PARAMS,
        "n_li": int(li.sum()),
        "n_na": int(na.sum()),
        "n_total": int(len(y)),
        "n_groups_li": int(len(set(gl))),
        "n_groups_na": int(len(set(group[na]))),
        "ion_counts": {k: int(v) for k, v in pd.Series(ion).value_counts().items()},
        "voltage_stats": {
            k: {
                "mean": float(y[m].mean()),
                "std": float(y[m].std()),
                "min": float(y[m].min()),
                "max": float(y[m].max()),
            }
            for k, m in [("Li", li), ("Na", na)]
        },
    }

    # 1. Li inside its range
    cv_group, dummy = group_cv(Xl, yl, gl)
    cv_random = np.zeros_like(yl)
    for tr, te in KFold(n_splits=5, shuffle=True, random_state=SEED).split(Xl):
        cv_random[te] = model().fit(Xl[tr], yl[tr]).predict(Xl[te])
    out["li_cv"] = {
        "group_mae": mae(yl, cv_group),
        "group_r2": r2(yl, cv_group),
        "random_mae": mae(yl, cv_random),
        "random_r2": r2(yl, cv_random),
        "dummy_mae": mae(yl, dummy),
        "frac_err_gt_1V": float(np.mean(np.abs(cv_group - yl) > 1.0)),
    }

    # 2. Li -> Na
    t0 = time.time()
    m_li = model().fit(Xl, yl)
    out["fit_seconds_li"] = time.time() - t0
    p_na = m_li.predict(X[na])
    err_na = p_na - y[na]
    out["li_to_na"] = {
        "mae": mae(y[na], p_na),
        "r2": r2(y[na], p_na),
        "bias": float(err_na.mean()),
        "dummy_mae": mae(y[na], np.full(na.sum(), yl.mean())),
        "frac_err_gt_1V": float(np.mean(np.abs(err_na) > 1.0)),
    }

    # 3. other ions -> Na
    other = ~na
    m_oth = model().fit(X[other], y[other])
    p_na_oth = m_oth.predict(X[na])
    out["others_to_na"] = {
        "mae": mae(y[na], p_na_oth),
        "r2": r2(y[na], p_na_oth),
        "bias": float((p_na_oth - y[na]).mean()),
        "n_train": int(other.sum()),
    }

    # 4. matched hosts
    P = matched_pairs(clean, m_li.predict(X), cv_group)
    P.to_csv(pairs_csv, index=False, lineterminator="\n")
    out["pairs"] = {
        "n": int(len(P)),
        "data_mean_vna_minus_vli": float((P.v_na - P.v_li).mean()),
        "data_std_vna_minus_vli": float((P.v_na - P.v_li).std()),
        "frac_na_lower": float(np.mean(P.v_na < P.v_li)),
        "pred_na_minus_data_li": float((P.pred_na - P.v_li).mean()),
        "pred_na_minus_pred_li_insample": float((P.pred_na - P.pred_li_insample).mean()),
        "pred_na_minus_pred_li_oof": float((P.pred_na - P.pred_li_oof).mean()),
    }

    # which features of Na leave the Li range; importance in the Li model
    lo, hi = Xl.min(axis=0), Xl.max(axis=0)
    outside = (X[na] < lo) | (X[na] > hi)
    frac_out = outside.mean(axis=0)
    out["features_outside_li_range"] = [
        {"name": str(names[i]), "frac_na_outside": float(frac_out[i])} for i in np.argsort(-frac_out, kind="stable")[:8]
    ]
    out["frac_na_with_any_feature_outside"] = float(outside.any(axis=1).mean())
    imp = m_li.feature_importances_
    out["top_features_li_model"] = [
        {"name": str(names[i]), "importance": float(imp[i])} for i in np.argsort(-imp, kind="stable")[:10]
    ]

    # 5. learning curve, other ions on the same held-out sets, size control
    li_idx, other_idx = np.where(li)[0], np.where(other)[0]
    lc = {k: [] for k in LC_K}
    oth_same, oth_size = [], []
    for s, (test, pool) in enumerate(lc_splits(group, na)):
        for k in LC_K:
            tr = np.concatenate([li_idx, pool[:k]]).astype(int)
            lc[k].append(mae(y[test], model().fit(X[tr], y[tr]).predict(X[test])))
        oth_same.append(mae(y[test], m_oth.predict(X[test])))
        sub = np.sort(np.random.default_rng(200 + s).choice(other_idx, size=len(li_idx), replace=False))
        oth_size.append(mae(y[test], model().fit(X[sub], y[sub]).predict(X[test])))
    out["learning_curve"] = {
        "k": LC_K,
        "mae_mean": [float(np.mean(lc[k])) for k in LC_K],
        "mae_std": [float(np.std(lc[k])) for k in LC_K],
        "splits": LC_SPLITS,
        "test_share_of_na_hosts": LC_TEST_SHARE,
        "others_mae_mean": float(np.mean(oth_same)),
        "others_mae_std": float(np.std(oth_same)),
        "others_size_matched_mae_mean": float(np.mean(oth_size)),
        "others_size_matched_mae_std": float(np.std(oth_size)),
        "size_matched_n_train": int(len(li_idx)),
    }

    # 6. speed (machine dependent) and the toy
    Xbig = np.repeat(X[na], 40, axis=0)[:10000]
    t0 = time.time()
    m_li.predict(Xbig)
    t_pred = (time.time() - t0) / len(Xbig)
    rx = list(zip(clean.formula_discharge[:200], clean.formula_charge[:200], strict=True))
    t0 = time.time()
    for fd, fc in rx:
        ft.magpie_vector(fd)
        ft.magpie_vector(fc)
    t_feat = (time.time() - t0) / len(rx)
    out["speed"] = {
        "predict_seconds_per_reaction": t_pred,
        "featurize_seconds_per_reaction": t_feat,
        "seconds_per_reaction": t_pred + t_feat,
        "note": "two compositions featurized per reaction",
    }
    xt = np.linspace(0, 1, 50)[:, None]
    toy = model().fit(xt, 2 * xt.ravel())
    xq = np.linspace(0, 2, 81)[:, None]
    out["toy"] = {"pred_at_2": float(toy.predict([[2.0]])[0]), "true_at_2": 4.0, "train_y_max": 2.0}

    lookup = {}
    for f in ["LiCoO2", "NaCoO2", "LiFePO4", "NaFePO4"]:
        s = clean[clean.formula_discharge == f]
        lookup[f] = [{"charged": c, "v": float(v)} for c, v in zip(s.formula_charge, s.average_voltage, strict=True)]
    out["lookup"] = lookup

    with open(out_json, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
    np.savez(
        arrays_npz,
        yl=yl,
        cv_pred_group=cv_group,
        cv_pred_random=cv_random,
        y_na=y[na],
        p_na=p_na,
        p_na_oth=p_na_oth,
        xq=xq.ravel(),
        toy_pred=toy.predict(xq),
        xt=xt.ravel(),
        imp=imp,
    )
    if verbose:
        print(
            json.dumps({k: out[k] for k in ["li_cv", "li_to_na", "others_to_na", "pairs", "learning_curve"]}, indent=1)
        )
    return out
