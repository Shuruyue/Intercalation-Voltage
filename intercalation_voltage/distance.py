"""Distance to the training data: a warning light for inputs unlike anything the Li model was trained on.

Distance of a reaction = mean Euclidean distance to its k = 5 nearest Li training reactions in standardized feature
space. Features that are constant over the Li data (28 of 266) are left out: a standard deviation of zero cannot
scale them. Threshold = 95th percentile of the distances of Li test reactions to their own GroupKFold training folds
(the folds of learning.group_cv), so by construction 5 % of unseen Li hosts get a red light.
"""

import json

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from . import data as dt
from . import paths

DISTANCE_JSON = paths.result("distance.json")
DISTANCE_ARRAYS_NPZ = paths.result("distance_arrays.npz")

K_NN = 5
PERCENTILE = 95
N_BINS = 6


def kept_features(Xl):
    return Xl.std(axis=0) >= 1e-9


def fit_reference(Xl):
    """Scaler and neighbour index on all Li reactions (what the web page uses)."""
    keep = kept_features(Xl)
    sc = StandardScaler().fit(Xl[:, keep])
    nn = NearestNeighbors(n_neighbors=K_NN).fit(sc.transform(Xl[:, keep]))
    return keep, sc, nn


def cv_distances(Xl, yl, gl):
    keep = kept_features(Xl)
    d = np.zeros(len(yl))
    for tr, te in GroupKFold(n_splits=5).split(Xl, yl, gl):
        k_tr = keep & (Xl[tr].std(axis=0) > 1e-9)
        sc = StandardScaler().fit(Xl[tr][:, k_tr])
        nn = NearestNeighbors(n_neighbors=K_NN).fit(sc.transform(Xl[tr][:, k_tr]))
        d[te] = nn.kneighbors(sc.transform(Xl[te][:, k_tr]))[0].mean(axis=1)
    return d


def run(out_json=DISTANCE_JSON, arrays_npz=DISTANCE_ARRAYS_NPZ, verbose=True):
    d = dt.load()
    X, y, ion, group, names = d["X"], d["y"], d["ion"], d["group"], d["names"]
    li, na = ion == "Li", ion == "Na"
    Xl, yl, gl = X[li], y[li], group[li]
    A = np.load(paths.result("learning_arrays.npz"))
    err_li, err_na = A["cv_pred_group"] - yl, A["p_na"] - y[na]

    keep, sc, nn = fit_reference(Xl)
    const = ~keep
    cv_dist = cv_distances(Xl, yl, gl)
    d_na = nn.kneighbors(sc.transform(X[na][:, keep]))[0].mean(axis=1)
    thr = float(np.percentile(cv_dist, PERCENTILE))
    flag_li, flag_na = cv_dist > thr, d_na > thr
    D = {
        "k": K_NN,
        "percentile": PERCENTILE,
        "n_features_used": int(keep.sum()),
        "n_constant_in_li": int(const.sum()),
        "constant_features_in_li": [str(n) for n in names[const]],
        "threshold": thr,
        "median_li_cv": float(np.median(cv_dist)),
        "median_na": float(np.median(d_na)),
        "frac_na_flagged": float(flag_na.mean()),
        "frac_li_flagged": float(flag_li.mean()),
        "mae_na_flagged": float(np.mean(np.abs(err_na[flag_na]))),
        "mae_na_not_flagged": float(np.mean(np.abs(err_na[~flag_na]))),
        "mae_li_flagged": float(np.mean(np.abs(err_li[flag_li]))),
        "mae_li_not_flagged": float(np.mean(np.abs(err_li[~flag_li]))),
        "spearman_dist_abs_err_na": float(pd.Series(d_na).corr(pd.Series(np.abs(err_na)), method="spearman")),
        "spearman_dist_abs_err_li": float(pd.Series(cv_dist).corr(pd.Series(np.abs(err_li)), method="spearman")),
        "frac_na_differs_in_a_constant_feature": float(
            (np.abs(X[na][:, const] - Xl[0, const]) > 1e-9).any(axis=1).mean()
        ),
    }
    # pooled bins of equal count (Li test + Na); the Na share of each bin is reported because it changes along them
    all_d, all_e = np.concatenate([cv_dist, d_na]), np.abs(np.concatenate([err_li, err_na]))
    is_na = np.r_[np.zeros(len(cv_dist)), np.ones(len(d_na))]
    edges = np.quantile(all_d, np.linspace(0, 1, N_BINS + 1))
    D["bins"] = []
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        s = (all_d >= a) & (all_d <= b)
        D["bins"].append(
            {
                "d_lo": float(a),
                "d_hi": float(b),
                "mae": float(all_e[s].mean()),
                "n": int(s.sum()),
                "frac_na": float(is_na[s].mean()),
                "mae_li": float(all_e[s & (is_na == 0)].mean()) if np.any(s & (is_na == 0)) else None,
            }
        )
    pca = PCA(n_components=2, random_state=0).fit(sc.transform(Xl[:, keep]))
    D["pca_explained"] = [float(v) for v in pca.explained_variance_ratio_]
    with open(out_json, "w", encoding="utf-8", newline="\n") as f:
        json.dump(D, f, indent=1)
    np.savez(
        arrays_npz,
        cv_dist=cv_dist,
        d_na=d_na,
        pc_li=pca.transform(sc.transform(Xl[:, keep])),
        pc_na=pca.transform(sc.transform(X[na][:, keep])),
    )
    if verbose:
        print(json.dumps({k: v for k, v in D.items() if k not in ("bins", "constant_features_in_li")}, indent=1))
    return D
