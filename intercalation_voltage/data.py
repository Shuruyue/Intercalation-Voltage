"""Cleaning of the Materials Project insertion-electrode snapshot and the feature matrix.

Input : data/mp_insertion_electrodes_2019-12-03.csv, the Materials Project battery explorer as of 2019-12-03 (5,133
        voltage steps; distributed with github.com/nissy-dev/machine-learning-the-voltage-of-electrode-materials,
        a reproduction of Joshi et al. 2019). Every voltage in it is a GGA/GGA+U average voltage, not a measurement.
Output: data/clean.csv (one row per unique reaction step), data/features.npz (X, y, ion, group, names) and
        results/data_log.json (row counts after each filter).
"""

import json

import numpy as np
import pandas as pd

from . import features as ft
from . import paths

DATA_LOG_JSON = paths.result("data_log.json")

V_MIN, V_MAX = 0.0, 6.0  # average voltage vs the metal, V: 0 < V <= 6
E_HULL_MAX = 0.2  # eV/atom above the convex hull, for both end members


def clean(raw):
    """Filter and de-duplicate; returns (table, log)."""
    log = {"raw_rows": int(len(raw))}
    d = raw[(raw.average_voltage > V_MIN) & (raw.average_voltage <= V_MAX)].copy()
    log["after_voltage_filter"] = int(len(d))
    d = d[(d.stability_charge <= E_HULL_MAX) & (d.stability_discharge <= E_HULL_MAX)].copy()
    log["after_stability_filter"] = int(len(d))
    # polymorphs with the same (charged, discharged, ion) cannot be told apart by composition: keep the median voltage
    d["key"] = d.formula_charge + "|" + d.formula_discharge + "|" + d.working_ion
    agg = (
        d.groupby("key")
        .agg(
            formula_charge=("formula_charge", "first"),
            formula_discharge=("formula_discharge", "first"),
            working_ion=("working_ion", "first"),
            average_voltage=("average_voltage", "median"),
            n_polymorphs=("average_voltage", "size"),
            battery_id=("battery_id", "first"),
        )
        .reset_index(drop=True)
    )
    log["unique_steps"] = int(len(agg))
    agg["group"] = [ft.host_system(f, i) for f, i in zip(agg.formula_discharge, agg.working_ion, strict=True)]
    return agg, log


def featurize(agg):
    Xd = np.vstack([ft.magpie_vector(f) for f in agg.formula_discharge])
    Xc = np.vstack([ft.magpie_vector(f) for f in agg.formula_charge])
    extra = np.array([ft.ion_terms(r.formula_discharge, r.formula_charge, r.working_ion) for r in agg.itertuples()])
    return np.hstack([Xd, Xc, extra])


def build(raw_csv=paths.RAW_CSV, clean_csv=paths.CLEAN_CSV, features_npz=paths.FEATURES, log_json=DATA_LOG_JSON):
    raw = pd.read_csv(raw_csv, index_col=0)
    agg, log = clean(raw)
    X = featurize(agg)
    ok = np.isfinite(X).all(axis=1)
    log["dropped_nonfinite"] = int((~ok).sum())
    agg, X = agg[ok].reset_index(drop=True), X[ok]
    agg.to_csv(clean_csv, index=False, lineterminator="\n")
    np.savez(
        features_npz,
        X=X,
        y=agg.average_voltage.values,
        ion=agg.working_ion.values.astype(str),
        group=agg.group.values.astype(str),
        names=np.array(ft.feature_names()),
    )
    log["final_rows"] = int(len(agg))
    log["per_ion"] = {k: int(v) for k, v in agg.working_ion.value_counts().items()}
    log["n_features"] = int(X.shape[1])
    log["filters"] = {"V_min_exclusive": V_MIN, "V_max": V_MAX, "E_hull_max_eV_per_atom": E_HULL_MAX}
    with open(log_json, "w", encoding="utf-8", newline="\n") as f:
        json.dump(log, f, indent=1)
    return log


def load():
    """Feature matrix and labels as a dict: X, y, ion, group, names (plain arrays, no pickle)."""
    z = np.load(paths.FEATURES)
    return {k: z[k] for k in z.files}


def load_clean():
    return pd.read_csv(paths.CLEAN_CSV)
