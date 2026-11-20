"""Composition features and the keys that pair hosts across working ions."""

import numpy as np
import pytest

from intercalation_voltage import data as dt
from intercalation_voltage import features as ft


def test_ion_terms_full_and_partial_insertion():
    assert ft.ion_terms("LiCoO2", "CoO2", "Li") == pytest.approx((0.25, 0.25))
    # Li0.5CoO2 -> LiCoO2: half a Li per formula unit of 4 atoms
    assert ft.ion_terms("LiCoO2", "Li(CoO2)2", "Li") == pytest.approx((0.125, 0.25))
    # the charged formula is scaled to the host amount of the discharged one
    assert ft.ion_terms("Li2Co2O4", "CoO2", "Li") == pytest.approx((0.25, 0.25))


def test_host_system_and_ion_swap():
    assert ft.host_system("NaCoO2", "Na") == "Co-O"
    assert ft.host_system("Li3V2(PO4)3", "Li") == "O-P-V"
    assert ft.swap_ion("Na3V2(PO4)3", "Na", "Li") == "Li3V2(PO4)3"
    assert ft.swap_ion("CoO2", "Na", "Li") == "CoO2"


def test_feature_vector_layout():
    names = ft.feature_names()
    assert len(names) == 2 * ft.N_MAGPIE + 2 == 266
    x = ft.reaction_vector("LiCoO2", "CoO2", "Li")
    assert x.shape == (266,) and np.isfinite(x).all()
    assert names[264] == "inserted ion per atom" and x[264] == pytest.approx(0.25)


def test_features_match_the_stored_matrix():
    # one reaction recomputed from its formulas equals its row of data/features.npz (scripts/prep_data.py)
    d, clean = dt.load(), dt.load_clean()
    i = int(np.where((clean.formula_discharge == "LiCoO2") & (clean.formula_charge == "Li(CoO2)2"))[0][0])
    x = ft.reaction_vector(clean.formula_discharge[i], clean.formula_charge[i], clean.working_ion[i])
    assert np.array_equal(x, d["X"][i])
