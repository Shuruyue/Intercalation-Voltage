"""Screening statistics on cases whose answer is known, and the transition-metal grouping."""

import numpy as np
import pytest

from intercalation_voltage import screening as sc


def test_perfect_and_reversed_rankings():
    y = np.where(np.arange(100) >= 66, 4.5, 3.0)  # 34 reactions at or above 4.0 V
    p = y + 1e-3 * np.arange(100)  # ranks every high reaction first, no ties
    perfect = sc.screen(y, p)
    assert perfect["n_target"] == 34 and perfect["base_rate"] == pytest.approx(0.34)
    assert perfect["top"]["n_selected"] == 10
    assert perfect["top"]["precision"] == 1.0 and perfect["top"]["recall"] == pytest.approx(10 / 34)
    assert perfect["top"]["enrichment"] == pytest.approx(1 / 0.34)
    assert perfect["predicted_above"]["precision"] == 1.0 and perfect["predicted_above"]["recall"] == 1.0
    worst = sc.screen(y, -p)
    assert worst["top"]["precision"] == 0.0 and worst["top"]["enrichment"] == 0.0
    assert worst["predicted_above"]["n_selected"] == 0 and worst["predicted_above"]["precision"] is None


def test_gain_curve_runs_from_zero_to_one():
    y = np.array([4.5, 3.0, 4.2, 2.0])
    g = sc.gain_curve(y, np.array([0.9, 0.8, 0.1, 0.0]))
    assert g.tolist() == [0.0, 0.5, 0.5, 1.0, 1.0]


def test_error_shares():
    e = sc.error_shares(np.zeros(4), np.array([0.1, 0.4, 0.8, 2.0]))
    assert e["within_0.3"] == 0.25 and e["within_0.5"] == 0.5 and e["within_1.0"] == 0.75
    assert e["median"] == pytest.approx(0.6)


def test_transition_metals_counts_each_formula_once():
    sets, frequent = sc.transition_metals(["LiCoO2", "LiCo0.5Ni0.5O2", "LiNiO2", "LiFePO4"], min_n=2)
    assert sets[1] == {"Co", "Ni"} and sets[3] == {"Fe"}
    assert frequent == ["Co", "Ni"]
