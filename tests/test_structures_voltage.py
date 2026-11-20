"""Starting structures and the voltage formula."""

import numpy as np
import pytest

from intercalation_voltage import structures as S
from intercalation_voltage import voltage as vt


def shortest(s, a, b):
    i = next(k for k, x in enumerate(s) if x.specie.symbol == a)
    return min(n.nn_distance for n in s.get_neighbors(s[i], 3.0) if n.specie.symbol == b)


def test_licoo2_has_short_co_o_and_long_li_o_bonds():
    # Co on 3b with z(O) = 0.2396: Co-O 1.92 A, Li-O 2.09 A (Akimoto et al. 1998). With Li and Co exchanged, as in
    # the first runs of this project, the two bond lengths swap.
    s = S.licoo2()
    assert len(s) == 4 and s.composition.reduced_formula == "LiCoO2"
    assert shortest(s, "Co", "O") == pytest.approx(1.922, abs=0.005)
    assert shortest(s, "Li", "O") == pytest.approx(2.092, abs=0.005)


def test_o3_coo2_keeps_the_parent_lattice_and_o1_is_one_formula_unit():
    o3, parent = S.coo2_o3(S.licoo2()), S.licoo2()
    assert o3.composition.reduced_formula == "CoO2" and len(o3) == 3
    assert np.allclose(o3.lattice.abc, parent.lattice.abc)
    o1 = S.coo2_o1()
    assert len(o1) == 3 and o1.lattice.c == pytest.approx(4.29)


def test_metals_are_one_atom_bcc_cells():
    for s, a in ((S.li_metal(), 3.51), (S.na_metal(), 4.29)):
        assert len(s) == 1
        assert s.volume == pytest.approx(a**3 / 2)


def test_average_voltage_sign_and_value():
    # PBE+U energies of the first run, per formula unit: CoO2 (O1), Li, LiCoO2
    assert vt.average_voltage(-18.495957, -1.894275, -24.269971) == pytest.approx(3.879739, abs=1e-6)
    # x Li into a host: per Li
    assert vt.average_voltage(-10.0, -2.0, -15.0, n=2) == pytest.approx(0.5)


def test_dft_results_give_every_voltage():
    V = vt.dft_voltages(vt.load_dft())
    for key in ("PBE|O1|base", "PBE+U|O1|base", "PBE+U|O3|base", "PBE|O1|converged", "PBE+U|O1|converged"):
        assert 3.0 < V[key] < 4.5
