"""Composition features of one insertion reaction, and the keys that pair a host across working ions.

A reaction is (discharged composition, charged composition, working ion). Its feature vector is the Magpie element
statistics (22 element properties x 6 statistics = 132 numbers, Ward et al. 2016, via matminer) of the discharged
composition, the same 132 of the charged composition, and two ion terms: the working ion inserted per atom of the
discharged composition, and the atomic fraction of the working ion in it. 266 numbers in all. The web page computes
the same vector in JavaScript (web/src/core.js); tests/web_model.test.js checks the two against each other.
"""

from functools import lru_cache

import numpy as np
from pymatgen.core import Composition

WORKING_IONS = ("Li", "Na", "K", "Mg", "Ca", "Zn", "Al", "Y", "Rb", "Cs")
N_MAGPIE = 132


@lru_cache(maxsize=1)
def magpie():
    from matminer.featurizers.composition import ElementProperty

    return ElementProperty.from_preset("magpie")


def magpie_vector(formula):
    return np.array(magpie().featurize(Composition(formula)), dtype=float)


def ion_terms(discharged, charged, ion):
    """(ion inserted per atom of the discharged composition, ion fraction of the discharged composition).

    The charged composition is scaled to the host amount of the discharged one before the ion counts are compared.
    """
    cd, cc = Composition(discharged), Composition(charged)
    host_d = sum(v for k, v in cd.items() if str(k) != ion)
    host_c = sum(v for k, v in cc.items() if str(k) != ion)
    n_ion_c = cc[ion] * host_d / host_c if host_c > 0 else 0.0
    return (cd[ion] - n_ion_c) / cd.num_atoms, cd.get_atomic_fraction(ion)


def reaction_vector(discharged, charged, ion):
    return np.concatenate(
        [magpie_vector(discharged), magpie_vector(charged), list(ion_terms(discharged, charged, ion))]
    )


def feature_names():
    labels = magpie().feature_labels()
    return (
        [f"discharged: {s}" for s in labels]
        + [f"charged: {s}" for s in labels]
        + ["inserted ion per atom", "ion fraction (discharged)"]
    )


def host_system(formula, ion):
    """Chemical system of the host: the elements other than the working ion, e.g. 'Co-O' for LiCoO2."""
    return "-".join(sorted(str(e) for e in Composition(formula).elements if str(e) != ion))


def swap_ion(formula, ion, new):
    """Reduced formula with the working ion replaced, e.g. swap_ion('NaCoO2', 'Na', 'Li') = 'LiCoO2'."""
    c = Composition(formula)
    return Composition({(new if str(k) == ion else str(k)): v for k, v in c.items()}).reduced_formula
