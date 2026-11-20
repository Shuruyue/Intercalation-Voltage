"""Starting structures at experimental lattice parameters (pymatgen; only pymatgen is needed, so the GPAW
environment can import this module).
"""

from pymatgen.core import Lattice, Structure


def layered_o3(alkali, a, c, z_o):
    """O3-type alpha-NaFeO2 structure (R-3m), primitive cell (one formula unit).

    Alkali on 3a (0,0,0), Co on 3b (0,0,1/2), O on 6c (0,0,z). With z(O) = 0.2396 the O planes sit 1.03 A from the
    Co planes and 1.31 A from the Li planes, as in the single-crystal structure of Akimoto et al. (1998).
    """
    s = Structure.from_spacegroup(
        "R-3m", Lattice.hexagonal(a, c), [alkali, "Co", "O"], [[0, 0, 0], [0, 0, 0.5], [0, 0, z_o]]
    )
    return s.get_primitive_structure()


def licoo2():
    """LiCoO2, a = 2.8156 A, c = 14.0542 A, z(O) = 0.2396 (Akimoto, Gotoh & Oosawa 1998)."""
    return layered_o3("Li", 2.8156, 14.0542, 0.2396)


def nacoo2():
    """O3-NaCoO2, a = 2.889 A, c = 15.60 A (Takahashi et al. 2003); z(O) = 0.232 as a start, relaxed anyway."""
    return layered_o3("Na", 2.889, 15.60, 0.232)


def coo2_o3(parent):
    """CoO2 with O3 stacking: the alkali removed from a layered parent, the parent's lattice kept."""
    s = parent.copy()
    s.remove_species([el for el in ("Li", "Na") if el in s.symbol_set])
    return s


def coo2_o1():
    """CdI2-type (O1) CoO2, the end member of full delithiation (Amatucci, Tarascon & Klein 1996):
    P-3m1, a = 2.822 A, c = 4.29 A, z(O) = 0.24 as a start."""
    return Structure.from_spacegroup(
        "P-3m1", Lattice.hexagonal(2.822, 4.29), ["Co", "O"], [[0, 0, 0], [1 / 3, 2 / 3, 0.24]]
    )


def bcc(el, a):
    return Structure(Lattice.cubic(a), [el, el], [[0, 0, 0], [0.5, 0.5, 0.5]]).get_primitive_structure()


def li_metal():
    return bcc("Li", 3.51)


def na_metal():
    return bcc("Na", 4.29)


BUILDERS = {
    "LiCoO2": licoo2,
    "NaCoO2": nacoo2,
    "CoO2-O3": lambda: coo2_o3(licoo2()),
    "CoO2-O1": coo2_o1,
    "Li": li_metal,
    "Na": na_metal,
}
