"""CHGNet average voltages of LiCoO2 and NaCoO2 (CoO2 + metal -> ACoO2), two ways.

cell relaxed : atoms and cell relaxed together (fmax 0.02 eV/A, up to 500 steps), as in the first run.
fixed lattice: experimental lattices kept, atoms relaxed (the setting of calc/dft/dft_voltage.py). CHGNet, trained on
               PBE(+U) without van der Waals binding, lets layered CoO2 swell along c when the cell is free; this
               variant shows how much of the voltage that swelling carries.

CoO2 is taken as the lower-energy of the O1 and O3 stackings in each mode; V with each stacking is stored too.

swapped start: ACoO2 relaxed (cell free) from a start with the alkali and Co sites exchanged, the O planes on the
               wrong side, as in the first run. LiCoO2 returns to the same structure; NaCoO2 ends in a second
               minimum with Co-O = 2.04 A (1.97 A from the right start), 78 meV lower in CHGNet, which moves V_Na.
Usage: python calc/chgnet/chgnet_voltage.py            -> calc/chgnet/chgnet_voltage.json
"""

import json
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

import chgnet  # noqa: E402
import torch  # noqa: E402
from chgnet.model import CHGNet, StructOptimizer  # noqa: E402
from pymatgen.core import Lattice, Structure  # noqa: E402

from intercalation_voltage import structures as S  # noqa: E402
from intercalation_voltage.voltage import average_voltage  # noqa: E402

NAMES = ["LiCoO2", "NaCoO2", "CoO2-O3", "CoO2-O1", "Li", "Na"]


def swapped_start(alkali, a, c, z_o):
    """The first run's start: alkali on 3b (0,0,1/2) and Co on 3a (0,0,0), O planes nearer the alkali."""
    return Structure.from_spacegroup(
        "R-3m", Lattice.hexagonal(a, c), [alkali, "Co", "O"], [[0, 0, 0.5], [0, 0, 0], [0, 0, z_o]]
    ).get_primitive_structure()


def co_o_distance(s):
    co = next(i for i, x in enumerate(s) if x.specie.symbol == "Co")
    return float(min(n.nn_distance for n in s.get_neighbors(s[co], 3.0) if n.specie.symbol == "O"))


def relax(opt, structure, relax_cell):
    t0 = time.time()
    r = opt.relax(structure, fmax=0.02, steps=500, relax_cell=relax_cell, verbose=False)
    fs = r["final_structure"]
    e_atom = float(r["trajectory"].energies[-1]) / len(fs)
    n_fu = fs.composition.get_reduced_composition_and_factor()[1]
    return {
        "formula": fs.composition.reduced_formula,
        "n_atoms": len(fs),
        "E_per_atom": e_atom,
        "Co_O_A": co_o_distance(fs) if "Co" in fs.symbol_set else None,
        "E_per_fu": e_atom * len(fs) / n_fu,
        "volume_per_atom": fs.volume / len(fs),
        "volume_per_atom_start": structure.volume / len(structure),
        "lattice_abc": list(fs.lattice.abc),
        "lattice_abc_start": list(structure.lattice.abc),
        "steps": len(r["trajectory"].energies),
        "seconds": time.time() - t0,
    }


def voltages(E):
    out = {}
    for A in ("Li", "Na"):
        for st in ("O1", "O3"):
            out[f"V_{A}_{st}"] = average_voltage(
                E[f"CoO2-{st}"]["E_per_fu"], E[A]["E_per_fu"], E[f"{A}CoO2"]["E_per_fu"]
            )
    ground = "O3" if E["CoO2-O3"]["E_per_fu"] <= E["CoO2-O1"]["E_per_fu"] else "O1"
    for A in ("Li", "Na"):
        out[f"V_{A}"] = out[f"V_{A}_{ground}"]
    out["CoO2_ground"] = ground
    return out


def main():
    torch.set_num_threads(4)
    model = CHGNet.load()
    opt = StructOptimizer(model=model, use_device="cpu")
    res = {
        "versions": {
            "chgnet": chgnet.__version__,
            "chgnet_model": model.version,
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "device": "cpu",
    }
    for mode, cell in (("cell_relaxed", True), ("fixed_lattice", False)):
        E = {n: relax(opt, S.BUILDERS[n](), cell) for n in NAMES}
        res[mode] = {"energies": E, **voltages(E)}
        print(mode, {k: v for k, v in res[mode].items() if k != "energies"}, flush=True)
    # the first run's start for ACoO2; CoO2 and the metals as in "cell_relaxed"
    E = dict(res["cell_relaxed"]["energies"])
    E["LiCoO2"] = relax(opt, swapped_start("Li", 2.8156, 14.0542, 0.2396), True)
    E["NaCoO2"] = relax(opt, swapped_start("Na", 2.889, 15.60, 0.232), True)
    res["swapped_start"] = {"energies": {k: E[k] for k in ("LiCoO2", "NaCoO2")}, **voltages(E)}
    print("swapped_start", {k: v for k, v in res["swapped_start"].items() if k != "energies"}, flush=True)
    out = HERE / "chgnet_voltage.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8", newline="\n")
    print("written", out)


if __name__ == "__main__":
    main()
