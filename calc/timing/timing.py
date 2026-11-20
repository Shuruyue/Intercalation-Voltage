"""Cost of one energy evaluation against the number of atoms, GPAW (one PBE SCF) and CHGNet, on one CPU thread.

    python calc/timing/timing.py dft [max atoms]     (GPAW; Linux or WSL, OMP_NUM_THREADS=1)  -> timing_dft.json
    python calc/timing/timing.py chgnet              (CHGNet on the CPU)                       -> timing_chgnet.json

LiCoO2 supercells of the conventional hexagonal cell (12 atoms), plus the 4-atom primitive cell, with k meshes of
similar density. Run both on the same machine with nothing else running: the numbers are wall-clock times.
"""

import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pymatgen.core import Lattice, Structure  # noqa: E402
from pymatgen.io.ase import AseAtomsAdaptor  # noqa: E402

HERE = Path(__file__).resolve().parent
DFT_CELLS = [
    ("prim", None, (7, 7, 7)),
    ("conv", (1, 1, 1), (7, 7, 2)),
    ("2x1x1", (2, 1, 1), (4, 7, 2)),
    ("2x2x1", (2, 2, 1), (4, 4, 2)),
]
CHGNET_CELLS = [None, (1, 1, 1), (2, 2, 1), (3, 3, 1), (4, 4, 2), (6, 6, 2)]


def conventional_licoo2():
    """Hexagonal LiCoO2 (12 atoms): Li on 3a, Co on 3b, O on 6c with z = 0.2396 (Akimoto et al. 1998)."""
    return Structure.from_spacegroup(
        "R-3m", Lattice.hexagonal(2.8156, 14.0542), ["Li", "Co", "O"], [[0, 0, 0], [0, 0, 0.5], [0, 0, 0.2396]]
    )


def cell(rep):
    s = conventional_licoo2()
    return s.get_primitive_structure() if rep is None else s * rep


def machine():
    return {"platform": platform.platform(), "processor": platform.processor(), "python": platform.python_version()}


def dft(max_atoms):
    from gpaw import GPAW, PW, FermiDirac

    out = {"machine": machine(), "runs": []}
    for name, rep, k in DFT_CELLS:
        s = cell(rep)
        if len(s) > max_atoms:
            continue
        at = AseAtomsAdaptor.get_atoms(s)
        at.calc = GPAW(
            mode=PW(520),
            xc="PBE",
            kpts={"size": k, "gamma": True},
            occupations=FermiDirac(0.05),
            txt=str(HERE / f"timing_{name}.txt"),
            convergence={"energy": 1e-5},
        )
        t0 = time.time()
        e = at.get_potential_energy()
        out["runs"].append(
            {"cell": name, "n_atoms": len(at), "kmesh": list(k), "seconds": time.time() - t0, "E_per_atom": e / len(at)}
        )
        print(out["runs"][-1], flush=True)
        (HERE / "timing_dft.json").write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")


def chgnet():
    import torch
    from chgnet.model import CHGNet

    torch.set_num_threads(1)
    m = CHGNet.load(use_device="cpu")
    out = {"machine": machine(), "runs": []}
    for rep in CHGNET_CELLS:
        s = cell(rep)
        m.predict_structure(s)  # warm-up
        n = 5
        t0 = time.time()
        for _ in range(n):
            m.predict_structure(s)
        out["runs"].append({"n_atoms": len(s), "seconds": (time.time() - t0) / n})
        print(out["runs"][-1], flush=True)
    (HERE / "timing_chgnet.json").write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    if sys.argv[1] == "dft":
        dft(int(sys.argv[2]) if len(sys.argv) > 2 else 48)
    else:
        chgnet()
