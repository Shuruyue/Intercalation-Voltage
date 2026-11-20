"""GPAW total energy of one structure for the LiCoO2 voltage, PBE or PBE+U (runs on Linux; on Windows use WSL).

    python calc/dft/dft_voltage.py <tag> <U> [<ecut eV> <k points added>]      tag: LiCoO2, CoO2-O1, CoO2-O3, Li

Settings: plane waves (520 eV), Gamma-centred Monkhorst-Pack meshes (LiCoO2 and O3 CoO2 7x7x7, O1 CoO2 9x9x6,
Li 12x12x12; "k points added" n adds n per direction, 2n for meshes of 10 or more), Fermi-Dirac smearing 0.05 eV
(0.1 eV for Li metal), SCF energy convergence 1e-5 eV. Experimental lattices are kept and only atomic positions are
relaxed (BFGS to 0.03 eV/A): PBE has no van der Waals binding, so a free cell lets layered CoO2 swell along c.
CoO2 is spin-polarised (1 muB on Co to start). +U: Dudarev U_eff on Co 3d (3.32 eV, the Materials Project value).
Writes dft_<tag>_U<U>[_e<ecut>_k<n>].json, the GPAW text output and the BFGS log into the current folder.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ase.optimize import BFGS  # noqa: E402
from gpaw import GPAW, PW, FermiDirac  # noqa: E402
from pymatgen.io.ase import AseAtomsAdaptor  # noqa: E402

from intercalation_voltage.structures import BUILDERS  # noqa: E402

KMESH = {"LiCoO2": (7, 7, 7), "CoO2-O3": (7, 7, 7), "CoO2-O1": (9, 9, 6), "Li": (12, 12, 12)}
FMAX, MAX_STEPS = 0.03, 60


def main():
    tag, U = sys.argv[1], float(sys.argv[2])
    ecut = float(sys.argv[3]) if len(sys.argv) > 3 else 520.0
    kadd = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    sfx = "" if (ecut == 520.0 and kadd == 0) else f"_e{int(ecut)}_k{kadd}"
    atoms = AseAtomsAdaptor.get_atoms(BUILDERS[tag]())
    spin = tag.startswith("CoO2")
    if spin:
        atoms.set_initial_magnetic_moments([1.0 if s == "Co" else 0.0 for s in atoms.get_chemical_symbols()])
    setups = {"Co": f":d,{U}"} if (U > 0 and "Co" in atoms.get_chemical_symbols()) else "paw"
    kmesh = tuple(k + kadd * (2 if k >= 10 else 1) for k in KMESH[tag])
    atoms.calc = GPAW(
        mode=PW(ecut),
        xc="PBE",
        kpts={"size": kmesh, "gamma": True},
        occupations=FermiDirac(0.1 if tag == "Li" else 0.05),
        spinpol=spin,
        setups=setups,
        txt=f"gpaw_{tag}_U{U}{sfx}.txt",
        convergence={"energy": 1e-5},
    )
    t0 = time.time()
    opt = BFGS(atoms, logfile=f"relax_{tag}_U{U}{sfx}.log")
    converged = bool(opt.run(fmax=FMAX, steps=MAX_STEPS))
    e = atoms.get_potential_energy()
    res = {
        "tag": tag,
        "U": U,
        "ecut": ecut,
        "kadd": kadd,
        "E_total_eV": float(e),
        "n_atoms": len(atoms),
        "formula": atoms.get_chemical_formula(),
        "volume_A3": float(atoms.get_volume()),
        "cell_lengths": [float(x) for x in atoms.cell.cellpar()[:3]],
        "magmom": float(atoms.get_magnetic_moment()) if spin else 0.0,
        "relax_steps": opt.nsteps,
        "relax_converged": converged,
        "fmax_eV_per_A": float((atoms.get_forces() ** 2).sum(axis=1).max() ** 0.5),
        "positions_A": atoms.get_positions().tolist(),
        "symbols": atoms.get_chemical_symbols(),
        "kmesh": list(kmesh),
        "minutes": (time.time() - t0) / 60,
    }
    Path(f"dft_{tag}_U{U}{sfx}.json").write_text(json.dumps(res, indent=1), encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in res.items() if k not in ("positions_A", "symbols")}))
    if not converged:
        sys.exit(f"relaxation not converged in {MAX_STEPS} steps")


if __name__ == "__main__":
    main()
