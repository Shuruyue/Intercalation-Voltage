"""CHGNet molecular dynamics of Li in alpha-Li3N: one NVT run at one temperature.

    python calc/md/md_li3n.py 600 --seed 1 --out calc/md/runs/600K_seed1
    python calc/md/md_li3n.py 600 --seed 0 --steps 500 --out <folder>     # timing / smoke test

System: alpha-Li3N (P6/mmm, a = 3.648 A, c = 3.875 A; Rabenau & Schulz 1976), 3x3x3 supercell (108 atoms), two Li
removed from the Li2N plane at z = 0 (vacancies) -> 106 atoms. CHGNet (pretrained model shipped with the package),
NVT with ASE's Berendsen thermostat (CHGNet's default for "nvt": taut = 100 time steps, centre-of-mass momentum kept
at zero), 2 fs time step, 12,500 steps = 25 ps, a frame every 10 steps. Initial velocities: Maxwell-Boltzmann at T
from the given seed (the first seven runs, *_laptop*, were made with this same set-up but without a recorded seed).

Writes md_<T>.traj, md_<T>.log and run.json (settings, wall time, versions) into --out. The MSD analysis is separate:
intercalation_voltage.diffusion (scripts/analyze_md.py), so a run can be analysed again without re-running it.
"""

import argparse
import json
import os
import platform
import time

import ase
import chgnet
import numpy as np
import torch
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from chgnet.model import CHGNet
from chgnet.model.dynamics import MolecularDynamics
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor

TIMESTEP_FS, LOG_INTERVAL, STEPS_25PS = 2.0, 10, 12_500


def li3n_with_vacancies():
    s = Structure.from_spacegroup(
        "P6/mmm", Lattice.hexagonal(3.648, 3.875), ["N", "Li", "Li"], [[0, 0, 0], [0, 0, 0.5], [1 / 3, 2 / 3, 0]]
    )
    s.make_supercell([3, 3, 3])
    plane = [i for i, site in enumerate(s) if site.specie.symbol == "Li" and abs(site.frac_coords[2]) < 1e-3]
    s.remove_sites([plane[0], plane[len(plane) // 2]])
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("T", type=float, help="temperature in K")
    ap.add_argument("--seed", type=int, required=True, help="seed for the initial Maxwell-Boltzmann velocities")
    ap.add_argument("--steps", type=int, default=STEPS_25PS, help="MD steps (default 12,500 = 25 ps)")
    ap.add_argument("--out", required=True, help="output folder (created; must not already hold a trajectory)")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads for torch (graph construction)")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    tag = f"md_{int(a.T)}"
    traj, log = os.path.join(a.out, f"{tag}.traj"), os.path.join(a.out, f"{tag}.log")
    if os.path.exists(traj):
        raise SystemExit(f"{traj} exists; use a fresh folder (ASE would append to the log)")
    torch.set_num_threads(a.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    structure = li3n_with_vacancies()
    atoms = AseAtomsAdaptor.get_atoms(structure)
    MaxwellBoltzmannDistribution(atoms, temperature_K=a.T, rng=np.random.default_rng(a.seed))
    model = CHGNet.load()
    md = MolecularDynamics(
        atoms=atoms,
        model=model,
        ensemble="nvt",
        temperature=a.T,
        timestep=TIMESTEP_FS,
        trajectory=traj,
        logfile=log,
        loginterval=LOG_INTERVAL,
        use_device=device,
    )
    t0 = time.time()
    md.run(a.steps)
    minutes = (time.time() - t0) / 60
    info = {
        "T": a.T,
        "seed": a.seed,
        "steps": a.steps,
        "timestep_fs": TIMESTEP_FS,
        "log_interval": LOG_INTERVAL,
        "n_atoms": len(atoms),
        "ensemble": "NVT Berendsen (chgnet default), taut = 100 steps",
        "minutes": minutes,
        "seconds_per_step": minutes * 60 / a.steps,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "versions": {
            "chgnet": chgnet.__version__,
            "chgnet_model": model.version,
            "torch": torch.__version__,
            "ase": ase.__version__,
            "python": platform.python_version(),
        },
    }
    with open(os.path.join(a.out, "run.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(info, f, indent=1)
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
