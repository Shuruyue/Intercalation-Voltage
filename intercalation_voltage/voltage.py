"""Average intercalation voltage from total energies, and the DFT / CHGNet results in calc/.

For x Li going into a host between x1 and x2 (Aydinol et al. 1997):

    V = -[E(Li_x2 Host) - E(Li_x1 Host) - (x2 - x1) E(Li metal)] / ((x2 - x1) e)

For LiCoO2 over 0 <= x <= 1, per formula unit: e V = E(CoO2) + E(Li) - E(LiCoO2).
"""

import json
from pathlib import Path

from . import paths

DFT_DIR = paths.CALC / "dft" / "output"
CHGNET_JSON = paths.CALC / "chgnet" / "chgnet_voltage.json"
U_CO = 3.32  # U_eff(Co), eV: the Materials Project value (Jain et al. 2011)
SETTINGS = {"base": (520, 0), "converged": (650, 2)}  # (plane-wave cutoff in eV, k points added per direction)


def average_voltage(e_host, e_metal, e_lithiated, n=1.0):
    """Average voltage (V) from energies per formula unit (eV): charged host, metal, discharged host."""
    return (e_host + n * e_metal - e_lithiated) / n


def load_dft(folder=DFT_DIR):
    """{(tag, U, ecut, kadd): record} from the dft_*.json files that calc/dft/dft_voltage.py writes."""
    out = {}
    for f in sorted(Path(folder).glob("dft_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        out[(d["tag"], float(d["U"]), int(d.get("ecut", 520)), int(d.get("kadd", 0)))] = d
    return out


def dft_voltages(records):
    """Voltages of every complete set: PBE and PBE+U, O1 (and O3) CoO2, base and converged settings."""
    E = {k: v["E_total_eV"] for k, v in records.items()}
    V = {}
    for name, (ecut, kadd) in SETTINGS.items():
        for U, xc in ((0.0, "PBE"), (U_CO, "PBE+U")):
            for coo2 in ("CoO2-O1", "CoO2-O3"):
                need = [(coo2, U, ecut, kadd), ("Li", 0.0, ecut, kadd), ("LiCoO2", U, ecut, kadd)]
                if all(n in E for n in need):
                    V[f"{xc}|{coo2[-2:]}|{name}"] = average_voltage(*(E[n] for n in need))
    return V


def load_chgnet(path=CHGNET_JSON):
    return json.loads(Path(path).read_text(encoding="utf-8"))
