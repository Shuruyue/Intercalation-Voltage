"""Li diffusion from MD trajectories: unwrapped mean-squared displacement, D, and the Arrhenius fit.

MSD(t) is averaged over all Li atoms and all time origins after the first 2.5 ps; D = slope / 6 from a straight-line
fit for 1 ps < t <= half the analysed trajectory. In-plane (ab, the Li2N layers) and c-axis parts use the same fit with
2D and 1D factors: D_ab = slope_ab / 4, D_c = slope_c / 2. Unit: 1 A^2/ps = 1e-4 cm^2/s.
"""

import json
from pathlib import Path

import numpy as np

KB_EV = 8.617333262e-5
TIMESTEP_FS, LOG_INTERVAL, EQUILIBRATION_PS, FIT_FROM_PS = 2.0, 10, 2.5, 1.0
A2_PER_PS_TO_CM2_PER_S = 1e-4


def unwrap(frac):
    """Unwrap fractional coordinates (frames, atoms, 3) across periodic boundaries; frames must be close in time."""
    d = np.diff(frac, axis=0)
    d -= np.round(d)
    return np.concatenate([frac[:1], frac[:1] + np.cumsum(d, axis=0)])


def msd_curve(p, dims=slice(None)):
    """MSD over all atoms and time origins, lags 0 .. len(p)//2 - 1 (frames)."""
    nlag = len(p) // 2
    q = p[..., dims]
    return np.array([0.0] + [np.mean(np.sum((q[lag:] - q[:-lag]) ** 2, axis=-1)) for lag in range(1, nlag)])


def analyze(traj_file, log_file=None):
    """D, D_ab, D_c and the MSD curve of one run (trajectory written every LOG_INTERVAL steps)."""
    from ase.io import read

    traj = read(str(traj_file), index=":")
    li = np.array([a.symbol == "Li" for a in traj[0]])
    cell = traj[0].cell.array
    pos = np.array([a.get_positions() for a in traj])[:, li, :]
    frac = np.linalg.solve(cell.T, pos.reshape(-1, 3).T).T.reshape(pos.shape)
    p = unwrap(frac) @ cell
    dt_ps = LOG_INTERVAL * TIMESTEP_FS / 1000
    skip = int(EQUILIBRATION_PS / dt_ps)
    p = p[skip:]
    msd, msd_ab = msd_curve(p), msd_curve(p, slice(0, 2))
    t = np.arange(len(msd)) * dt_ps
    sel = t > FIT_FROM_PS
    slope = float(np.polyfit(t[sel], msd[sel], 1)[0])
    slope_ab = float(np.polyfit(t[sel], msd_ab[sel], 1)[0])
    out = {
        "n_frames": len(traj),
        "n_atoms": len(traj[0]),
        "n_li": int(li.sum()),
        "D_cm2_s": slope / 6 * A2_PER_PS_TO_CM2_PER_S,
        "D_ab_cm2_s": slope_ab / 4 * A2_PER_PS_TO_CM2_PER_S,
        "D_c_cm2_s": (slope - slope_ab) / 2 * A2_PER_PS_TO_CM2_PER_S,
        "t_ps": t[::5].tolist(),
        "msd_A2": msd[::5].tolist(),
    }
    if log_file is not None:
        T = last_segment_temperatures(log_file)
        out["mean_T_after_eq_K"] = float(np.mean(T[skip:]))
    return out


def last_segment_temperatures(log_file):
    """Temperatures of the last run in an ASE MD log (a log is appended to, so an aborted run can precede it)."""
    rows = []
    for line in Path(log_file).read_text().splitlines():
        if line.startswith("Time"):
            rows = []
            continue
        parts = line.split()
        if len(parts) >= 5:
            rows.append(float(parts[4]))
    return np.array(rows)


def arrhenius(T, D):
    """ln D = ln D0 - Ea / (kB T): Ea (eV), its standard error (None for two points), ln D0."""
    T, D = np.asarray(T, float), np.asarray(D, float)
    if len(T) > 2:
        fit, cov = np.polyfit(1 / T, np.log(D), 1, cov=True)
        se = float(np.sqrt(cov[0, 0]) * KB_EV)
    else:
        fit, se = np.polyfit(1 / T, np.log(D), 1), None
    return {"Ea_eV": float(-fit[0] * KB_EV), "Ea_stderr_eV": se, "lnD0": float(fit[1])}


def pool(runs):
    """Runs grouped by temperature: mean D, sample s.d., Arrhenius on the means and on every run."""
    by_t = {}
    for r in runs:
        by_t.setdefault(float(r["T"]), []).append(r)
    temps = []
    for T in sorted(by_t):
        D = np.array([r["D_cm2_s"] for r in by_t[T]])
        temps.append(
            {
                "T": T,
                "n_runs": len(D),
                "D_cm2_s": float(D.mean()),
                "D_sd_cm2_s": float(D.std(ddof=1)) if len(D) > 1 else None,
                "D_each_cm2_s": D.tolist(),
                "runs": [r["name"] for r in by_t[T]],
                "D_ab_cm2_s": float(np.mean([r["D_ab_cm2_s"] for r in by_t[T]])),
                "D_c_cm2_s": float(np.mean([r["D_c_cm2_s"] for r in by_t[T]])),
            }
        )
    out = {"temperatures": temps}
    if len(temps) >= 2:
        a = arrhenius([x["T"] for x in temps], [x["D_cm2_s"] for x in temps])
        allT = [r["T"] for r in runs]
        allD = [r["D_cm2_s"] for r in runs]
        b = arrhenius(allT, allD)
        a["Ea_all_runs_eV"], a["Ea_all_runs_stderr_eV"] = b["Ea_eV"], b["Ea_stderr_eV"]
        a["D_300K_extrapolated_cm2_s"] = float(np.exp(a["lnD0"] - a["Ea_eV"] / (KB_EV * 300.0)))
        out["arrhenius"] = a
    return out


def save(obj, path):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1)
