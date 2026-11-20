"""Analyse every MD run in calc/md/runs/ that still has its trajectory, and pool them by temperature.

Each run folder holds md_<T>.traj (not in git: 9 MB each), md_<T>.log and run.json; this script writes analysis.json
into the folder and results/diffusion.json with the pooled D(T) and Arrhenius fits: all runs, and the seven runs of the
first set (laptop GPU) alone. Runs whose trajectory is missing are pooled from their analysis.json.

Usage: python scripts/analyze_md.py
"""

import json

import _path  # noqa: F401

from intercalation_voltage import diffusion as df
from intercalation_voltage import paths

RUNS = paths.CALC / "md" / "runs"


def main():
    runs = []
    for d in sorted(p for p in RUNS.iterdir() if p.is_dir()):
        if not (d / "run.json").exists():  # still running
            continue
        info = json.loads((d / "run.json").read_text(encoding="utf-8"))
        T = int(info["T"])
        traj, log = d / f"md_{T}.traj", d / f"md_{T}.log"
        if info.get("steps") != 12_500:
            continue
        if traj.exists():
            a = df.analyze(traj, log)
            df.save(a, d / "analysis.json")
        else:
            a = json.loads((d / "analysis.json").read_text(encoding="utf-8"))
        runs.append({"name": d.name, "T": float(T), "seed": info.get("seed"), "first": "laptop" in d.name, **a})
        print(f"{d.name:14s} D = {a['D_cm2_s']:.4e} cm2/s  <T> = {a['mean_T_after_eq_K']:.1f} K", flush=True)
    out = {
        "all_runs": df.pool(runs),
        "first_runs": df.pool([r for r in runs if r["first"]]),
        "runs": [
            {k: r[k] for k in ("name", "T", "seed", "D_cm2_s", "D_ab_cm2_s", "D_c_cm2_s", "mean_T_after_eq_K")}
            for r in runs
        ],
    }
    df.save(out, paths.result("diffusion.json"))
    for key in ("first_runs", "all_runs"):
        a = out[key].get("arrhenius", {})
        print(
            key,
            {t["T"]: (t["n_runs"], f"{t['D_cm2_s']:.3e}") for t in out[key]["temperatures"]},
            f"Ea = {a.get('Ea_eV', float('nan')):.4f} +- {a.get('Ea_stderr_eV') or float('nan'):.4f} eV",
        )


if __name__ == "__main__":
    main()
