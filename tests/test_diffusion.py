"""MSD analysis on trajectories whose answer is known."""

import numpy as np
import pytest

from intercalation_voltage import diffusion as df


def test_unwrap_follows_an_atom_across_the_boundary():
    # one atom moving +0.1 in fractional x per frame, stored wrapped into [0, 1)
    true = np.zeros((30, 1, 3))
    true[:, 0, 0] = 0.05 + 0.1 * np.arange(30)
    wrapped = true % 1.0
    assert np.allclose(df.unwrap(wrapped), true)


def test_msd_of_uniform_motion_is_quadratic_and_of_a_random_walk_linear():
    v = 0.3
    p = np.zeros((101, 2, 3))
    p[:, :, 0] = v * np.arange(101)[:, None]
    msd = df.msd_curve(p)
    lags = np.arange(len(msd))
    assert np.allclose(msd, (v * lags) ** 2)
    rng = np.random.default_rng(0)
    walk = np.cumsum(rng.normal(0, 0.2, size=(1001, 300, 3)), axis=0)  # variance 0.04 per step and axis
    m = df.msd_curve(walk)[:100]
    slope = np.polyfit(np.arange(100), m, 1)[0]
    assert slope == pytest.approx(3 * 0.04, rel=0.05)


def test_arrhenius_recovers_the_activation_energy():
    T = np.array([500.0, 600.0, 700.0, 800.0])
    D = 1e-3 * np.exp(-0.15 / (df.KB_EV * T))
    a = df.arrhenius(T, D)
    assert a["Ea_eV"] == pytest.approx(0.15, abs=1e-9)
    assert a["lnD0"] == pytest.approx(np.log(1e-3), abs=1e-9)


def test_pool_groups_runs_by_temperature():
    runs = [
        {"name": f"r{k}", "T": T, "D_cm2_s": D, "D_ab_cm2_s": D, "D_c_cm2_s": D}
        for k, (T, D) in enumerate([(500, 1e-6), (500, 3e-6), (800, 1e-5)])
    ]
    p = df.pool(runs)
    assert [t["n_runs"] for t in p["temperatures"]] == [2, 1]
    assert p["temperatures"][0]["D_cm2_s"] == pytest.approx(2e-6)
    assert p["temperatures"][0]["D_sd_cm2_s"] == pytest.approx(np.std([1e-6, 3e-6], ddof=1))


def test_last_segment_of_an_appended_log(tmp_path):
    log = tmp_path / "md.log"
    head = "Time[ps]      Etot[eV]     Epot[eV]     Ekin[eV]    T[K]\n"
    log.write_text(head + "0.0 1 1 1 300.0\n0.1 1 1 1 310.0\n" + head + "0.0 1 1 1 600.0\n0.1 1 1 1 590.0\n")
    assert df.last_segment_temperatures(log).tolist() == [600.0, 590.0]
