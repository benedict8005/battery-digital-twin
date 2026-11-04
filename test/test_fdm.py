"""Compare the NumPy finite-difference solver with analytical heat-equation cases."""

import numpy as np
import pytest

from battery_pinn.fdm import solve_heat

RHO_CP = 2.0e6  # J/(m3*K)


def test_no_source_stays_at_ambient():
    _, _, temp = solve_heat(100.0, 25.0, 0.0, newton_coeff=5.0)
    assert np.allclose(temp, 25.0)


def test_uniform_heating_is_linear_in_time():
    # Insulated, no cooling, uniform source q: T(t) = T0 + q t / (rho cp), exactly.
    q, t_end = 1.0e4, 600.0
    _, t, temp = solve_heat(t_end, 20.0, q, n_t=100, thermal_mass=RHO_CP)
    expected = 20.0 + q * t / RHO_CP
    assert np.allclose(temp, expected[:, None], rtol=1e-9)


def test_cooling_to_ambient_matches_exponential():
    # Uniform start at 60 C, h > 0, no source: T - Ta = (T0 - Ta) exp(-h t / rho cp).
    h, t_end = 2.0e3, 1.0e3
    _, t, temp = solve_heat(
        t_end,
        25.0,
        0.0,
        newton_coeff=h,
        thermal_mass=RHO_CP,
        t_initial_c=60.0,
        n_t=2000,
    )
    expected = 25.0 + 35.0 * np.exp(-h * t / RHO_CP)
    assert np.allclose(temp[:, 0], expected, atol=0.05)


def test_steady_state_with_newton_cooling():
    # Source q balanced by cooling: T_inf = Ta + q / h.
    q, h = 1.0e3, 5.0e2
    _, _, temp = solve_heat(2.0e5, 25.0, q, newton_coeff=h, thermal_mass=RHO_CP, n_t=50)
    assert np.allclose(temp[-1], 25.0 + q / h, atol=1e-6)


def test_dirichlet_steady_state_is_parabola():
    # k T'' + q = 0, T(0)=T(1)=Ta: T = Ta + q x (1 - x) / (2 k).
    q, k = 8.0, 2.0
    x, _, temp = solve_heat(
        1.0e9,
        25.0,
        q,
        k_cond=k,
        thermal_mass=RHO_CP,
        n_x=41,
        n_t=5,
        boundary="dirichlet",
    )
    expected = 25.0 + q * x * (1 - x) / (2 * k)
    assert np.allclose(temp[-1], expected, atol=1e-3)


def test_time_dependent_source_callable():
    _, _, temp = solve_heat(100.0, 0.0, lambda s: 2.0e6, n_t=100, thermal_mass=RHO_CP)
    assert temp[-1, 0] == pytest.approx(100.0, rel=1e-6)


@pytest.mark.parametrize(
    "kwargs",
    [{"boundary": "robin"}, {"n_x": 2}, {"thermal_mass": 0.0}, {"newton_coeff": -1.0}],
)
def test_bad_arguments_raise(kwargs):
    with pytest.raises(ValueError):
        solve_heat(10.0, 25.0, **kwargs)


def test_joule_source_is_current_squared_times_resistance():
    from battery_pinn.fdm import joule_source

    assert joule_source(3.0, 0.5) == pytest.approx(4.5)
    assert joule_source(-3.0, 0.5) == pytest.approx(4.5)


def test_simulate_command_prints_uniform_heating_profile(capsys):
    import time

    from battery_pinn import config
    from battery_pinn.cli import main

    start = time.perf_counter()
    main(["simulate", "--current", "10", "--resistance", "2", "--ambient", "25", "--duration", "100"])
    assert time.perf_counter() - start < 1.0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if not ln.startswith("#")]
    temps = [float(ln.split()[1]) for ln in lines]
    assert len(temps) == 11
    expected = 25.0 + 200.0 * 100.0 / (config.RHO * config.CP)  # insulated: uniform ramp
    assert temps == pytest.approx([expected] * 11, abs=1e-3)
