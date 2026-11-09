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


def test_sei_off_matches_plain_solver():
    from battery_pinn.fdm import solve_heat_sei

    _, _, temp = solve_heat(300.0, 25.0, 5.0e3, n_t=50)
    _, _, temp2, c = solve_heat_sei(300.0, 25.0, 0.0, 5.0e3, n_t=50)
    assert np.array_equal(temp, temp2)
    assert np.all(c == 1.0)


def test_sei_decay_matches_exponential_at_fixed_temperature():
    # Huge conduction and thermal mass keep T pinned at ambient, so dc/dt = -k c with
    # k = A exp(-Ea / (R T)) and c(t) = exp(-k t) (explicit Euler, small dt).
    from battery_pinn import config
    from battery_pinn.fdm import solve_heat_sei

    t_kelvin = 300.0
    k_rate = 2.0e-3
    a_sei = k_rate / np.exp(-config.EA_SEI / (config.R_GAS * t_kelvin))
    t_end = 600.0
    _, t, temp, c = solve_heat_sei(t_end, t_kelvin - 273.15, a_sei, 0.0, n_t=6000, thermal_mass=1e30, h_sei=0.0)
    assert np.allclose(temp, t_kelvin - 273.15)
    assert np.allclose(c[:, 0], np.exp(-k_rate * t), atol=2e-3)


def test_sei_reaction_heats_the_cell_by_enthalpy_released():
    # Insulated, no cooling: rho cp dT = h_sei * (c0 - c) per unit volume, so
    # T_end - T0 = h_sei * (1 - c_end) / rho_cp, with the (tiny) T feedback on the rate ignored
    # by choosing a rate that is nearly constant in temperature over the run.
    from battery_pinn import config
    from battery_pinn.fdm import solve_heat_sei

    rho_cp, h_sei = 2.0e6, 1.0e6
    a_sei = 1.0e-3 / np.exp(-config.EA_SEI / (config.R_GAS * 298.15))
    _, _, temp, c = solve_heat_sei(200.0, 25.0, a_sei, 0.0, n_t=4000, thermal_mass=rho_cp, h_sei=h_sei)
    assert np.allclose(temp[-1] - 25.0, h_sei * (1.0 - c[-1]) / rho_cp, rtol=0.02)


def test_sei_argument_validation():
    from battery_pinn.fdm import solve_heat_sei

    with pytest.raises(ValueError):
        solve_heat_sei(10.0, 25.0, -1.0)
    with pytest.raises(ValueError):
        solve_heat_sei(10.0, 25.0, 1.0, c_initial=1.5)


def test_simulate_without_sei_option_has_two_columns(capsys):
    from battery_pinn.cli import main

    main(["simulate", "--duration", "50"])
    lines = [ln for ln in capsys.readouterr().out.splitlines() if not ln.startswith("#")]
    assert all(len(ln.split()) == 2 for ln in lines)


def test_simulate_sei_option_matches_solver(capsys):
    from battery_pinn.cli import main
    from battery_pinn.fdm import joule_source, solve_heat_sei

    log_a = 12.0
    main(["simulate", "--current", "5", "--resistance", "1", "--duration", "300", "--log-a-sei", str(log_a)])
    lines = [ln for ln in capsys.readouterr().out.splitlines() if not ln.startswith("#")]
    rows = np.array([[float(v) for v in ln.split()] for ln in lines])
    assert rows.shape == (11, 3)

    _, _, temp, c = solve_heat_sei(300.0, 25.0, 10.0**log_a, joule_source(5.0, 1.0), n_x=11, n_t=200)
    assert rows[:, 1] == pytest.approx(temp[-1], abs=1e-4)
    assert rows[:, 2] == pytest.approx(c[-1], abs=1e-6)
    assert np.all(rows[:, 2] <= 1.0)


@pytest.mark.parametrize(
    "extra",
    [
        ["--newton-cooling", "--boundary", "dirichlet"],
        ["--log-a-sei", "-2.0"],
        ["--newton-cooling", "--log-a-sei", "-2.0", "--duration", "600"],
    ],
)
def test_simulate_option_variants_run_under_a_second(extra, capsys):
    import time

    from battery_pinn.cli import main

    start = time.perf_counter()
    main(["simulate", "--current", "5", "--resistance", "2", *extra])
    assert time.perf_counter() - start < 1.0
    assert capsys.readouterr().out.strip()


def test_cooling_error_shrinks_in_proportion_to_step_size():
    # docs/finite_difference_check.md states the scheme is first order in time:
    # doubling n_t should roughly halve the error against the exponential.
    h, t_end = 2.0e3, 1.0e3

    def max_error(n_t):
        _, t, temp = solve_heat(t_end, 25.0, 0.0, newton_coeff=h, thermal_mass=RHO_CP, t_initial_c=60.0, n_t=n_t)
        return np.max(np.abs(temp[:, 0] - (25.0 + 35.0 * np.exp(-h * t / RHO_CP))))

    coarse, fine = max_error(500), max_error(1000)
    assert fine < coarse
    assert 1.8 < coarse / fine < 2.2
