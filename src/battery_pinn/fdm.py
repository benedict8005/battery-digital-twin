"""Plain NumPy finite-difference solver for the one-dimensional heat equation.

It is an independent check of the thermal part of the PDE in :mod:`battery_pinn.physics`::

    rho*cp * dT/dt = k * d2T/dx2 + q(t) - h * (T - T_ambient)

with ``x`` the normalised space coordinate in [0, 1] (as in the network inputs), ``q`` a
volumetric heat source in W/m3 (Joule heat plus any SEI heat) and ``h`` the optional Newton
cooling coefficient (0 disables it). The default boundary condition is zero flux (insulated)
at both ends; ``boundary="dirichlet"`` instead pins both ends to the ambient temperature.
Time stepping is backward Euler, so it is unconditionally stable. No PyTorch is needed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from battery_pinn import config

FloatArray: TypeAlias = NDArray[np.float64]
Source = float | Callable[[float], float]


def joule_source(current: float, resistance: float) -> float:
    """Constant Joule heat ``I * (V - U_ocv) = I**2 * R`` as used by the PDE (W/m3)."""
    return float(current) ** 2 * float(resistance)


def _march(
    t_end: float,
    ambient_c: float,
    source: Source = 0.0,
    *,
    n_x: int = 21,
    n_t: int = 200,
    k_cond: float = config.K_COND,
    thermal_mass: float = config.RHO * config.CP,
    newton_coeff: float = 0.0,
    t_initial_c: float | None = None,
    boundary: str = "neumann",
    a_sei: float = 0.0,
    c_initial: float = 1.0,
    ea_sei: float = config.EA_SEI,
    h_sei: float = config.H_SEI,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Shared time-marching loop; returns ``(x, t, T, c)``."""
    if boundary not in ("neumann", "dirichlet"):
        raise ValueError(f"boundary must be 'neumann' or 'dirichlet', got {boundary!r}")
    if n_x < 3 or n_t < 1:
        raise ValueError("need n_x >= 3 and n_t >= 1")
    if a_sei < 0 or not 0.0 <= c_initial <= 1.0:
        raise ValueError("a_sei must be >= 0 and c_initial in [0, 1]")
    if t_end <= 0 or k_cond < 0 or thermal_mass <= 0 or newton_coeff < 0:
        raise ValueError("t_end and thermal_mass must be positive; k_cond and newton_coeff >= 0")

    q = source if callable(source) else (lambda _t, _v=float(source): _v)
    x = np.linspace(0.0, 1.0, n_x)
    t = np.linspace(0.0, t_end, n_t + 1)
    dx, dt = x[1] - x[0], t[1] - t[0]

    # Second-difference operator L (so d2T/dx2 ~ L @ T); ghost-node mirror for zero flux.
    lap = np.zeros((n_x, n_x))
    for i in range(1, n_x - 1):
        lap[i, i - 1], lap[i, i], lap[i, i + 1] = 1.0, -2.0, 1.0
    lap /= dx**2
    if boundary == "neumann":
        lap[0, 0], lap[0, 1] = -2.0 / dx**2, 2.0 / dx**2
        lap[-1, -1], lap[-1, -2] = -2.0 / dx**2, 2.0 / dx**2

    system = np.eye(n_x) * (thermal_mass / dt + newton_coeff) - k_cond * lap
    if boundary == "dirichlet":
        system[0, :] = 0.0
        system[-1, :] = 0.0
        system[0, 0] = system[-1, -1] = 1.0

    temp = np.full((n_t + 1, n_x), float(ambient_c if t_initial_c is None else t_initial_c))
    stab = np.full((n_t + 1, n_x), float(c_initial))
    for n in range(n_t):
        # SEI terms are lagged one step (explicit): rate = A c exp(-Ea / (R T)), dc/dt = -rate.
        t_kelvin = np.clip(temp[n] + 273.15, 273.15, 600.0)
        rate = a_sei * stab[n] * np.exp(-ea_sei / (config.R_GAS * t_kelvin))
        stab[n + 1] = np.clip(stab[n] - dt * rate, 0.0, 1.0)
        rhs = thermal_mass / dt * temp[n] + q(t[n + 1]) + h_sei * rate + newton_coeff * ambient_c
        if boundary == "dirichlet":
            rhs[0] = rhs[-1] = ambient_c
        temp[n + 1] = np.linalg.solve(system, rhs)
    return x, t, temp, stab


def solve_heat(
    t_end: float,
    ambient_c: float,
    source: Source = 0.0,
    *,
    n_x: int = 21,
    n_t: int = 200,
    k_cond: float = config.K_COND,
    thermal_mass: float = config.RHO * config.CP,
    newton_coeff: float = 0.0,
    t_initial_c: float | None = None,
    boundary: str = "neumann",
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Integrate the heat equation from ``t=0`` to ``t_end`` seconds.

    ``source`` is a constant or a function of time in seconds (W/m3). The initial temperature
    defaults to ``ambient_c``. Returns ``(x, t, T)`` with ``T`` of shape ``(n_t + 1, n_x)`` in
    degrees Celsius.
    """
    x, t, temp, _ = _march(
        t_end,
        ambient_c,
        source,
        n_x=n_x,
        n_t=n_t,
        k_cond=k_cond,
        thermal_mass=thermal_mass,
        newton_coeff=newton_coeff,
        t_initial_c=t_initial_c,
        boundary=boundary,
    )
    return x, t, temp


def solve_heat_sei(
    t_end: float,
    ambient_c: float,
    a_sei: float,
    source: Source = 0.0,
    *,
    c_initial: float = 1.0,
    ea_sei: float = config.EA_SEI,
    h_sei: float = config.H_SEI,
    **kwargs: float | str | None,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Heat equation coupled to SEI decay, mirroring ``sei_thermal_pde``.

    Adds the heat ``h_sei * rate`` to the source and integrates ``dc/dt = -rate`` with
    ``rate = a_sei * c * exp(-ea_sei / (R * T))`` (T in kelvin, clamped to [273.15, 600]).
    Other keyword arguments are those of :func:`solve_heat`. Returns ``(x, t, T, c)``.
    """
    return _march(
        t_end,
        ambient_c,
        source,
        a_sei=a_sei,
        c_initial=c_initial,
        ea_sei=ea_sei,
        h_sei=h_sei,
        **kwargs,  # type: ignore[arg-type]
    )
