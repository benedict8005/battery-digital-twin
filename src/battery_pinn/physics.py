"""Coupled thermal / SEI-stability PDE system and Arrhenius helpers."""

import deepxde as dde
import numpy as np
import torch

from battery_pinn import config
from battery_pinn.ocv import ocv_from_soc_torch


def arrhenius_exponents(temps_c, ea=config.EA_SEI, r_gas=config.R_GAS):
    """Unclamped Arrhenius exponent -Ea/(R*T) at the given Celsius temperatures."""
    temps_k = np.asarray(temps_c, dtype=float) + 273.15
    return -ea / (r_gas * temps_k)


def recommended_log_a_sei(ea=config.EA_SEI, r_gas=config.R_GAS, t_ref_k=config.REFERENCE_TEMP_K):
    """Starting log10(A_sei) that makes the reaction rate O(1) at the reference temperature."""
    exponent = -ea / (r_gas * t_ref_k)
    return float(-exponent / np.log(10))


def make_thermal_variables():
    """Trainable log10 thermal constants (``--learn-thermal``), initialised at the config values.

    Returns ``{"log10_k_cond", "log10_thermal_mass", "log10_newton_coeff"}`` as dde.Variables;
    pass the dict to :func:`make_pde` and its values to ``external_trainable_variables``.
    """
    return {
        "log10_k_cond": dde.Variable(float(np.log10(config.K_COND))),
        "log10_thermal_mass": dde.Variable(float(np.log10(config.RHO * config.CP))),
        "log10_newton_coeff": dde.Variable(float(np.log10(config.NEWTON_COOLING_COEFF))),
    }


def make_pde(profile, ocv_coeffs, log_a_sei, t_max_seconds, newton_cooling=False, thermal=None):
    """Build the PDE residual function for DeepXDE.

    Input columns are [space(0), SoH(1), Re(2), Ambient(3), time(4)];
    outputs are [T_delta, c]. ``log_a_sei`` is a dde.Variable (trainable) or a fixed tensor.

    Opt-in variants, both off by default so the residual is exactly the original one:

    * ``newton_cooling`` adds a surface heat-loss sink ``-h (T - T_ambient)`` to the heat
      sources, with ``h = NEWTON_COOLING_COEFF`` (or its learnable counterpart);
    * ``thermal`` (from :func:`make_thermal_variables`) replaces ``K_COND``, ``RHO * CP`` and
      ``h`` by trainable log10 variables.
    """

    def constants():
        # Evaluated inside the residual so the autograd graph is rebuilt on every call.
        if thermal is None:
            return float(config.K_COND), float(config.RHO * config.CP), float(config.NEWTON_COOLING_COEFF)
        return (
            torch.pow(10.0, thermal["log10_k_cond"]),
            torch.pow(10.0, thermal["log10_thermal_mass"]),
            torch.pow(10.0, thermal["log10_newton_coeff"]),
        )

    def sei_thermal_pde(x, y):
        k_cond, thermal_mass, newton_coeff = constants()
        T_delta = y[:, 0:1]
        c = y[:, 1:2]

        T_ref_point = x[:, 3:4] + 273.15
        T = T_delta * config.TEMP_SCALE_C + T_ref_point
        T_safe = torch.clamp(T, min=273.15, max=600.0)
        c_safe = torch.clamp(c, min=0.0, max=1.0)

        dT_t = dde.grad.jacobian(y, x, i=0, j=4) / t_max_seconds
        dc_t = dde.grad.jacobian(y, x, i=1, j=4) / t_max_seconds
        dT_xx = dde.grad.hessian(y, x, i=0, j=0)

        I, V, soc = profile.at(x[:, 4:5])
        U_ocv = ocv_from_soc_torch(soc / 100.0, ocv_coeffs)
        actual_A_sei = torch.pow(10.0, log_a_sei)
        joule_heat = torch.abs(I * (V - U_ocv))

        exponent = -config.EA_SEI / (config.R_GAS * T_safe + 1e-6)
        rate = (
            actual_A_sei
            * c_safe
            * torch.exp(torch.clamp(exponent, min=config.EXPONENT_CLAMP_MIN, max=config.EXPONENT_CLAMP_MAX))
        )

        sources = k_cond * dT_xx * config.TEMP_SCALE_C + joule_heat + config.H_SEI * rate
        if newton_cooling:
            sources = sources - newton_coeff * (T - T_ref_point)
        res_thermal = (dT_t * config.TEMP_SCALE_C) - sources / (thermal_mass + 1e-6)
        res_sei = dc_t + rate
        return [res_thermal, res_sei]

    return sei_thermal_pde
