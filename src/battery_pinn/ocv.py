"""Open-circuit-voltage curve: polynomial fit and torch evaluation."""

import numpy as np
import torch


def fit_ocv_coeffs(df_agg, degree=4):
    """Fit OCV(SoC) with a polynomial against the representative discharge curve.

    SoC is normalised to [0, 1]; coefficients are returned highest power first.
    """
    soc = np.clip(df_agg["SoC"].values / 100.0, 1e-3, 1 - 1e-3)
    return np.polyfit(soc, df_agg["Voltage_measured"].values, deg=degree)


def ocv_from_soc_torch(soc_tensor, coeffs):
    """Evaluate the fitted OCV polynomial on a tensor of SoC fractions in [0, 1]."""
    soc = torch.clamp(soc_tensor, 1e-3, 1 - 1e-3)
    ocv = torch.zeros_like(soc)
    degree = len(coeffs) - 1
    for power, coeff in zip(range(degree, -1, -1), coeffs):
        ocv = ocv + float(coeff) * torch.pow(soc, power)
    return ocv
