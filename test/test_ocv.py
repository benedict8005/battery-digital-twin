import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from battery_pinn.ocv import fit_ocv_coeffs, ocv_from_soc_torch


def test_fit_recovers_known_polynomial():
    soc = np.linspace(0.01, 0.99, 200)
    true = np.array([0.5, -1.0, 0.8, 1.2, 3.0])
    df = pd.DataFrame({"SoC": soc * 100.0, "Voltage_measured": np.polyval(true, soc)})
    np.testing.assert_allclose(fit_ocv_coeffs(df), true, atol=1e-6)


def test_torch_evaluation_matches_polyval():
    coeffs = np.array([0.5, -1.0, 0.8, 1.2, 3.0])
    soc = np.linspace(0.05, 0.95, 30)
    out = ocv_from_soc_torch(torch.tensor(soc, dtype=torch.float32), coeffs).numpy()
    np.testing.assert_allclose(out, np.polyval(coeffs, soc), atol=1e-4)


def test_input_is_clamped_to_valid_range():
    coeffs = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
    out = ocv_from_soc_torch(torch.tensor([-1.0, 2.0]), coeffs).numpy()
    assert out[0] == pytest.approx(1e-3, abs=1e-6)
    assert out[1] == pytest.approx(1 - 1e-3, abs=1e-6)
