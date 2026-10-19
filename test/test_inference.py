import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("deepxde")

from battery_pinn import config, synthetic
from battery_pinn.inference import abuse_test


class StubTwin:
    """Predicts a fixed T_delta and c so the verdict logic can be tested in isolation."""

    def __init__(self, t_delta, c, in_domain_c=1.0):
        self.t_delta, self.c, self.in_domain_c = t_delta, c, in_domain_c

    def predict(self, points):
        n = len(points)
        out = np.zeros((n, 2))
        out[:, 0] = self.t_delta
        out[:, 1] = np.where(points[:, 0] == 0.0, self.c, self.in_domain_c)
        return out


@pytest.fixture(scope="module")
def train_df():
    df, _ = synthetic.generate_train_and_held_out(n_batteries=3, n_held_out=1, cycles_per_battery=3)
    return df


def test_safe(train_df):
    r = abuse_test(StubTwin(t_delta=0.0, c=1.0), train_df)
    assert r["verdict"] == "SAFE"


def test_danger_from_core_temperature(train_df):
    r = abuse_test(StubTwin(t_delta=1.0, c=1.0), train_df)
    assert r["verdict"] == "DANGER"


def test_warning_band(train_df):
    r = abuse_test(StubTwin(t_delta=0.4, c=1.0), train_df)
    assert r["verdict"] == "WARNING"


def test_danger_from_sei_collapse(train_df):
    r = abuse_test(StubTwin(t_delta=0.0, c=0.0), train_df)
    assert r["verdict"] == "DANGER"


def test_extrapolated_points_do_not_drive_verdict(train_df):
    r = abuse_test(StubTwin(t_delta=0.0, c=1.0), train_df)
    assert (~r["in_domain"]).any()
    assert r["core_temp"].max() > config.CORE_TEMP_DANGER_C
    assert r["verdict"] == "SAFE"


def test_classify_reports_the_deciding_criterion():
    from battery_pinn.inference import CRITERIA, classify

    assert classify(50.0, 0.9, 0.8) == ("SAFE", "none")
    assert classify(65.0, 0.9, 0.8) == ("WARNING", "core_temp_warning")
    assert classify(85.0, 0.9, 0.8) == ("DANGER", "core_temp_danger")
    assert classify(50.0, 0.7, 0.8) == ("DANGER", "sei_below_threshold")
    assert classify(85.0, 0.7, 0.8) == ("DANGER", "both")
    assert set(CRITERIA) == {"core_temp_danger", "sei_below_threshold", "both", "core_temp_warning", "none"}
    assert "unvalidated" in CRITERIA["core_temp_danger"]


def test_abuse_test_result_carries_criterion_and_caveats(train_df):
    r = abuse_test(StubTwin(t_delta=0.0, c=0.0), train_df)
    assert r["verdict"] == "DANGER" and r["criterion"] == "sei_below_threshold"
    assert r["ambient_max_in_domain_c"] == config.AMBIENT_TRAINED_MAX_C
    assert "UNVALIDATED" in r["caveats"] and "EXCLUDES ambient > 45" in r["caveats"]


def test_ensemble_summary_counts_verdicts_criteria_and_threshold_spread(train_df):
    from battery_pinn.inference import format_ensemble_summary, summarise_ensemble

    results = {
        "seed0": abuse_test(StubTwin(t_delta=0.0, c=1.0), train_df),   # SAFE
        "seed1": abuse_test(StubTwin(t_delta=0.4, c=1.0), train_df),   # WARNING (core T: 20 + 45 = 65 C)
        "seed2": abuse_test(StubTwin(t_delta=1.2, c=1.0), train_df),   # DANGER (core T)
        "seed3": abuse_test(StubTwin(t_delta=0.0, c=0.0), train_df),   # DANGER (SEI)
    }
    s = summarise_ensemble(results)
    assert s["n"] == 4 and s["verdict_counts"] == {"DANGER": 2, "WARNING": 1, "SAFE": 1}
    assert s["criterion_counts"]["core_temp_danger"] == 1 and s["criterion_counts"]["sei_below_threshold"] == 1
    assert s["criterion_counts"]["core_temp_warning"] == 1 and s["criterion_counts"]["none"] == 1
    assert s["consistent"] is False
    assert s["threshold"]["min"] <= s["threshold"]["mean"] <= s["threshold"]["max"]
    text = format_ensemble_summary(s)
    assert text.startswith("Abuse-test ensemble (4 models): 2 DANGER / 1 WARNING / 1 SAFE")
    assert "NOT consistent" in text and "Deciding criterion:" in text
    assert "SEI threshold" in text and "UNVALIDATED" in text and "EXCLUDES ambient > 45" in text
