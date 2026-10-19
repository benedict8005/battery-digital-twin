"""A_sei identifiability assessment across an ensemble of seeds (pure Python)."""
import pytest

from battery_pinn import config
from battery_pinn.identifiability import assess_log_a_sei, format_assessment, spread_initial_values


def test_spread_initial_values():
    assert spread_initial_values(3, 1.0, 22.0) == [21.0, 22.0, 23.0]
    assert spread_initial_values(4, 0.5, 22.0) == pytest.approx([21.5, 21.8333333, 22.1666667, 22.5])
    assert spread_initial_values(3, 0.0, 22.0) == [22.0, 22.0, 22.0]
    assert spread_initial_values(1, 2.0, 22.0) == [22.0]


def test_stuck_at_initial_value_is_not_identifiable():
    a = assess_log_a_sei([22.88, 22.87, 22.89], [22.88, 22.88, 22.88])
    assert a["status"] == "not identifiable" and "insensitive" in a["reason"]
    assert a["shift"]["max"] == pytest.approx(0.01)


def test_seed_dependent_values_are_not_identifiable():
    a = assess_log_a_sei([21.0, 22.5, 24.0], [22.0, 22.0, 22.0])
    assert a["status"] == "not identifiable" and "depends on the seed" in a["reason"]
    assert a["final"]["range"] == pytest.approx(3.0)


def test_convergence_from_spread_starts_is_identifiable():
    a = assess_log_a_sei([22.10, 22.05, 22.20], [21.0, 22.0, 23.0])
    assert a["status"] == "identifiable"
    assert "started 2.00 decades apart" in a["reason"]
    assert a["final"]["mean"] == pytest.approx(22.1166667)


def test_agreement_from_a_common_start_is_only_consistency():
    a = assess_log_a_sei([22.3, 22.4, 22.35], [22.88, 22.88, 22.88])
    assert a["status"] == "consistent across seeds"
    assert "--a-sei-init-spread" in a["reason"]
    # Without known starting points the verdict is also only consistency.
    b = assess_log_a_sei([22.3, 22.4, 22.35])
    assert b["status"] == "consistent across seeds" and "initial values unknown" in b["reason"]
    assert b["initial"] is None and b["shift"] is None


def test_thresholds_come_from_config_and_can_be_overridden():
    finals, inits = [22.0, 22.4], [21.0, 23.0]
    assert assess_log_a_sei(finals, inits)["status"] == "identifiable"   # 0.4 < 0.5 default
    assert assess_log_a_sei(finals, inits, max_spread=0.3)["status"] == "not identifiable"
    assert assess_log_a_sei([22.0, 22.0], [22.0, 22.05], min_movement=0.1)["status"] == "not identifiable"
    assert config.IDENTIFIABLE_MAX_SPREAD_DECADES == 0.5 and config.IDENTIFIABLE_MIN_MOVEMENT_DECADES == 0.1


def test_single_seed_is_inconclusive_and_format_is_readable():
    a = assess_log_a_sei([22.0], [21.0])
    assert a["status"] == "inconclusive"
    text = format_assessment(assess_log_a_sei([22.1, 22.0, 22.2], [21.0, 22.0, 23.0]))
    assert text.startswith("A_sei identifiability (3 seeds): IDENTIFIABLE")
    assert "final log10(A_sei): mean 22.100" in text and "initial log10(A_sei): range [21.000, 23.000]" in text


def test_runaway_in_one_direction_is_not_identifiable_even_if_seeds_agree():
    a = assess_log_a_sei([14.7, 14.9, 14.8], [21.9, 22.9, 23.9])
    assert a["status"] == "not identifiable" and a["reason"].startswith("runaway")
    assert "towards A_sei = 0" in a["reason"]
    # Mixed directions or small drifts are judged by the other rules.
    assert assess_log_a_sei([18.0, 26.0], [22.0, 22.0])["reason"].startswith("seeds disagree")
    assert assess_log_a_sei([22.1, 22.0], [21.0, 23.0])["status"] == "identifiable"
    assert config.IDENTIFIABLE_MAX_RUNAWAY_DECADES == 3.0
    assert assess_log_a_sei([14.7, 14.9], [21.9, 22.9], max_runaway=10.0)["status"] == "identifiable"
