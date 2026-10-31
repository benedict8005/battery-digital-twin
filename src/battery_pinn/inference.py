"""Restore a trained checkpoint, validate on held-out batteries, and run the abuse test."""

import deepxde as dde
import numpy as np

from battery_pinn import config
from battery_pinn.checkpoint import load_checkpoint
from battery_pinn.data import build_points_and_targets
from battery_pinn.training import predict


class DigitalTwin:
    """A restored network plus its calibrated parameters."""

    def __init__(self, checkpoint_path):
        checkpoint = load_checkpoint(checkpoint_path)
        self.net = dde.nn.FNN(config.NETWORK_LAYERS, "tanh", "Glorot normal")
        self.net.float()
        self.net.load_state_dict(checkpoint["model_state_dict"])
        self.log_a_sei = checkpoint["log_A_sei"]
        self.ocv_coeffs = np.array(checkpoint["ocv_coeffs"])
        self.checkpoint_path = checkpoint_path

    @property
    def a_sei(self):
        return 10**self.log_a_sei

    def predict(self, points):
        return predict(self.net, points)


def danger_threshold(twin, train_df, n_std=3):
    """Data-driven SEI-stability threshold: mean - n_std * std of clipped in-domain predictions."""
    points, _ = build_points_and_targets(train_df)
    baseline_c = np.clip(twin.predict(points)[:, 1], 0.0, 1.0)
    return float(max(baseline_c.mean() - n_std * baseline_c.std(), 0.0))


def evaluate(twin, df):
    """Temperature MAE / RMSE in approximate degrees C, overall and per battery."""
    points, targets = build_points_and_targets(df)
    err = (targets - twin.predict(points)[:, 0]) * config.TEMP_SCALE_C
    per_battery = {
        bid: float(np.abs(err[df["battery_id"].values == bid]).mean()) for bid in sorted(df["battery_id"].unique())
    }
    return {
        "mae_c": float(np.abs(err).mean()),
        "rmse_c": float(np.sqrt((err**2).mean())),
        "per_battery_mae_c": per_battery,
    }


def abuse_test(twin, train_df, soh=0.8, re=0.5, ambient_start=25.0, ambient_end=150.0, n_points=100):
    """Ambient-stress sweep at full exposure; the verdict uses in-domain points only."""
    threshold = danger_threshold(twin, train_df)
    ambient = np.linspace(ambient_start, ambient_end, n_points)
    in_domain = ambient <= config.AMBIENT_TRAINED_MAX_C
    if not in_domain.any():
        raise ValueError("The sweep has no points inside the trained ambient domain.")

    inputs = np.column_stack(
        (
            np.zeros(n_points),
            np.full(n_points, soh),
            np.full(n_points, re),
            ambient,
            np.full(n_points, 1.0),
        )
    ).astype(np.float32)
    res = twin.predict(inputs)
    core_temp = res[:, 0] * config.TEMP_SCALE_C + ambient
    sei_health = np.clip(res[:, 1], 0.0, 1.0)

    temp_max = float(core_temp[in_domain].max())
    sei_min = float(sei_health[in_domain].min())
    verdict, criterion = classify(temp_max, sei_min, threshold)
    return {
        "verdict": verdict,
        "criterion": criterion,
        "core_temp_max_c": temp_max,
        "sei_min": sei_min,
        "threshold": threshold,
        "ambient": ambient,
        "in_domain": in_domain,
        "ambient_max_in_domain_c": float(config.AMBIENT_TRAINED_MAX_C),
        "core_temp": core_temp,
        "sei_health": sei_health,
        "caveats": CAVEATS,
    }


CAVEATS = (
    f"core temperature is UNVALIDATED (inferred through the physics, never compared with an internal sensor); "
    f"the verdict EXCLUDES ambient > {config.AMBIENT_TRAINED_MAX_C:.0f} °C (outside the trained domain, extrapolation only)"
)

CRITERIA = {
    "core_temp_danger": f"core temperature (unvalidated) > {config.CORE_TEMP_DANGER_C:.0f} °C",
    "sei_below_threshold": "SEI stability min < data-driven threshold",
    "both": f"core temperature (unvalidated) > {config.CORE_TEMP_DANGER_C:.0f} °C and SEI min < threshold",
    "core_temp_warning": f"core temperature (unvalidated) > {config.CORE_TEMP_WARNING_C:.0f} °C",
    "none": "no criterion triggered",
}


def classify(temp_max, sei_min, threshold):
    """The verdict and the criterion key (see ``CRITERIA``) that decided it."""
    temp_danger = temp_max > config.CORE_TEMP_DANGER_C
    sei_danger = sei_min < threshold
    if temp_danger and sei_danger:
        return "DANGER", "both"
    if temp_danger:
        return "DANGER", "core_temp_danger"
    if sei_danger:
        return "DANGER", "sei_below_threshold"
    if temp_max > config.CORE_TEMP_WARNING_C:
        return "WARNING", "core_temp_warning"
    return "SAFE", "none"


def summarise_ensemble(results):
    """Counts per verdict, per criterion, and the spread of the data-driven threshold.

    ``results`` maps a name (e.g. ``seed3``) to an :func:`abuse_test` result. Returns a dict
    with ``verdict_counts``, ``criterion_counts``, ``threshold`` statistics, ``core_temp_max_c``
    and ``sei_min`` statistics, ``consistent`` and the ``caveats`` string.
    """
    verdicts = [r["verdict"] for r in results.values()]
    out = {
        "n": len(results),
        "verdict_counts": {v: verdicts.count(v) for v in ("DANGER", "WARNING", "SAFE")},
        "criterion_counts": {k: sum(1 for r in results.values() if r["criterion"] == k) for k in CRITERIA},
        "consistent": len(set(verdicts)) <= 1,
        "caveats": CAVEATS,
    }
    for key in ("threshold", "core_temp_max_c", "sei_min"):
        values = np.array([r[key] for r in results.values()], dtype=float)
        out[key] = (
            {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
            }
            if len(values)
            else None
        )
    return out


def format_ensemble_summary(summary):
    c, k = summary["verdict_counts"], summary["criterion_counts"]
    lines = [
        f"Abuse-test ensemble ({summary['n']} models): {c['DANGER']} DANGER / {c['WARNING']} WARNING / {c['SAFE']} SAFE"
        + ("  (consistent)" if summary["consistent"] else "  <- NOT consistent across seeds"),
        "Deciding criterion: " + ", ".join(f"{k[key]} x {CRITERIA[key]}" for key in CRITERIA if k[key]),
    ]
    if summary["threshold"]:
        t = summary["threshold"]
        lines.append(
            f"SEI threshold (mean - 3 std of each model's own in-domain c): "
            f"{t['min']:.3f} .. {t['max']:.3f} (mean {t['mean']:.3f}, std {t['std']:.3f})"
        )
        ct, sm = summary["core_temp_max_c"], summary["sei_min"]
        lines.append(
            f"Core temperature max (UNVALIDATED): {ct['min']:.1f} .. {ct['max']:.1f} °C; "
            f"SEI min: {sm['min']:.3f} .. {sm['max']:.3f}"
        )
    lines.append("Caveats: " + CAVEATS + ".")
    return "\n".join(lines)


def plot_abuse_test(results, path):
    """Save an overlay of SEI health vs core temperature (in-domain solid, extrapolated dashed)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    colors = plt.cm.tab10.colors
    for i, (name, r) in enumerate(results.items()):
        m = r["in_domain"]
        color = colors[i % len(colors)]
        plt.plot(r["core_temp"][m], r["sei_health"][m], color=color, linewidth=2, label=f"{name} ({r['verdict']})")
        if not m.all():
            k = max(int(np.argmax(~m)) - 1, 0)
            plt.plot(r["core_temp"][k:], r["sei_health"][k:], "--", color=color, alpha=0.4)
    plt.axvline(config.CORE_TEMP_DANGER_C, color="r", linestyle="--", label="Critical temperature")
    plt.xlabel("Internal (core) temperature, degC -- UNVALIDATED, inferred through the physics")
    plt.ylabel("Chemical stability (c)")
    plt.title(
        f"Abuse test: verdict excludes ambient > {config.AMBIENT_TRAINED_MAX_C:.0f} C "
        "(dashed = extrapolated beyond the trained ambient range)"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
