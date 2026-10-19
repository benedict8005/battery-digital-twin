#!/usr/bin/env python
"""Compile trained-checkpoint metrics into ``docs/RESULTS.md``.

For every run (a folder of ``battery_pinn_model_seed*.pt`` checkpoints plus the processed
tables it was trained on) the script restores each seed's digital twin and collects:

* checkpoint metrics -- calibrated ``A_sei``, temperature MAE / RMSE on the training and
  held-out tables;
* per-battery temperature MAE on both tables;
* the abuse-test verdict (core-temperature maximum, minimum SEI stability, the data-driven
  threshold) and, across seeds, the verdict spread.

Several runs are rendered side by side so that, for example, the legacy 1.4 Ah capacity floor
(19 training / 5 held-out batteries) can be compared with the corrected dynamic floor (24 / 6).

Usage::

    python scripts/make_results_table.py \\
        --run legacy  runs/legacy/artifacts  runs/legacy/data \\
        --run dynamic runs/dynamic/artifacts runs/dynamic/data \\
        --out docs/RESULTS.md --json docs/RESULTS.json --protocol "1000 Adam iterations, ..."

Each ``--run`` is ``NAME ARTIFACTS_DIR DATA_DIR``; ``DATA_DIR`` must hold
``train.csv`` and ``held_out.csv`` (or the pre-rename table names) as written by
``battery-pinn prepare-data``.
"""
import argparse
import datetime as dt
import glob
import json
import math
import os
import re
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.isdir(os.path.join(_REPO_ROOT, "src")):
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from battery_pinn import config  # noqa: E402
from battery_pinn.data import battery_report, load_dataset  # noqa: E402
from battery_pinn.identifiability import assess_log_a_sei  # noqa: E402

CHECKPOINT_PATTERN = re.compile(r"battery_pinn_model_seed(\d+)\.pt$")

# The only numeric accuracy figures recorded for the Databricks pipeline (v9 notebook run,
# legacy 1.4 Ah floor -> 19 training / 5 held-out batteries, full-anchor protocol,
# 15000 Adam + 2000 L-BFGS). Surface-temperature MAE in degrees C.
DATABRICKS_V9 = {
    "label": "Databricks v9 (19/5 split, full anchors, 15000 + 2000)",
    "train_mae_c": 1.2,
    "held_out_mae_c": 7.65,
    "per_battery_held_out_mae_c": {"B0025": 4.2, "B0030": 1.1, "B0036": 11.4, "B0042": 0.9, "B0047": 7.3},
}
CORE_T_LABEL = "core T max (°C, unvalidated)"
VERDICT_NOTE = (f"Core temperature is **unvalidated** (inferred through the physics, never compared with an "
                f"internal sensor). The verdict uses in-domain sweep points only, i.e. it **excludes ambient > "
                f"{config.AMBIENT_TRAINED_MAX_C:.0f} °C**; it is a heuristic, not a certified safety result.")
VERDICTS = ("DANGER", "WARNING", "SAFE")
SEED_METRICS = ("train_mae_c", "train_rmse_c", "held_out_mae_c", "held_out_rmse_c",
                "core_temp_max_c", "sei_min", "threshold")


# ----------------------------------------------------------------------------------------
# Collection
# ----------------------------------------------------------------------------------------
def find_checkpoints(artifacts_dir):
    """``{seed: path}`` for every ``battery_pinn_model_seed*.pt`` in the folder, sorted by seed."""
    found = {}
    for path in glob.glob(os.path.join(artifacts_dir, "battery_pinn_model_seed*.pt")):
        match = CHECKPOINT_PATTERN.search(os.path.basename(path))
        if match:
            found[int(match.group(1))] = path
    return dict(sorted(found.items()))


def _coverage(path):
    """Batteries / cycles / rows of a processed table, before and after cleaning."""
    import pandas as pd

    raw = pd.read_csv(path)
    clean = load_dataset(path, log=None)
    report = battery_report(raw, clean)
    return {
        "batteries": sorted(report),
        "cycles": int(sum(r["cycles_kept"] for r in report.values())),
        "rows": int(len(clean)),
        "rows_in_file": int(len(raw)),
        "cycles_dropped_on_load": int(sum(r["cycles_dropped"] for r in report.values())),
    }, clean


def _training_from_history(artifacts_dir, seed):
    """Initial log10(A_sei) and final loss terms from ``loss_history_seed<k>.csv`` (a partially
    finished ensemble has these even before ``train_summary.json`` exists)."""
    path = os.path.join(artifacts_dir, f"loss_history_seed{seed}.csv")
    if not os.path.exists(path):
        return {}
    import pandas as pd

    history = pd.read_csv(path)
    if history.empty:
        return {}
    terms = [c for c in config.LOSS_TERM_NAMES if c in history.columns]
    return {
        "seed": seed,
        "log10_A_sei_init": float(history["log10_A_sei"].iloc[0]) if "log10_A_sei" in history else None,
        "final_losses": {c: float(history[c].iloc[-1]) for c in terms},
    }


def per_ambient_mae(twin, df):
    """Temperature MAE (°C) per ambient-temperature group, keys like ``"24 °C"``."""
    import numpy as np

    from battery_pinn.data import build_points_and_targets

    points, targets = build_points_and_targets(df)
    err = np.abs((targets - twin.predict(points)[:, 0]) * config.TEMP_SCALE_C)
    ambient = df["ambient_temperature"].values
    return {f"{int(round(a))} °C": float(err[ambient == a].mean()) for a in sorted(set(ambient))}


def collect_run(name, artifacts_dir, data_dir, twin_factory=None, log=print):
    """Restore every seed of a run and compute its metrics. Returns a JSON-serialisable dict."""
    from battery_pinn.inference import DigitalTwin, abuse_test, evaluate
    from battery_pinn.physics import recommended_log_a_sei

    twin_factory = twin_factory or DigitalTwin
    train_path = config.find_table(data_dir, config.TRAIN_TABLE)
    held_out_path = config.find_table(data_dir, config.HELD_OUT_TABLE)
    train_cov, train_df = _coverage(train_path)
    held_out_cov, held_out_df = _coverage(held_out_path)

    checkpoints = find_checkpoints(artifacts_dir)
    if not checkpoints:
        raise FileNotFoundError(f"{name}: no battery_pinn_model_seed*.pt in {artifacts_dir}")
    summary_path = os.path.join(artifacts_dir, "train_summary.json")
    training_summary = {}
    if os.path.exists(summary_path):
        with open(summary_path, encoding="utf-8") as f:
            training_summary = {int(r["seed"]): r for r in json.load(f)}

    seeds = []
    for seed, path in checkpoints.items():
        if log:
            log(f"[{name}] seed {seed}: {os.path.basename(path)}")
        twin = twin_factory(path)
        tr, ho = evaluate(twin, train_df), evaluate(twin, held_out_df)
        ab = abuse_test(twin, train_df)
        amb_tr, amb_ho = per_ambient_mae(twin, train_df), per_ambient_mae(twin, held_out_df)
        training = training_summary.get(seed) or _training_from_history(artifacts_dir, seed)
        # Runs trained before the init was recorded always started from the recommended value.
        init_assumed = training.get("log10_A_sei_init") is None
        seeds.append({
            "seed": seed,
            "checkpoint": path,
            "A_sei": float(twin.a_sei),
            "log10_A_sei": math.log10(float(twin.a_sei)) if float(twin.a_sei) > 0 else None,
            "log10_A_sei_init": recommended_log_a_sei() if init_assumed else training["log10_A_sei_init"],
            "log10_A_sei_init_assumed": init_assumed,
            "final_losses": training.get("final_losses") or {},
            "train_mae_c": tr["mae_c"], "train_rmse_c": tr["rmse_c"],
            "held_out_mae_c": ho["mae_c"], "held_out_rmse_c": ho["rmse_c"],
            "per_battery_train_mae_c": tr["per_battery_mae_c"],
            "per_battery_held_out_mae_c": ho["per_battery_mae_c"],
            "per_ambient_train_mae_c": amb_tr,
            "per_ambient_held_out_mae_c": amb_ho,
            "verdict": ab["verdict"],
            "criterion": ab.get("criterion", "unknown"),
            "core_temp_max_c": ab["core_temp_max_c"],
            "sei_min": ab["sei_min"],
            "threshold": ab["threshold"],
            "training": training,
        })
    return {
        "name": name,
        "artifacts_dir": artifacts_dir,
        "data_dir": data_dir,
        "train": train_cov,
        "held_out": held_out_cov,
        "seeds": seeds,
    }


# ----------------------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------------------
def mean_std(values):
    values = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not values:
        return None, None
    mean = sum(values) / len(values)
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)) if len(values) > 1 else 0.0
    return mean, std


def verdict_spread(seeds):
    counts = {v: 0 for v in VERDICTS}
    for s in seeds:
        counts[s["verdict"]] = counts.get(s["verdict"], 0) + 1
    return counts


def aggregate(run):
    """Mean / std across seeds of every scalar metric, the verdict spread and the per-battery means."""
    seeds = run["seeds"]
    out = {"n_seeds": len(seeds), "metrics": {}, "verdicts": verdict_spread(seeds)}
    for key in SEED_METRICS:
        out["metrics"][key] = mean_std([s[key] for s in seeds])
    out["metrics"]["log10_A_sei"] = mean_std([math.log10(s["A_sei"]) for s in seeds if s["A_sei"] > 0])
    gap = [s["held_out_mae_c"] - s["train_mae_c"] for s in seeds]
    out["metrics"]["generalisation_gap_c"] = mean_std(gap)
    out["verdict_consistent"] = len({s["verdict"] for s in seeds}) <= 1
    for table in ("train", "held_out"):
        key = f"per_battery_{table}_mae_c"
        batteries = sorted({b for s in seeds for b in s[key]})
        out[key] = {b: mean_std([s[key][b] for s in seeds if b in s[key]]) for b in batteries}
        akey = f"per_ambient_{table}_mae_c"
        groups = sorted({g for s in seeds for g in s.get(akey, {})}, key=lambda g: float(g.split()[0]))
        out[akey] = {g: mean_std([s[akey][g] for s in seeds if g in s.get(akey, {})]) for g in groups}
    out["criterion_counts"] = {}
    for s in seeds:
        out["criterion_counts"][s.get("criterion", "unknown")] = out["criterion_counts"].get(s.get("criterion", "unknown"), 0) + 1
    out["threshold"] = {k: v for k, v in zip(("mean", "std"), mean_std([s["threshold"] for s in seeds]))}
    thresholds = [s["threshold"] for s in seeds if s["threshold"] is not None]
    out["threshold"].update({"min": min(thresholds) if thresholds else None, "max": max(thresholds) if thresholds else None})
    finals = [s["log10_A_sei"] for s in seeds if s.get("log10_A_sei") is not None]
    inits = [s.get("log10_A_sei_init") for s in seeds if s.get("log10_A_sei") is not None]
    fixed = [s.get("training", {}).get("physics", {}).get("fix_log_a_sei") for s in seeds]
    if seeds and all(f is not None for f in fixed):
        out["identifiability"] = {"n_seeds": len(seeds), "status": "fixed (not trained)",
                                  "reason": f"log10(A_sei) held at {fixed[0]:g} with --fix-log-a-sei",
                                  "initial": None, "final": None, "shift": None}
    else:
        out["identifiability"] = assess_log_a_sei(finals, inits if any(i is not None for i in inits) else None)
    terms = {}
    for s in seeds:
        for k, v in (s.get("final_losses") or {}).items():
            terms.setdefault(k, []).append(v)
    out["loss_terms"] = {k: mean_std(v) for k, v in terms.items()}
    if out["loss_terms"]:
        total = sum(m for m, _ in out["loss_terms"].values()) or 1.0
        out["loss_shares"] = {k: m / total for k, (m, _) in out["loss_terms"].items()}
        out["dominant_term"] = max(out["loss_shares"], key=out["loss_shares"].get)
    else:
        out["loss_shares"], out["dominant_term"] = {}, None
    return out


def option_decision(baseline, option, n_worst=2):
    """Does a physics option earn its place? It must lower the held-out MAE of the baseline's
    ``n_worst`` worst batteries by more than the seed spread (the larger of the two runs' per-
    battery std) without degrading any other battery by more than its seed spread.

    Returns a dict with the per-battery verdicts and ``adopt`` (bool) plus ``reason``.
    """
    base, opt = aggregate(baseline)["per_battery_held_out_mae_c"], aggregate(option)["per_battery_held_out_mae_c"]
    common = [b for b in base if b in opt]
    worst = sorted(common, key=lambda b: -base[b][0])[:n_worst]
    rows = []
    for b in common:
        spread = max(base[b][1], opt[b][1])
        delta = opt[b][0] - base[b][0]
        if b in worst:
            status = "improved beyond spread" if delta < -spread else "not improved beyond spread"
        else:
            status = "unchanged within spread" if delta <= spread else "DEGRADED beyond spread"
        rows.append({"battery": b, "worst": b in worst, "baseline": base[b][0], "option": opt[b][0],
                     "delta": delta, "spread": spread, "status": status})
    worst_ok = all(r["status"] == "improved beyond spread" for r in rows if r["worst"])
    others_ok = not any(r["status"].startswith("DEGRADED") for r in rows)
    adopt = worst_ok and others_ok
    if adopt:
        reason = f"lowers the worst batteries ({', '.join(worst)}) beyond the seed spread without degrading the others"
    elif not worst_ok:
        missed = [r["battery"] for r in rows if r["worst"] and r["status"] != "improved beyond spread"]
        reason = f"does not lower {', '.join(missed)} beyond the seed spread" + ("" if others_ok else "; and degrades others")
    else:
        reason = "degrades " + ", ".join(r["battery"] for r in rows if r["status"].startswith("DEGRADED"))
    return {"baseline": baseline["name"], "option": option["name"], "worst": worst, "rows": rows,
            "adopt": adopt, "reason": reason}


def render_option_grid(baseline, options):
    parts = ["## Physics options grid (opt-in; decision rule: adopt only if the worst held-out batteries improve "
             "by more than the seed spread and no other battery degrades beyond its spread)", ""]
    for option in options:
        d = option_decision(baseline, option)
        parts.append(f"### {option['name']} vs {baseline['name']}: **{'ADOPT' if d['adopt'] else 'REJECTED (negative result, left off)'}** -- {d['reason']}")
        parts.append("")
        parts.append(_table(["held-out battery", baseline["name"], option["name"], "delta", "seed spread", "status"],
                            [[r["battery"] + (" (worst)" if r["worst"] else ""), fmt(r["baseline"]), fmt(r["option"]),
                              f"{r['delta']:+.2f}", fmt(r["spread"]), r["status"]] for r in d["rows"]]))
        parts.append("")
    return parts


# ----------------------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------------------
def fmt(value, digits=2):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    return f"{value:.{digits}f}"


def fmt_ms(pair, digits=2):
    mean, std = pair
    if mean is None:
        return "--"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def fmt_sci(value):
    return "--" if value is None else f"{value:.2e}"


def _table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


COMPARE_METRICS = [
    ("train_mae_c", "train MAE (°C)"), ("train_rmse_c", "train RMSE (°C)"),
    ("held_out_mae_c", "held-out MAE (°C)"), ("held_out_rmse_c", "held-out RMSE (°C)"),
    ("core_temp_max_c", CORE_T_LABEL), ("sei_min", "SEI min"), ("log10_A_sei", "log10 A_sei"),
]


def compare_runs(run_a, run_b, seed=None):
    """Metric-by-metric comparison of two runs: per-seed values when ``seed`` is given (or when
    both runs share exactly one seed), otherwise the means over seeds. Returns a list of rows
    ``(label, value_a, value_b, delta)`` plus the per-battery held-out deltas."""
    agg_a, agg_b = aggregate(run_a), aggregate(run_b)
    common = sorted({s["seed"] for s in run_a["seeds"]} & {s["seed"] for s in run_b["seeds"]})
    if seed is None and len(common) == 1:
        seed = common[0]
    if seed is not None:
        pick = lambda run: next(s for s in run["seeds"] if s["seed"] == seed)  # noqa: E731
        sa, sb = pick(run_a), pick(run_b)
        values = lambda key: (sa.get(key), sb.get(key))  # noqa: E731
        basis = f"seed {seed} of each run"
    else:
        values = lambda key: (agg_a["metrics"][key][0], agg_b["metrics"][key][0])  # noqa: E731
        basis = f"means over {agg_a['n_seeds']} and {agg_b['n_seeds']} seeds"
    rows = []
    for key, label in COMPARE_METRICS:
        va, vb = values(key)
        delta = None if va is None or vb is None else vb - va
        rows.append((label, va, vb, delta))
    per_a, per_b = agg_a["per_battery_held_out_mae_c"], agg_b["per_battery_held_out_mae_c"]
    if seed is not None:
        per_a = {k: (v, 0.0) for k, v in sa["per_battery_held_out_mae_c"].items()}
        per_b = {k: (v, 0.0) for k, v in sb["per_battery_held_out_mae_c"].items()}
    batteries = sorted(set(per_a) | set(per_b))
    per_battery = [(b, per_a.get(b, (None,))[0], per_b.get(b, (None,))[0]) for b in batteries]
    return {"a": run_a["name"], "b": run_b["name"], "basis": basis, "rows": rows, "per_battery": per_battery}


def render_comparison(cmp):
    head = [f"### {cmp['a']} vs {cmp['b']} ({cmp['basis']})", ""]
    table = _table(["metric", cmp["a"], cmp["b"], f"delta ({cmp['b']} − {cmp['a']})"],
                   [[label, fmt(va, 3 if "log10" in label or "SEI" in label else 2),
                     fmt(vb, 3 if "log10" in label or "SEI" in label else 2),
                     ("--" if d is None else f"{d:+.3f}" if ("log10" in label or "SEI" in label) else f"{d:+.2f}")]
                    for label, va, vb, d in cmp["rows"]])
    per = _table(["held-out battery", cmp["a"], cmp["b"], "delta"],
                 [[b, fmt(va), fmt(vb), "--" if va is None or vb is None else f"{vb - va:+.2f}"]
                  for b, va, vb in cmp["per_battery"]])
    return "\n".join(head + [table, "", per, ""])


def render_reference(runs, aggregates, reference=DATABRICKS_V9):
    """The Databricks reference figures next to every local run, per held-out battery."""
    batteries = list(reference["per_battery_held_out_mae_c"]) + sorted(
        {b for r in runs for b in aggregates[r["name"]]["per_battery_held_out_mae_c"]}
        - set(reference["per_battery_held_out_mae_c"]))
    header = ["held-out MAE (°C)", reference["label"]] + [r["name"] for r in runs]
    rows = [["**overall**", fmt(reference["held_out_mae_c"])]
            + [fmt_ms(aggregates[r["name"]]["metrics"]["held_out_mae_c"]) for r in runs],
            ["train MAE (for reference)", fmt(reference["train_mae_c"])]
            + [fmt_ms(aggregates[r["name"]]["metrics"]["train_mae_c"]) for r in runs]]
    for b in batteries:
        ref = reference["per_battery_held_out_mae_c"].get(b)
        row = [b, fmt(ref) if ref is not None else "-- (absent in v9)"]
        for r in runs:
            cell = aggregates[r["name"]]["per_battery_held_out_mae_c"].get(b)
            row.append(fmt_ms(cell) if cell else "-- (absent)")
        rows.append(row)
    return ["## Databricks reference result (v9) vs local runs", "",
            "The v9 notebook run is the only Databricks run with recorded accuracy figures. It used the legacy "
            "1.4 Ah capacity floor (19 training / 5 held-out batteries -- B0056 had no data), every observation "
            "point as anchor and the full 15000 + 2000 budget; the local runs below use the tables and protocols "
            "described under Protocol, so the per-battery numbers are the like-for-like comparison and the overall "
            "held-out mean is not (the local held-out set contains B0056).", "",
            _table(header, rows), ""]


def render_markdown(runs, title="Results", protocol=None, generated=None, notes=None, compare=(),
                    grid=None, reference=True):
    """The full RESULTS.md text for a list of collected runs.

    ``notes`` is appended verbatim; ``compare`` is a list of ``(name_a, name_b)`` pairs rendered as
    delta tables; ``grid`` is ``(baseline_name, [option_names])`` for the physics-option decision
    tables; ``reference`` adds the Databricks v9 section.
    """
    aggregates = {run["name"]: aggregate(run) for run in runs}
    by_name = {run["name"]: run for run in runs}
    generated = generated or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parts = [f"# {title}", "",
             f"Generated {generated} by `scripts/make_results_table.py`. Temperature errors are the "
             "surface-temperature MAE / RMSE in °C against the measured values. " + VERDICT_NOTE, ""]
    if protocol:
        parts += ["## Protocol", "", protocol, ""]

    parts += ["## Runs", "", _table(
        ["run", "train batteries", "train cycles", "train rows", "held-out batteries",
         "held-out cycles", "held-out rows", "seeds", "artifacts"],
        [[r["name"], len(r["train"]["batteries"]), r["train"]["cycles"], r["train"]["rows"],
          len(r["held_out"]["batteries"]), r["held_out"]["cycles"], r["held_out"]["rows"],
          ", ".join(str(s["seed"]) for s in r["seeds"]), f"`{r['artifacts_dir']}`"] for r in runs]), ""]
    for r in runs:
        missing = sorted(set(config.HELD_OUT_BATTERIES) - set(r["held_out"]["batteries"]))
        if missing:
            parts.append(f"- **{r['name']}**: held-out table lacks designated batteries {missing}.")
        if r["train"]["cycles_dropped_on_load"] or r["held_out"]["cycles_dropped_on_load"]:
            parts.append(f"- **{r['name']}**: {r['train']['cycles_dropped_on_load']} training and "
                         f"{r['held_out']['cycles_dropped_on_load']} held-out cycles are dropped on load "
                         "(missing Re or other required values).")
    parts.append("")

    if reference:
        parts += render_reference(runs, aggregates)

    # Per-battery and per-ambient errors come first.
    for table, heading in (("held_out", "Per-battery held-out MAE (°C, mean ± std over seeds)"),
                           ("train", "Per-battery training MAE (°C, mean ± std over seeds)")):
        key = f"per_battery_{table}_mae_c"
        batteries = sorted({b for r in runs for b in aggregates[r["name"]][key]})
        rows = []
        for b in batteries:
            row = [b + (" [HELD-OUT]" if table == "train" and b in config.HELD_OUT_BATTERIES else "")]
            for r in runs:
                cell = aggregates[r["name"]][key].get(b)
                row.append(fmt_ms(cell) if cell else "-- (absent)")
            rows.append(row)
        parts += [f"## {heading}", "", _table(["battery"] + [r["name"] for r in runs], rows), ""]
    for table, heading in (("held_out", "Per-ambient held-out MAE (°C, mean ± std over seeds)"),
                           ("train", "Per-ambient training MAE (°C, mean ± std over seeds)")):
        key = f"per_ambient_{table}_mae_c"
        groups = sorted({g for r in runs for g in aggregates[r["name"]].get(key, {})}, key=lambda g: float(g.split()[0]))
        if groups:
            rows = [[g] + [fmt_ms(aggregates[r["name"]][key][g]) if g in aggregates[r["name"]].get(key, {}) else "--"
                           for r in runs] for g in groups]
            parts += [f"## {heading}", "", _table(["ambient"] + [r["name"] for r in runs], rows), ""]

    parts += ["## Summary across seeds (mean ± std)", "", _table(
        ["run", "seeds", "log10 A_sei", "train MAE (°C)", "train RMSE (°C)", "held-out MAE (°C)",
         "held-out RMSE (°C)", "gap (held-out − train)", CORE_T_LABEL, "SEI min", "verdicts (ambient ≤ 45 °C only)"],
        [[r["name"], a["n_seeds"], fmt_ms(a["metrics"]["log10_A_sei"]),
          fmt_ms(a["metrics"]["train_mae_c"]), fmt_ms(a["metrics"]["train_rmse_c"]),
          fmt_ms(a["metrics"]["held_out_mae_c"]), fmt_ms(a["metrics"]["held_out_rmse_c"]),
          fmt_ms(a["metrics"]["generalisation_gap_c"]), fmt_ms(a["metrics"]["core_temp_max_c"], 1),
          fmt_ms(a["metrics"]["sei_min"], 3),
          " / ".join(f"{a['verdicts'][v]} {v}" for v in VERDICTS if a["verdicts"][v])
          + ("" if a["verdict_consistent"] else " **(inconsistent)**")]
         for r in runs for a in [aggregates[r["name"]]]]), ""]

    parts += ["## Per-seed checkpoint metrics", "", _table(
        ["run", "seed", "A_sei", "train MAE", "train RMSE", "held-out MAE", "held-out RMSE",
         CORE_T_LABEL, "SEI min", "threshold", "verdict", "decided by"],
        [[r["name"], s["seed"], fmt_sci(s["A_sei"]), fmt(s["train_mae_c"]), fmt(s["train_rmse_c"]),
          fmt(s["held_out_mae_c"]), fmt(s["held_out_rmse_c"]), fmt(s["core_temp_max_c"], 1),
          fmt(s["sei_min"], 3), fmt(s["threshold"], 3), s["verdict"], s.get("criterion", "--")]
         for r in runs for s in r["seeds"]]), ""]

    parts += ["## A_sei identifiability across seeds", "", _table(
        ["run", "seeds", "initial log10 A_sei (min..max)", "final log10 A_sei (mean ± std)", "final range",
         "status", "reason"],
        [[r["name"], a["n_seeds"],
          "--" if not a["initial"] else f"{a['initial']['min']:.2f}..{a['initial']['max']:.2f}",
          "--" if not a["final"] else f"{a['final']['mean']:.2f} ± {a['final']['std']:.2f}",
          "--" if not a["final"] else f"{a['final']['range']:.2f}",
          f"**{a['status']}**", a["reason"]
          + (" (initial value not recorded; assumed the recommended default)"
             if any(s.get("log10_A_sei_init_assumed") for s in r["seeds"]) else "")]
         for r in runs for a in [aggregates[r["name"]]["identifiability"]]]), "",
        f"Thresholds: seeds must end within {config.IDENTIFIABLE_MAX_SPREAD_DECADES} decades of each other and "
        f"have moved at least {config.IDENTIFIABLE_MIN_MOVEMENT_DECADES} decades from their start; only seeds "
        "that *started* apart (`train --a-sei-init-spread`) can show identifiability rather than consistency. "
        "Runs trained with `--fix-log-a-sei` hold A_sei constant by construction.", ""]

    term_runs = [r for r in runs if aggregates[r["name"]]["loss_terms"]]
    if term_runs:
        names = config.LOSS_TERM_NAMES
        parts += ["## Final weighted loss terms (mean over seeds; share of total in brackets)", "", _table(
            ["run"] + names + ["dominant"],
            [[r["name"]] + [
                (f"{aggregates[r['name']]['loss_terms'][n][0]:.2e} ({100 * aggregates[r['name']]['loss_shares'][n]:.0f}%)"
                 if n in aggregates[r["name"]]["loss_terms"] else "--") for n in names]
             + [aggregates[r["name"]]["dominant_term"]] for r in term_runs]), "",
            "Weights " + ", ".join(f"{n}={w}" for n, w in zip(names, config.LOSS_WEIGHTS))
            + "; per-step histories are in each run's `loss_history_seed*.csv`.", ""]

    if grid:
        baseline_name, option_names = grid
        parts += render_option_grid(by_name[baseline_name], [by_name[n] for n in option_names])

    parts += ["## Abuse-test verdict spread (ensemble per run)", "", VERDICT_NOTE, "", _table(
        ["run", "models"] + list(VERDICTS) + ["consistent", "decided by (count x criterion)",
                                               "SEI threshold min..max (mean ± std)", CORE_T_LABEL + " min..max"],
        [[r["name"], a["n_seeds"]] + [a["verdicts"][v] for v in VERDICTS]
         + ["yes" if a["verdict_consistent"] else "**no**",
            ", ".join(f"{n} x {k}" for k, n in a["criterion_counts"].items()),
            "--" if a["threshold"]["min"] is None else
            f"{a['threshold']['min']:.3f}..{a['threshold']['max']:.3f} ({a['threshold']['mean']:.3f} ± {a['threshold']['std']:.3f})",
            f"{min(s['core_temp_max_c'] for s in r['seeds']):.1f}..{max(s['core_temp_max_c'] for s in r['seeds']):.1f}"]
         for r in runs for a in [aggregates[r["name"]]]]), ""]
    if compare:
        parts += ["## Run-to-run comparisons", ""]
        for name_a, name_b in compare:
            if name_a not in by_name or name_b not in by_name:
                raise KeyError(f"--compare {name_a} {name_b}: unknown run name (have {sorted(by_name)})")
            parts.append(render_comparison(compare_runs(by_name[name_a], by_name[name_b])))
    if notes:
        parts += ["## Notes", "", notes.strip(), ""]
    return "\n".join(parts)


# ----------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", nargs=3, action="append", metavar=("NAME", "ARTIFACTS_DIR", "DATA_DIR"),
                   required=True, help="a run to include; repeatable")
    p.add_argument("--out", default=os.path.join(_REPO_ROOT, "docs", "RESULTS.md"))
    p.add_argument("--json", default=None, help="also dump the collected metrics as JSON")
    p.add_argument("--title", default="Baseline results on the NASA battery data set")
    p.add_argument("--protocol", default=None, help="free text describing the training protocol")
    p.add_argument("--protocol-file", default=None, help="file whose contents describe the protocol")
    p.add_argument("--notes-file", default=None,
                   help="Markdown appended as a 'Notes' section (interpretation of the numbers)")
    p.add_argument("--grid", nargs="+", default=None, metavar="RUN",
                   help="physics-option decision tables: the first run is the baseline, the rest are options")
    p.add_argument("--no-reference", action="store_true", help="omit the Databricks v9 reference section")
    p.add_argument("--compare", nargs=2, action="append", default=[], metavar=("RUN_A", "RUN_B"),
                   help="render a metric-by-metric delta table for two runs (same seed if they share exactly "
                        "one, else means); repeatable")
    return p


def main(argv=None, twin_factory=None, log=print):
    args = build_parser().parse_args(argv)
    protocol = args.protocol
    if args.protocol_file:
        with open(args.protocol_file, encoding="utf-8") as f:
            protocol = f.read().strip()
    notes = None
    if args.notes_file:
        with open(args.notes_file, encoding="utf-8") as f:
            notes = f.read()
    runs = [collect_run(name, artifacts, data, twin_factory=twin_factory, log=log)
            for name, artifacts, data in args.run]
    grid = (args.grid[0], args.grid[1:]) if args.grid else None
    text = render_markdown(runs, title=args.title, protocol=protocol, notes=notes, compare=args.compare,
                           grid=grid, reference=not args.no_reference)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"runs": runs, "aggregates": {r["name"]: aggregate(r) for r in runs}}, f, indent=2)
    if log:
        log(f"Wrote {args.out}" + (f" and {args.json}" if args.json else ""))
    return runs


if __name__ == "__main__":
    main()
