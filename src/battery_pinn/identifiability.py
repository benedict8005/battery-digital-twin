"""Is the calibrated SEI frequency factor ``A_sei`` identified by the data?

The PINN treats ``log10(A_sei)`` as a trainable variable. Three outcomes are possible once an
ensemble of seeds has been trained:

* **stuck** -- every seed still sits at (or within ``IDENTIFIABLE_MIN_MOVEMENT_DECADES`` of)
  its initial value: the loss is insensitive to ``A_sei`` and the "calibrated" number is just
  the initialisation;
* **runaway** -- every seed drifted more than ``IDENTIFIABLE_MAX_RUNAWAY_DECADES`` in the
  same direction: the optimiser is switching the SEI reaction off (``A_sei -> 0`` makes the
  SEI residual vanish for any data) or blowing it up, not calibrating it. Seeds that agree
  *there* agree on a degenerate minimum, so this is not identifiability either;
* **seed-dependent** -- the seeds moved but disagree by more than
  ``IDENTIFIABLE_MAX_SPREAD_DECADES``: the data do not pin the value down;
* **identifiable** -- the seeds moved and agree. This is strong evidence only when the seeds
  *started* from different values (``train --a-sei-init-spread``) and still converged to the
  same one; with a common starting point it is merely consistency.
"""
import math

from battery_pinn import config


def spread_initial_values(n_seeds, spread_decades, centre):
    """Per-seed initial ``log10(A_sei)`` values spaced evenly across ``centre +- spread``."""
    if n_seeds <= 1 or not spread_decades:
        return [float(centre)] * n_seeds
    step = 2.0 * spread_decades / (n_seeds - 1)
    return [float(centre - spread_decades + i * step) for i in range(n_seeds)]


def _stats(values):
    values = [float(v) for v in values]
    mean = sum(values) / len(values)
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)) if len(values) > 1 else 0.0
    return {"mean": mean, "std": std, "min": min(values), "max": max(values), "range": max(values) - min(values)}


def assess_log_a_sei(finals, inits=None, max_spread=None, min_movement=None, max_runaway=None):
    """Classify an ensemble's final ``log10(A_sei)`` values. Returns a JSON-serialisable dict."""
    if max_spread is None:
        max_spread = config.IDENTIFIABLE_MAX_SPREAD_DECADES
    if min_movement is None:
        min_movement = config.IDENTIFIABLE_MIN_MOVEMENT_DECADES
    if max_runaway is None:
        max_runaway = config.IDENTIFIABLE_MAX_RUNAWAY_DECADES
    finals = [float(v) for v in finals]
    if inits is None:
        inits = [None] * len(finals)
    out = {"n_seeds": len(finals), "final": _stats(finals) if finals else None,
           "max_spread_decades": max_spread, "min_movement_decades": min_movement,
           "max_runaway_decades": max_runaway}
    known_inits = [i for i in inits if i is not None]
    out["initial"] = _stats(known_inits) if known_inits else None
    shifts = [f - i for f, i in zip(finals, inits) if i is not None]
    out["shift"] = _stats(shifts) if shifts else None

    if len(finals) < 2:
        out["status"], out["reason"] = "inconclusive", "fewer than two seeds"
        return out
    moved = bool(shifts) and max(abs(s) for s in shifts) >= min_movement
    if shifts and not moved:
        out["status"] = "not identifiable"
        out["reason"] = (f"every seed stayed within {min_movement} decades of its initial value "
                         f"(max shift {max(abs(s) for s in shifts):.3f}); the loss is insensitive to A_sei")
        return out
    if shifts and min(abs(s) for s in shifts) > max_runaway and len({s > 0 for s in shifts}) == 1:
        direction = "down (towards A_sei = 0, which zeroes the SEI residual)" if shifts[0] < 0 else "up"
        out["status"] = "not identifiable"
        out["reason"] = (f"runaway: every seed moved {min(abs(s) for s in shifts):.1f}-{max(abs(s) for s in shifts):.1f} "
                         f"decades {direction} regardless of its start; the optimiser is switching the SEI term off, "
                         f"not calibrating it")
        return out
    if out["final"]["range"] > max_spread:
        out["status"] = "not identifiable"
        out["reason"] = (f"seeds disagree: final log10(A_sei) spans {out['final']['range']:.2f} decades "
                         f"(> {max_spread}); the value depends on the seed")
        return out
    spread_starts = out["initial"] is not None and out["initial"]["range"] > max_spread
    out["status"] = "identifiable" if spread_starts else "consistent across seeds"
    out["reason"] = (
        f"seeds started {out['initial']['range']:.2f} decades apart and ended within "
        f"{out['final']['range']:.2f} decades (mean {out['final']['mean']:.2f})" if spread_starts else
        f"final log10(A_sei) agree within {out['final']['range']:.2f} decades (mean {out['final']['mean']:.2f})"
        + ("" if shifts else "; initial values unknown")
        + ("; all seeds started from the same value, so this shows consistency, not identifiability "
           "-- rerun with --a-sei-init-spread" if out["initial"] is not None and not spread_starts else "")
    )
    return out


def format_assessment(a):
    lines = [f"A_sei identifiability ({a['n_seeds']} seeds): {a['status'].upper()} -- {a['reason']}"]
    if a.get("final"):
        f = a["final"]
        lines.append(f"  final log10(A_sei): mean {f['mean']:.3f}, std {f['std']:.3f}, range [{f['min']:.3f}, {f['max']:.3f}]")
    if a.get("initial"):
        i = a["initial"]
        lines.append(f"  initial log10(A_sei): range [{i['min']:.3f}, {i['max']:.3f}]")
    return "\n".join(lines)
