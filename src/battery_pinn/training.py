"""Train the PINN (Adam, then optional L-BFGS) and write checkpoints locally."""
import json
import os

import deepxde as dde
import numpy as np
import pandas as pd
import torch

from battery_pinn import config
from battery_pinn import device as device_selection
from battery_pinn.checkpoint import save_checkpoint
from battery_pinn.data import build_points_and_targets, representative_profile
from battery_pinn.identifiability import assess_log_a_sei, format_assessment, spread_initial_values
from battery_pinn.interpolation import DischargeProfile
from battery_pinn.ocv import fit_ocv_coeffs
from battery_pinn.physics import make_pde, make_thermal_variables, recommended_log_a_sei


class LossTermLogger(dde.callbacks.Callback):
    """Record the weighted loss terms and log10(A_sei) every ``period`` steps.

    DeepXDE refreshes ``train_state.loss_train`` at its display steps, so ``period`` should be
    the ``display_every`` passed to ``model.train``.
    """

    def __init__(self, log_a_sei, period):
        super().__init__()
        self.log_a_sei, self.period, self.rows = log_a_sei, period, []

    def on_epoch_end(self):
        if self.model.train_state.step % self.period == 0:
            self._record()

    def _record(self):
        state = self.model.train_state
        if state.loss_train is not None:
            losses = [float(v) for v in np.ravel(state.loss_train)]
            self.rows.append({"step": int(state.step), "log10_A_sei": float(self.log_a_sei.detach().item()),
                              **dict(zip(config.LOSS_TERM_NAMES, losses))})

    on_train_begin = on_train_end = _record  # the initial state and the last step, whatever the period

    def frame(self):
        return pd.DataFrame(self.rows).drop_duplicates("step", keep="last")


def loss_breakdown(final_losses):
    """Share of each weighted term in the total, and which one dominates."""
    total = sum(final_losses.values()) or 1.0
    shares = {k: v / total for k, v in final_losses.items()}
    dominant = max(shares, key=shares.get)
    return {"total": total, "shares": shares, "dominant": dominant, "dominant_share": shares[dominant]}


def select_anchor_subset(n_points, anchor_subset, seed):
    """Indices of the observation points used as PDE anchors / data-fit points (None = all)."""
    if anchor_subset is None or anchor_subset >= n_points:
        return None
    if anchor_subset <= 0:
        raise ValueError("anchor_subset must be a positive number of points")
    return np.sort(np.random.default_rng(seed).choice(n_points, size=int(anchor_subset), replace=False))


def predict(net, points):
    """Forward pass through the network; returns an (N, 2) array of [T_delta, c]."""
    device = next(net.parameters()).device
    net.eval()
    with torch.no_grad():
        out = net(torch.as_tensor(points, dtype=torch.float32, device=device))
    return out.cpu().numpy()


def train_one_seed(df, seed, out_dir, iterations=15000, lbfgs_iterations=2000,
                   num_domain=5000, num_boundary=1000, num_test=2000, display_every=1000,
                   device="auto", anchor_subset=None, log_a_sei_init=None, num_initial=0,
                   newton_cooling=False, learn_thermal=False, fix_log_a_sei=None):
    """Train one model on ``df`` and save ``battery_pinn_model_seed{seed}.pt`` in ``out_dir``.

    ``device`` is ``"auto"`` (CUDA when usable, else CPU), ``"cpu"`` or ``"cuda"`` (falls back
    to CPU with a warning when CUDA is unavailable); see :mod:`battery_pinn.device`.

    ``anchor_subset`` (opt-in) uses a seeded random subset of that many observation points as
    the PDE anchors *and* the temperature data-fit points instead of all of them. DeepXDE
    evaluates the PDE residual (with its Jacobian and Hessian) on every input point, data
    points included, so the cost per iteration scales with the full observation set (about
    86k rows for the NASA tables); a few thousand points keep the same distribution over
    batteries and conditions at a fraction of the cost. Evaluation always uses every row.

    ``log_a_sei_init`` overrides the initial ``log10(A_sei)`` (default: the value that makes
    the SEI rate O(1) at the reference temperature), which lets an ensemble start its seeds
    from different values for the identifiability check.

    ``num_initial`` adds that many explicit ``t = 0`` points (the notebook used none: its
    initial-condition points are the random domain/boundary points with ``t <= 0.01``, about
    1 % of them, so very small point budgets can end up with an empty IC set, whose NaN loss
    makes DeepXDE stop training silently).

    Opt-in physics variants (off by default, see :func:`battery_pinn.physics.make_pde`):
    ``newton_cooling`` adds a surface heat-loss term, ``learn_thermal`` makes the thermal
    constants trainable, and ``fix_log_a_sei`` freezes ``log10(A_sei)`` at the given value
    instead of training it.

    The weighted loss terms and ``log10(A_sei)`` are recorded every ``display_every`` steps
    and written to ``loss_history_seed{seed}.csv``; the returned dict carries the final terms,
    their shares of the total and the dominant term.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device_selection.activate(device_selection.resolve_device(device))
    print(f"Training on {device_selection.describe(device)}")

    df_agg = representative_profile(df)
    profile = DischargeProfile(df_agg, device=device)
    ocv_coeffs = fit_ocv_coeffs(df_agg)
    t_max_seconds = float(df["Time"].max())
    if fix_log_a_sei is not None:
        log_a_sei_init = float(fix_log_a_sei)
        log_a_sei = torch.tensor(log_a_sei_init, dtype=torch.float32)
        trainable = []
    else:
        log_a_sei_init = float(recommended_log_a_sei() if log_a_sei_init is None else log_a_sei_init)
        log_a_sei = dde.Variable(log_a_sei_init)
        trainable = [log_a_sei]
    thermal = make_thermal_variables() if learn_thermal else None
    if thermal:
        trainable += list(thermal.values())
    pde = make_pde(profile, ocv_coeffs, log_a_sei, t_max_seconds, newton_cooling=newton_cooling, thermal=thermal)
    options = {"newton_cooling": bool(newton_cooling), "learn_thermal": bool(learn_thermal),
               "fix_log_a_sei": fix_log_a_sei}
    if any(options.values()) or fix_log_a_sei is not None:
        print("Physics options: " + ", ".join(f"{k}={v}" for k, v in options.items()))

    param_geom = dde.geometry.Hypercube(
        xmin=config.PARAM_BOUNDS_MIN, xmax=config.PARAM_BOUNDS_MAX
    )
    geomtime = dde.geometry.GeometryXTime(param_geom, dde.geometry.TimeDomain(0, 1))

    def on_initial_condition(x, on_initial):
        return on_initial or (x[4] <= 0.01)

    ic_T = dde.icbc.IC(geomtime, lambda x: 0, on_initial_condition, component=0)
    ic_c = dde.icbc.IC(geomtime, lambda x: 1.0, on_initial_condition, component=1)

    obs_points, obs_values = build_points_and_targets(df)
    subset = select_anchor_subset(len(obs_points), anchor_subset, seed)
    fit_points, fit_values = (obs_points, obs_values) if subset is None else (obs_points[subset], obs_values[subset])
    print(f"Observation points: {len(obs_points)}; used as anchors / data fit: {len(fit_points)}"
          + ("" if subset is None else f" (anchor subset, seed {seed})"))
    observe_T = dde.icbc.PointSetBC(fit_points, fit_values.reshape(-1, 1), component=0)

    data = dde.data.TimePDE(
        geomtime, pde, [ic_T, ic_c, observe_T],
        num_domain=num_domain, num_boundary=num_boundary, num_initial=num_initial,
        anchors=fit_points, num_test=num_test,
    )
    net = dde.nn.FNN(config.NETWORK_LAYERS, "tanh", "Glorot normal")
    net.float()
    net.to(device)
    model = dde.model.Model(data, net)

    logger = LossTermLogger(log_a_sei, display_every)
    model.compile("adam", lr=1e-4, loss_weights=config.LOSS_WEIGHTS,
                  external_trainable_variables=trainable)
    model.train(iterations=iterations, display_every=display_every, callbacks=[logger])

    lbfgs_error = None
    if lbfgs_iterations:
        # PyTorch's L-BFGS line search raises (IndexError / RuntimeError) when the loss turns
        # non-finite and can leave the parameters mid-line-search; keep the Adam result then.
        adam_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        adam_extra = [v.detach().clone() for v in trainable]
        model.compile("L-BFGS", loss_weights=config.LOSS_WEIGHTS,
                      external_trainable_variables=trainable)
        dde.optimizers.set_LBFGS_options(maxiter=lbfgs_iterations, ftol=1e-9)
        try:
            model.train(display_every=display_every, callbacks=[logger])
            if not all(torch.isfinite(v).all() for v in net.state_dict().values()):
                raise FloatingPointError("non-finite network weights after L-BFGS")
        except (IndexError, RuntimeError, FloatingPointError) as exc:
            lbfgs_error = f"{type(exc).__name__}: {exc}"
            print(f"WARNING: L-BFGS refinement aborted ({lbfgs_error}); keeping the Adam result.")
            net.load_state_dict(adam_state)
            with torch.no_grad():
                for var, saved in zip(trainable, adam_extra):
                    var.copy_(saved)

    pred = predict(net, obs_points)
    train_mae_c = float(np.mean(np.abs(obs_values - pred[:, 0])) * config.TEMP_SCALE_C)

    os.makedirs(out_dir, exist_ok=True)
    history = logger.frame()
    history.to_csv(os.path.join(out_dir, f"loss_history_seed{seed}.csv"), index=False)
    final_losses = {k: float(history.iloc[-1][k]) for k in config.LOSS_TERM_NAMES} if len(history) else {}
    breakdown = loss_breakdown(final_losses) if final_losses else None
    if breakdown:
        print("Final weighted loss terms: " + ", ".join(f"{k}={v:.3e}" for k, v in final_losses.items()))
        print(f"Dominant term: {breakdown['dominant']} ({100 * breakdown['dominant_share']:.0f}% of the total)")

    path = os.path.join(out_dir, f"battery_pinn_model_seed{seed}.pt")
    # Checkpoints are always stored as CPU tensors so they load anywhere.
    state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    log_a_sei_final = float(log_a_sei.detach().item())
    thermal_final = {k: float(v.detach().item()) for k, v in (thermal or {}).items()}
    save_checkpoint(path, state, log_a_sei_final, ocv_coeffs,
                    physics={**options, "thermal": thermal_final or None})
    return {
        "seed": seed,
        "checkpoint": path,
        "device": str(device),
        "physics": options,
        "thermal": thermal_final or None,
        "anchor_subset": None if subset is None else int(len(subset)),
        "log10_A_sei_init": log_a_sei_init,
        "log10_A_sei": log_a_sei_final,
        "A_sei": float(10 ** log_a_sei_final),
        "train_mae_c": train_mae_c,
        "lbfgs_error": lbfgs_error,
        "final_losses": final_losses,
        "loss_breakdown": breakdown,
    }


def train_ensemble(df, seeds, out_dir, a_sei_init_spread=0.0, log_a_sei_init=None, **kwargs):
    """Train one model per seed; with ``a_sei_init_spread`` the seeds start their log10(A_sei)
    spread evenly across ``recommended +- spread`` decades so the ensemble doubles as an
    identifiability experiment (``log_a_sei_init`` pins one explicit start for every seed,
    e.g. when the seeds of one ensemble are trained in separate processes). Writes
    ``train_summary.json`` and ``identifiability.json``."""
    if log_a_sei_init is not None:
        inits = [float(log_a_sei_init)] * len(seeds)
    else:
        inits = spread_initial_values(len(seeds), a_sei_init_spread, recommended_log_a_sei())
    results = []
    for i, (seed, init) in enumerate(zip(seeds, inits)):
        print(f"\n=== Training seed {seed} ({i + 1}/{len(seeds)}), log10(A_sei) init {init:.3f} ===")
        res = train_one_seed(df, seed, out_dir, log_a_sei_init=init, **kwargs)
        print(f"Seed {seed}: A_sei={res['A_sei']:.2e} (log10 {res['log10_A_sei']:.3f}, started {init:.3f}), "
              f"train MAE ~{res['train_mae_c']:.2f} C")
        results.append(res)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "train_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    assessment = assess_log_a_sei([r["log10_A_sei"] for r in results], [r["log10_A_sei_init"] for r in results])
    with open(os.path.join(out_dir, "identifiability.json"), "w") as f:
        json.dump(assessment, f, indent=2)
    print("\n" + format_assessment(assessment))
    return results
