"""Command-line interface: convert-mat, prepare-data, generate-data, train, evaluate, abuse-test, demo."""
import argparse
import glob
import os

from battery_pinn import config, synthetic
from battery_pinn.data import load_dataset



def _cmd_convert_mat(args):
    from battery_pinn.nasa_mat import convert_mat_dir

    convert_mat_dir(args.mat_dir, args.out, battery_ids=args.batteries)


def _anchor_subset(text):
    if text.lower() == "all":
        return None
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive number of points or 'all'")
    return value


def _fraction(text):
    """``--min-capacity-fraction`` value: a float, or ``none`` for the legacy absolute floor."""
    if text.lower() == "none":
        return None
    value = float(text)
    if not 0.0 < value <= 1.0:
        raise argparse.ArgumentTypeError("must be in (0, 1] or 'none'")
    return value


def _cmd_prepare(args):
    from battery_pinn.etl import run_pipeline

    run_pipeline(args.raw, args.out, min_capacity_fraction=args.min_capacity_fraction,
                 allow_missing_held_out=args.allow_missing_held_out)


def _held_out_expectation(args):
    """Which batteries the held-out file must contain (None disables the check)."""
    if getattr(args, "no_battery_check", False):
        return None
    return args.held_out_batteries


def _cmd_generate(args):
    train, held_out = synthetic.generate_train_and_held_out(
        n_batteries=args.batteries, n_held_out=args.held_out,
        cycles_per_battery=args.cycles, seed=args.seed,
    )
    os.makedirs(args.out, exist_ok=True)
    train.to_csv(os.path.join(args.out, "train.csv"), index=False)
    held_out.to_csv(os.path.join(args.out, "held_out.csv"), index=False)
    print(f"Wrote {len(train)} train rows and {len(held_out)} held-out rows to {args.out}")
    print("NOTE: synthetic data, not NASA measurements.")


def _cmd_train(args):
    from battery_pinn.training import train_ensemble

    df = load_dataset(args.data)
    train_ensemble(
        df, args.seeds, args.out,
        iterations=args.iterations, lbfgs_iterations=args.lbfgs_iterations,
        num_domain=args.num_domain, num_boundary=args.num_boundary,
        device=args.device, anchor_subset=args.anchor_subset, num_initial=args.num_initial,
        a_sei_init_spread=args.a_sei_init_spread, log_a_sei_init=args.log_a_sei_init,
        display_every=args.display_every, newton_cooling=args.newton_cooling,
        learn_thermal=args.learn_thermal, fix_log_a_sei=args.fix_log_a_sei,
    )


def _checkpoints(out_dir):
    paths = sorted(glob.glob(os.path.join(out_dir, "battery_pinn_model_seed*.pt")))
    if not paths:
        raise SystemExit(f"No checkpoints found in {out_dir}. Run `train` first.")
    return paths


def _cmd_evaluate(args):
    from battery_pinn.inference import DigitalTwin, evaluate

    train_df = load_dataset(args.data)
    held_out_df = load_dataset(args.held_out_data, expected_batteries=_held_out_expectation(args))
    for path in _checkpoints(args.checkpoints):
        twin = DigitalTwin(path)
        tr, ho = evaluate(twin, train_df), evaluate(twin, held_out_df)
        print(f"{os.path.basename(path)}: train MAE {tr['mae_c']:.2f} C / RMSE {tr['rmse_c']:.2f} C | "
              f"held-out MAE {ho['mae_c']:.2f} C / RMSE {ho['rmse_c']:.2f} C")
        for bid, mae in ho["per_battery_mae_c"].items():
            print(f"    {bid}: MAE {mae:.2f} C")


def _cmd_abuse(args):
    import json

    from battery_pinn.inference import (CRITERIA, DigitalTwin, abuse_test, format_ensemble_summary,
                                        plot_abuse_test, summarise_ensemble)

    train_df = load_dataset(args.data)
    paths = _checkpoints(args.checkpoints)
    if len(paths) < args.min_models:
        raise SystemExit(f"{len(paths)} checkpoint(s) in {args.checkpoints}; the ensemble verdict needs at least "
                         f"{args.min_models} (train more seeds, or pass --min-models).")
    results = {}
    for path in paths:
        name = os.path.basename(path).replace("battery_pinn_model_", "").replace(".pt", "")
        r = abuse_test(DigitalTwin(path), train_df)
        results[name] = r
        print(f"{name}: {r['verdict']} -- {CRITERIA[r['criterion']]} (core temp max {r['core_temp_max_c']:.1f} C "
              f"[unvalidated], SEI min {r['sei_min']:.3f}, threshold {r['threshold']:.3f})")
    summary = summarise_ensemble(results)
    print(format_ensemble_summary(summary))
    os.makedirs(args.checkpoints, exist_ok=True)
    with open(os.path.join(args.checkpoints, "abuse_test_summary.json"), "w") as f:
        json.dump({"summary": summary,
                   "models": {k: {kk: v for kk, v in r.items() if kk in ("verdict", "criterion", "core_temp_max_c",
                                                                          "sei_min", "threshold")}
                              for k, r in results.items()}}, f, indent=2)
    plot_path = os.path.join(args.checkpoints, "abuse_test.png")
    plot_abuse_test(results, plot_path)
    print(f"Saved plot to {plot_path} and abuse_test_summary.json")


def _cmd_demo(args):
    data_dir = os.path.join(args.workdir, "data")
    out_dir = os.path.join(args.workdir, config.DEFAULT_ARTIFACTS_DIR)
    _cmd_generate(argparse.Namespace(out=data_dir, batteries=8, held_out=2, cycles=12, seed=0))
    _cmd_train(argparse.Namespace(
        data=os.path.join(data_dir, "train.csv"), out=out_dir, seeds=args.seeds,
        iterations=args.iterations, lbfgs_iterations=args.lbfgs_iterations,
        num_domain=500, num_boundary=100, device=args.device, anchor_subset=None, num_initial=0,
        a_sei_init_spread=0.0, log_a_sei_init=None, display_every=1000,
        newton_cooling=False, learn_thermal=False, fix_log_a_sei=None,
    ))
    _cmd_evaluate(argparse.Namespace(
        checkpoints=out_dir, data=os.path.join(data_dir, "train.csv"),
        held_out_data=os.path.join(data_dir, "held_out.csv"),
        held_out_batteries=None, no_battery_check=True,  # synthetic IDs, no designated batteries
    ))
    _cmd_abuse(argparse.Namespace(checkpoints=out_dir, data=os.path.join(data_dir, "train.csv"), min_models=1))


def build_parser():
    p = argparse.ArgumentParser(prog="battery-pinn", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    cm = sub.add_parser(
        "convert-mat",
        help="convert NASA's B00xx.mat files into the raw CSV layout prepare-data reads",
        description=(
            "Reads every B*.mat in --mat-dir and writes <out>/metadata.csv plus "
            f"<out>/{config.CYCLE_DATA_SUBDIR}/<uid>.csv, the cleaned layout that prepare-data expects."
        ),
    )
    cm.add_argument("--mat-dir", default=config.DEFAULT_MAT_DIR,
                    help="folder holding the .mat files (default: %(default)s)")
    cm.add_argument("--out", default=config.DEFAULT_RAW_DATA_DIR,
                    help="raw dataset folder to write (default: %(default)s)")
    cm.add_argument("--batteries", nargs="+", default=None, help="convert only these battery IDs")
    cm.set_defaults(func=_cmd_convert_mat)

    pd_ = sub.add_parser(
        "prepare-data",
        help="build train.csv / held_out.csv from the raw NASA dataset (.mat folder or per-cycle CSVs)",
        description=(
            "Raw-to-processed pipeline (bronze -> silver -> gold -> feature tables). <raw> is either a "
            f"folder of NASA's B00xx.mat files (converted first into <out>/{config.MAT_CONVERSION_SUBDIR}/) "
            f"or the per-cycle CSV layout (<raw>/{config.METADATA_FILENAME} + <raw>/{config.CYCLE_DATA_SUBDIR}/). "
            f"Writes <out>/{config.TRAIN_FILE} and <out>/{config.HELD_OUT_FILE} plus the silver and gold "
            "tables. Battery lists, cutoff voltages and thresholds come from battery_pinn/config.py."
        ),
    )
    pd_.add_argument("--raw", default=config.DEFAULT_RAW_DATA_DIR,
                     help="folder of B*.mat files, or of metadata.csv + data/ (default: %(default)s)")
    pd_.add_argument("--out", default=config.DEFAULT_PROCESSED_DIR,
                     help="where to write the processed CSV tables (default: %(default)s)")
    pd_.add_argument("--min-capacity-fraction", type=_fraction, default=config.MIN_CAPACITY_FRACTION_OF_MAX,
                     metavar="FRACTION|none",
                     help="capacity floor = min(MIN_VALID_CAPACITY_AH, FRACTION x battery's best cycle); "
                          "'none' restores the legacy absolute floor (default: %(default)s)")
    pd_.add_argument("--allow-missing-held-out", action="store_true",
                     help="write the tables even if a held-out battery has no data (warning instead "
                          "of error); only for legacy-comparison runs")
    pd_.set_defaults(func=_cmd_prepare)

    g = sub.add_parser("generate-data", help="write a synthetic train/held-out dataset")
    g.add_argument("--out", default=config.DEFAULT_SYNTHETIC_DIR)
    g.add_argument("--batteries", type=int, default=8)
    g.add_argument("--held-out", type=int, default=2)
    g.add_argument("--cycles", type=int, default=12)
    g.add_argument("--seed", type=int, default=0)
    g.set_defaults(func=_cmd_generate)

    t = sub.add_parser("train", help="train one model per seed")
    d_ = config.TRAIN_DEFAULTS
    t.add_argument("--data", default=config.DEFAULT_TRAIN_CSV)
    t.add_argument("--out", default=config.DEFAULT_ARTIFACTS_DIR)
    t.add_argument("--seeds", type=int, nargs="+", default=config.ENSEMBLE_SEEDS,
                   help="default: ten seeds, the size of the abuse-test ensemble")
    t.add_argument("--iterations", type=int, default=d_["iterations"],
                   help="Adam iterations (default: %(default)s; the notebook used 15000)")
    t.add_argument("--lbfgs-iterations", type=int, default=d_["lbfgs_iterations"],
                   help="L-BFGS iterations, 0 disables (default: %(default)s; the notebook used 2000)")
    t.add_argument("--num-domain", type=int, default=d_["num_domain"])
    t.add_argument("--num-boundary", type=int, default=d_["num_boundary"])
    t.add_argument("--num-initial", type=int, default=0,
                   help="explicit t=0 points for the initial conditions (default: 0, as in the notebook)")
    t.add_argument("--anchor-subset", type=_anchor_subset, default=d_["anchor_subset"], metavar="N|all",
                   help="seeded random subset of N observation points used as PDE anchors and data-fit "
                        "points; 'all' uses every point as the notebook did (default: %(default)s)")
    t.add_argument("--a-sei-init-spread", type=float, default=0.0, metavar="DECADES",
                   help="start the seeds' log10(A_sei) evenly spread across init +- DECADES so the "
                        "ensemble tests identifiability (default: 0, all seeds share the init)")
    t.add_argument("--log-a-sei-init", type=float, default=None, metavar="LOG10",
                   help="explicit initial log10(A_sei) for every seed in this run (overrides the spread; "
                        "use it when one ensemble's seeds are trained in separate processes)")
    t.add_argument("--newton-cooling", action="store_true",
                   help="opt-in: add a Newton surface heat-loss term to the thermal PDE")
    t.add_argument("--learn-thermal", action="store_true",
                   help="opt-in: make k_cond, rho*Cp (and the Newton coefficient) trainable")
    t.add_argument("--fix-log-a-sei", type=float, default=None, metavar="LOG10",
                   help="freeze log10(A_sei) at this value instead of training it")
    t.add_argument("--display-every", type=int, default=d_["display_every"],
                   help="print and log the loss terms every N steps (default: %(default)s)")
    t.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                   help="auto = CUDA when it works, else CPU; cuda falls back to CPU with a warning "
                        "if CUDA is unavailable (default: %(default)s)")
    t.set_defaults(func=_cmd_train)

    e = sub.add_parser("evaluate", help="train vs held-out temperature error")
    e.add_argument("--checkpoints", default=config.DEFAULT_ARTIFACTS_DIR)
    e.add_argument("--data", default=config.DEFAULT_TRAIN_CSV)
    e.add_argument("--held-out-data", default=config.DEFAULT_HELD_OUT_CSV)
    e.add_argument("--held-out-batteries", nargs="+", default=config.HELD_OUT_BATTERIES,
                   help="batteries the held-out file must contain with surviving cycles; "
                        "missing or empty ones are an error (default: config.HELD_OUT_BATTERIES)")
    e.add_argument("--no-battery-check", action="store_true",
                   help="skip that check (for synthetic data, whose battery IDs are not designated)")
    e.set_defaults(func=_cmd_evaluate)

    a = sub.add_parser("abuse-test", help=f"ambient-stress sweep verdict over an ensemble of seeds "
                       f"(core temperature unvalidated; verdict excludes ambient > {config.AMBIENT_TRAINED_MAX_C:.0f} C)")
    a.add_argument("--checkpoints", default=config.DEFAULT_ARTIFACTS_DIR)
    a.add_argument("--data", default=config.DEFAULT_TRAIN_CSV,
                   help="training table used for the data-driven SEI threshold (default: %(default)s)")
    a.add_argument("--min-models", type=int, default=1,
                   help=f"refuse to run with fewer checkpoints than this (the intended ensemble is "
                        f"{config.ABUSE_ENSEMBLE_SIZE} seeds)")
    a.set_defaults(func=_cmd_abuse)

    d = sub.add_parser("demo", help="generate data, train briefly, evaluate, abuse-test")
    d.add_argument("--workdir", default=".")
    d.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    d.add_argument("--iterations", type=int, default=1000)
    d.add_argument("--lbfgs-iterations", type=int, default=0)
    d.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    d.set_defaults(func=_cmd_demo)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
