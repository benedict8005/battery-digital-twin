"""CLI wiring. The parser and the data commands need neither PyTorch nor DeepXDE."""

import pandas as pd
import pytest

from battery_pinn import config
from battery_pinn.cli import build_parser, main
from conftest import RAW_SPEC, build_raw_dataset


def test_parser_requires_a_command():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_prepare_data_defaults_come_from_config():
    args = build_parser().parse_args(["prepare-data"])
    assert args.raw == config.DEFAULT_RAW_DATA_DIR
    assert args.out == config.DEFAULT_PROCESSED_DIR


def test_train_defaults_are_the_fast_protocol_on_processed_data():
    args = build_parser().parse_args(["train"])
    assert args.seeds == config.ENSEMBLE_SEEDS == list(range(10))
    assert args.out == config.DEFAULT_ARTIFACTS_DIR
    assert args.data == config.DEFAULT_TRAIN_CSV and args.data.endswith("train.csv")
    assert (
        args.iterations,
        args.lbfgs_iterations,
        args.num_domain,
        args.num_boundary,
        args.anchor_subset,
        args.display_every,
    ) == (2000, 200, 1000, 200, 2000, 100)
    assert args.device == "auto"
    assert build_parser().parse_args(["train", "--anchor-subset", "all"]).anchor_subset is None
    with pytest.raises(SystemExit):
        build_parser().parse_args(["train", "--anchor-subset", "0"])
    assert not args.newton_cooling and not args.learn_thermal and args.fix_log_a_sei is None
    opts = build_parser().parse_args(["train", "--newton-cooling", "--learn-thermal", "--fix-log-a-sei", "18"])
    assert opts.newton_cooling and opts.learn_thermal and opts.fix_log_a_sei == 18.0


def test_evaluate_and_abuse_test_default_to_processed_tables():
    e = build_parser().parse_args(["evaluate"])
    assert e.data == config.DEFAULT_TRAIN_CSV and e.held_out_data == config.DEFAULT_HELD_OUT_CSV
    a = build_parser().parse_args(["abuse-test"])
    assert a.data == config.DEFAULT_TRAIN_CSV and a.data.endswith("train.csv") and a.min_models == 1


def test_train_convergence_flags():
    args = build_parser().parse_args(["train"])
    assert args.a_sei_init_spread == 0.0 and args.log_a_sei_init is None and args.num_initial == 0
    args = build_parser().parse_args(
        [
            "train",
            "--anchor-subset",
            "4000",
            "--a-sei-init-spread",
            "1.0",
            "--log-a-sei-init",
            "21.5",
            "--display-every",
            "100",
        ]
    )
    assert (args.anchor_subset, args.a_sei_init_spread, args.log_a_sei_init, args.display_every) == (
        4000,
        1.0,
        21.5,
        100,
    )


@pytest.mark.parametrize("command", ["train", "demo"])
def test_device_option_accepts_auto_cpu_cuda_only(command):
    parser = build_parser()
    for choice in ("auto", "cpu", "cuda"):
        assert parser.parse_args([command, "--device", choice]).device == choice
    with pytest.raises(SystemExit):
        parser.parse_args([command, "--device", "mps"])


def test_generate_data_writes_both_files(tmp_path):
    main(["generate-data", "--out", str(tmp_path), "--batteries", "4", "--held-out", "1", "--cycles", "3"])
    assert (tmp_path / "train.csv").exists()
    assert (tmp_path / "held_out.csv").exists()


def test_prepare_data_writes_processed_tables(raw_dataset, tmp_path, capsys):
    out = tmp_path / "processed"
    main(["prepare-data", "--raw", raw_dataset, "--out", str(out)])
    train = pd.read_csv(out / config.TRAIN_FILE)
    held_out = pd.read_csv(out / config.HELD_OUT_FILE)
    assert list(train.columns) == list(held_out.columns) == config.PROCESSED_COLUMNS
    assert set(train["battery_id"]).isdisjoint(held_out["battery_id"])
    assert set(held_out["battery_id"]) <= set(config.HELD_OUT_BATTERIES)
    assert set(held_out["battery_id"]) == set(config.HELD_OUT_BATTERIES)
    out_text = capsys.readouterr().out
    assert "--- Cycle processing summary ---" in out_text
    assert "--- Per-battery cycle survival (every designated battery) ---" in out_text
    assert "B0056 [HELD-OUT]: 2/3 cycles kept" in out_text


def test_prepare_data_exits_with_error_when_held_out_battery_missing(tmp_path):
    from battery_pinn.data import BatteryCoverageError

    root = tmp_path / "raw"
    build_raw_dataset(str(root), spec={k: v for k, v in RAW_SPEC.items() if k != "B0047"})
    with pytest.raises(BatteryCoverageError, match="B0047: not present in the raw metadata"):
        main(["prepare-data", "--raw", str(root), "--out", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


def test_prepare_data_capacity_floor_flags(raw_dataset, tmp_path, capsys):
    parser = build_parser()
    assert parser.parse_args(["prepare-data"]).min_capacity_fraction == config.MIN_CAPACITY_FRACTION_OF_MAX
    assert parser.parse_args(["prepare-data", "--min-capacity-fraction", "none"]).min_capacity_fraction is None
    assert parser.parse_args(["prepare-data", "--min-capacity-fraction", "0.5"]).min_capacity_fraction == 0.5
    for bad in ("0", "1.5", "abc"):
        with pytest.raises(SystemExit):
            parser.parse_args(["prepare-data", "--min-capacity-fraction", bad])

    # The legacy absolute floor empties B0056 in the fixture: a hard error ...
    from battery_pinn.data import BatteryCoverageError

    with pytest.raises(BatteryCoverageError, match="B0056"):
        main(
            ["prepare-data", "--raw", raw_dataset, "--out", str(tmp_path / "legacy"), "--min-capacity-fraction", "none"]
        )
    assert not (tmp_path / "legacy").exists()
    # ... unless explicitly allowed for legacy-comparison runs, which downgrades it to a warning.
    main(
        [
            "prepare-data",
            "--raw",
            raw_dataset,
            "--out",
            str(tmp_path / "legacy"),
            "--min-capacity-fraction",
            "none",
            "--allow-missing-held-out",
        ]
    )
    held_out = pd.read_csv(tmp_path / "legacy" / config.HELD_OUT_FILE)
    assert "B0056" not in set(held_out["battery_id"])
    assert "WARNING: --allow-missing-held-out set" in capsys.readouterr().out


def test_evaluate_parser_held_out_check_defaults():
    args = build_parser().parse_args(["evaluate"])
    assert args.held_out_batteries == config.HELD_OUT_BATTERIES and args.no_battery_check is False
    args = build_parser().parse_args(["evaluate", "--held-out-batteries", "B0025", "B0056", "--no-battery-check"])
    assert args.held_out_batteries == ["B0025", "B0056"] and args.no_battery_check is True


def test_convert_mat_then_prepare_data(tmp_path, capsys):
    """A tiny NASA-style .mat file round-trips through convert-mat into the raw layout."""
    from conftest import write_fake_mat

    mat_dir = tmp_path / "mat"
    write_fake_mat(mat_dir / "B0005.mat", "B0005")
    main(["convert-mat", "--mat-dir", str(mat_dir), "--out", str(tmp_path / "raw")])
    meta = pd.read_csv(tmp_path / "raw" / "metadata.csv")
    assert list(meta.columns) == [
        "type",
        "start_time",
        "ambient_temperature",
        "battery_id",
        "test_id",
        "uid",
        "filename",
        "Capacity",
        "Re",
        "Rct",
    ]
    assert meta["type"].tolist() == ["charge", "impedance", "discharge", "discharge"]
    assert meta["start_time"].iloc[2].startswith("[")
    assert (tmp_path / "raw" / "data" / "00003.csv").is_file()
    cycle = pd.read_csv(tmp_path / "raw" / "data" / "00003.csv")
    assert list(cycle.columns) == [
        "Voltage_measured",
        "Current_measured",
        "Temperature_measured",
        "Current_load",
        "Voltage_load",
        "Time",
    ]
    assert "B0005: 4 cycles" in capsys.readouterr().out


def test_prepare_data_and_generate_data_share_one_schema(raw_dataset, tmp_path):
    main(["prepare-data", "--raw", raw_dataset, "--out", str(tmp_path / "real")])
    main(
        ["generate-data", "--out", str(tmp_path / "synthetic"), "--batteries", "3", "--held-out", "1", "--cycles", "2"]
    )
    real = pd.read_csv(tmp_path / "real" / config.TRAIN_FILE)
    synthetic = pd.read_csv(tmp_path / "synthetic" / "train.csv")
    assert list(real.columns) == list(synthetic.columns)
    assert real.dtypes.to_dict() == synthetic.dtypes.to_dict()


def test_prepare_data_missing_raw_folder_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="metadata.csv"):
        main(["prepare-data", "--raw", str(tmp_path / "nope"), "--out", str(tmp_path / "out")])
