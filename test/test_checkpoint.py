import pytest

torch = pytest.importorskip("torch")

from battery_pinn.checkpoint import load_checkpoint, save_checkpoint


def test_round_trip(tmp_path):
    state = {"w": torch.arange(4, dtype=torch.float32)}
    path = save_checkpoint(str(tmp_path / "sub" / "m.pt"), state, 22.5, [1.0, 2.0, 3.0])
    ckpt = load_checkpoint(path)
    assert ckpt["log_A_sei"] == pytest.approx(22.5)
    assert ckpt["ocv_coeffs"] == [1.0, 2.0, 3.0]
    assert torch.equal(ckpt["model_state_dict"]["w"], state["w"])


def test_legacy_weights_only_checkpoint_is_rejected(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save({"w": torch.zeros(1)}, str(path))
    with pytest.raises(RuntimeError, match="incompatible"):
        load_checkpoint(str(path))
