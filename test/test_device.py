"""``--device`` resolution: auto-selection with a safe fallback to CPU."""
import pytest

torch = pytest.importorskip("torch")

from battery_pinn import device as dev  # noqa: E402


def test_cpu_is_always_cpu():
    assert dev.resolve_device("cpu") == torch.device("cpu")
    assert dev.resolve_device("CPU", log=None) == torch.device("cpu")


def test_invalid_choice_is_rejected():
    with pytest.raises(ValueError, match="auto"):
        dev.resolve_device("tpu")


def test_auto_without_cuda_is_cpu_and_silent(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    messages = []
    assert dev.resolve_device("auto", log=messages.append) == torch.device("cpu")
    assert dev.resolve_device(None, log=messages.append) == torch.device("cpu")
    assert messages == []


def test_cuda_requested_without_cuda_falls_back_with_warning(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    messages = []
    assert dev.resolve_device("cuda", log=messages.append) == torch.device("cpu")
    assert len(messages) == 1 and "falling back to CPU" in messages[0] and "--device cuda" in messages[0]


def test_cuda_reported_but_unusable_falls_back_with_warning(monkeypatch):
    """torch.cuda.is_available() can be True on a box whose driver cannot initialise."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(dev, "probe", lambda device: "RuntimeError: CUDA driver initialization failed")
    messages = []
    for requested in ("auto", "cuda"):
        assert dev.resolve_device(requested, log=messages.append) == torch.device("cpu")
    assert len(messages) == 2 and all("cannot run a tensor op" in m for m in messages)


def test_usable_cuda_is_selected(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(dev, "probe", lambda device: None)
    assert dev.resolve_device("auto").type == "cuda"
    assert dev.resolve_device("cuda").type == "cuda"


def test_probe_reports_failures_instead_of_raising():
    assert dev.probe("cpu") is None
    error = dev.probe("cuda:99")
    assert error is not None and isinstance(error, str)


def test_activate_sets_the_default_device():
    device = dev.activate("cpu")
    assert device == torch.device("cpu")
    assert torch.zeros(1).device.type == "cpu"
    assert dev.describe(device) == "cpu"
