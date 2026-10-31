"""Compute-device selection for training: ``auto`` / ``cpu`` / ``cuda`` with a safe CPU fallback.

DeepXDE's PyTorch backend makes CUDA (or Apple MPS) the *default* tensor device at import
time whenever it is detected, so honouring a device choice means (1) checking the device can
actually run a tensor op -- ``torch.cuda.is_available()`` is true on machines whose driver
cannot initialise -- and (2) resetting PyTorch's default device so that every tensor DeepXDE
creates lands there too. ``auto`` therefore means CUDA when it works and CPU otherwise; MPS
is never selected automatically because DeepXDE's L-BFGS path is not reliable on it.
"""

from collections.abc import Callable

import torch

DEVICE_CHOICES = ("auto", "cpu", "cuda")


def probe(device: str | torch.device) -> str | None:
    """Run one tiny tensor op on ``device``; returns the error message, or None if it works."""
    try:
        (torch.zeros(1, device=device) + 1).item()
        return None
    except Exception as exc:  # noqa: BLE001 - any failure means "do not use this device"
        return f"{type(exc).__name__}: {exc}"


def resolve_device(requested: str | None = "auto", log: Callable[[str], object] | None = print) -> torch.device:
    """Map a ``--device`` choice to a usable :class:`torch.device`.

    * ``cpu``  -> CPU.
    * ``cuda`` -> CUDA if PyTorch reports it available *and* a probe op succeeds; otherwise a
      warning is logged and CPU is used.
    * ``auto`` -> CUDA under the same conditions, silently CPU otherwise.
    """
    requested = (requested or "auto").lower()
    if requested not in DEVICE_CHOICES:
        raise ValueError(f"device must be one of {DEVICE_CHOICES}, got {requested!r}")
    if requested == "cpu":
        return torch.device("cpu")

    if not torch.cuda.is_available():
        if requested == "cuda" and log is not None:
            log("WARNING: --device cuda requested but PyTorch reports no usable CUDA device; falling back to CPU.")
        return torch.device("cpu")

    cuda = torch.device("cuda")
    error = probe(cuda)
    if error is not None:
        if log is not None:
            log(f"WARNING: CUDA is reported available but cannot run a tensor op ({error}); falling back to CPU.")
        return torch.device("cpu")
    return cuda


def activate(device: str | torch.device) -> torch.device:
    """Make ``device`` PyTorch's default tensor device (so DeepXDE builds the model on it)."""
    device = torch.device(device)
    torch.set_default_device(device)
    return device


def describe(device: str | torch.device) -> str:
    """Short human-readable name of ``device`` (includes the GPU model for CUDA)."""
    device = torch.device(device)
    if device.type == "cuda":
        try:
            return f"cuda ({torch.cuda.get_device_name(device)})"
        except Exception:  # noqa: BLE001
            return "cuda"
    return str(device.type)
