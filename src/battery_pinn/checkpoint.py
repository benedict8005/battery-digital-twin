"""Checkpoint save / load for the trained network and its calibrated parameters."""

import os

import torch

REQUIRED_KEYS = ("model_state_dict", "log_A_sei", "ocv_coeffs")


def save_checkpoint(path, state_dict, log_a_sei, ocv_coeffs, physics=None):
    """Write a checkpoint and force it to disk before returning.

    ``physics`` (optional) records the opt-in physics options and learned thermal constants
    the model was trained with; it is informational and not needed to run the network.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    checkpoint = {
        "model_state_dict": state_dict,
        "log_A_sei": float(log_a_sei),
        "ocv_coeffs": [float(c) for c in ocv_coeffs],
    }
    if physics:
        checkpoint["physics"] = physics
    with open(path, "wb") as f:
        torch.save(checkpoint, f)
        f.flush()
        os.fsync(f.fileno())
    return path


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location=torch.device("cpu"))
    if not (isinstance(checkpoint, dict) and all(k in checkpoint for k in REQUIRED_KEYS)):
        raise RuntimeError(f"{path}: incompatible checkpoint (expected keys {REQUIRED_KEYS}). Retrain to produce one.")
    return checkpoint
