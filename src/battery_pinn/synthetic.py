"""Synthetic stand-in for the processed NASA dataset.

This is NOT real battery data. It reproduces the schema (and rough magnitudes) of the
processed table so the pipeline can run end to end without the NASA files. Results from it
say nothing about real cells.
"""
import numpy as np
import pandas as pd

from battery_pinn.config import AMBIENT_GROUPS_C, BINS_PER_CYCLE, PROCESSED_COLUMNS

DISCHARGE_CURRENT_A = -2.0


def _ocv(soc_frac):
    s = np.clip(soc_frac, 0.0, 1.0)
    return 3.0 + 1.2 * s - 0.25 * np.exp(-20.0 * s) + 0.1 * s ** 4


def generate_cells(n_batteries=8, cycles_per_battery=12, seed=0, noise_c=0.2):
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_batteries):
        battery_id = f"S{b + 1:04d}"
        ambient = AMBIENT_GROUPS_C[b % len(AMBIENT_GROUPS_C)]
        re0 = rng.uniform(0.03, 0.08)
        for cyc in range(cycles_per_battery):
            frac = cyc / max(cycles_per_battery - 1, 1)
            capacity = 2.0 * (1.0 - 0.3 * frac) * rng.uniform(0.98, 1.02)
            soh = capacity / 2.0 * 100.0
            re = re0 * (1.0 + 1.5 * frac)
            duration = capacity / abs(DISCHARGE_CURRENT_A) * 3600.0
            # Identical normalised grid for every cycle, so the pooled profile has exactly
            # BINS_PER_CYCLE breakpoints (torch_interp also copes with near-duplicates now).
            tn = np.arange(1, BINS_PER_CYCLE + 1) / BINS_PER_CYCLE
            t = tn * duration
            soc = 100.0 * (1.0 - t / duration)
            volt = _ocv(soc / 100.0) - abs(DISCHARGE_CURRENT_A) * re * 4.0
            heat = (3.0 + 40.0 * re) * (1.0 - np.exp(-3.0 * tn))
            temp = ambient + heat + rng.normal(0.0, noise_c, size=t.shape)
            rows.append(pd.DataFrame({
                "bin": np.arange(BINS_PER_CYCLE),
                "Time": t,
                "Voltage_measured": volt,
                "Current_measured": DISCHARGE_CURRENT_A,
                "Temperature_measured": temp,
                "SoC": soc,
                "battery_id": battery_id,
                "cycle_number": cyc + 1,
                "SoH": soh,
                "Re": re,
                "Rct": re * 2.0,
                "ambient_temperature": ambient,
                "Time_normalized": tn,
            }))
    return pd.concat(rows, ignore_index=True)[PROCESSED_COLUMNS]


def generate_train_and_held_out(n_batteries=8, n_held_out=2, cycles_per_battery=12, seed=0):
    """Split whole batteries (never individual cycles) into train / held-out sets."""
    df = generate_cells(n_batteries, cycles_per_battery, seed)
    ids = sorted(df["battery_id"].unique())
    held_out_ids = ids[-n_held_out:] if n_held_out else []
    held_out = df[df["battery_id"].isin(held_out_ids)].reset_index(drop=True)
    train = df[~df["battery_id"].isin(held_out_ids)].reset_index(drop=True)
    return train, held_out
