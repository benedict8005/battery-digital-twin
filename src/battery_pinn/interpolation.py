"""GPU-native linear interpolation of the representative discharge profile."""

import torch

# Two breakpoints closer than this fraction of the breakpoint range are treated as the same
# point. The pooled profile's Time_normalized values come from many cycles, so values that are
# equal up to float32 rounding (one ulp apart) are common; dividing by such a gap produced
# slopes of order 1e6 and wrecked the extrapolation at t=0 and t=1.
MIN_RELATIVE_SPACING = 1e-6


def default_device():
    """CUDA when usable, otherwise CPU (see :func:`battery_pinn.device.resolve_device`)."""
    from battery_pinn.device import resolve_device

    return resolve_device("auto", log=None)


def _min_spacing(xp, min_spacing):
    """Gap below which two breakpoints count as coincident (a 0-d tensor, no host sync)."""
    if min_spacing is not None:
        return torch.as_tensor(float(min_spacing), dtype=xp.dtype, device=xp.device)
    span = xp[-1] - xp[0]
    # Never below a few ulps of the breakpoint magnitude, so float rounding noise always merges.
    ulp = torch.finfo(xp.dtype).eps * xp.abs().max().clamp(min=1.0)
    return torch.maximum(MIN_RELATIVE_SPACING * span, 4.0 * ulp)


def _safe_divide(num, den, tol):
    """``num / den`` where ``den >= tol``, else 0 (a step instead of a near-infinite slope)."""
    degenerate = den < tol
    return torch.where(degenerate, torch.zeros_like(num), num / torch.where(degenerate, torch.ones_like(den), den))


def _edge_slopes(xp, fp, tol):
    """Extrapolation slopes at both ends, taken to the nearest breakpoint at least ``tol`` away.

    Zero (constant extrapolation) when every breakpoint lies within ``tol`` of that end.
    """
    n = len(xp)
    j = torch.searchsorted(xp, (xp[0] + tol).reshape(1), right=True).clamp(max=n - 1)[0]
    lower = _safe_divide(fp[j] - fp[0], xp[j] - xp[0], tol)
    k = (torch.searchsorted(xp, (xp[-1] - tol).reshape(1), right=False) - 1).clamp(min=0)[0]
    upper = _safe_divide(fp[-1] - fp[k], xp[-1] - xp[k], tol)
    return lower, upper


def torch_interp(x, xp, fp, min_spacing=None):
    """1D linear interpolation with linear extrapolation at the edges.

    ``x``, ``xp``, ``fp`` are torch tensors; ``xp`` must be sorted ascending (duplicates and
    near-duplicates allowed). Breakpoints closer than ``min_spacing`` (default: a fraction
    ``MIN_RELATIVE_SPACING`` of the breakpoint range, but at least a few ulps) are treated as
    coincident: inside such a gap the left value is returned (a step, never a division by a
    tiny number), and the extrapolation slopes at both ends are taken to the nearest
    *well-separated* neighbour, so near-identical end breakpoints cannot blow them up. If all
    breakpoints coincide the function is constant. Everything is tensor arithmetic, so no
    host/device synchronisation happens inside the PDE residual.
    """
    x = x.reshape(-1).contiguous()  # searchsorted wants contiguous values (x often is a column slice)
    tol = _min_spacing(xp, min_spacing)
    idx = torch.searchsorted(xp, x, right=True)
    idx = torch.clamp(idx, 1, len(xp) - 1)

    x0 = xp[idx - 1]
    x1 = xp[idx]
    y0 = fp[idx - 1]
    y1 = fp[idx]

    out = y0 + _safe_divide(y1 - y0, x1 - x0, tol) * (x - x0)

    lower_slope, upper_slope = _edge_slopes(xp, fp, tol)
    out = torch.where(x < xp[0], fp[0] + lower_slope * (x - xp[0]), out)
    out = torch.where(x > xp[-1], fp[-1] + upper_slope * (x - xp[-1]), out)
    return out.view(-1, 1)


class DischargeProfile:
    """Lookup tables (time -> current, voltage, SoC) held on one device."""

    def __init__(self, df_agg, device=None):
        self.device = torch.device(device) if device is not None else default_device()

        def tensor(col):
            return torch.tensor(df_agg[col].values, dtype=torch.float32, device=self.device)

        self.t = tensor("Time_normalized")
        self.current = tensor("Current_measured")
        self.voltage = tensor("Voltage_measured")
        self.soc = tensor("SoC")

    def at(self, t_norm):
        """Return (I, V, SoC) at normalised times, shaped (N, 1) on t_norm's device."""
        t_flat = t_norm.view(-1).to(self.device)
        out_device = t_norm.device
        return (
            torch_interp(t_flat, self.t, self.current).to(out_device),
            torch_interp(t_flat, self.t, self.voltage).to(out_device),
            torch_interp(t_flat, self.t, self.soc).to(out_device),
        )
