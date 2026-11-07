"""XPU compatibility helpers for GPUs without native FP64 (e.g. Intel Arc A770).

torch.amp.GradScaler computes the scale reciprocal in float64 (unscale_) and
uses a fused native kernel that also needs float64 (update). Both raise
"Required aspect fp64 is not supported on the device" on such GPUs. The loss
scale is always a power of two, so an fp32 reciprocal is exact.

Call `install()` before ultralytics creates its trainer; it replaces
`torch.amp.GradScaler` with `XPUGradScaler`.

Written against torch 2.15 and relies on GradScaler private attributes, so
re-check this if torch is upgraded.
"""
from collections import defaultdict

import torch
from torch.amp.grad_scaler import GradScaler, OptState, _refresh_per_optimizer_state


class XPUGradScaler(GradScaler):
    def unscale_(self, optimizer):
        if not self._enabled:
            return
        self._check_scale_growth_tracker("unscale_")
        state = self._per_optimizer_states[id(optimizer)]
        if state["stage"] is OptState.UNSCALED:
            raise RuntimeError("unscale_() has already been called on this optimizer since the last update().")
        elif state["stage"] is OptState.STEPPED:
            raise RuntimeError("unscale_() is being called after step().")
        assert self._scale is not None
        inv_scale = self._scale.reciprocal()  # fp32; exact because scale is a power of two
        found_inf = torch.full((), 0.0, dtype=torch.float32, device=self._scale.device)
        state["found_inf_per_device"] = self._unscale_grads_(optimizer, inv_scale, found_inf, False)
        state["stage"] = OptState.UNSCALED

    def update(self, new_scale=None):
        # torch._amp_update_scale_ is a fused native kernel that needs fp64 on XPU, so reimplement
        # the same growth/backoff rule with fp32/int32 tensor ops (no host sync).
        if not self._enabled:
            return
        if new_scale is not None:
            return super().update(new_scale)
        scale, tracker = self._check_scale_growth_tracker("update")
        found_infs = [
            f.to(device=scale.device, non_blocking=True)
            for s in self._per_optimizer_states.values()
            for f in s["found_inf_per_device"].values()
        ]
        if not found_infs:
            raise AssertionError("No inf checks were recorded prior to update.")
        found = sum(found_infs[1:], found_infs[0]) > 0
        nxt = torch.where(found, torch.zeros_like(tracker), tracker + 1)
        grow = (~found) & (nxt == self._growth_interval)
        new_s = torch.where(found, scale * self._backoff_factor, torch.where(grow, scale * self._growth_factor, scale))
        tracker.copy_(torch.where(grow, torch.zeros_like(nxt), nxt))
        scale.copy_(new_s)
        self._per_optimizer_states = defaultdict(_refresh_per_optimizer_state)


def install():
    """Make `torch.amp.GradScaler(...)` return an `XPUGradScaler`."""
    torch.amp.GradScaler = XPUGradScaler
