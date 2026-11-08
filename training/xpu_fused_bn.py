"""Fused training-mode BatchNorm2d + SiLU (or identity) for channels_last tensors, as Triton kernels.

Ultralytics' Conv block is conv -> BatchNorm2d -> SiLU. On the A770 the stock XPU kernels for that BN + SiLU pair are
six kernels and 13 passes over the activation per training step (BN statistics, BN transform, SiLU; SiLU backward,
BN backward reduce, BN backward elementwise). Here it is 8 passes:

    forward   stats (per-channel shifted sums) -> finalize (mean, invstd, running stats) -> apply (affine + SiLU)
    backward  reduce (sum dz, sum dz*xhat; z and SiLU' recomputed from x) -> finalize -> dx

The BN output z is not saved for SiLU's backward any more (it is recomputed from x), which also saves memory.

Semantics follow torch.nn.BatchNorm2d in training mode (track_running_stats, float momentum): batch mean and biased
variance normalize; running_mean/running_var get the momentum update with the unbiased variance; num_batches_tracked
+= 1. All arithmetic is fp32; the output has the input's dtype. The variance uses sums of (x - running_mean), which is
exact enough unless the batch mean is >~300 standard deviations from the running mean. Results are not bit-identical
to the stock kernels (different reduction order; SiLU sees the fp32 pre-activation instead of its fp16 rounding): in
fp16 they are closer to an fp32 reference than the stock kernels are (bench_step.py mode=bnparity).

Tile sizes are autotuned per (shape, channels) on first use; results are cached on disk by Triton.
"""
import torch
import triton
import triton.language as tl
from triton.runtime import driver as _driver

# --- tuning -------------------------------------------------------------------------------------------------------

_RED_TILES = [(8, 128, 8), (16, 64, 8), (32, 128, 8), (32, 64, 8), (64, 64, 8), (64, 32, 8), (128, 32, 16),
              (128, 16, 8), (256, 16, 8)]  # (BLOCK_M, BLOCK_C, num_warps) for the column reductions
_FIN_G, _FIN_C, _FIN_W = 256, 16, 8  # finalize kernels: [row groups x channels] tile per loop step, warps
_EW_TILES = [(1024, 4), (1024, 8), (2048, 4), (2048, 8), (4096, 8)]  # (BLOCK, num_warps) for the elementwise passes


def _prune_red(configs, named_args, **kwargs):
    c = triton.next_power_of_2(named_args["C"])
    keep = [cf for cf in configs if cf.kwargs["BLOCK_C"] <= max(c, 16)]
    return keep or configs[:1]


_red_configs = [triton.Config({"BLOCK_M": m, "BLOCK_C": c}, num_warps=w) for m, c, w in _RED_TILES]
_ew_configs = [triton.Config({"BLOCK": b}, num_warps=w) for b, w in _EW_TILES]


def _red_autotune(fn):
    return triton.autotune(_red_configs, key=["M", "C"], prune_configs_by={"early_config_prune": _prune_red},
                           cache_results=True)(fn)


def _ew_autotune(fn):
    return triton.autotune(_ew_configs, key=["total", "C"], cache_results=True)(fn)


# --- kernels ------------------------------------------------------------------------------------------------------


@_red_autotune
@triton.jit
def _stats(x_ptr, shift_ptr, part_ptr, M, C, G, ROWS, BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr):
    """part[0, g, c] = sum (x - shift[c]), part[1, g, c] = sum (x - shift[c])^2 over program g's rows."""
    pid_g = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    cmask = cols < C
    k = tl.load(shift_ptr + cols, mask=cmask, other=0.0)
    s1 = tl.zeros([BLOCK_M, BLOCK_C], dtype=tl.float32)
    s2 = tl.zeros([BLOCK_M, BLOCK_C], dtype=tl.float32)
    r0 = pid_g * ROWS
    r1 = tl.minimum(r0 + ROWS, M)
    for r in range(r0, r1, BLOCK_M):
        rows = r + tl.arange(0, BLOCK_M)
        mask = (rows < r1)[:, None] & cmask[None, :]
        x = tl.load(x_ptr + rows.to(tl.int64)[:, None] * C + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        d = tl.where(mask, x - k[None, :], 0.0)
        s1 += d
        s2 += d * d
    tl.store(part_ptr + pid_g * C + cols, tl.sum(s1, 0), mask=cmask)
    tl.store(part_ptr + (G + pid_g) * C + cols, tl.sum(s2, 0), mask=cmask)


@triton.jit
def _stats_finalize(part_ptr, coef_ptr, w_ptr, b_ptr, rmean_ptr, rvar_ptr, nbt_ptr, M, C, G, eps, momentum,
                    BLOCK_G: tl.constexpr, BLOCK_C: tl.constexpr):
    """coef = [mean, invstd, scale, shift] (4, C); running stats momentum update; num_batches_tracked += 1."""
    pid = tl.program_id(0)
    cols = pid * BLOCK_C + tl.arange(0, BLOCK_C)
    cmask = cols < C
    s1 = tl.zeros([BLOCK_C], dtype=tl.float32)
    s2 = tl.zeros([BLOCK_C], dtype=tl.float32)
    for g0 in range(0, G, BLOCK_G):
        g = g0 + tl.arange(0, BLOCK_G)
        mask = (g < G)[:, None] & cmask[None, :]
        off = g[:, None] * C + cols[None, :]
        s1 += tl.sum(tl.load(part_ptr + off, mask=mask, other=0.0), 0)
        s2 += tl.sum(tl.load(part_ptr + G * C + off, mask=mask, other=0.0), 0)
    rm = tl.load(rmean_ptr + cols, mask=cmask, other=0.0)
    rv = tl.load(rvar_ptr + cols, mask=cmask, other=0.0)
    dm = s1 / M
    mean = rm + dm
    var = tl.maximum(s2 / M - dm * dm, 0.0)
    invstd = 1.0 / tl.sqrt(var + eps)
    scale = invstd * tl.load(w_ptr + cols, mask=cmask, other=0.0)
    tl.store(coef_ptr + cols, mean, mask=cmask)
    tl.store(coef_ptr + C + cols, invstd, mask=cmask)
    tl.store(coef_ptr + 2 * C + cols, scale, mask=cmask)
    tl.store(coef_ptr + 3 * C + cols, tl.load(b_ptr + cols, mask=cmask, other=0.0) - mean * scale, mask=cmask)
    tl.store(rmean_ptr + cols, rm * (1.0 - momentum) + mean * momentum, mask=cmask)
    tl.store(rvar_ptr + cols, rv * (1.0 - momentum) + var * (M / (M - 1.0)) * momentum, mask=cmask)
    if pid == 0:
        tl.store(nbt_ptr, tl.load(nbt_ptr) + 1)


@_ew_autotune
@triton.jit
def _apply(x_ptr, y_ptr, coef_ptr, total, C: tl.constexpr, ACT: tl.constexpr, BLOCK: tl.constexpr):
    """y = act(x * scale[c] + shift[c]) over the flat channels_last storage."""
    o = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = o < total
    c = o % C
    z = tl.load(x_ptr + o, mask=m, other=0.0).to(tl.float32) * tl.load(coef_ptr + 2 * C + c) \
        + tl.load(coef_ptr + 3 * C + c)
    if ACT == 1:
        z = z * tl.sigmoid(z)
    tl.store(y_ptr + o, z.to(y_ptr.dtype.element_ty), mask=m)


@_red_autotune
@triton.jit
def _bwd_reduce(dy_ptr, x_ptr, coef_ptr, part_ptr, M, C, G, ROWS,
                ACT: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr):
    """part[0, g, c] = sum dz, part[1, g, c] = sum dz * xhat over program g's rows (dz = grad of the BN output)."""
    pid_g = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    cmask = cols < C
    mean = tl.load(coef_ptr + cols, mask=cmask, other=0.0)[None, :]
    invstd = tl.load(coef_ptr + C + cols, mask=cmask, other=0.0)[None, :]
    scale = tl.load(coef_ptr + 2 * C + cols, mask=cmask, other=0.0)[None, :]
    shift = tl.load(coef_ptr + 3 * C + cols, mask=cmask, other=0.0)[None, :]
    sdb = tl.zeros([BLOCK_M, BLOCK_C], dtype=tl.float32)
    sdw = tl.zeros([BLOCK_M, BLOCK_C], dtype=tl.float32)
    r0 = pid_g * ROWS
    r1 = tl.minimum(r0 + ROWS, M)
    for r in range(r0, r1, BLOCK_M):
        rows = r + tl.arange(0, BLOCK_M)
        mask = (rows < r1)[:, None] & cmask[None, :]
        off = rows.to(tl.int64)[:, None] * C + cols[None, :]
        x = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
        dz = tl.load(dy_ptr + off, mask=mask, other=0.0).to(tl.float32)
        if ACT == 1:
            s = tl.sigmoid(x * scale + shift)
            dz = dz * s * (1.0 + (x * scale + shift) * (1.0 - s))
        sdb += dz
        sdw += dz * ((x - mean) * invstd)
    tl.store(part_ptr + pid_g * C + cols, tl.sum(sdb, 0), mask=cmask)
    tl.store(part_ptr + (G + pid_g) * C + cols, tl.sum(sdw, 0), mask=cmask)


@triton.jit
def _bwd_finalize(part_ptr, coef_ptr, w_ptr, db_ptr, dw_ptr, M, C, G, BLOCK_G: tl.constexpr, BLOCK_C: tl.constexpr):
    """db, dw (the bias/weight grads) and coef[4:7] = [A, R, T] so that dx = A * dz + R * x + T."""
    pid = tl.program_id(0)
    cols = pid * BLOCK_C + tl.arange(0, BLOCK_C)
    cmask = cols < C
    sdb = tl.zeros([BLOCK_C], dtype=tl.float32)
    sdw = tl.zeros([BLOCK_C], dtype=tl.float32)
    for g0 in range(0, G, BLOCK_G):
        g = g0 + tl.arange(0, BLOCK_G)
        mask = (g < G)[:, None] & cmask[None, :]
        off = g[:, None] * C + cols[None, :]
        sdb += tl.sum(tl.load(part_ptr + off, mask=mask, other=0.0), 0)
        sdw += tl.sum(tl.load(part_ptr + G * C + off, mask=mask, other=0.0), 0)
    tl.store(db_ptr + cols, sdb, mask=cmask)
    tl.store(dw_ptr + cols, sdw, mask=cmask)
    mean = tl.load(coef_ptr + cols, mask=cmask, other=0.0)
    invstd = tl.load(coef_ptr + C + cols, mask=cmask, other=0.0)
    a = tl.load(w_ptr + cols, mask=cmask, other=0.0) * invstd
    # dx = a * (dz - sdb/M - xhat * sdw/M), xhat = (x - mean) * invstd
    p = invstd * sdw / M
    tl.store(coef_ptr + 4 * C + cols, a, mask=cmask)
    tl.store(coef_ptr + 5 * C + cols, -a * p, mask=cmask)
    tl.store(coef_ptr + 6 * C + cols, a * (mean * p - sdb / M), mask=cmask)


@_ew_autotune
@triton.jit
def _bwd_dx(dy_ptr, x_ptr, dx_ptr, coef_ptr, total, C: tl.constexpr, ACT: tl.constexpr, BLOCK: tl.constexpr):
    o = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = o < total
    c = o % C
    x = tl.load(x_ptr + o, mask=m, other=0.0).to(tl.float32)
    dz = tl.load(dy_ptr + o, mask=m, other=0.0).to(tl.float32)
    if ACT == 1:
        z = x * tl.load(coef_ptr + 2 * C + c) + tl.load(coef_ptr + 3 * C + c)
        s = tl.sigmoid(z)
        dz = dz * s * (1.0 + z * (1.0 - s))
    dx = tl.load(coef_ptr + 4 * C + c) * dz + tl.load(coef_ptr + 5 * C + c) * x + tl.load(coef_ptr + 6 * C + c)
    tl.store(dx_ptr + o, dx.to(dx_ptr.dtype.element_ty), mask=m)


# --- autograd -----------------------------------------------------------------------------------------------------


def _groups(M, C):
    """Row groups for the column reductions: ~32k elements per group, between 64 and 1024 groups."""
    g = max(64, min(1024, triton.cdiv(M * C, 32768)))
    g = min(g, M)
    rows = triton.cdiv(M, g)
    return triton.cdiv(M, rows), rows


def _red_grid(g, C):
    return lambda meta: (g, triton.cdiv(C, meta["BLOCK_C"]))


def _ew_grid(total):
    return lambda meta: (triton.cdiv(total, meta["BLOCK"]),)


class _Launch:
    """Launch a (possibly autotuned) Triton kernel, after the first call for a key straight through the compiled
    kernel's launcher. Triton's per-call argument binding, specialization and autotuner bookkeeping cost ~26 us of
    CPU per launch here, and a training step has ~700 of these launches.

    Only the tensor arguments may change between calls with the same key: the key must determine every integer and
    constexpr argument (they are baked into the cached launch). Calls with a tensor not 16-byte aligned, or while
    Triton launch hooks are set (e.g. by a profiler), take Triton's normal path.
    """

    def __init__(self, kernel):
        self.kernel = kernel
        self.names = list(kernel.arg_names)
        self.cache = {}

    def __call__(self, grid, key, *args, **const):
        aligned = all(a.data_ptr() % 16 == 0 for a in args if torch.is_tensor(a))
        hooks = triton.knobs.runtime.launch_enter_hook.calls or triton.knobs.runtime.launch_exit_hook.calls
        hit = self.cache.get(key) if aligned and not hooks else None
        if hit is not None:
            ck, (g0, g1, g2), tail = hit
            dev = _driver.active.get_current_device()
            ck.run(g0, g1, g2, _driver.active.get_current_stream(dev), ck.function, ck.packed_metadata, None, None, None,
                   *args, *tail)
            return
        ck = self.kernel[grid](*args, **const)
        if aligned and not hooks:
            cfg = getattr(self.kernel, "best_config", None)
            meta = {**const, **(cfg.kwargs if cfg is not None else {})}
            vals = dict(zip(self.names, args), **meta)
            tail = tuple(vals[n] for n in self.names[len(args):])
            g = tuple(grid(meta) if callable(grid) else grid) + (1, 1)
            self.cache[key] = (ck, g[:3], tail)


_L = {}  # kernel name -> _Launch, filled below


class _BNAct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, running_mean, running_var, num_batches_tracked, momentum, eps, act):
        N, C, H, W = x.shape
        M = N * H * W
        g, rows = _groups(M, C)
        key = (M, C, x.dtype, act)
        part = torch.empty((2, g, C), dtype=torch.float32, device=x.device)
        _L["stats"](_red_grid(g, C), key, x, running_mean, part, M, C, g, rows)
        coef = torch.empty((7, C), dtype=torch.float32, device=x.device)  # rows 4..6 are filled in backward
        _L["stats_finalize"]((triton.cdiv(C, _FIN_C),), (key, eps, momentum), part, coef, weight, bias, running_mean,
                             running_var, num_batches_tracked, M, C, g, eps, momentum, BLOCK_G=_FIN_G, BLOCK_C=_FIN_C,
                             num_warps=_FIN_W)
        y = torch.empty_like(x, memory_format=torch.channels_last)
        _L["apply"](_ew_grid(x.numel()), key, x, y, coef, x.numel(), C=C, ACT=act)
        ctx.save_for_backward(x, weight, coef)
        ctx.act = act
        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight, coef = ctx.saved_tensors
        if not dy.is_contiguous(memory_format=torch.channels_last):
            dy = dy.contiguous(memory_format=torch.channels_last)
        N, C, H, W = x.shape
        M = N * H * W
        g, rows = _groups(M, C)
        key = (M, C, x.dtype, dy.dtype, ctx.act)
        part = torch.empty((2, g, C), dtype=torch.float32, device=x.device)
        _L["bwd_reduce"](_red_grid(g, C), key, dy, x, coef, part, M, C, g, rows, ACT=ctx.act)
        db = torch.empty(C, dtype=torch.float32, device=x.device)
        dw = torch.empty(C, dtype=torch.float32, device=x.device)
        coef = coef.clone()  # backward may run more than once (retain_graph); keep the saved coef pristine
        _L["bwd_finalize"]((triton.cdiv(C, _FIN_C),), key, part, coef, weight, db, dw, M, C, g, BLOCK_G=_FIN_G,
                           BLOCK_C=_FIN_C, num_warps=_FIN_W)
        dx = torch.empty_like(x, memory_format=torch.channels_last)
        _L["bwd_dx"](_ew_grid(x.numel()), key, dy, x, dx, coef, x.numel(), C=C, ACT=ctx.act)
        return (dx, dw if ctx.needs_input_grad[1] else None, db if ctx.needs_input_grad[2] else None,
                None, None, None, None, None, None)


_L.update(stats=_Launch(_stats), stats_finalize=_Launch(_stats_finalize), apply=_Launch(_apply),
          bwd_reduce=_Launch(_bwd_reduce), bwd_finalize=_Launch(_bwd_finalize), bwd_dx=_Launch(_bwd_dx))


def supported(x, bn, act):
    """True when bn_act(x, bn, act) takes the fused path; otherwise use act(bn(x))."""
    return (
        bn.training
        and torch.is_grad_enabled()
        and x.device.type == "xpu"
        and x.dim() == 4
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and x.is_contiguous(memory_format=torch.channels_last)
        and type(bn) is torch.nn.BatchNorm2d
        and bn.track_running_stats
        and bn.running_mean is not None
        and bn.affine
        and isinstance(bn.momentum, float)
        and bn.weight.dtype == torch.float32
        and x.shape[0] * x.shape[2] * x.shape[3] > 1
        and type(act) in (torch.nn.SiLU, torch.nn.Identity)
    )


def bn_act(x, bn, act):
    """act(bn(x)) for a training-mode BatchNorm2d and a SiLU/Identity act, fused (see the module docstring)."""
    return _BNAct.apply(x, bn.weight, bn.bias, bn.running_mean, bn.running_var, bn.num_batches_tracked,
                        float(bn.momentum), float(bn.eps), 1 if type(act) is torch.nn.SiLU else 0)
