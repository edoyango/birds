"""Channel concatenation of channels_last (NHWC) tensors as one Triton kernel, with a backward that hands every input
a contiguous gradient.

torch.cat along the channels of NHWC tensors runs at 66-320 GB/s on the A770 (its generic batched-copy kernel); this
kernel copies each input's contiguous storage into its channel range of the output at 290-410 GB/s (yolo26s shapes,
2-4x faster). Stock autograd's backward for cat returns channel-slice views of the output gradient, which every
consumer then copies (.contiguous() in the conv and BatchNorm backward) or adds with a slow strided kernel; here the
backward is the same kernel in reverse and writes contiguous gradients. Values are copied, so results are identical.

Used by xpu_speedups (fast_cat) for the C2f/C3k2, C3, SPPF, C2PSA and Concat modules.
"""
import torch
import triton
import triton.language as tl

from xpu_fused_bn import _Launch

MAX_INPUTS = 4
_configs = [triton.Config({"BLOCK": b}, num_warps=w) for b, w in ((1024, 4), (2048, 8), (4096, 16))]


@triton.jit
def _part(out_ptr, p, total, j, C: tl.constexpr, Ci: tl.constexpr, off: tl.constexpr):
    m = j < total
    tl.store(out_ptr + (j // Ci) * C + off + (j % Ci), tl.load(p + j, mask=m), mask=m)


@triton.autotune(_configs, key=["total0", "total1", "total2", "total3", "C0", "C1", "C2", "C3"], cache_results=True)
@triton.jit
def _cat_nhwc(out_ptr, p0, p1, p2, p3, total0, total1, total2, total3,
              C: tl.constexpr, C0: tl.constexpr, C1: tl.constexpr, C2: tl.constexpr, C3: tl.constexpr,
              BLOCK: tl.constexpr):
    """out[row, off_i + c] = in_i[row, c]; program axis 1 = input i, axis 0 = blocks of in_i's (contiguous) storage."""
    i = tl.program_id(1)
    j = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    if i == 0:
        _part(out_ptr, p0, total0, j, C, C0, 0)
    elif i == 1:
        _part(out_ptr, p1, total1, j, C, C1, C0)
    elif i == 2:
        _part(out_ptr, p2, total2, j, C, C2, C0 + C1)
    else:
        _part(out_ptr, p3, total3, j, C, C3, C0 + C1 + C2)


@triton.autotune(_configs, key=["total", "C0", "C1", "C2", "C3"], cache_results=True)
@triton.jit
def _split_nhwc(g_ptr, p0, p1, p2, p3, total, C: tl.constexpr, C0: tl.constexpr, C1: tl.constexpr,
                C2: tl.constexpr, C3: tl.constexpr, BLOCK: tl.constexpr):
    """in_i[row, c] = g[row, off_i + c]: one pass over g, masked stores. (The mirror of _cat_nhwc, gathering per
    input inside if/elif branches, crashes Triton 3.9's XPU compiler.)"""
    o = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = o < total
    c = o % C
    row = o // C
    v = tl.load(g_ptr + o, mask=m)
    tl.store(p0 + row * C0 + c, v, mask=m & (c < C0))
    if C1 > 0:
        tl.store(p1 + row * C1 + (c - C0), v, mask=m & (c >= C0) & (c < C0 + C1))
    if C2 > 0:
        tl.store(p2 + row * C2 + (c - C0 - C1), v, mask=m & (c >= C0 + C1) & (c < C0 + C1 + C2))
    if C3 > 0:
        tl.store(p3 + row * C3 + (c - C0 - C1 - C2), v, mask=m & (c >= C0 + C1 + C2))


_launch_cat = _Launch(_cat_nhwc)
_launch_split = _Launch(_split_nhwc)


def _run(out, parts, split):
    n = len(parts)
    cs = [p.shape[1] for p in parts] + [0] * (MAX_INPUTS - n)
    ptrs = list(parts) + [parts[0]] * (MAX_INPUTS - n)
    const = dict(C=sum(cs), C0=cs[0], C1=cs[1], C2=cs[2], C3=cs[3])
    if split:
        total = out.numel()
        _launch_split(lambda meta: (triton.cdiv(total, meta["BLOCK"]),), (tuple(cs), total, out.dtype), out, *ptrs,
                      total, **const)
    else:
        tot = [p.numel() for p in parts] + [0] * (MAX_INPUTS - n)
        mx = max(tot)
        _launch_cat(lambda meta: (triton.cdiv(mx, meta["BLOCK"]), n), (tuple(cs), tuple(tot), out.dtype), out, *ptrs,
                    *tot, **const)


class _CatNHWC(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *xs):
        N, _, H, W = xs[0].shape
        cs = [x.shape[1] for x in xs]
        out = torch.empty((N, sum(cs), H, W), dtype=xs[0].dtype, device=xs[0].device,
                          memory_format=torch.channels_last)
        _run(out, xs, 0)
        ctx.cs = cs
        return out

    @staticmethod
    def backward(ctx, g):
        if not g.is_contiguous(memory_format=torch.channels_last):
            g = g.contiguous(memory_format=torch.channels_last)
        N, _, H, W = g.shape
        grads = [torch.empty((N, c, H, W), dtype=g.dtype, device=g.device, memory_format=torch.channels_last)
                 for c in ctx.cs]
        _run(g, grads, 1)
        return tuple(gr if need else None for gr, need in zip(grads, ctx.needs_input_grad))


def supported(xs):
    if not 2 <= len(xs) <= MAX_INPUTS:
        return False
    x0 = xs[0]
    if not (torch.is_tensor(x0) and x0.device.type == "xpu" and x0.dim() == 4
            and x0.dtype in (torch.float16, torch.bfloat16, torch.float32)):
        return False
    n, _, h, w = x0.shape
    return all(
        torch.is_tensor(x) and x.dim() == 4 and x.dtype == x0.dtype and x.device == x0.device
        and x.shape[0] == n and x.shape[2] == h and x.shape[3] == w and x.shape[1] > 0
        and x.is_contiguous(memory_format=torch.channels_last)
        for x in xs
    ) and (n * h * w > 1)


def cat(xs):
    """torch.cat(xs, 1), through the Triton kernel when every input is a channels_last XPU tensor of one dtype."""
    xs = list(xs)
    if supported(xs):
        return _CatNHWC.apply(*xs)
    return torch.cat(xs, 1)
