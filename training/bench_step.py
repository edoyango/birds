#!/usr/bin/env python
"""Time Ultralytics training steps on an Intel Arc GPU, outside Ultralytics' epoch loop.

The trainer is built exactly like train_xpu.py builds it (same cfg, dataset, XPUGradScaler), then steps are run by a
loop that mirrors BaseTrainer._do_train (autocast forward, loss, scaled backward, optimizer_step every `accumulate`
batches, the progress bar's per-step loss formatting).

Usage (key=value like train_xpu.py; bench keys below are not passed to Ultralytics):

    python training/bench_step.py cfg=training/cfg-yolo26.yaml data_root=../birds-3-57 model=yolo26s.pt mode=fixed

Bench keys:
    mode      fixed  cycle `nbatches` batches read once (no data loader in the timing; host->device copy still done)
              real   iterate the real train loader
              loader iterate the train loader only (no GPU), report images/s
              opt    time the pieces of optimizer_step for MuSGD and AdamW, and print MuSGD's Newton-Schulz batches
              parity loss/gradients on one batch: fp32 (twice), fp16, fp16+channels_last, fp32+channels_last, and
                     fp16+channels_last with fused_bn / half_glue / fast_topk / all three, against fp32
              lossparity  original vs sync-free loss (xpu_speedups) on the same predictions, `nbatches` batches
              optparity   original vs fast_musgd optimizer steps from identical states and gradients (incl. an inf
                          step that must be skipped): params and optimizer state compared bit for bit
    steps     timed steps (default 100), after `warmup` untimed steps (default 20)
    stages    true: synchronize after every stage and report per-stage times (slows the step a little)
    profile   directory: record a torch.profiler trace (CPU + XPU) of `steps` steps there, print the top kernels
    pbar      false: skip the per-step loss formatting that Ultralytics' progress bar does (a host sync)
    channels_last  true: model weights and input in NHWC (xpu_speedups; Ultralytics only does this on CUDA)
    sync_free true: xpu_speedups' host-sync-free detection loss
    fast_musgd true: xpu_speedups' MuSGD queued before GradScaler's inf check
    fused_bn  true: xpu_speedups' fused BatchNorm + SiLU Triton kernels (xpu_fused_bn)
    half_glue true: xpu_speedups' fp16 input image and fp16 nearest upsample
    fast_topk true: xpu_speedups' argmax-based top-k in the task-aligned assigner
    launch_env true: xpu_speedups.set_launch_env() (UR_L0_USE_IMMEDIATE_COMMANDLISTS=0; with fused_bn also
              UR_L0_COMMANDLISTS_CLEANUP_THRESHOLD=-1)
    out       write the summary as JSON here
Unset Ultralytics keys default to the 120-epoch recipe (epochs=120 -> optimizer=auto picks MuSGD).
"""
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import xpu_compat  # noqa: E402
import xpu_speedups  # noqa: E402
from train_xpu import make_data_yaml, parse_args  # noqa: E402

BENCH_DEFAULTS = {
    "mode": "fixed",
    "steps": 100,
    "warmup": 20,
    "nbatches": 8,
    "stages": False,
    "profile": None,
    "pbar": True,
    "channels_last": False,
    "sync_free": False,
    "fast_musgd": False,
    "fused_bn": False,
    "half_glue": False,
    "fast_topk": False,
    "launch_env": False,
    "out": None,
}


def sync():
    torch.xpu.synchronize()


def build_trainer(uargs):
    from ultralytics.models.yolo.detect import DetectionTrainer

    t = DetectionTrainer(overrides=uargs)
    t._setup_train()
    return t


class Stopwatch:
    """Per-stage wall times; with sync=True each stage is closed with a device synchronize."""

    def __init__(self, sync_stages):
        self.sync_stages = sync_stages
        self.times = {}
        self.t = None

    def start(self):
        if self.sync_stages:
            sync()
        self.t = time.perf_counter()

    def lap(self, name):
        if self.sync_stages:
            sync()
        now = time.perf_counter()
        self.times.setdefault(name, []).append((now - self.t) * 1e3)
        self.t = now


def train_step(t, batch, ni, sw, state, pbar):
    """One iteration of BaseTrainer._do_train's inner loop (without warmup lr ramps)."""
    from ultralytics.utils.torch_utils import autocast, unwrap_model

    rf = torch.profiler.record_function
    with autocast(torch.bfloat16 if t.args.amp == "bf16" else t.amp, device=t.device.type):
        with rf("bench/preprocess"):
            batch = t.preprocess_batch(batch)
        sw.lap("h2d+preprocess")
        with rf("bench/forward"):
            preds = t.model(batch["img"])
        sw.lap("forward")
        with rf("bench/loss"):
            loss, items = unwrap_model(t.model).loss(batch, preds)
            loss = loss.sum()
        sw.lap("loss")
    with rf("bench/backward"):
        t.scaler.scale(loss).backward()
    sw.lap("backward")
    if ni - state["last_opt"] >= t.accumulate:
        with rf("bench/optimizer_step"):
            t.optimizer_step()
        state["last_opt"] = ni
        sw.lap("optimizer_step")
    if pbar:  # BaseTrainer formats the running-mean losses every batch: one host sync per step
        i = state["i"]
        tl = state["tloss"]
        state["tloss"] = items if tl is None else {k: (tl[k] * i + v) / (i + 1) for k, v in items.items()}
        state["desc"] = ("%11.4g" * len(state["tloss"])) % tuple(state["tloss"].values())
        sw.lap("pbar")
    state["i"] += 1


def run_steps(t, batches, n, sw, state, pbar, data_wait):
    for _ in range(n):
        tw = time.perf_counter()
        batch = next(batches)
        data_wait.append((time.perf_counter() - tw) * 1e3)
        sw.start()
        train_step(t, batch, state["ni"], sw, state, pbar)
        state["ni"] += 1


def bench_train(t, b):
    imgs = t.batch_size
    if b["mode"] == "fixed":
        it = iter(t.train_loader)
        cached = [next(it) for _ in range(b["nbatches"])]
        t.train_loader.close()  # stop the workers so they do not compete for the CPU

        def cycle():
            k = 0
            while True:
                yield dict(cached[k % len(cached)])  # shallow copy: preprocess_batch replaces tensors in the dict
                k += 1

        batches = cycle()
    else:
        batches = iter(t.train_loader)

    state = {"ni": 0, "i": 0, "last_opt": -1, "tloss": None}
    sw = Stopwatch(b["stages"])
    wait = []
    tw0 = time.perf_counter()
    run_steps(t, batches, b["warmup"], sw, state, b["pbar"], wait)
    sync()
    warmup_s = time.perf_counter() - tw0
    sw.times.clear()
    wait.clear()
    torch.xpu.reset_peak_memory_stats()

    prof = None
    if b["profile"]:
        from torch.profiler import ProfilerActivity, profile

        prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU])
        prof.__enter__()
    t0 = time.perf_counter()
    run_steps(t, batches, b["steps"], sw, state, b["pbar"], wait)
    sync()
    wall = time.perf_counter() - t0
    if prof is not None:
        prof.__exit__(None, None, None)
        out = Path(b["profile"])
        out.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(out / "trace.json"))
        ka = prof.key_averages()
        for key in ("self_xpu_time_total", "self_device_time_total"):
            try:
                table = ka.table(sort_by=key, row_limit=60, max_name_column_width=90)
                break
            except Exception:
                continue
        (out / "top_kernels.txt").write_text(table)
        (out / "top_cpu.txt").write_text(ka.table(sort_by="self_cpu_time_total", row_limit=60, max_name_column_width=90))
        print(table)

    ms = wall / b["steps"] * 1e3
    res = {
        "mode": b["mode"],
        "steps": b["steps"],
        "batch": imgs,
        "accumulate": t.accumulate,
        "optimizer": type(t.optimizer).__name__,
        "ms_per_step": round(ms, 1),
        "warmup_s": round(warmup_s, 1),
        "img_per_s": round(imgs * 1e3 / ms, 1),
        "epoch_s_estimate": round(ms * len(t.train_loader) / 1e3, 1),
        "data_wait_ms_mean": round(statistics.mean(wait), 2),
        "peak_alloc_gb": round(torch.xpu.max_memory_allocated() / 2**30, 2),
        "peak_reserved_gb": round(torch.xpu.max_memory_reserved() / 2**30, 2),
        "loss_items": {k: float(v) for k, v in (state["tloss"] or {}).items()},
    }
    if sw.times:
        res["stages_ms_mean"] = {k: round(statistics.mean(v), 1) for k, v in sw.times.items()}
        # optimizer_step only runs every `accumulate` steps: report its per-call mean and amortized share
        if "optimizer_step" in sw.times:
            res["optimizer_step_ms_amortized"] = round(sum(sw.times["optimizer_step"]) / b["steps"], 1)
    return res


def newton_schulz_batches(opt):
    """Shapes of the batched Newton-Schulz calls muon_update makes (mirrors its bucketing)."""
    rows = []
    for g in opt.param_groups:
        if not g.get("use_muon"):
            continue
        buckets = {}
        for p in g["params"]:
            m = p.flatten(1) if p.ndim > 2 else p
            scale = max(1, p.size(-2) / p.size(-1)) ** 0.5
            buckets.setdefault((m.size(1), scale), []).append(m.size(0))
        for (cols, _), rs in buckets.items():
            rs.sort(reverse=True)
            start = 0
            for j in range(1, len(rs) + 1):
                if j == len(rs) or rs[start] > 16 * rs[j]:
                    part = rs[start:j]
                    n, r = len(part), part[0]
                    rows.append({"group_lr": g["lr"], "n": n, "rows": r, "cols": cols,
                                 "pad_frac": round(1 - sum(part) / (n * r), 3)})
                    start = j
    return rows


def bench_opt(t, b, uargs):
    """Time the pieces of optimizer_step with identical finite gradients for MuSGD, AdamW and fused AdamW.

    The GradScaler is bypassed (unscaled backward, optimizer.step() called directly): with a fresh scaler the first
    fp16 steps overflow and GradScaler skips optimizer.step(), which would time nothing. Each piece is timed twice:
    'sync' closes it with a device synchronize (GPU+CPU time), 'cpu' does not (host dispatch time only).
    """
    from ultralytics.utils.torch_utils import autocast, unwrap_model

    it = iter(t.train_loader)
    batch = next(it)
    t.train_loader.close()
    with autocast(t.amp, device=t.device.type):
        batch = t.preprocess_batch(batch)
        loss, _ = unwrap_model(t.model).loss(batch, t.model(batch["img"]))
    loss.sum().backward()
    params = [p for p in t.model.parameters() if p.requires_grad]
    grads = [p.grad.detach().clone() if p.grad is not None else None for p in params]
    finite = all(bool(torch.isfinite(g).all()) for g in grads if g is not None)
    print(f"params with grad: {sum(g is not None for g in grads)} / {len(params)}, "
          f"numel {sum(p.numel() for p in params) / 1e6:.2f}M, all finite: {finite}")

    decay = t.args.weight_decay * t.batch_size * t.accumulate / t.args.nbs
    results = {}
    for name in ("MuSGD", "AdamW", "AdamW-fused"):
        base = name.split("-")[0]
        lr = 0.01 if base == "MuSGD" else 0.000909
        opt = t.build_optimizer(t.model, name=base, lr=lr, momentum=0.9, decay=decay, iterations=1e5)
        if name.endswith("fused"):
            opt = torch.optim.AdamW(
                [{k: v for k, v in g.items() if k != "param_group"} for g in opt.param_groups], fused=True
            )
        if base == "MuSGD":
            results["newton_schulz_batches"] = newton_schulz_batches(opt)
        for mode in ("sync", "cpu"):
            times = {}
            for rep in range(b["warmup"] + 30):
                for p, g in zip(params, grads):
                    p.grad = None if g is None else g.clone()
                sync()
                laps = {}
                t0 = time.perf_counter()

                def lap(k):
                    nonlocal t0
                    if mode == "sync":
                        sync()
                    now = time.perf_counter()
                    laps[k] = (now - t0) * 1e3
                    t0 = now

                torch.nn.utils.clip_grad_norm_(t.model.parameters(), max_norm=10.0)
                lap("clip_grad_norm")
                opt.step()
                lap("optimizer.step")
                opt.zero_grad()
                lap("zero_grad")
                t.ema.update(t.model)
                lap("ema.update")
                sync()
                if rep >= b["warmup"]:
                    for k, v in laps.items():
                        times.setdefault(k, []).append(v)
            summary = {k: round(statistics.median(v), 2) for k, v in times.items()}
            summary["total"] = round(sum(summary.values()), 2)
            results[f"{name}/{mode}"] = summary
            print(f"{name}/{mode}", summary)
    print("newton-schulz batches:", results["newton_schulz_batches"])
    return results


def bench_parity(t, b):
    """Loss and gradients on one batch for several variants, compared with fp32 NCHW.

    The fp16 NCHW variant is today's training setup, so its error vs fp32 is the noise floor a change has to stay
    within. Backward uses a fixed loss scale of 1024 (no GradScaler) so fp16 gradients do not underflow.
    """
    from ultralytics.utils.torch_utils import autocast, unwrap_model

    it = iter(t.train_loader)
    batch_cpu = next(it)
    t.train_loader.close()
    model = t.model
    params = [p for p in model.parameters() if p.requires_grad]
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    scale = 1024.0

    def run(amp, cl, flags):
        saved = dict(xpu_speedups._flags)
        xpu_speedups.install(**{**saved, **flags})  # the _flags keys are install()'s arguments
        model.to(memory_format=torch.channels_last if cl else torch.contiguous_format)
        batch = dict(batch_cpu)
        try:
            with autocast(amp, device=t.device.type):
                batch = DetectionPre.preprocess(t, batch)
                if cl:
                    batch["img"] = batch["img"].contiguous(memory_format=torch.channels_last)
                loss, items = unwrap_model(model).loss(batch, model(batch["img"]))
            sc = scale
            while True:  # halve the loss scale until no fp16 gradient overflows (what GradScaler would do)
                model.zero_grad(set_to_none=True)
                (loss.sum() * sc).backward(retain_graph=True)
                g = [p.grad.detach().float().flatten() / sc for p in params]
                if all(bool(torch.isfinite(x).all()) for x in g) or sc < 1:
                    break
                sc /= 2
        finally:
            xpu_speedups.install(**saved)
        return loss.detach().float().cpu(), {k: float(v) for k, v in items.items()}, g, sc

    # name: (amp, channels_last, xpu_speedups flags for this run on top of the command line's)
    new = {"fused_bn": True, "half_glue": True, "fast_topk": True}
    variants = {"fp32": (False, False, {}), "fp32_again": (False, False, {}), "fp16": (True, False, {}),
                "fp16_cl": (True, True, {}), "fp32_cl": (False, True, {}),
                "fp16_cl_fbn": (True, True, {"fused_bn": True}), "fp16_cl_glue": (True, True, {"half_glue": True}),
                "fp16_cl_topk": (True, True, {"fast_topk": True}), "fp16_cl_new": (True, True, new)}
    out = {}
    res = {}
    for name, (amp, cl, flags) in variants.items():
        out[name] = run(amp, cl, flags)
    pairs = [("fp32_again", "fp32"), ("fp16", "fp32"), ("fp16_cl", "fp32"), ("fp16_cl", "fp16"), ("fp32_cl", "fp32"),
             ("fp16_cl_fbn", "fp32"), ("fp16_cl_glue", "fp32"), ("fp16_cl_topk", "fp32"), ("fp16_cl_new", "fp32"),
             ("fp16_cl_glue", "fp16_cl"), ("fp16_cl_topk", "fp16_cl")]
    for name, refname in pairs:
        if name not in out or refname not in out:
            continue
        loss, items, g, sc = out[name]
        rloss, _, rg, _ = out[refname]
        gnorm = statistics.median([r.norm().item() for r in rg])
        rel = [((a - r).norm() / (r.norm() + 1e-12)).item() for a, r in zip(g, rg)]
        cos = torch.nn.functional.cosine_similarity(torch.cat(g), torch.cat(rg), dim=0).item()
        worst = sorted(zip(rel, names, g, rg), key=lambda x: -x[0])[:3]
        key = f"{name}_vs_{refname}"
        res[key] = {
            "loss": items,
            "loss_scale": sc,
            "loss_rel_err": float(((loss - rloss).abs() / rloss.abs()).max()),
            "grad_rel_err_median": statistics.median(rel),
            "grad_rel_err_p90": sorted(rel)[int(0.9 * len(rel))],
            "grad_global_rel_err": (torch.cat(g) - torch.cat(rg)).norm().item() / torch.cat(rg).norm().item(),
            "grad_cosine_all": cos,
            "worst (name, rel err, |ref grad| / median |grad|)": [
                (n, round(r, 4), round(b.norm().item() / gnorm, 5)) for r, n, a, b in worst
            ],
        }
        print(key, json.dumps(res[key]))
    model.to(memory_format=torch.contiguous_format)
    return res


def _leafify(obj, leaves):
    """Copy a nested preds structure with every float tensor replaced by a detached leaf that requires grad."""
    if torch.is_tensor(obj):
        if obj.is_floating_point():
            leaf = obj.detach().clone().requires_grad_(True)
            leaves.append(leaf)
            return leaf
        return obj
    if isinstance(obj, dict):
        return {k: _leafify(v, leaves) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_leafify(v, leaves) for v in obj)
    return obj


def bench_lossparity(t, b):
    """Original vs sync-free loss on the same predictions for `nbatches` real batches: loss items and d loss/d preds."""
    from ultralytics.utils.torch_utils import autocast, unwrap_model

    model = unwrap_model(t.model)
    it = iter(t.train_loader)
    cl = xpu_speedups._flags["channels_last"]
    rows = []
    for k in range(b["nbatches"]):
        xpu_speedups.install(channels_last=cl, sync_free_loss=True)  # adds batch["n_max_boxes"]
        batch = t.preprocess_batch(next(it))
        with torch.no_grad(), autocast(t.amp, device=t.device.type):
            preds = t.model(batch["img"])
        got = {}
        for name, on in (("orig", False), ("sync_free", True), ("sync_free_again", True)):
            xpu_speedups.install(channels_last=cl, sync_free_loss=on)
            leaves = []
            p = _leafify(preds, leaves)
            with autocast(t.amp, device=t.device.type):
                loss, items = model.loss(batch, p)
            grads = torch.autograd.grad(loss.sum(), leaves, allow_unused=True)
            g = torch.cat([(x if x is not None else torch.zeros_like(l)).float().flatten() for x, l in zip(grads, leaves)])
            got[name] = (loss.detach().float(), {kk: float(v) for kk, v in items.items()}, g)
        ref_l, ref_i, ref_g = got["orig"]
        row = {"batch": k, "n_boxes": int(batch["batch_idx"].numel()), "n_max_boxes": batch.get("n_max_boxes"),
               "orig": ref_i, "orig_grad_finite": bool(torch.isfinite(ref_g).all())}
        for name in ("sync_free", "sync_free_again"):
            l, i, g = got[name]
            row[name] = {
                "loss_max_rel": float(((l - ref_l).abs() / ref_l.abs().clamp(min=1e-12)).max()),
                "items_max_rel": max(abs(i[kk] - ref_i[kk]) / max(abs(ref_i[kk]), 1e-12) for kk in ref_i),
                "grad_rel": float((g - ref_g).norm() / ref_g.norm().clamp(min=1e-30)),
                "grad_max_abs": float((g - ref_g).abs().max()),
                "grad_finite": bool(torch.isfinite(g).all()),
            }
        print(json.dumps(row))
        rows.append(row)
    t.train_loader.close()
    return {"rows": rows}


def bench_optparity(t, b):
    import copy

    from ultralytics.utils.torch_utils import autocast, unwrap_model
    from xpu_compat import XPUGradScaler

    model, opt = t.model, t.optimizer
    it = iter(t.train_loader)
    scale = 1.0  # fp16 grads of a fresh 7-class head overflow at larger scales, and every step would be skipped
    gradsets = []
    for _ in range(b["nbatches"]):
        batch = t.preprocess_batch(next(it))
        model.zero_grad(set_to_none=True)
        with autocast(t.amp, device=t.device.type):
            loss, _ = unwrap_model(model).loss(batch, model(batch["img"]))
        (loss.sum() * scale).backward()
        gradsets.append([p.grad.detach().clone() for p in model.parameters()])
    t.train_loader.close()
    model.zero_grad(set_to_none=True)
    finite = [all(bool(torch.isfinite(g).all()) for g in gs) for gs in gradsets]
    bad = [g.clone() for g in gradsets[0]]
    bad[7].view(-1)[0] = float("inf")
    gradsets.insert(2, bad)  # GradScaler must skip this step in both paths
    decay = t.args.weight_decay * t.batch_size * t.accumulate / t.args.nbs
    copies = {}
    for name in ("orig", "fast"):
        m = copy.deepcopy(model)
        o = t.build_optimizer(m, name="MuSGD", lr=0.01, momentum=0.9, decay=decay, iterations=1e5)
        copies[name] = (m, o)
    flags = dict(xpu_speedups._flags)
    for name, (m, o) in copies.items():
        xpu_speedups.install(**{**{k: flags[k] for k in ("channels_last", "sync_free_loss", "fast_nms")},
                                "fast_musgd": name == "fast"})
        sc = XPUGradScaler("xpu", init_scale=scale)
        sc.scale(torch.ones((), device=t.device))  # GradScaler allocates its scale lazily, on the first scale()
        for gs in gradsets:
            for p, g in zip(m.parameters(), gs):
                p.grad = g.clone()
            sc.unscale_(o)
            torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=10.0)
            sc.step(o)
            sc.update()
            o.zero_grad()
        sync()
    (ma, oa), (mb, ob) = copies["orig"], copies["fast"]
    pa, pb = list(ma.parameters()), list(mb.parameters())
    n_param_diff = sum(not torch.equal(x, y) for x, y in zip(pa, pb))
    sa = [v for p in pa for v in oa.state[p].values() if torch.is_tensor(v)]
    sb = [v for p in pb for v in ob.state[p].values() if torch.is_tensor(v)]
    n_state_diff = sum(not torch.equal(x, y) for x, y in zip(sa, sb))
    moved = sum(not torch.equal(x, y) for x, y in zip(pa, model.parameters()))
    res = {"optimizer": type(oa).__name__, "steps": len(gradsets), "skipped_steps_orig": None,
           "finite_gradsets": finite, "params": len(pa), "params_differing": n_param_diff,
           "state_tensors": len(sa), "state_tensors_differing": n_state_diff,
           "params_changed_vs_start": moved,
           "max_abs_param_diff": max(float((x - y).abs().max()) for x, y in zip(pa, pb))}
    print("OPTPARITY", json.dumps(res))
    return res


class DetectionPre:
    @staticmethod
    def preprocess(t, batch):  # the trainer's preprocess without bench-only layout changes
        from ultralytics.models.yolo.detect import DetectionTrainer

        return DetectionTrainer.preprocess_batch(t, batch)


def main():
    args = parse_args(sys.argv[1:])
    b = {k: args.pop(k, v) for k, v in BENCH_DEFAULTS.items()}
    cfg = args.pop("cfg", None)
    if cfg:
        import yaml

        base = yaml.safe_load(Path(cfg).read_text())
        base.update(args)
        args = base
    data_root = args.pop("data_root", None)
    names_yaml = args.pop("names_yaml", None)
    if data_root:
        args["data"] = make_data_yaml(data_root, names_yaml)
    args.setdefault("device", "xpu")
    args.setdefault("amp", True)
    args.setdefault("epochs", 120)
    args.update(val=False, plots=False, exist_ok=True, name=args.get("name", "bench"))
    if b["mode"] == "loader":
        args["device"] = "cpu"  # the loader does not touch the GPU; skip AMP checks and the GPU model
    elif str(args["device"]).startswith("xpu"):
        if b["launch_env"]:
            xpu_speedups.set_launch_env(deep_queue=bool(b["fused_bn"]))
        xpu_compat.install()
        xpu_speedups.install(channels_last=b["channels_last"], sync_free_loss=b["sync_free"], fast_nms=True,
                             fast_musgd=b["fast_musgd"], fused_bn=b["fused_bn"],
                             half_glue=b["half_glue"], fast_topk=b["fast_topk"])

    if b["mode"] == "loader":
        from ultralytics.models.yolo.detect import DetectionTrainer

        t = DetectionTrainer(overrides=args)
        t.args.workers = args.get("workers", 8)  # Ultralytics sets workers=0 for device=cpu
        t.model = torch.nn.Module()  # build_dataset only reads the model's stride
        t.model.stride = torch.tensor([8.0, 16.0, 32.0])
        loader = t.get_dataloader(t.data["train"], batch_size=t.batch_size, rank=-1, mode="train")
        it = iter(loader)
        for _ in range(b["warmup"]):
            next(it)
        t0 = time.perf_counter()
        for _ in range(b["steps"]):
            next(it)
        wall = time.perf_counter() - t0
        res = {"mode": "loader", "workers": loader.num_workers, "batch": t.batch_size,
               "img_per_s": round(t.batch_size * b["steps"] / wall, 1),
               "ms_per_batch": round(wall / b["steps"] * 1e3, 1)}
        loader.close()
    else:
        t = build_trainer(args)
        modes = {
            "opt": lambda: bench_opt(t, b, args),
            "parity": lambda: bench_parity(t, b),
            "lossparity": lambda: bench_lossparity(t, b),
            "optparity": lambda: bench_optparity(t, b),
        }
        res = modes.get(b["mode"], lambda: bench_train(t, b))()
    res.update({k: b[k] for k in ("channels_last", "sync_free", "fast_musgd", "fused_bn", "half_glue", "fast_topk",
                                  "launch_env")})
    res["env"] = {k: v for k, v in os.environ.items() if k.startswith(("UR_", "SYCL", "ZE_", "TORCHINDUCTOR"))}
    res["args"] = {k: args.get(k) for k in ("model", "imgsz", "batch", "workers", "amp", "compile", "optimizer")}
    print("RESULT", json.dumps(res))
    if b["out"]:
        Path(b["out"]).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
