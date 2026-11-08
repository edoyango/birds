"""Training-speed patches for Ultralytics on Intel Arc GPUs (XPU). Measured with training/bench_step.py.

What limits a YOLO26 training step on the A770 is mostly host-side work, not GPU math: every kernel launch costs
CPU time, and every host sync leaves the GPU idle while Python catches up. The patches:

launch_env   UR_L0_USE_IMMEDIATE_COMMANDLISTS=0. With the default immediate command lists, each kernel launch
             costs ~22 us of CPU when the GPU is idle and ~157 us while it is busy; batched command lists cost
             ~5-36 us. Bit-identical results. Must be set before the first XPU call (set_launch_env()).
             UR_L0_COMMANDLISTS_CLEANUP_THRESHOLD=-1 (set_launch_env(deep_queue=True), with fused_bn): in batched
             mode every launch otherwise scans the command lists still in flight once there are more than 20, so a
             launch costs more the further the CPU runs ahead of the GPU (~10 -> ~100 us), which stops the CPU from
             building the lead it needs to cover the loss's many small ops. Completed lists are still recycled at
             the per-step .item() syncs (600 steps: flat step time and memory).
channels_last  NHWC weights and input on XPU (Ultralytics enables it only on CUDA). oneDNN then skips the
             NCHW<->blocked reorders around every convolution and BatchNorm uses its channels-last kernels.
             Same math, different summation order (fp32 grads within 1e-4 of NCHW).
sync_free_loss  Removes the 16 host syncs per step in the YOLO26 detection loss (8 per head):
             - make_anchors: strides read back from the GPU (3 syncs) -> cached Python floats + cached anchors
             - target preprocessing: unique() + counts.max() (3 syncs) -> max boxes per image counted on the
               CPU before the batch moves to the GPU
             - TaskAlignedAssigner.get_box_metrics: mask.nonzero() -> the same CIoU/score math on the dense
               (batch, boxes, anchors) grid, masked with torch.where (bit-identical)
             - BboxLoss: fg_mask.nonzero() -> dense over all anchors, off-foreground anchors replaced by a constant
               box with weight 0 (same terms, different summation order); L1 head (reg_max=1, YOLO26) only,
               DFL heads fall back to the original
             - host data turned into device tensors in the loss/assigner (torch.tensor(shape), a Python-list index,
               torch.tensor(stride): 6 blocking host->device copies per step, each a full drain of the GPU queue)
               -> cached device constants
fast_nms     Validation NMS on XPU through torchvision's XPU kernel. Ultralytics sends XPU boxes through TorchNMS,
             a Python loop (~80 ms/image for YOLO26 here) that trips its time limit and silently drops detections:
             the final eval of an 11-epoch run reported mAP50-95 0.408 for weights that score 0.644. Same keep
             indices as TorchNMS (checked on 30 real validation batches); ~3 ms/image.
fast_musgd   MuSGD (optimizer=auto for long runs) queues ~1150 small device ops per step (~75 ms of CPU) only after
             GradScaler's inf check, a host sync that waits for the whole backward pass, so the GPU idles while they
             are queued. With this patch the momentum + Newton-Schulz part is queued before that check (into
             temporaries, by Ultralytics' own muon_update) and only applied after it, or dropped if the step is
             skipped. Same ops on the same values: bit-identical params and buffers.
             (XPU graphs would remove the CPU cost entirely but the A770 driver has no graph record/replay.)
fused_bn     Ultralytics Conv's BatchNorm + SiLU in training mode as Triton kernels (xpu_fused_bn.py): 8 passes over
             the activation instead of 13, BN output not stored. fp32 inside, so not bit-identical (closer to fp32).
             Eval/validation/export use the stock modules.
half_glue    fp16 where autocast needlessly promoted: the input image (uint8 -> fp16 channels_last in one kernel) and
             nearest upsampling (+ the Concat after it). Forward bit-identical.
fast_topk    TaskAlignedAssigner's topk (3-15 ms per call on XPU) as k argmax passes; same set except among ties.
fast_val     Validation NMS + metrics on the host. The stock per-image loop makes ~12k tiny device->host copies per
             validation pass (each a queue drain), and per-image XPU NMS costs ~1.7 ms vs ~0.2 ms on the CPU for the
             ~20 candidate boxes a trained model leaves at conf=0.001. Now: one copy of the raw predictions per
             batch, Ultralytics' own NMS and metric code on host tensors (images with > NMS_HOST_MAX boxes, e.g.
             early in training, go back to the XPU kernel). Identical NMS output and metrics; 15.2 -> 6.6 s per
             validation of the 2164-image val set.
fast_cast    All conv weights cast to fp16 in one multi-tensor kernel per forward (and their fp16 grads back to
             fp32 in one), instead of autocast's ~140 + ~140 per-weight cast kernels. Bit-identical casts.
fast_cat     C2f/C3k2, C3, SPPF, C2PSA and Concat: the split after cv1 is written by the fused BN kernel as two
             contiguous tensors (no channel-slice views, which the next conv had to copy), and concatenations go
             through xpu_cat.py (Triton, 2-4x faster than torch.cat on NHWC, and its backward hands each input a
             contiguous gradient instead of a slice that consumers copy or add with a slow strided kernel). Same
             values; gradients move by ~1e-3 relative (different kernels sum in a different order), 200x below the
             fp16-vs-fp32 difference, which is unchanged (0.2094 vs 0.2095).
fast_attn    Attention (C2PSA / PSABlock) as out^T = attn @ v^T with an fp16 softmax and a channels_last output:
             no transposed copy of the (B, heads, N, N) gradient in softmax's backward, no fp32 copy of the scores.
             Forward bit-identical; 8.2 -> 4.7 ms per attention block (fwd + bwd).

Call set_launch_env() before any XPU work, then install(...) before the trainer is built. Written against
Ultralytics 8.4.204 / torch 2.15; the patched functions fall back to the originals for shapes they do not cover.
"""
import os
import weakref

import torch

LAUNCH_ENV = {"UR_L0_USE_IMMEDIATE_COMMANDLISTS": "0"}
DEEP_QUEUE_ENV = {"UR_L0_COMMANDLISTS_CLEANUP_THRESHOLD": "-1"}  # see launch_env above; only with fused_bn

_orig = {}
_flags = {"channels_last": False, "sync_free_loss": False, "fast_nms": False, "fast_musgd": False, "fused_bn": False,
          "half_glue": False, "fast_topk": False, "fast_val": False, "fast_cast": False,
          "fast_cat": False, "fast_attn": False}


def set_launch_env(deep_queue=True):
    """Set Level Zero launch settings; they only take effect if the XPU runtime is not initialized yet.

    deep_queue: also UR_L0_COMMANDLISTS_CLEANUP_THRESHOLD=-1. Measured with fused_bn on (yolo26s, b32: 348 -> 332
    ms/step, stable over 600 steps without a queue synchronize, no host or device memory growth). With stock
    BatchNorm the same setting made steps slower and erratic (434 -> 444-510 ms), so pass fused_bn's value here.
    """
    env = {**LAUNCH_ENV, **(DEEP_QUEUE_ENV if deep_queue else {})}
    for k, v in env.items():
        os.environ.setdefault(k, v)
    if torch.xpu.is_initialized():
        print(f"xpu_speedups: WARNING XPU already initialized, {list(env)} may not take effect")


# ---------------------------------------------------------------- channels_last + host-side box counts (trainer)


def _uses_channels_last(trainer):
    return _flags["channels_last"] and trainer.device.type == "xpu" and trainer.args.channels_last is not False


def _set_model_attributes(self):
    _orig["set_model_attributes"](self)
    if _uses_channels_last(self):
        self.model = self.model.to(memory_format=torch.channels_last)


def _preprocess_batch(self, batch):
    n_max = None
    if _flags["sync_free_loss"]:
        bi = batch.get("batch_idx")
        if torch.is_tensor(bi) and bi.device.type == "cpu":
            n_max = int(torch.bincount(bi.long()).max()) if bi.numel() else 0
    img = batch.get("img")
    if (_flags["half_glue"] and _uses_channels_last(self) and self.amp and self.args.amp in (True, "fp16")
            and self.args.multi_scale <= 0 and torch.is_tensor(img) and img.dtype == torch.uint8 and img.dim() == 4):
        # What the first conv sees is autocast's fp16 rounding of float(img) / 255: compute exactly that, in one
        # kernel, straight into a channels_last fp16 tensor (the stock path is uint8 -> fp32 -> /255 -> NHWC -> fp16)
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(self.device, non_blocking=True)
        u8 = batch["img"]
        out = torch.empty(u8.shape, dtype=torch.float16, device=u8.device, memory_format=torch.channels_last)
        batch["img"] = torch.div(u8, 255, out=out)
    else:
        batch = _orig["preprocess_batch"](self, batch)
        if _uses_channels_last(self):
            batch["img"] = batch["img"].contiguous(memory_format=torch.channels_last)
    if n_max is not None:
        batch["n_max_boxes"] = n_max
    return batch


# ---------------------------------------------------------------- sync-free detection loss

_anchor_cache = {}


def _make_anchors(feats, strides, grid_cell_offset=0.5):
    from ultralytics.utils import tal

    if not (isinstance(feats, list) and feats and torch.is_tensor(feats[0])):
        return tal.make_anchors(feats, strides, grid_cell_offset)
    if torch.is_tensor(strides):
        sv = getattr(strides, "_xpu_speedups_values", None)
        if sv is None:  # one sync per stride tensor, ever
            sv = tuple(float(s) for s in strides.tolist())
            strides._xpu_speedups_values = sv
    else:
        sv = tuple(float(s) for s in strides)
    key = (tuple(tuple(f.shape[2:]) for f in feats), sv, grid_cell_offset, feats[0].dtype, feats[0].device)
    hit = _anchor_cache.get(key)
    if hit is None:
        if len(_anchor_cache) >= 64:  # e.g. multi_scale training: a new image size every batch
            _anchor_cache.clear()
        hit = _anchor_cache[key] = tal.make_anchors(feats, list(sv), grid_cell_offset)
    return hit[0].clone(), hit[1].clone()  # callers own their copies, as with a fresh make_anchors call


_imgsz_cache = {}


def _imgsz_consts(self, feats, dtype):
    """(imgsz, imgsz[[1, 0, 1, 0]]) as cached device tensors. Ultralytics builds them every call from host data
    (torch.tensor(shape) and a Python-list index): two blocking host->device copies per loss head per step."""
    key = (tuple(feats[0].shape[2:]), dtype, str(self.device), self.stride.data_ptr())
    c = _imgsz_cache.get(key)
    if c is None:
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        c = _imgsz_cache[key] = (imgsz, imgsz[[1, 0, 1, 0]])
    return c


def _get_assigned_targets_and_loss(self, preds, batch, **assign_kwargs):
    """v8DetectionLoss.get_assigned_targets_and_loss (Ultralytics 8.4.204) with cached imgsz constants; it also hands
    the host-side box count (batch["n_max_boxes"], see _preprocess_batch) to _preprocess_targets."""
    from ultralytics.utils import loss as L

    self._n_max_boxes = batch.get("n_max_boxes") if isinstance(batch, dict) else None
    try:
        if type(self) is not L.v8DetectionLoss or self.device.type != "xpu":
            return _orig["get_assigned_targets_and_loss"](self, preds, batch, **assign_kwargs)
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = L.make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz, scale_tensor = _imgsz_consts(self, preds["feats"], dtype)

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=scale_tensor)
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
            **assign_kwargs,
        )

        target_scores_sum = target_scores.sum().clamp_(min=1)

        bce_loss = self.bce(pred_scores, target_scores.to(dtype))  # (bs, num_anchors, nc)
        if self.class_weights is not None:
            bce_loss *= self.class_weights
        loss[1] = bce_loss.sum() / target_scores_sum  # BCE

        loss[0], loss[2] = self.bbox_loss(
            pred_distri,
            pred_bboxes,
            anchor_points,
            target_bboxes / stride_tensor,
            target_scores,
            target_scores_sum,
            fg_mask,
            imgsz,
            stride_tensor,
        )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            dict(zip(self.loss_names, loss.detach())),
        )
    finally:
        self._n_max_boxes = None



class _ConstTensorTorch:
    """Stands in for the `torch` module inside ultralytics.utils.tal (every attribute is torch's), except that
    torch.tensor(<constant>, device=<xpu>, [dtype=]) of the given Python types returns a cached device tensor.

    A torch.tensor(..., device=xpu) of host data is a blocking copy queued behind all pending GPU work: the host waits
    for the GPU to drain, then the GPU idles through the loss's host-heavy small ops. Reviewed call sites (Ultralytics
    8.4.204), both read-only uses of the new tensor: tal.py `torch.where(mask, torch.tensor(self.stride_val, ...), x)`
    (349, 518). The loss's own constants are handled in _get_assigned_targets_and_loss.
    """

    def __init__(self, cache_types):
        self._types = cache_types
        self._cache = {}

    def __getattr__(self, name):
        return getattr(torch, name)

    def tensor(self, data, *args, **kw):
        dev = kw.get("device")
        if (_flags["sync_free_loss"] and not args and type(data) in self._types and dev is not None
                and set(kw) <= {"device", "dtype"} and torch.device(dev).type == "xpu"):
            key = (type(data), tuple(data) if isinstance(data, torch.Size) else data, kw.get("dtype"), str(dev))
            t = self._cache.get(key)
            if t is None:
                t = self._cache[key] = torch.tensor(data, **kw)
            return t
        return torch.tensor(data, *args, **kw)


def _preprocess_targets(self, targets, batch_size, scale_tensor):
    from ultralytics.utils.ops import xywh2xyxy

    n = getattr(self, "_n_max_boxes", None)
    nl, ne = targets.shape
    if n is None or nl == 0 or self.device.type == "mps":
        return _orig["preprocess"](self, targets, batch_size, scale_tensor)
    # Ultralytics' code minus unique()/counts.max(): n comes from the CPU-side batch_idx
    batch_idx = targets[:, 0].long()
    out = torch.zeros(batch_size, n, ne - 1, device=self.device)
    offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=self.device)
    offsets.scatter_add_(0, batch_idx + 1, torch.ones_like(batch_idx))
    offsets = offsets.cumsum(0)
    within_idx = torch.arange(nl, device=self.device) - offsets[batch_idx]
    out[batch_idx, within_idx] = targets[:, 1:]
    out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
    return out


def _get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt, **extra):
    from ultralytics.utils.tal import TaskAlignedAssigner

    cls = type(self)
    if extra or cls.pair_overlaps is not TaskAlignedAssigner.pair_overlaps or (
        cls.iou_calculation is not TaskAlignedAssigner.iou_calculation
    ):
        return _orig["get_box_metrics"](self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt, **extra)
    mask = mask_gt.bool()  # (b, n_max_boxes, anchors)
    labels = gt_labels[..., 0].long()  # (b, n_max_boxes)
    na = pd_scores.shape[1]
    scores = pd_scores.permute(0, 2, 1).gather(1, labels.unsqueeze(-1).expand(-1, -1, na))  # (b, n, anchors)
    ov = self.iou_calculation(gt_bboxes.unsqueeze(2), pd_bboxes.unsqueeze(1))  # (b, n, anchors)
    align = scores.pow(self.alpha) * ov.pow(self.beta)
    overlaps = torch.where(mask, ov, 0).to(pd_bboxes.dtype)
    align_metric = torch.where(mask, align, 0)
    return align_metric, overlaps


def _bbox_loss_forward(
    self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask, imgsz,
    stride,
):
    import torch.nn.functional as F

    from ultralytics.utils.metrics import bbox_iou
    from ultralytics.utils.tal import bbox2dist

    if self.dfl_loss is not None:
        return _orig["bbox_forward"](
            self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask,
            imgsz, stride,
        )
    fg = fg_mask.bool()
    fg4 = fg.unsqueeze(-1)
    weight = target_scores.sum(-1) * fg  # (b, anchors), zero off the foreground
    # Off the foreground both boxes become a constant unit box, so torch.where sends no gradient there. Weight 0
    # alone is not enough: fp16 predictions can have zero height (h + 1e-7 underflows), and the CIoU aspect term's
    # derivative is then 0 * inf = NaN.
    unit = torch.zeros_like(target_bboxes)
    unit[..., 2:] = 1.0
    pb = torch.where(fg4, pred_bboxes, unit.to(pred_bboxes.dtype))
    tb = torch.where(fg4, target_bboxes, unit)
    iou = bbox_iou(pb, tb, xywh=False, CIoU=True).squeeze(-1)
    loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
    target_ltrb = bbox2dist(anchor_points, tb)
    target_ltrb = target_ltrb * stride
    target_ltrb[..., 0::2] /= imgsz[1]
    target_ltrb[..., 1::2] /= imgsz[0]
    pred_dist = pred_dist * stride
    pred_dist[..., 0::2] /= imgsz[1]
    pred_dist[..., 1::2] /= imgsz[0]
    loss_l1 = (F.l1_loss(pred_dist, target_ltrb, reduction="none").mean(-1) * weight).sum() / target_scores_sum
    return loss_iou, loss_l1


# ---------------------------------------------------------------- validation NMS


NMS_HOST_MAX = 2000  # boxes per image above which host NMS hands the boxes to the XPU kernel (see fast_val)


def _nms_boxes(boxes, scores, iou, rotated=False):
    if _flags["fast_nms"] and not rotated and boxes.device.type == "xpu":
        import torchvision

        return torchvision.ops.nms(boxes, scores, iou)
    if (_flags["fast_val"] and not rotated and boxes.device.type == "cpu" and boxes.shape[0] > NMS_HOST_MAX
            and torch.xpu.is_initialized()):
        import torchvision

        return torchvision.ops.nms(boxes.to("xpu"), scores.to("xpu"), iou).cpu()
    return _orig["nms_boxes"](boxes, scores, iou, rotated)


# ---------------------------------------------------------------- validation: NMS + metrics on the host


def _val_preprocess(self, batch):
    """DetectionValidator.preprocess; with fast_val, keeps the loader's host copies of the targets for
    update_metrics (they are moved to the GPU for the validation loss as before)."""
    if _flags["fast_val"] and self.device.type == "xpu" and not self.args.tile:
        self._xpu_host_targets = {k: batch[k] for k in ("batch_idx", "cls", "bboxes")
                                  if torch.is_tensor(batch.get(k)) and batch[k].device.type == "cpu"}
    return _orig["val_preprocess"](self, batch)


def _val_postprocess(self, preds):
    """DetectionValidator.postprocess (NMS) on a host copy of the raw predictions (one device->host copy per batch)."""
    if _flags["fast_val"] and self.device.type == "xpu" and not self.args.tile:
        p = preds[0] if isinstance(preds, (list, tuple)) else preds
        if torch.is_tensor(p) and p.device.type == "xpu":
            preds = p.cpu()
    return _orig["val_postprocess"](self, preds)


def _val_update_metrics(self, preds, batch):
    """DetectionValidator.update_metrics on host tensors: the stock per-image loop then makes no device syncs."""
    host = self.__dict__.pop("_xpu_host_targets", None)
    if (host is not None and len(host) == 3 and self.device.type == "xpu"
            and all(torch.is_tensor(v) and v.device.type == "cpu" for p in preds for v in p.values())):
        dev = self.device
        self.device = torch.device("cpu")  # _prepare_batch builds its scale tensor on self.device
        try:
            return _orig["val_update_metrics"](self, preds, {**batch, **host})
        finally:
            self.device = dev
    return _orig["val_update_metrics"](self, preds, batch)


# ---------------------------------------------------------------- MuSGD: queue Newton-Schulz before the inf check


@torch.no_grad()
def _musgd_prepare(opt):
    """Queue MuSGD's muon part (momentum + orthogonalization) into temporaries; nothing in opt is modified."""
    from ultralytics.optim.muon import muon_update

    prep = []
    for group in opt.param_groups:
        params = [p for p in group["params"] if p.grad is not None]
        if not group["use_muon"] or not params:
            prep.append(None)
            continue
        for p in params:  # the lazy state init of MuSGD.step
            if len(opt.state[p]) == 0:
                opt.state[p]["momentum_buffer"] = torch.zeros_like(p)
                opt.state[p]["momentum_buffer_SGD"] = torch.zeros_like(p)
        moms = list(torch._foreach_mul([opt.state[p]["momentum_buffer"] for p in params], 1.0))  # exact copies
        updates = muon_update([p.grad for p in params], moms, beta=group["momentum"], nesterov=group["nesterov"])
        prep.append((params, moms, updates))
    opt._xpu_prepared = prep


@torch.no_grad()
def _musgd_step(self, closure=None):
    """MuSGD.step, applying the updates _musgd_prepare queued (if any). Otherwise the original step."""
    prep = self.__dict__.pop("_xpu_prepared", None)
    if prep is None or closure is not None:
        return _orig["musgd_step"](self, closure)
    for group, pg in zip(self.param_groups, prep):
        params = [p for p in group["params"] if p.grad is not None]
        if not params:
            continue
        lr, momentum, nesterov = group["lr"], group["momentum"], group["nesterov"]
        for p in params:
            if len(self.state[p]) == 0:
                self.state[p]["momentum_buffer"] = torch.zeros_like(p)
                if group["use_muon"]:
                    self.state[p]["momentum_buffer_SGD"] = torch.zeros_like(p)
        if group["use_muon"]:
            pparams, moms, updates = pg
            if len(pparams) != len(params) or any(a is not b for a, b in zip(pparams, params)):
                raise RuntimeError("xpu_speedups: MuSGD params changed between prepare and step")
            torch._foreach_copy_([self.state[p]["momentum_buffer"] for p in params], moms)
            torch._foreach_add_(params, updates, alpha=-(lr * self.muon))
            buffers = [self.state[p]["momentum_buffer_SGD"] for p in params]
            lr *= self.sgd
        else:
            buffers = [self.state[p]["momentum_buffer"] for p in params]
        # SGD update, as in MuSGD.step
        grads = [p.grad for p in params]
        if group["weight_decay"] != 0:
            grads = torch._foreach_add(grads, params, alpha=group["weight_decay"])
        torch._foreach_mul_(buffers, momentum)
        torch._foreach_add_(buffers, grads)
        updates = torch._foreach_add(grads, buffers, alpha=momentum) if nesterov else buffers
        torch._foreach_add_(params, updates, alpha=-lr)
    return None


def _scaler_step(self, optimizer, *args, **kwargs):
    from torch.amp.grad_scaler import OptState

    from ultralytics.optim import MuSGD

    if (
        _flags["fast_musgd"]
        and self._enabled
        and isinstance(optimizer, MuSGD)
        and not args
        and not kwargs
        and self._per_optimizer_states[id(optimizer)]["stage"] is OptState.UNSCALED  # grads final (unscaled, clipped)
    ):
        _musgd_prepare(optimizer)
    try:
        return _orig["scaler_step"](self, optimizer, *args, **kwargs)  # host sync on found_inf, then optimizer.step()
    finally:
        optimizer.__dict__.pop("_xpu_prepared", None)  # dropped if the step was skipped


# ---------------------------------------------------------------- fp16 nearest upsample


def _upsample_forward(self, x):
    """nn.Upsample; nearest mode on an fp16 input under autocast runs in fp16 (autocast would run it in fp32).

    Nearest upsampling only copies values, so the output is the same numbers; keeping fp16 also keeps the Concat that
    follows (and its backward) in fp16 instead of promoting the whole concatenation to fp32 and casting it back.
    """
    if (_flags["half_glue"] and self.mode == "nearest" and x.dtype == torch.float16
            and torch.is_autocast_enabled(x.device.type)):
        with torch.autocast(x.device.type, enabled=False):
            return _orig["upsample_forward"](self, x)
    return _orig["upsample_forward"](self, x)


# ---------------------------------------------------------------- fused BatchNorm + SiLU (Triton)


def _conv_forward(self, x):
    """Ultralytics Conv.forward with the BN + act pair fused in training (xpu_fused_bn); unchanged otherwise."""
    y = self.conv(x)
    if _flags["fused_bn"]:
        import xpu_fused_bn

        if xpu_fused_bn.supported(y, self.bn, self.act):
            return xpu_fused_bn.bn_act(y, self.bn, self.act)
    return self.act(self.bn(y))


# ---------------------------------------------------------------- fp16 conv weights in one multi-tensor cast


class _CastHalf(torch.autograd.Function):
    """fp32 tensors -> fp16 copies in one multi-tensor kernel; backward casts the fp16 grads back the same way.

    Autocast does the same casts (and ToCopyBackward their grads) one kernel per weight: ~280 tiny kernels per step.
    Both directions are exact roundings/widenings, so results are bit-identical.
    """

    @staticmethod
    def forward(ctx, *ws):
        ctx.set_materialize_grads(False)
        out = [torch.empty_like(w, dtype=torch.float16) for w in ws]
        torch._foreach_copy_(out, list(ws))
        return tuple(out)

    @staticmethod
    def backward(ctx, *gs):
        idx = [i for i, g in enumerate(gs) if g is not None]
        res = [None] * len(gs)
        if idx:
            src = [gs[i] for i in idx]
            dst = [torch.empty_like(g, dtype=torch.float32) for g in src]
            torch._foreach_copy_(dst, src)
            for i, d in zip(idx, dst):
                res[i] = d
        return tuple(res)


_cast_conv_cache = weakref.WeakKeyDictionary()  # model -> its nn.Conv2d modules (kept out of the model's pickle)


def _cast_convs(model):
    """Modules whose weights get pre-cast: every nn.Conv2d with fp32 params (cached per model)."""
    convs = _cast_conv_cache.get(model)
    if convs is None:
        convs = [m for m in model.modules() if type(m) is torch.nn.Conv2d and m.weight.dtype == torch.float32
                 and (m.bias is None or m.bias.dtype == torch.float32)]
        _cast_conv_cache[model] = convs
    return convs


def _predict_once(self, x, *args, **kwargs):
    """BaseModel._predict_once; with fast_cast, under fp16 autocast with grad on, first casts all conv weights (and
    biases) to fp16 in one _CastHalf call and hands each conv its copy (autocast then has nothing to cast)."""
    if not (_flags["fast_cast"] and self.training and torch.is_grad_enabled() and x.device.type == "xpu"
            and torch.is_autocast_enabled("xpu") and torch.get_autocast_dtype("xpu") == torch.float16):
        return _orig["predict_once"](self, x, *args, **kwargs)
    convs = _cast_convs(self)
    ws = []
    for m in convs:
        ws.append(m.weight)
        if m.bias is not None:
            ws.append(m.bias)
    half = iter(_CastHalf.apply(*ws))
    for m in convs:
        m.__dict__["_xpu_half"] = (next(half), next(half) if m.bias is not None else None)
    try:
        return _orig["predict_once"](self, x, *args, **kwargs)
    finally:
        for m in convs:
            m.__dict__.pop("_xpu_half", None)


def _conv2d_forward(self, input):
    """nn.Conv2d.forward, using the fp16 weights _predict_once prepared if there are any."""
    half = self.__dict__.get("_xpu_half")
    if half is not None:
        return self._conv_forward(input, half[0], half[1])
    return _orig["conv2d_forward"](self, input)


# ---------------------------------------------------------------- channel split / concat without strided copies


def _cat(xs):
    if _flags["fast_cat"]:
        import xpu_cat

        return xpu_cat.cat(xs)
    return torch.cat(list(xs), 1)


def _conv_split2(cv, x, c):
    """cv(x).split((c, c), 1) for an Ultralytics Conv; with fused_bn in training the BN kernel writes the two halves
    as separate contiguous tensors (same values) instead of returning channel-slice views."""
    from ultralytics.nn.modules import conv as CM

    if _flags["fused_bn"] and type(cv) is CM.Conv and hasattr(cv, "bn"):
        import xpu_fused_bn

        y = cv.conv(x)
        if xpu_fused_bn.supported(y, cv.bn, cv.act) and y.shape[1] == 2 * c:
            return xpu_fused_bn.bn_act(y, cv.bn, cv.act, split=c)
        return cv.act(cv.bn(y)).split((c, c), 1)
    return cv(x).split((c, c), 1)


def _fast_cat_on(x):
    return _flags["fast_cat"] and torch.is_tensor(x) and x.device.type == "xpu"


def _c2f_forward(self, x):
    """C2f.forward (also C3k2): halves from the BN kernel, concatenation by xpu_cat."""
    if not _fast_cat_on(x):
        return _orig["c2f_forward"](self, x)
    y = list(_conv_split2(self.cv1, x, self.c))
    y.extend(m(y[-1]) for m in self.m)
    return self.cv2(_cat(y))


def _c3_forward(self, x):
    if not _fast_cat_on(x):
        return _orig["c3_forward"](self, x)
    return self.cv3(_cat((self.m(self.cv1(x)), self.cv2(x))))


def _sppf_forward(self, x):
    if not _fast_cat_on(x):
        return _orig["sppf_forward"](self, x)
    y = [self.cv1(x)]
    y.extend(self.m(y[-1]) for _ in range(getattr(self, "n", 3)))
    y = self.cv2(_cat(y))
    return y + x if getattr(self, "add", False) else y


def _c2psa_forward(self, x):
    if not _fast_cat_on(x):
        return _orig["c2psa_forward"](self, x)
    a, b = _conv_split2(self.cv1, x, self.c)
    b = self.m(b)
    return self.cv2(_cat((a, b)))


def _concat_forward(self, x):
    if self.d == 1 and isinstance(x, (list, tuple)) and x and _fast_cat_on(x[0]):
        return _cat(x)
    return _orig["concat_forward"](self, x)


# ---------------------------------------------------------------- attention: no transposed copies, fp16 softmax


def _attention_forward(self, x):
    """Ultralytics Attention.forward with the same math in a layout the XPU kernels like (fp16 autocast only):

    - out^T = attn @ v^T instead of out = v @ attn^T, so softmax's backward gets a contiguous gradient (the stock
      order hands it the transpose of a (B, heads, N, N) matrix, which it copies: ~1.7 ms per call at N=484);
    - softmax on the fp16 scores, fp32 inside (autocast would cast the scores to an fp32 copy, softmax that, and the
      next matmul would cast the result back to fp16: the same numbers);
    - the output is written channels_last, so the add with pe and the proj conv need no format conversion.
    Forward output bit-identical to the stock order on a yolo26s-shaped test; gradient error vs fp32 unchanged.
    """
    if not (_flags["fast_attn"] and x.device.type == "xpu" and self.format is None
            and torch.is_autocast_enabled("xpu") and torch.get_autocast_dtype("xpu") == torch.float16):
        return _orig["attention_forward"](self, x)
    B, C, H, W = x.shape
    N = H * W
    qkv = self.qkv(x)
    q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
        [self.key_dim, self.key_dim, self.head_dim], dim=2
    )
    s = (q * self.scale).transpose(-2, -1) @ k
    if s.dtype == torch.float16:
        with torch.autocast("xpu", enabled=False):
            attn = s.softmax(dim=-1)
    else:
        attn = s.softmax(dim=-1)
    o = attn @ v.transpose(-2, -1)  # (B, heads, N, head_dim)
    o = o.transpose(1, 2).reshape(B, H, W, C).permute(0, 3, 1, 2)  # channels_last (B, C, H, W)
    return self.proj(o + self.pe(v.reshape(B, C, H, W)))


# ---------------------------------------------------------------- task-aligned assigner top-k


def _topk_idx(x, k):
    """Indices of the k largest values along the last dim (same set as torch.topk up to ties, which topk leaves
    unspecified; here ties go to the lowest index). The XPU topk kernel takes 3-15 ms on the assigner's
    (batch, boxes, 10164) metric; k argmax passes take 0.1-3 ms."""
    if k == 1:
        return x.argmax(-1, keepdim=True)
    x = x.clone()
    idx = []
    for _ in range(k):
        i = x.argmax(-1, keepdim=True)
        idx.append(i)
        x.scatter_(-1, i, float("-inf"))
    return torch.cat(idx, -1)


def _select_topk_candidates(self, metrics, topk_mask=None):
    if topk_mask is None or metrics.device.type != "xpu":
        return _orig["select_topk_candidates"](self, metrics, topk_mask)
    topk_idxs = _topk_idx(metrics, self.topk)
    topk_idxs.masked_fill_(~topk_mask, 0)
    count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
    count_tensor.scatter_add_(-1, topk_idxs, torch.ones_like(topk_idxs, dtype=torch.int8))
    count_tensor.masked_fill_(count_tensor > 1, 0)
    return count_tensor


def _select_highest_overlaps(self, mask_pos, overlaps, n_max_boxes, align_metric):
    if self.topk2 == self.topk or mask_pos.device.type != "xpu":
        return _orig["select_highest_overlaps"](self, mask_pos, overlaps, n_max_boxes, align_metric)
    fg_mask = mask_pos.sum(-2)
    mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)
    max_overlaps_idx = overlaps.max(1).indices
    is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
    is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)
    mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos)
    align_metric = align_metric * mask_pos
    max_overlaps_idx = _topk_idx(align_metric, self.topk2)
    topk_idx = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
    topk_idx.scatter_(-1, max_overlaps_idx, 1.0)
    mask_pos *= topk_idx
    fg_mask, target_gt_idx = mask_pos.max(-2)
    return target_gt_idx, fg_mask, mask_pos


# ---------------------------------------------------------------- install / uninstall


def install(channels_last=True, sync_free_loss=True, fast_nms=True, fast_musgd=True, fused_bn=True, half_glue=True,
            fast_topk=True, fast_val=True, fast_cast=True, fast_cat=True, fast_attn=True):
    """Patch Ultralytics in place. Safe to call again with different flags. fast_musgd needs xpu_compat's scaler."""
    import xpu_compat

    from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
    from ultralytics.nn import tasks as TK
    from ultralytics.nn.modules import block as BM
    from ultralytics.nn.modules import conv as CM
    from ultralytics.optim import MuSGD
    from ultralytics.utils import loss as L
    from ultralytics.utils import nms as N
    from ultralytics.utils import tal as T

    if not _orig:
        _orig.update(
            set_model_attributes=DetectionTrainer.set_model_attributes,
            preprocess_batch=DetectionTrainer.preprocess_batch,
            make_anchors=L.make_anchors,
            get_assigned_targets_and_loss=L.v8DetectionLoss.get_assigned_targets_and_loss,
            preprocess=L.v8DetectionLoss.preprocess,
            get_box_metrics=T.TaskAlignedAssigner.get_box_metrics,
            bbox_forward=L.BboxLoss.forward,
            nms_boxes=N.nms_boxes,
            musgd_step=MuSGD.step,
            scaler_step=xpu_compat.XPUGradScaler.step,
            conv_forward=CM.Conv.forward,
            tal_torch=T.torch,
            upsample_forward=torch.nn.Upsample.forward,
            select_topk_candidates=T.TaskAlignedAssigner.select_topk_candidates,
            select_highest_overlaps=T.TaskAlignedAssigner.select_highest_overlaps,
            val_preprocess=DetectionValidator.preprocess,
            val_postprocess=DetectionValidator.postprocess,
            val_update_metrics=DetectionValidator.update_metrics,
            predict_once=TK.BaseModel._predict_once,
            conv2d_forward=torch.nn.Conv2d.forward,
            c2f_forward=BM.C2f.forward,
            c3_forward=BM.C3.forward,
            sppf_forward=BM.SPPF.forward,
            c2psa_forward=BM.C2PSA.forward,
            concat_forward=CM.Concat.forward,
            attention_forward=BM.Attention.forward,
        )
        BM.Attention.forward = _attention_forward
        BM.C2f.forward = _c2f_forward  # with fast_cat off these are the original forwards
        BM.C3.forward = _c3_forward
        BM.SPPF.forward = _sppf_forward
        BM.C2PSA.forward = _c2psa_forward
        CM.Concat.forward = _concat_forward
        TK.BaseModel._predict_once = _predict_once  # with fast_cast off these are the original methods
        torch.nn.Conv2d.forward = _conv2d_forward
        DetectionValidator.preprocess = _val_preprocess  # with fast_val off these are the original methods
        DetectionValidator.postprocess = _val_postprocess
        DetectionValidator.update_metrics = _val_update_metrics
        MuSGD.step = _musgd_step  # without a prepared update it is the original step
        xpu_compat.XPUGradScaler.step = _scaler_step
        DetectionTrainer.set_model_attributes = _set_model_attributes
        DetectionTrainer.preprocess_batch = _preprocess_batch
        N.nms_boxes = _nms_boxes  # non_max_suppression looks nms_boxes up in its module at call time
        CM.Conv.forward = _conv_forward  # also DWConv; with fused_bn off it is the original forward
        torch.nn.Upsample.forward = _upsample_forward  # with half_glue off it is the original forward
    _flags.update(channels_last=channels_last, sync_free_loss=sync_free_loss, fast_nms=fast_nms, fast_musgd=fast_musgd,
                  fused_bn=fused_bn, half_glue=half_glue, fast_topk=fast_topk, fast_val=fast_val,
                  fast_cast=fast_cast, fast_cat=fast_cat, fast_attn=fast_attn)
    if fast_topk:
        T.TaskAlignedAssigner.select_topk_candidates = _select_topk_candidates
        T.TaskAlignedAssigner.select_highest_overlaps = _select_highest_overlaps
    else:
        T.TaskAlignedAssigner.select_topk_candidates = _orig["select_topk_candidates"]
        T.TaskAlignedAssigner.select_highest_overlaps = _orig["select_highest_overlaps"]
    if sync_free_loss:
        if not isinstance(T.torch, _ConstTensorTorch):
            T.torch = _ConstTensorTorch((float, int))
        L.make_anchors = _make_anchors
        L.v8DetectionLoss.get_assigned_targets_and_loss = _get_assigned_targets_and_loss
        L.v8DetectionLoss.preprocess = _preprocess_targets
        T.TaskAlignedAssigner.get_box_metrics = _get_box_metrics
        L.BboxLoss.forward = _bbox_loss_forward
    else:
        T.torch = _orig["tal_torch"]
        L.make_anchors = _orig["make_anchors"]
        L.v8DetectionLoss.get_assigned_targets_and_loss = _orig["get_assigned_targets_and_loss"]
        L.v8DetectionLoss.preprocess = _orig["preprocess"]
        T.TaskAlignedAssigner.get_box_metrics = _orig["get_box_metrics"]
        L.BboxLoss.forward = _orig["bbox_forward"]
