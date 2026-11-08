"""Training-speed patches for Ultralytics on Intel Arc GPUs (XPU). Measured with training/bench_step.py.

What limits a YOLO26 training step on the A770 is mostly host-side work, not GPU math: every kernel launch costs
CPU time, and every host sync leaves the GPU idle while Python catches up. The patches:

launch_env   UR_L0_USE_IMMEDIATE_COMMANDLISTS=0. With the default immediate command lists, each kernel launch
             costs ~22 us of CPU when the GPU is idle and ~157 us while it is busy; batched command lists cost
             ~5-36 us. Bit-identical results. Must be set before the first XPU call (set_launch_env()).
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

Call set_launch_env() before any XPU work, then install(...) before the trainer is built. Written against
Ultralytics 8.4.204 / torch 2.15; the patched functions fall back to the originals for shapes they do not cover.
"""
import os

import torch

LAUNCH_ENV = {"UR_L0_USE_IMMEDIATE_COMMANDLISTS": "0"}

_orig = {}
_flags = {"channels_last": False, "sync_free_loss": False, "fast_nms": False, "fast_musgd": False}


def set_launch_env():
    """Set Level Zero launch settings; they only take effect if the XPU runtime is not initialized yet."""
    for k, v in LAUNCH_ENV.items():
        os.environ.setdefault(k, v)
    if torch.xpu.is_initialized():
        print(f"xpu_speedups: WARNING XPU already initialized, {list(LAUNCH_ENV)} may not take effect")


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


def _get_assigned_targets_and_loss(self, preds, batch, **kw):
    self._n_max_boxes = batch.get("n_max_boxes") if isinstance(batch, dict) else None
    try:
        return _orig["get_assigned_targets_and_loss"](self, preds, batch, **kw)
    finally:
        self._n_max_boxes = None


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


def _nms_boxes(boxes, scores, iou, rotated=False):
    if _flags["fast_nms"] and not rotated and boxes.device.type == "xpu":
        import torchvision

        return torchvision.ops.nms(boxes, scores, iou)
    return _orig["nms_boxes"](boxes, scores, iou, rotated)


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


# ---------------------------------------------------------------- install / uninstall


def install(channels_last=True, sync_free_loss=True, fast_nms=True, fast_musgd=True):
    """Patch Ultralytics in place. Safe to call again with different flags. fast_musgd needs xpu_compat's scaler."""
    import xpu_compat

    from ultralytics.models.yolo.detect import DetectionTrainer
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
        )
        MuSGD.step = _musgd_step  # without a prepared update it is the original step
        xpu_compat.XPUGradScaler.step = _scaler_step
        DetectionTrainer.set_model_attributes = _set_model_attributes
        DetectionTrainer.preprocess_batch = _preprocess_batch
        N.nms_boxes = _nms_boxes  # non_max_suppression looks nms_boxes up in its module at call time
    _flags.update(channels_last=channels_last, sync_free_loss=sync_free_loss, fast_nms=fast_nms, fast_musgd=fast_musgd)
    if sync_free_loss:
        L.make_anchors = _make_anchors
        L.v8DetectionLoss.get_assigned_targets_and_loss = _get_assigned_targets_and_loss
        L.v8DetectionLoss.preprocess = _preprocess_targets
        T.TaskAlignedAssigner.get_box_metrics = _get_box_metrics
        L.BboxLoss.forward = _bbox_loss_forward
    else:
        L.make_anchors = _orig["make_anchors"]
        L.v8DetectionLoss.get_assigned_targets_and_loss = _orig["get_assigned_targets_and_loss"]
        L.v8DetectionLoss.preprocess = _orig["preprocess"]
        T.TaskAlignedAssigner.get_box_metrics = _orig["get_box_metrics"]
        L.BboxLoss.forward = _orig["bbox_forward"]
