#!/usr/bin/env python
"""Export a trained YOLO26 checkpoint to ONNX for RKNN conversion, with the NMS-free (one-to-one) head.

    python training/export_onnx.py runs/detect/<run>/weights/last.pt --imgsz 448 768 --data-root /path/to/birds-3-57

Writes <weights>.onnx and <weights>.outputs.json next to the checkpoint (or --out). The head gets clips that keep INT8
conversion working after long training, with the class-feature bound measured on --data-root training images and all
clips checked there to change no detection (see clip_head_levels and training/README.md). The json names the
tensors `convert_rknn.py` cuts the graph at:

    scores  (1, nc, N)  class probabilities, sigmoid already applied
    boxes   (1, 4,  N)  x1, y1, x2, y2 in input pixels, already decoded (anchor grid and stride applied)

Everything after them in the exported graph (ReduceMax -> TopK -> GatherElements) is the part that would pick the top
300 detections; TopK is not an NPU op, so the board does that step on the CPU (a threshold + sort, no NMS).

IMPORTANT: this must be exported with nms=False. Recent Ultralytics versions export the one-to-many head by default,
which needs NMS; the Rockchip YOLO26 example's plain `yolo export ... format=onnx` no longer yields the NMS-free graph.

Run in the Ultralytics venv (needs ultralytics, onnx).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnx.numpy_helper
from onnx import shape_inference


def use_scale_after_matmul_attention():
    """Export attention as softmax((q^T k) * scale) instead of softmax((q * scale)^T k).

    The two are mathematically identical. Recent Ultralytics multiplies the scale into q before the matmul; with that
    arrangement rknn-toolkit2 2.5.0 crashes in its SDPA fusion rule, and the workaround (disabling the rule) leaves
    unfused Softmax ops that fail to run on the NPU (RKNPU driver 0.9.6: `failed to submit! ... exSoftmax13`). The
    scale-after-matmul arrangement is what YOLO11 exports and what the toolkit's attention fusion is written for.
    """
    from ultralytics.nn.modules import block

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )
        attn = ((q.transpose(-2, -1) @ k) * self.scale).softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        return self.proj(x)

    block.Attention.forward = forward


def reorder_input_channels(yolo):
    """Make the model take BGR input instead of RGB by reversing the input channels of its first convolution.

    Only the first conv sees the image, and a conv is linear in its input channels, so this gives the same outputs on
    BGR input as the original model on RGB input (up to float rounding: 1e-3 on the final checkpoint's outputs, which
    are pixel coordinates up to ~768). The board can then feed OpenCV's BGR frames without a colour conversion."""
    import torch

    conv = yolo.model.model[0].conv
    convs_on_image = [n for n, m in yolo.model.named_modules() if isinstance(m, torch.nn.Conv2d) and m.in_channels == 3]
    if conv.in_channels != 3 or convs_on_image != ["model.0.conv"]:
        raise SystemExit(f"expected model.0.conv to be the only conv on the 3-channel image, found {convs_on_image}")
    with torch.no_grad():
        conv.weight.copy_(conv.weight[:, [2, 1, 0]].clone())


def find_truncation_outputs(onnx_path, nc):
    """Locate the scores and boxes tensors by graph structure (robust to node renaming by onnxslim)."""
    model = shape_inference.infer_shapes(onnx.load(str(onnx_path)))
    g = model.graph
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim] for v in list(g.value_info) + list(g.output)}
    producer = {o: n for n in g.node for o in n.output}

    # The end-to-end head ends with: Concat(boxes (1,4,N), scores (1,nc,N)) -> Transpose -> Split -> ReduceMax -> TopK ->
    # GatherElements. Walk upstream from the ReduceMax (max class prob per anchor) to that Concat; its two inputs are
    # the tensors to cut at. Rockchip's example names them /model.23/Mul_2 (boxes) and /model.23/Sigmoid (scores).
    reduce_max = [n for n in g.node if n.op_type == "ReduceMax"]
    if not reduce_max:
        raise SystemExit("no ReduceMax found; was the model exported with nms=False?")
    tensor = reduce_max[-1].input[0]
    while producer.get(tensor) is not None and producer[tensor].op_type in ("Split", "Transpose", "Identity"):
        tensor = producer[tensor].input[0]
    concat = producer.get(tensor)
    if concat is None or concat.op_type != "Concat" or len(concat.input) != 2:
        raise SystemExit(f"unexpected head structure above ReduceMax (reached {concat.op_type if concat else tensor})")

    def channels(t):
        s = shapes.get(t, [])
        return s[1] if len(s) == 3 else None

    a, b = concat.input
    if (channels(a), channels(b)) == (4, nc):
        boxes, scores = a, b
    elif (channels(b), channels(a)) == (4, nc):
        boxes, scores = b, a
    else:
        raise SystemExit(f"Concat inputs have channels {channels(a)}, {channels(b)}; expected 4 and nc={nc}")
    if producer[scores].op_type != "Sigmoid":
        raise SystemExit(f"scores tensor is produced by {producer[scores].op_type}, expected Sigmoid")
    n_anchors = shapes[scores][2]
    return scores, boxes, n_anchors


def find_distance_tensor(onnx_path, boxes):
    """The raw (1, 4, N) left/top/right/bottom distances that the graph turns into `boxes`:
    boxes = Concat(anchor - lt, anchor + rb) * stride, with lt and rb Slices of this one tensor."""
    model = onnx.load(str(onnx_path))
    producer = {o: n for n in model.graph.node for o in n.output}
    mul = producer[boxes]
    concat = producer[mul.input[0]] if producer[mul.input[0]].op_type == "Concat" else producer[mul.input[1]]
    sources = set()
    for branch in concat.input:
        node = producer[branch]  # Sub or Add
        for i in node.input:
            p = producer.get(i)
            if p is not None and p.op_type == "Slice":
                sources.add(p.input[0])
    if len(sources) != 1:
        raise SystemExit(f"could not identify the distance tensor behind {boxes}: {sources}")
    return sources.pop()


def head_concats(model, scores, dist):
    """The Concats that join the per-level class logits (behind `scores`) and the per-level distances (`dist`)."""
    producer = {o: n for n in model.graph.node for o in n.output}
    logit_cat, dist_cat = producer[producer[scores].input[0]], producer[dist]
    for cat in (logit_cat, dist_cat):
        if cat.op_type != "Concat":
            raise SystemExit(f"expected a Concat of per-level tensors, found {cat.op_type} ({cat.name})")
    return producer, logit_cat, dist_cat


def class_conv_inputs(model, scores, dist):
    """Per level, the final class Conv node (behind Reshape -> logit Concat) and the feature tensor it reads."""
    producer, logit_cat, _ = head_concats(model, scores, dist)
    convs = []
    for name in logit_cat.input:
        conv = producer[producer[name].input[0]]
        if conv.op_type != "Conv":
            raise SystemExit(f"expected the final class Conv behind {name}, found {conv.op_type}")
        convs.append(conv)
    return convs


def clip_head_levels(model, scores, dist, logit_lim, dist_hi, cls_feat_hi):
    """In each detection level, clip the class features that feed the final class conv to [-1, cls_feat_hi], and, before
    the per-level tensors are concatenated into the two cut outputs, the class logits to [-logit_lim, logit_lim] and the
    ltrb distances to [-1, dist_hi]. A bound of 0 leaves that tensor alone. Edits `model` in place.

    INT8 gives each tensor one scale, and long training lets the head's background positions drift to huge values that
    then set that scale. yolo26s after 120 epochs:
    - the stride-32 level never fires; its logits reached -4e7 and its distances -1091..5854 (real detections: logits
      < 3.5, distances 0.1-8.5 grid units). Through the Concats this rounded every real box distance to 0 on the board
      (mAP50 0.000).
    - the class features reached 212-308 at background positions while detections use at most ~8. At a step of ~1.2
      the SiLU's small negative outputs all became 0; with negative weights on them in the final conv, every bird's
      logit fell by ~2.3 (score 0.96 -> 0.70; board recall at conf 0.5 0.92 -> 0.58).
    Clipping in the graph keeps the ONNX reference, the simulator and the board on the same function. Scores below
    sigmoid(-logit_lim) are below any threshold we use; `check_clips_unchanged` verifies the rest on real images."""
    _, logit_cat, dist_cat = head_concats(model, scores, dist)
    if cls_feat_hi > 0:
        lo_name, hi_name = "cls_feat_clip_lo", "cls_feat_clip_hi"
        model.graph.initializer.extend([
            onnx.numpy_helper.from_array(np.array(-1.0, np.float32), lo_name),  # SiLU >= -0.28: no lower clipping
            onnx.numpy_helper.from_array(np.array(cls_feat_hi, np.float32), hi_name),
        ])
        for conv in class_conv_inputs(model, scores, dist):
            out = f"{conv.input[0]}_clip"
            idx = list(model.graph.node).index(conv)
            model.graph.node.insert(idx, onnx.helper.make_node("Clip", [conv.input[0], lo_name, hi_name], [out], name=out))
            conv.input[0] = out
    # distances keep down to -1: detections can sit marginally outside their box (seen: -0.005 grid units)
    for cat, lo, hi in ((logit_cat, -logit_lim, logit_lim), (dist_cat, -1.0, dist_hi)):
        if hi <= 0:
            continue
        if cat.op_type != "Concat":
            raise SystemExit(f"expected a Concat of per-level tensors, found {cat.op_type} ({cat.name})")
        lo_name, hi_name = cat.name + "_clip_lo", cat.name + "_clip_hi"
        model.graph.initializer.extend([
            onnx.numpy_helper.from_array(np.array(lo, np.float32), lo_name),
            onnx.numpy_helper.from_array(np.array(hi, np.float32), hi_name),
        ])
        idx = list(model.graph.node).index(cat)
        for i, name in enumerate(list(cat.input)):
            out = f"{name}_clip"
            model.graph.node.insert(idx, onnx.helper.make_node("Clip", [name, lo_name, hi_name], [out], name=out))
            idx += 1
            cat.input[i] = out
    onnx.checker.check_model(model)


DET_SCORE = 0.05  # anchors whose best class scores above this count as detections when measuring feature ranges
OUTLIER_RATIO = 4.0  # clip the class features only when the background reaches this many times the detections' max
FEAT_MARGIN = 2.0  # the bound is this many times the largest class feature seen at a detection


def sample_images(data_root, n):
    """n evenly spaced training images (not validation, which the evaluation uses)."""
    files = sorted(p for p in (Path(data_root) / "train" / "images").iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not files:
        raise SystemExit(f"no images in {Path(data_root) / 'train' / 'images'}")
    return files[:: max(1, len(files) // n)][:n]


def run_images(model, outputs, images, hw, order):
    """Yield the requested tensors of `model` (an in-memory ModelProto; intermediate tensors allowed) for each image,
    letterboxed exactly as the evaluation and the board do."""
    import onnxruntime as ort

    import preproc

    m = onnx.ModelProto()
    m.CopyFrom(model)
    have = {o.name for o in m.graph.output}
    m.graph.output.extend([onnx.helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None) for t in outputs if t not in have])
    sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    for f in images:
        canvas, *_ = preproc.letterbox(preproc.read_image(f, order), hw[1], hw[0])
        yield sess.run(outputs, {name: (canvas[None].transpose(0, 3, 1, 2) / 255).astype(np.float32)})


def measure_class_features(model, scores, dist, images, hw, order):
    """Largest class feature (input of the final class convs) at detection anchors and anywhere, over `images`."""
    """Per level: largest class feature (input of the final class conv) at detection anchors and anywhere, over `images`,
    and the number of detection anchors."""
    feats = [conv.input[0] for conv in class_conv_inputs(model, scores, dist)]
    det_max, all_max, n_det = [0.0] * len(feats), [0.0] * len(feats), [0] * len(feats)
    for out in run_images(model, [scores] + feats, images, hw, order):
        best = out[0][0].max(0)  # best class score per anchor, all levels concatenated
        start = 0
        for lv, f in enumerate(out[1:]):
            f = f[0].reshape(f.shape[1], -1)  # (channels, anchors of this level)
            det = best[start : start + f.shape[1]] > DET_SCORE
            start += f.shape[1]
            all_max[lv] = max(all_max[lv], float(f.max()))
            if det.any():
                det_max[lv] = max(det_max[lv], float(f[:, det].max()))
                n_det[lv] += int(det.sum())
    return det_max, all_max, n_det


def check_clips_unchanged(raw, clipped, scores, dist, images, hw, order, tol=1e-3):
    """The clips must not change what the model detects: compare the scores of every anchor/class above 0.001 and the
    distances of the detection anchors, raw vs clipped graph. Returns the largest differences."""
    worst_s = worst_d = 0.0
    for (s0, d0), (s1, d1) in zip(run_images(raw, [scores, dist], images, hw, order), run_images(clipped, [scores, dist], images, hw, order)):
        keep = s0[0] > 0.001
        if keep.any():
            worst_s = max(worst_s, float(np.abs(s1[0] - s0[0])[keep].max()))
        det = s0[0].max(0) > DET_SCORE
        if det.any():
            worst_d = max(worst_d, float(np.abs(d1[0][:, det] - d0[0][:, det]).max()))
    if worst_s > tol or worst_d > tol:
        raise SystemExit(
            f"the head clips change detections on training images (max score change {worst_s:.4g}, max distance change "
            f"{worst_d:.4g}). Pass larger bounds (--clip-cls-feat, --clip-dist, --clip-logit) or 0 to disable one."
        )
    return worst_s, worst_d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights")
    ap.add_argument("--imgsz", type=int, nargs="+", default=[704], help="N for square, or H W")
    ap.add_argument("--opset", type=int, default=18, help="rknn-toolkit2 supports ONNX opset 12-19")
    ap.add_argument("--simplify", action="store_true", help="run onnxslim (node names may change; outputs are re-detected)")
    ap.add_argument("--out", help="output .onnx path (default: next to the weights)")
    ap.add_argument(
        "--input-order",
        choices=["rgb", "bgr"],
        default="rgb",
        help="channel order the exported model takes. bgr reverses the first conv's input channels, so the board feeds "
        "OpenCV frames as they are (recorded in the json, and by convert_rknn.py in the .rknn)",
    )
    ap.add_argument(
        "--keep-q-scaling",
        action="store_true",
        help="export Ultralytics' own attention arrangement (scale applied to q) instead of scale after the matmul",
    )
    ap.add_argument(
        "--clip-dist",
        type=float,
        default=16.0,
        help="clip each level's ltrb distances to [-1, this] grid units before they are concatenated, so a level that "
        "never fires cannot blow up the INT8 range (real detections stay below ~9; 0 disables)",
    )
    ap.add_argument(
        "--clip-logit", type=float, default=8.0, help="clip each level's class logits to [-this, this] (0 disables)"
    )
    ap.add_argument(
        "--clip-cls-feat",
        default="auto",
        help="upper bound for the class features that feed each level's final class conv: 'auto' (measure it on "
        "--data-root training images, see README), a number, or 0 to disable",
    )
    ap.add_argument("--data-root", help="dataset dir with train/images, for --clip-cls-feat auto and the clip check")
    ap.add_argument("--clip-images", type=int, default=200, help="training images to measure and check the clips on")
    args = ap.parse_args()
    if args.clip_cls_feat == "auto" and not args.data_root:
        ap.error("--clip-cls-feat auto needs --data-root (or pass a number, or 0)")
    cls_feat = None if args.clip_cls_feat == "auto" else float(args.clip_cls_feat)

    from ultralytics import YOLO

    if not args.keep_q_scaling:
        use_scale_after_matmul_attention()
    yolo = YOLO(args.weights)
    if args.input_order == "bgr":
        reorder_input_channels(yolo)
    nc = len(yolo.names)
    imgsz = args.imgsz[0] if len(args.imgsz) == 1 else args.imgsz
    path = Path(
        yolo.export(format="onnx", imgsz=imgsz, opset=args.opset, nms=False, simplify=args.simplify, batch=1, device="cpu")
    )
    if args.out:
        path = path.rename(args.out)

    scores, boxes, n_anchors = find_truncation_outputs(path, nc)
    if n_anchors % 16:
        print(
            f"WARNING: the head has {n_anchors} anchors, which is not a multiple of 16. On an RK3588 (RKNPU driver 0.9.6, "
            "runtime 2.5.0) a 704x704 model (10164 anchors) returned unstable values in a few box outputs at row edges, "
            "while 640x640 (8400), 896x512 (9408) and 768x448 (7056) did not. Prefer an input size whose anchor count is "
            "a multiple of 16: (H/8)*(W/8) + (H/16)*(W/16) + (H/32)*(W/32). See training/README.md."
        )
    hw = [imgsz, imgsz] if isinstance(imgsz, int) else imgsz
    dist = find_distance_tensor(path, boxes)

    raw = onnx.load(str(path))
    images = sample_images(args.data_root, args.clip_images) if args.data_root else []
    clip_info = {}
    if cls_feat is None:  # auto
        det_lv, all_lv, n_lv = measure_class_features(raw, scores, dist, images, hw, args.input_order)
        clip_info = {"images": len(images), "det_anchors_per_level": n_lv,
                     "det_max_per_level": [round(v, 3) for v in det_lv], "all_max_per_level": [round(v, 3) for v in all_lv]}
        for lv, (d, a, n) in enumerate(zip(det_lv, all_lv, n_lv)):
            print(f"  class features, stride {8 << lv}: max {a:.4g} overall, {d:.4g} at {n} detection anchors")
        det_max = max(det_lv)
        if sum(n_lv) == 0:
            raise SystemExit(f"no detections above {DET_SCORE} on {len(images)} training images; pass --clip-cls-feat")
        if max(all_lv) > OUTLIER_RATIO * det_max:
            cls_feat = float(np.ceil(FEAT_MARGIN * det_max))
            why = f"background features reach more than {OUTLIER_RATIO:g}x the detections' max of {det_max:.2f}"
        else:
            cls_feat = 0.0
            why = f"no background outliers to remove (detections use features up to {det_max:.2f})"
        print(f"class-feature clip: {cls_feat:g} ({why}; {sum(n_lv)} detection anchors on {len(images)} training images)")
    clipped = onnx.ModelProto()
    clipped.CopyFrom(raw)
    clip_head_levels(clipped, scores, dist, args.clip_logit, args.clip_dist, cls_feat)
    if images:
        ds, dd = check_clips_unchanged(raw, clipped, scores, dist, images, hw, args.input_order)
        clip_info.update(check_max_score_change=float(f"{ds:.3g}"), check_max_dist_change=float(f"{dd:.3g}"))
        print(f"clip check on {len(images)} training images: max score change {ds:.2g}, max distance change {dd:.2g}")
    else:
        print("WARNING: no --data-root, so the head clips were not checked against real images")
    onnx.save(clipped, str(path))

    meta = {
        "onnx": path.name,
        "scores": scores,
        "boxes": boxes,
        "dist": dist,  # the default cut: raw ltrb in grid units, see preproc.ltrb_to_boxes
        "clip_dist": args.clip_dist,
        "clip_logit": args.clip_logit,
        "clip_cls_feat": cls_feat,
        "clip_info": clip_info,
        "strides": [8, 16, 32],
        "num_anchors": n_anchors,
        "nc": nc,
        "names": [yolo.names[i] for i in range(nc)],
        "input_hw": hw,
        "input_order": args.input_order,
        "opset": args.opset,
    }
    meta_path = path.with_suffix(".outputs.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{path}\n{meta_path}\n  scores: {scores}\n  boxes:  {boxes}\n  anchors: {n_anchors}")


if __name__ == "__main__":
    main()
