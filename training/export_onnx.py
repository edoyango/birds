#!/usr/bin/env python
"""Export a trained YOLO26 checkpoint to ONNX for RKNN conversion, with the NMS-free (one-to-one) head.

    python training/export_onnx.py runs/detect/<run>/weights/best.pt --imgsz 704

Writes <weights>.onnx and <weights>.outputs.json next to the checkpoint (or --out). The json names the two
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

import onnx
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
    args = ap.parse_args()

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
    meta = {
        "onnx": path.name,
        "scores": scores,
        "boxes": boxes,
        "dist": find_distance_tensor(path, boxes),  # alternative cut: raw ltrb in grid units, see preproc.ltrb_to_boxes
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
