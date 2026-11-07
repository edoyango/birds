#!/usr/bin/env python
"""Check that the NPU-side graph cut + CPU decode reproduces the model's own end-to-end output.

Runs the full exported ONNX (which ends in ReduceMax -> TopK -> GatherElements and outputs (1, 300, 6)) and the same
graph cut at the two NPU output tensors (scores and raw ltrb distances) followed by preproc.ltrb_to_boxes + decode_yolo26, on the same letterboxed images (ONNX
Runtime, FP32), and compares the detections above --conf. This validates the cut points and the decode, which the C++
postprocess mirrors, independently of any quantization.

    python training/check_decode.py model.onnx --data-root /path/to/birds-3-57 --n 30

Runs in either venv with onnx, onnxruntime, cv2 and numpy.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

from preproc import decode_yolo26, letterbox, ltrb_to_boxes, read_image


def iou(a, b):
    x1, y1 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1])
    x2, y2 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter + 1e-9)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx")
    ap.add_argument("--meta")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split", default="valid")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    path = Path(args.onnx)
    meta = json.loads(Path(args.meta or path.with_suffix(".outputs.json")).read_text())
    h, w = meta["input_hw"]
    cut = path.with_suffix(".cut.onnx")
    onnx.utils.extract_model(str(path), str(cut), ["images"], [meta["scores"], meta["dist"]])
    full = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    part = ort.InferenceSession(str(cut), providers=["CPUExecutionProvider"])

    imgs = sorted(p for p in (Path(args.data_root) / args.split / "images").iterdir() if p.suffix.lower() in (".jpg", ".png", ".jpeg"))
    imgs = [imgs[i] for i in np.linspace(0, len(imgs) - 1, args.n).round().astype(int)]

    n_full = n_cut = matched_full = matched_cut = other_class = 0
    worst_score = 0.0
    for p in imgs:
        rgb = read_image(p, meta.get("input_order", "rgb"))  # in the model's channel order
        canvas, *_ = letterbox(rgb, w, h)
        x = canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        ref = full.run(None, {"images": x})[0][0]  # (300, 6): x1 y1 x2 y2 score cls, input pixels
        ref = ref[ref[:, 4] > args.conf]
        scores, dist = part.run(None, {"images": x})
        # scale=1, no padding: stay in model-input pixels so both sides are directly comparable
        mine = decode_yolo26(scores[0], ltrb_to_boxes(dist[0], h, w), args.conf, 1.0, 0, 0, w, h, max_det=300)
        n_full += len(ref)
        n_cut += len(mine)
        for d in mine:  # every decoded box should exist in the reference
            if len(ref):
                same = ref[:, 5] == d[5]
                k = iou(d[:4], ref[:, :4]) * same
                if k.max() > 0.95:
                    matched_cut += 1
                    worst_score = max(worst_score, abs(float(ref[k.argmax(), 4]) - float(d[4])))
        for r in ref:  # and vice versa
            if len(mine):
                k = iou(r[:4], mine[:, :4]) * (mine[:, 5] == r[5])
                found = k.max() > 0.95
                matched_full += int(found)
                if not found:  # same box present with a different class? (second-best class of the same anchor)
                    other_class += int((iou(r[:4], mine[:, :4]) > 0.95).any())
    print(f"{len(imgs)} images, conf > {args.conf}")
    print(f"reference (full graph): {n_full} detections, {matched_full} found by the cut+decode")
    print(f"cut + decode:           {n_cut} detections, {matched_cut} found in the reference")
    print(f"reference detections missed by the decode: {n_full - matched_full}, of which same box with another class: {other_class}")
    print(f"largest score difference among matches: {worst_score:.2e}")


if __name__ == "__main__":
    sys.exit(main())
