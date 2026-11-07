#!/usr/bin/env python
"""Convert the ONNX from export_onnx.py to an .rknn for the RK3588.

    # FP16 (no quantization), batch 1: a good reference for accuracy
    python training/convert_rknn.py model.onnx --dtype fp --out model-fp16.rknn
    # INT8 for deployment, batch 24 (the batch size is baked into the model and must match --batch-size at runtime)
    python training/convert_rknn.py model.onnx --dtype i8 --calib calib704/calib.txt --batch 24 --out model-i8-bs24.rknn

The graph is cut at two tensors named in <model>.outputs.json: the class scores and the raw left/top/right/bottom
box distances in grid units (--cut ltrb, the default). The anchor/stride arithmetic and the TopK then happen on the CPU
for the few surviving detections. Cutting at decoded pixel coordinates (--cut boxes) loses ~11 points of mAP50-95 in
INT8 because 8 bits then resolve only ~3.8 px. Output order is (scores, dist).

Input normalisation is baked in: mean 0, std 255 (the board feeds raw uint8 RGB).

Runs in the rknn venv (Python 3.12, rknn-toolkit2, torch<=2.4). Python 3.13+ is not supported by the toolkit.
"""
import argparse

from rknn_build import build, load_meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx")
    ap.add_argument("--meta", help="outputs json (default: <onnx>.outputs.json)")
    ap.add_argument("--dtype", choices=["fp", "i8"], default="i8")
    ap.add_argument("--calib", help="calibration image list (required for i8); see make_calib.py")
    ap.add_argument("--batch", type=int, default=1, help="rknn_batch_size baked into the model")
    ap.add_argument("--platform", default="rk3588")
    ap.add_argument("--algo", default="normal", choices=["normal", "mmse", "kl_divergence"], help="quantization algorithm")
    ap.add_argument(
        "--disable-rule",
        action="append",
        default=None,
        help="rknn-toolkit2 graph-fusion rule to disable (repeatable); none are disabled by default",
    )
    ap.add_argument(
        "--cut",
        choices=["ltrb", "boxes"],
        default="ltrb",
        help="where to cut the graph: raw ltrb distances (default, what src/yolo26 decodes) or decoded boxes",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    meta = load_meta(args.onnx, args.meta)
    rknn = build(
        args.onnx,
        meta,
        dtype=args.dtype,
        calib=args.calib,
        batch=args.batch,
        platform=args.platform,
        algo=args.algo,
        disable_rules=args.disable_rule,
        cut=args.cut,
        verbose=args.verbose,
    )
    ret = rknn.export_rknn(args.out)
    if ret != 0:
        raise SystemExit(f"export_rknn failed ({ret})")
    rknn.release()
    print(f"wrote {args.out}  (outputs: scores={meta['scores']}, {'dist=' + meta['dist'] if args.cut == 'ltrb' else 'boxes=' + meta['boxes']})")


if __name__ == "__main__":
    main()
