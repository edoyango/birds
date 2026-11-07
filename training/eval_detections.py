#!/usr/bin/env python
"""Run one inference backend over a dataset split and dump the decoded detections for eval_map.py.

Backends (all see identical letterboxed input and use the identical decode in preproc.decode_yolo26, which mirrors
src/yolo26/yolo26_postprocess.cc):
    onnx  ONNX Runtime, FP32, on the graph cut at the same two tensors the NPU model ends at. The reference.
    fp    rknn-toolkit2 simulator, FP16 build.
    i8    rknn-toolkit2 simulator, INT8 build (needs --calib).

    python training/eval_detections.py model.onnx --backend i8 --calib calib704/calib.txt \
        --data-root /path/to/birds-3-57 --limit 500 --out dets-i8.npz

The simulator can only run a model built in the same process, which is why this builds instead of loading an .rknn.
It emulates the NPU numerics on x86; it is not bit-exact with the board, and it is not fast.

Runs in the rknn venv (Python 3.12, rknn-toolkit2).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

from preproc import SUSPECT_RUN, constant_runs, decode_yolo26, letterbox, ltrb_to_boxes, read_image
from rknn_build import build, input_order, load_meta


def list_images(root, split, limit, list_file=None):
    if list_file:
        return [Path(p.strip()) for p in Path(list_file).read_text().splitlines() if p.strip()]
    imgs = sorted(p for p in (Path(root) / split / "images").iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if limit and limit < len(imgs):
        idx = np.linspace(0, len(imgs) - 1, limit).round().astype(int)  # evenly spaced, deterministic
        imgs = [imgs[i] for i in sorted(set(idx))]
    return imgs


class OnnxBackend:
    def __init__(self, onnx_path, meta, cut="ltrb"):
        import onnx
        import onnxruntime as ort

        tmp = Path(onnx_path).with_suffix(".cut.onnx")
        onnx.utils.extract_model(str(onnx_path), str(tmp), ["images"], [meta["scores"], meta["boxes" if cut == "boxes" else "dist"]])
        self.sess = ort.InferenceSession(str(tmp), providers=["CPUExecutionProvider"])
        self.name = self.sess.get_inputs()[0].name

    def __call__(self, canvas_rgb):
        x = canvas_rgb.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        scores, boxes = self.sess.run(None, {self.name: x})
        return scores[0], boxes[0]

    def release(self):
        pass


class RknnSimBackend:
    def __init__(self, onnx_path, meta, dtype, calib, verbose=False, cut="ltrb"):
        self.rknn = build(onnx_path, meta, dtype=dtype, calib=calib, batch=1, verbose=verbose, cut=cut)
        if self.rknn.init_runtime() != 0:  # no target given: run on the x86 simulator
            raise RuntimeError("init_runtime failed")

    def __call__(self, canvas_rgb):
        scores, boxes = self.rknn.inference(inputs=[canvas_rgb], data_format=["nhwc"])
        return scores[0], boxes[0]

    def release(self):
        self.rknn.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx")
    ap.add_argument("--meta")
    ap.add_argument("--backend", choices=["onnx", "fp", "i8"], required=True)
    ap.add_argument("--calib")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split", default="valid")
    ap.add_argument("--limit", type=int, default=0, help="use N evenly spaced images (default: all)")
    ap.add_argument("--list", help="text file of image paths to use instead of the split (same images across backends)")
    ap.add_argument("--cut", choices=["ltrb", "boxes"], default="ltrb", help="graph cut: raw ltrb distances (default) or decoded boxes")
    ap.add_argument("--swap-channels", action="store_true",
                    help="feed the opposite channel order from the one the model takes (to measure the cost of a mix-up)")
    ap.add_argument("--conf", type=float, default=0.001, help="score floor (low, as for mAP evaluation)")
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--min-run", type=int, default=SUSPECT_RUN,
                    help="output-fault guard: ignore class scores in runs of this many identical values (0: off)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    meta = load_meta(args.onnx, args.meta)
    h, w = meta["input_hw"]
    imgs = list_images(args.data_root, args.split, args.limit, args.list)
    feed_order = input_order(meta)
    if args.swap_channels:
        feed_order = "rgb" if feed_order == "bgr" else "bgr"
    print(f"{len(imgs)} images, backend={args.backend}, input {w}x{h}, model takes {input_order(meta)}, feeding {feed_order}", flush=True)

    if args.backend == "onnx":
        backend = OnnxBackend(args.onnx, meta, args.cut)
    else:
        backend = RknnSimBackend(args.onnx, meta, args.backend, args.calib, args.verbose, args.cut)

    rows, sizes, t_inf = [], [], 0.0
    guarded = guarded_imgs = 0  # anchors whose best class score above --conf the guard ignored
    for i, p in enumerate(imgs):
        rgb = read_image(p, feed_order)  # named rgb for brevity; in the model's channel order
        sizes.append((rgb.shape[1], rgb.shape[0]))
        canvas, scale, x_pad, y_pad = letterbox(rgb, w, h)
        t0 = time.time()
        scores, boxes = backend(canvas)
        if args.cut == "ltrb":
            boxes = ltrb_to_boxes(boxes, h, w)  # the CPU-side arithmetic that follows the cut
        t_inf += time.time() - t0
        if args.min_run > 0:
            top = scores.argmax(0)
            hit = int(((scores.max(0) > args.conf) & constant_runs(scores, args.min_run)[top, np.arange(scores.shape[1])]).sum())
            guarded += hit
            guarded_imgs += hit > 0
        d = decode_yolo26(scores, boxes, args.conf, scale, x_pad, y_pad, w, h, args.max_det, min_run=args.min_run)
        rows.append(np.concatenate([np.full((len(d), 1), i, np.float32), d], 1))
        if (i + 1) % 50 == 0 or i + 1 == len(imgs):
            print(f"  {i + 1}/{len(imgs)}  {t_inf / (i + 1):.2f} s/img", flush=True)
    backend.release()

    np.savez(
        args.out,
        dets=np.concatenate(rows, 0) if rows else np.zeros((0, 7), np.float32),
        files=np.array([str(p) for p in imgs]),
        sizes=np.array(sizes),
        backend=args.backend,
        model=str(args.onnx),
    )
    print(f"output-fault guard (min run {args.min_run}): ignored the best class of {guarded} anchors above conf "
          f"{args.conf} in {guarded_imgs} of {len(imgs)} images")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
