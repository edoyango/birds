#!/usr/bin/env python
"""Check that the C++ decode (src/yolo26/yolo26_postprocess.cc) matches preproc.decode_yolo26 on real model outputs.

Runs the ONNX graph cut at the NPU output tensors on real images (ONNX Runtime, FP32), writes the two tensors for each
image to a file, compiles src/yolo26/test/test_decode.cc with the library's post-processing for the host, and compares
detections one by one: same count, order, class, score (1e-6) and box (+-1 px: the C++ stores integers).

Each image is also decoded with injected output faults of the kind seen on the RK3588 (a 16-element chunk of one class
plane overwritten with the INT8 value 0): once in the background, where the guard must give exactly the clean result,
and once over the best detection's anchor in another class plane, which exercises the fall-back to the next best class.

    python training/check_cpp_decode.py model.onnx --data-root /path/to/birds-3-57 --conf 0.01

Needs a C++17 compiler. It uses $CXX or g++/clang++/c++ from PATH; otherwise `pip install ziglang` and pass
--cxx "python -m ziglang c++". Runs in a venv with onnx, onnxruntime, cv2 and numpy (the Ultralytics venv has them).
"""
import argparse
import json
import re
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

from preproc import decode_yolo26, letterbox, ltrb_to_boxes, read_image

SRC = Path(__file__).resolve().parent.parent / "src"


def find_cxx(arg):
    if arg:
        return shlex.split(arg)
    import os

    if os.environ.get("CXX"):
        return shlex.split(os.environ["CXX"])
    for c in ("g++", "clang++", "c++"):
        if shutil.which(c):
            return [c]
    raise SystemExit("no C++ compiler found; set CXX, or pip install ziglang and pass --cxx 'python -m ziglang c++'")


def define(header, name):
    return int(re.search(rf"#define\s+{name}\s+(\d+)", Path(header).read_text()).group(1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx")
    ap.add_argument("--meta")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split", default="valid")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--conf", type=float, default=0.01, help="low values exercise the OBJ_NUMB_MAX_SIZE cap")
    ap.add_argument("--cxx")
    args = ap.parse_args()

    path = Path(args.onnx)
    meta = json.loads(Path(args.meta or path.with_suffix(".outputs.json")).read_text())
    h, w = meta["input_hw"]
    max_det = define(SRC / "yolo26" / "model.h", "OBJ_NUMB_MAX_SIZE")
    obj_class_num = define(SRC / "yolo26" / "model.h", "OBJ_CLASS_NUM")
    if obj_class_num != meta["nc"]:
        raise SystemExit(f"model has {meta['nc']} classes but OBJ_CLASS_NUM={obj_class_num}")

    work = Path(tempfile.mkdtemp(prefix="check-cpp-"))
    exe = work / "test_decode"
    cxx = find_cxx(args.cxx)
    cmd = cxx + [
        "-std=c++17", "-O2", "-Wall",
        f"-I{SRC / 'yolo26'}", f"-I{SRC / 'utils'}", f"-I{SRC / '3rdparty/include/rknpu2'}",
        str(SRC / "yolo26/test/test_decode.cc"), str(SRC / "yolo26/yolo26_postprocess.cc"), "-o", str(exe),
    ]  # fmt: skip
    print("compiling:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    cut = work / "cut.onnx"
    onnx.utils.extract_model(str(path), str(cut), ["images"], [meta["scores"], meta["dist"]])
    sess = ort.InferenceSession(str(cut), providers=["CPUExecutionProvider"])
    imgs = sorted(p for p in (Path(args.data_root) / args.split / "images").iterdir() if p.suffix.lower() in (".jpg", ".png", ".jpeg"))
    imgs = [imgs[i] for i in np.linspace(0, len(imgs) - 1, args.n).round().astype(int)]

    # Each image becomes two cases: the float tensors, and the same tensors quantized to INT8 with the zero points and
    # scales of the same order as a converted model reports on the board (scores: 1/255, zp -128; distances: about 0.04
    # grid units per step). The INT8 reference is the float decode of the dequantized tensors, which the INT8 decode
    # must reproduce exactly.
    SCORES_Q, BOXES_Q = (np.float32(1 / 255), -128), (np.float32(0.0388), -115)

    def quantize(x, q):
        return np.clip(np.round(x / q[0]) + q[1], -128, 127).astype(np.int8)

    def dequantize(x, q):
        return (x.astype(np.float32) - np.float32(q[1])) * q[0]

    rng = np.random.default_rng(0)

    def inject(qs, at_anchor=None):
        """Copy of INT8 scores with one 16-element chunk (16-aligned, as on the board) of one class plane set to 0. In
        the background (all scores below the threshold) unless `at_anchor` is given; then over that anchor, in a class
        plane other than the anchor's best."""
        q = qs.copy()
        nc, n = q.shape
        thr = np.floor(args.conf / SCORES_Q[0] + SCORES_Q[1])
        if at_anchor is None:
            chunks = np.nonzero((q.max(0)[: n // 16 * 16].reshape(-1, 16) <= thr).all(1))[0]
            start, c = int(rng.choice(chunks)) * 16, int(rng.integers(nc))
        else:
            start = at_anchor // 16 * 16
            c = int(np.argsort(q[:, at_anchor])[-2])  # second best class: the fault outranks the true class
        q[c, start : start + 16] = 0
        return q

    def add_case(scores, boxes, quantized, ref_scores, ref_dist, kind):
        nonlocal cases
        head = struct.pack("<6i2f", nc, n, w, h, x_pad, y_pad, np.float32(scale), np.float32(args.conf))
        q = struct.pack("<i2f2i", 1, SCORES_Q[0], BOXES_Q[0], SCORES_Q[1], BOXES_Q[1]) if quantized else struct.pack("<i2f2i", 0, 1.0, 1.0, 0, 0)
        cases += head + q + scores.tobytes() + boxes.tobytes()
        refs.append(decode_yolo26(ref_scores, ltrb_to_boxes(ref_dist, h, w), args.conf, sc, x_pad, y_pad, w, h, max_det, as_int=True))
        kinds.append(kind)

    cases, refs, kinds = bytearray(), [], []
    clean_vs_bg = []  # indices of the clean int8 cases
    for p in imgs:
        rgb = read_image(p, meta.get("input_order", "rgb"))  # in the model's channel order
        canvas, scale, x_pad, y_pad = letterbox(rgb, w, h)
        scores, boxes = sess.run(None, {"images": canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255.0})
        scores, boxes = np.ascontiguousarray(scores[0], np.float32), np.ascontiguousarray(boxes[0], np.float32)  # boxes = dist
        nc, n = scores.shape
        sc = float(np.float32(scale))
        add_case(scores, boxes, False, scores, boxes, "float")
        qs, qb = quantize(scores, SCORES_Q), quantize(boxes, BOXES_Q)
        db = dequantize(qb, BOXES_Q)
        add_case(qs, qb, True, dequantize(qs, SCORES_Q), db, "int8")
        for kind, q in (("int8 fault bg", inject(qs)), ("int8 fault on det", inject(qs, int(qs.max(0).argmax())))):
            add_case(q, qb, True, dequantize(q, SCORES_Q), db, kind)
            if kind.endswith("bg"):  # the float path sees the runtime's dequantised fault
                fq = dequantize(q, SCORES_Q)
                add_case(fq, db, False, fq, db, "float fault bg")
                clean_vs_bg.append(len(refs) - 3)  # the clean int8 case; the int8 and float bg faults follow it
    # Real outputs rarely have more than OBJ_NUMB_MAX_SIZE anchors above even a tiny threshold, so add one synthetic
    # image of random scores (no constant runs) to exercise the cap and the ranking.
    noise = rng.random((nc, n), dtype=np.float32)
    add_case(noise, boxes, False, noise, boxes, "float random")
    qn = quantize(noise, SCORES_Q)
    add_case(qn, qb, True, dequantize(qn, SCORES_Q), db, "int8 random")
    (work / "cases.bin").write_bytes(struct.pack("<i", len(refs)) + bytes(cases))

    out = subprocess.run([str(exe), str(work / "cases.bin")], check=True, capture_output=True, text=True).stdout.splitlines()
    got, i, bad = [], 0, 0
    while i < len(out):
        _, _, count = out[i].split()
        got.append(np.array([[float(v) for v in line.split()] for line in out[i + 1 : i + 1 + int(count)]]).reshape(-1, 6))
        i += 1 + int(count)

    # A fault in the background must leave the detections exactly as without it. The float clean case is the FP32
    # tensor, not the dequantised one, so compare the faulted float case with the clean INT8 case.
    for k in clean_vs_bg:
        for j in (k + 1, k + 2):
            if not np.array_equal(got[k], got[j]):
                print(f"FAULT NOT REMOVED: case {j} ({kinds[j]}) gives {len(got[j])} detections, clean INT8 case {k} gives {len(got[k])}")
                bad += 1
    blank = quantize(np.zeros((meta["nc"], 1024), np.float32), SCORES_Q)
    faulted = dequantize(inject(blank), SCORES_Q)
    no_guard = len(decode_yolo26(faulted, np.ones((4, 1024), np.float32), args.conf, 1.0, 0, 0, w, h, max_det, min_run=0))
    # sanity check: without the guard the injected fault does produce detections
    if not no_guard:
        print("sanity check failed: an injected fault produced no detections with the guard off")
        bad += 1

    total = capped = 0
    worst_box = worst_score = 0.0
    for k, (g, r) in enumerate(zip(got, refs)):
        kind = kinds[k]
        total += len(r)
        capped += len(r) == max_det
        if len(g) != len(r) or (len(r) and (g[:, 5] != r[:, 5]).any()):
            bad += 1
            print(f"MISMATCH case {k} ({kind}): C++ {len(g)} detections vs reference {len(r)}")
            continue
        if len(r):
            worst_box = max(worst_box, float(np.abs(g[:, :4] - r[:, :4]).max()))
            worst_score = max(worst_score, float(np.abs(g[:, 4] - r[:, 4]).max()))
    print(f"{len(imgs)} images x (float, int8, 2 int8 faults, float fault) + 2 random = {len(refs)} cases, {total} detections (cap {max_det} reached on {capped} cases)")
    print(f"max |box diff| = {worst_box:.0f} px, max |score diff| = {worst_score:.2e}, mismatched cases = {bad}")
    ok = bad == 0 and worst_box <= 1 and worst_score <= 1e-6
    print("PASS" if ok else "FAIL")
    shutil.rmtree(work, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
