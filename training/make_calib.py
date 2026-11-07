#!/usr/bin/env python
"""Build an INT8 calibration set from the training images.

Picks images stratified by class (so rare classes such as Starling are represented), plus a share of background
(empty-label) images. Each image is letterboxed to the model input exactly like the board does at runtime, saved as
PNG, and listed in a text file for `convert_rknn.py --calib`. Pre-letterboxing matters: rknn-toolkit2 would otherwise
stretch the frame to the model input and calibrate on a different distribution from what the NPU sees.

    python training/make_calib.py --data-root /path/to/birds-3-57 --out calib704 --n 200

Runs in either venv (numpy + cv2 + pyyaml).
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from preproc import letterbox


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, help="dataset dir with train/images and train/labels")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True, help="output dir (created); writes <out>/images/*.png and <out>/calib.txt")
    ap.add_argument("--n", type=int, default=200, help="total images")
    ap.add_argument("--background-frac", type=float, default=0.1, help="share of empty-label images")
    ap.add_argument("--imgsz", type=int, nargs=2, default=[704, 704], metavar=("H", "W"), help="model input size")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.data_root)
    img_dir, lbl_dir = root / args.split / "images", root / args.split / "labels"
    rng = random.Random(args.seed)

    by_class, background = {}, []
    for lbl in sorted(lbl_dir.glob("*.txt")):
        classes = {int(float(line.split()[0])) for line in lbl.read_text().splitlines() if line.strip()}
        img = next((p for ext in (".jpg", ".jpeg", ".png") if (p := img_dir / (lbl.stem + ext)).exists()), None)
        if img is None:
            continue
        if not classes:
            background.append(img)
        for c in classes:
            by_class.setdefault(c, []).append(img)

    n_bg = min(int(round(args.n * args.background_frac)), len(background))
    n_fg = args.n - n_bg
    chosen, per_class = {}, {}
    classes = sorted(by_class)
    # round-robin over classes, rarest first, until n_fg images are picked: every class gets ~equal share
    pools = {c: rng.sample(by_class[c], len(by_class[c])) for c in classes}
    order = sorted(classes, key=lambda c: len(by_class[c]))
    while len(chosen) < n_fg and any(pools.values()):
        for c in order:
            while pools[c] and len(chosen) < n_fg:
                p = pools[c].pop()
                if p not in chosen:
                    chosen[p] = c
                    per_class[c] = per_class.get(c, 0) + 1
                    break
    for p in rng.sample(background, n_bg):
        chosen[p] = -1

    out = Path(args.out).resolve()
    (out / "images").mkdir(parents=True, exist_ok=True)
    h, w = args.imgsz
    paths = []
    for p in sorted(chosen):
        bgr = cv2.imread(str(p))
        canvas, *_ = letterbox(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), w, h)
        dst = out / "images" / (p.stem + ".png")
        cv2.imwrite(str(dst), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        paths.append(str(dst))
    (out / "calib.txt").write_text("\n".join(paths) + "\n")
    print(f"wrote {len(paths)} images to {out}/images; list: {out}/calib.txt")
    print("picked per class (an image counts toward the class that selected it):", dict(sorted(per_class.items())))
    print("background images:", n_bg)


if __name__ == "__main__":
    main()
