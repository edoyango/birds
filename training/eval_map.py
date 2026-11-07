#!/usr/bin/env python
"""Score detections written by eval_detections.py against YOLO-format labels (COCO-style mAP, as Ultralytics reports).

    python training/eval_map.py dets-onnx.npz dets-fp16.npz dets-i8.npz --names-from /path/to/birds-3-57/data.yaml

Several files can be given to compare backends side by side. Uses Ultralytics' own `ap_per_class`, so mAP50-95 here is
defined exactly as in `yolo val`. Image labels are found by swapping /images/ for /labels/ in the image path.

Runs in the Ultralytics venv.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics.utils.metrics import ap_per_class, box_iou, smooth

IOUV = torch.linspace(0.5, 0.95, 10)


def load_labels(img_path, w, h):
    lbl = Path(str(img_path).replace("/images/", "/labels/")).with_suffix(".txt")
    if not lbl.exists() or not lbl.read_text().strip():
        return np.zeros((0, 5), np.float32)
    a = np.array([[float(v) for v in line.split()[:5]] for line in lbl.read_text().splitlines() if line.strip()], np.float32)
    cls, xc, yc, bw, bh = a.T
    return np.stack([cls, (xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h], 1)


def match(pred_cls, gt_cls, iou):
    """Greedy one-to-one matching at each IoU threshold, same rule as Ultralytics' match_predictions."""
    correct = np.zeros((len(pred_cls), len(IOUV)), bool)
    iou = iou.numpy() * (gt_cls[:, None] == pred_cls)
    for i, thr in enumerate(IOUV.tolist()):
        m = np.array(np.nonzero(iou >= thr)).T
        if m.shape[0]:
            if m.shape[0] > 1:
                m = m[iou[m[:, 0], m[:, 1]].argsort()[::-1]]
                m = m[np.unique(m[:, 1], return_index=True)[1]]
                m = m[np.unique(m[:, 0], return_index=True)[1]]
            correct[m[:, 1], i] = True
    return correct


def evaluate(npz, names):
    z = np.load(npz, allow_pickle=False)
    dets, files, sizes = z["dets"], z["files"], z["sizes"]
    tp, conf, pcls, tcls = [], [], [], []
    for i, f in enumerate(files):
        w, h = sizes[i]
        gt = load_labels(f, w, h)
        d = dets[dets[:, 0] == i][:, 1:]  # x1 y1 x2 y2 score cls
        tcls.append(gt[:, 0])
        if len(d) == 0:
            continue
        if len(gt):
            iou = box_iou(torch.from_numpy(gt[:, 1:5]), torch.from_numpy(d[:, :4]))
            tp.append(match(d[:, 5], gt[:, 0], iou))
        else:
            tp.append(np.zeros((len(d), len(IOUV)), bool))
        conf.append(d[:, 4])
        pcls.append(d[:, 5])
    tp, conf, pcls, tcls = (np.concatenate(x) if x else np.zeros(0) for x in (tp, conf, pcls, tcls))
    res = ap_per_class(tp, conf, pcls, tcls)
    p, r, ap, classes = res[2], res[3], res[5], res[6]
    f1_curve, conf_axis = res[9], res[10]
    best_conf = float(conf_axis[smooth(f1_curve.mean(0), 0.1).argmax()])  # same rule Ultralytics uses for P and R
    return {
        "images": len(files),
        "labels": int(len(tcls)),
        "P": float(p.mean()),
        "R": float(r.mean()),
        "mAP50": float(ap[:, 0].mean()),
        "mAP50-95": float(ap.mean()),
        "best_conf": best_conf,
        "per_class_mAP50-95": {names[int(c)]: float(ap[j].mean()) for j, c in enumerate(classes)},
        "backend": str(z["backend"]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dets", nargs="+")
    ap.add_argument("--names-from", required=True, help="yaml with a `names:` list (e.g. the dataset's data.yaml)")
    args = ap.parse_args()
    names = yaml.safe_load(Path(args.names_from).read_text())["names"]
    results = {d: evaluate(d, names) for d in args.dets}

    print("P and R are at the score threshold that maximises F1 (best conf), averaged over classes.")
    print(f"{'file':32s} {'backend':8s} {'imgs':>5s} {'P':>6s} {'R':>6s} {'mAP50':>7s} {'mAP50-95':>9s} {'best conf':>10s}")
    for d, r in results.items():
        print(f"{Path(d).name:32s} {r['backend']:8s} {r['images']:5d} {r['P']:6.3f} {r['R']:6.3f} {r['mAP50']:7.3f} {r['mAP50-95']:9.3f} {r['best_conf']:10.2f}")
    print("\nmAP50-95 per class")
    cls_names = sorted({c for r in results.values() for c in r["per_class_mAP50-95"]})
    print(f"{'class':12s}" + "".join(f"{r['backend']:>10s}" for r in results.values()))
    for c in cls_names:
        print(f"{c:12s}" + "".join(f"{r['per_class_mAP50-95'].get(c, float('nan')):10.3f}" for r in results.values()))


if __name__ == "__main__":
    main()
