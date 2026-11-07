#!/usr/bin/env python
"""Train an Ultralytics model (YOLO26 by default) on an Intel Arc GPU.

Usage (key=value pairs mirror the `yolo train` CLI; see Ultralytics' cfg docs):

    python training/train_xpu.py model=yolo26n.pt data_root=/path/to/birds-3-57 \
        epochs=120 imgsz=704 batch=32 name=yolo26n_704

    python training/train_xpu.py cfg=cfg.yaml data_root=... model=yolo26s.pt

Extra keys handled here (not passed to Ultralytics):
    cfg         YAML of Ultralytics train args (e.g. the tuned cfg.yaml); other key=value pairs override it.
    data_root   Roboflow-style dataset dir with train/images, valid/images and data.yaml.
                A dataset yaml with absolute paths is generated for you.
    names_yaml  Take class names from this yaml instead of <data_root>/data.yaml.

Defaults: device=xpu, amp=True (fp16). fp64 never reaches the GPU (see xpu_compat.py).
"""
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import xpu_compat  # noqa: E402


def parse_args(argv):
    args = {}
    for tok in argv:
        if "=" not in tok:
            raise SystemExit(f"expected key=value, got {tok!r}")
        k, v = tok.split("=", 1)
        args[k] = yaml.safe_load(v)
    return args


def make_data_yaml(root, names_yaml=None):
    root = Path(root).resolve()
    names_src = Path(names_yaml) if names_yaml else root / "data.yaml"
    names = yaml.safe_load(names_src.read_text())["names"]
    data = {"path": str(root), "train": "train/images", "val": "valid/images", "names": names}
    if (root / "test" / "images").is_dir():
        data["test"] = "test/images"
    out = Path(tempfile.mkdtemp(prefix="birds-data-")) / "data.yaml"
    out.write_text(yaml.safe_dump(data))
    return str(out)


def main():
    args = parse_args(sys.argv[1:])
    cfg = args.pop("cfg", None)
    if cfg:
        base = yaml.safe_load(Path(cfg).read_text())
        base.update(args)
        args = base
    data_root = args.pop("data_root", None)
    names_yaml = args.pop("names_yaml", None)
    if data_root:
        args["data"] = make_data_yaml(data_root, names_yaml)
    if "data" not in args:
        raise SystemExit("need data=<yaml> or data_root=<dir>")
    args.setdefault("device", "xpu")
    args.setdefault("amp", True)
    model_name = args.pop("model", "yolo26n.pt")

    if str(args["device"]).startswith("xpu"):
        xpu_compat.install()  # before ultralytics builds its GradScaler

    from ultralytics import YOLO

    YOLO(model_name).train(**args)


if __name__ == "__main__":
    main()
