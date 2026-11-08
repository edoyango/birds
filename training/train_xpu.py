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
    speedups    true (default): xpu_speedups.py (Level Zero launch settings, channels_last, sync-free loss,
                torchvision NMS for validation, MuSGD queued before the inf check, fp16 input/upsample, argmax
                top-k in the assigner); yolo26s b32: 885 -> 332 ms/step with fused_bn (fixed batches).
                false: stock Ultralytics behaviour.
    fused_bn    true (default, with speedups): fused BatchNorm + SiLU Triton kernels (xpu_fused_bn.py). Training
                mode only; eval and export use the stock modules, so exported/RKNN models are unaffected.
                false: stock BatchNorm + SiLU.
    stop_epochs Stop after this many epochs while keeping the LR schedule of `epochs` (for short checks
                against the start of a full run). Ultralytics validates and saves on the stopping epoch.

Defaults: device=xpu, amp=True (fp16). fp64 never reaches the GPU (see xpu_compat.py).
"""
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import xpu_compat  # noqa: E402
import xpu_speedups  # noqa: E402


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
    speedups = args.pop("speedups", True)
    stop_epochs = args.pop("stop_epochs", None)
    fused_bn = args.pop("fused_bn", True)
    if data_root:
        args["data"] = make_data_yaml(data_root, names_yaml)
    if "data" not in args:
        raise SystemExit("need data=<yaml> or data_root=<dir>")
    args.setdefault("device", "xpu")
    args.setdefault("amp", True)
    model_name = args.pop("model", "yolo26n.pt")

    if str(args["device"]).startswith("xpu"):
        if speedups:
            xpu_speedups.set_launch_env(deep_queue=bool(fused_bn))  # before the first XPU call
        xpu_compat.install()  # before ultralytics builds its GradScaler
        if speedups:
            xpu_speedups.install(channels_last=True, sync_free_loss=True, fast_nms=True, fast_musgd=True,
                                 fused_bn=bool(fused_bn), half_glue=True, fast_topk=True)

    from ultralytics import YOLO

    model = YOLO(model_name)
    if stop_epochs:

        def stop_early(trainer):
            if trainer.epoch + 1 >= int(stop_epochs):
                trainer.stop = True

        model.add_callback("on_train_epoch_end", stop_early)
    model.train(**args)


if __name__ == "__main__":
    main()
