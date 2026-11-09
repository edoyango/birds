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
                top-k in the assigner, host-side validation metrics, one-kernel weight casts, Triton concat and
                split, reordered attention); yolo26s b32: 885 -> 301 ms/step with fused_bn (fixed batches),
                validation 15 -> 7 s. false: stock Ultralytics behaviour.
    fused_bn    true (default, with speedups): fused BatchNorm + SiLU Triton kernels (xpu_fused_bn.py). Training
                mode only; eval and export use the stock modules, so exported/RKNN models are unaffected.
                false: stock BatchNorm + SiLU.
    speedups_off  Comma-separated xpu_speedups patches to leave off, e.g. speedups_off=fast_cat,fast_attn (names
                as in xpu_speedups.install(); for A/B checks).
    stop_epochs Stop after this many epochs while keeping the LR schedule of `epochs` (for short checks
                against the start of a full run). Ultralytics validates and saves on the stopping epoch.
    val_workers Dataloader workers for validation (default: the training `workers`). Stock Ultralytics uses
                2x `workers`, capped at the CPU count.
    prefetch    Batches each training worker keeps ready (default 2; Ultralytics 4). val_prefetch: the same for
                validation workers (default 1; Ultralytics 2). At 640 px / batch 64 each training batch is ~79 MB of
                shared memory, so Ultralytics' 6 x 4 is ~1.9 GB.
    val_keep_workers  false (default): shut the validation workers down after each validation and fork new ones
                at the next. true: Ultralytics' behaviour, workers and their next batches stay for the whole run
                (~4 GB at 6 workers, batch 128).
    val_period  Validate only every this many epochs (epochs 5, 10, ... for 5). Ultralytics still validates the
                final epoch and any epoch where training stops. Skipped epochs have nan validation metrics in
                results.csv and never become best.pt. Pass it again when resuming.

Defaults: device=xpu, amp=True (fp16), workers=6 (training and validation). fp64 never reaches the GPU (see
xpu_compat.py). On this 15 GB machine, workers=8 (8 training + 12 validation processes, each a fork of a ~6 GB
main process) ran out of memory at epoch 91 of a 120-epoch run.
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


def set_loaders(val_workers, prefetch, val_prefetch, val_keep):
    """Dataloader settings for the detection trainer that stock Ultralytics hard-codes.

    val_workers: validation workers (Ultralytics: 2x `workers`). prefetch / val_prefetch: batches each training /
    validation worker keeps ready (Ultralytics: 4 / 2). val_keep=False shuts the validation workers down after each
    validation (Ultralytics keeps them, with their next batches, for the whole run).
    """
    import ultralytics.models.yolo.detect.train as detect_train
    from ultralytics.engine.trainer import BaseTrainer

    build = detect_train.build_dataloader

    def build_dataloader(dataset, *a, **kw):
        val = not dataset.augment  # build_yolo_dataset sets augment only for mode="train"
        if val:
            kw["workers"] = val_workers
        loader = build(dataset, *a, **kw)
        if loader.num_workers > 0:  # read when the workers start (first iteration), so it can be set here
            loader.prefetch_factor = val_prefetch if val else prefetch
        return loader

    detect_train.build_dataloader = build_dataloader

    if not val_keep:
        validate = BaseTrainer.validate

        def validate_and_release(self):
            try:
                return validate(self)
            finally:
                if hasattr(self.test_loader, "reset"):
                    self.test_loader.reset()  # join the workers; the next validation forks new ones

        BaseTrainer.validate = validate_and_release


def validate_every(n):
    """on_train_epoch_end callback: let Ultralytics validate only after every n-th epoch."""

    def callback(trainer):
        trainer.args.val = (trainer.epoch + 1) % n == 0
        if not trainer.args.val:  # validation is still forced on the final epoch or a stop, which overwrites these
            trainer.metrics = {k: float("nan") for k in trainer.metrics}  # not the previous validation's numbers
            trainer.fitness = None  # otherwise save_model copies this unvalidated epoch to best.pt

    return callback


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
    val_period = int(args.pop("val_period", 1))
    off = args.pop("speedups_off", None)
    off = [k for k in str(off).split(",") if k] if off else []
    if data_root:
        args["data"] = make_data_yaml(data_root, names_yaml)
    if "data" not in args:
        raise SystemExit("need data=<yaml> or data_root=<dir>")
    args.setdefault("device", "xpu")
    args.setdefault("amp", True)
    args.setdefault("workers", 6)  # also passed on resume=True, so it replaces the checkpoint's value
    val_workers = args.pop("val_workers", args["workers"])
    prefetch = int(args.pop("prefetch", 2))
    val_prefetch = int(args.pop("val_prefetch", 1))
    val_keep = bool(args.pop("val_keep_workers", False))
    model_name = args.pop("model", "yolo26n.pt")

    if str(args["device"]).startswith("xpu"):
        if speedups:
            xpu_speedups.set_launch_env(deep_queue=bool(fused_bn))  # before the first XPU call
        xpu_compat.install()  # before ultralytics builds its GradScaler
        if speedups:
            flags = dict(channels_last=True, sync_free_loss=True, fast_nms=True, fast_musgd=True,
                         fused_bn=bool(fused_bn), half_glue=True, fast_topk=True, fast_val=True, fast_cast=True,
                         fast_cat=True, fast_attn=True)
            unknown = [k for k in off if k not in flags]
            if unknown:
                raise SystemExit(f"speedups_off: unknown patch {unknown}; known: {list(flags)}")
            flags.update({k: False for k in off})
            xpu_speedups.install(**flags)

    from ultralytics import YOLO

    set_loaders(int(val_workers), prefetch, val_prefetch, val_keep)
    model = YOLO(model_name)
    if stop_epochs:

        def stop_early(trainer):
            if trainer.epoch + 1 >= int(stop_epochs):
                trainer.stop = True

        model.add_callback("on_train_epoch_end", stop_early)
    if val_period > 1:
        model.add_callback("on_train_epoch_end", validate_every(val_period))
    model.train(**args)


if __name__ == "__main__":
    main()
