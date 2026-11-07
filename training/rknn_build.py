"""Build an RKNN model from the ONNX exported by export_onnx.py. Shared by convert_rknn.py and eval_detections.py.

Needs the rknn venv (Python 3.12, rknn-toolkit2).
"""
import json
from pathlib import Path

from rknn.api import RKNN

# No fusion rules are disabled by default. export_onnx.py exports YOLO26's attention as softmax((q^T k) * scale), which
# the toolkit fuses into an NPU SDPA op. Ultralytics' own arrangement, softmax((q * scale)^T k) (export_onnx.py
# --keep-q-scaling), crashes rknn-toolkit2 2.5.0 in the SDPA fusion rule (IndexError in rules/matmul.py). Disabling that
# rule gets past the crash but leaves unfused Softmax ops that fail to run on an RKNPU 0.9.6 driver:
#   failed to submit! ... exSoftmax13 ...
DEFAULT_DISABLED_RULES = []


def input_order(meta):
    """'rgb' or 'bgr': the channel order the exported model takes (exports from before the option are RGB)."""
    return meta.get("input_order", "rgb")


def load_meta(onnx_path, meta_path=None):
    return json.loads(Path(meta_path or Path(onnx_path).with_suffix(".outputs.json")).read_text())


def build(
    onnx_path,
    meta,
    dtype="i8",
    calib=None,
    batch=1,
    platform="rk3588",
    algo="normal",
    disable_rules=None,
    verbose=False,
    cut="ltrb",
):
    """Return a built (not yet exported or released) RKNN object. Output order is (scores, dist) for cut="ltrb" (the
    default and the only one src/yolo26 decodes), where dist is the raw left/top/right/bottom distance in grid units, or
    (scores, boxes) for cut="boxes" (decoded pixel coordinates; INT8 resolves them only ~3.8 px at 704x704, kept for
    comparison experiments)."""
    if dtype == "i8" and not calib:
        raise ValueError("a calibration list is required for dtype='i8'")
    rknn = RKNN(verbose=verbose)
    order = input_order(meta)
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        # The toolkit loads calibration images (normal image files) as RGB; a BGR model must be calibrated on BGR.
        quant_img_RGB2BGR=order == "bgr",
        # src/yolo26 reads this at run time to decide whether the letterbox writes RGB or BGR.
        custom_string=f"input_order={order}",
        target_platform=platform,
        quantized_algorithm=algo,
        disable_rules=DEFAULT_DISABLED_RULES if disable_rules is None else disable_rules,
    )
    ret = rknn.load_onnx(model=str(onnx_path), outputs=[meta["scores"], meta["boxes" if cut == "boxes" else "dist"]])
    if ret != 0:
        raise RuntimeError(f"load_onnx failed ({ret})")
    ret = rknn.build(do_quantization=dtype == "i8", dataset=calib, rknn_batch_size=batch)
    if ret != 0:
        raise RuntimeError(f"build failed ({ret})")
    return rknn
