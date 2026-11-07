"""Pre/post-processing shared by the calibration, evaluation and conversion scripts.

`letterbox` mirrors `convert_image_with_letterbox` in src/utils/image_utils.c (centred, width aligned to a multiple of
4, height to an even number) except that it resizes with cv2 instead of the RGA hardware. The pad value is 114, which is
what Ultralytics trains and validates with. (The old YOLOv5/v7 path in src/yolov5 pads with 0.)

`decode_yolo26` is the reference implementation of src/yolo26/yolo26_postprocess.cc.

Only needs numpy and cv2, so it runs in both the rknn and the ultralytics environments.
"""
import cv2
import numpy as np

PAD_VALUE = 114


def read_image(path, order="rgb"):
    """Read an image file as HxWx3 uint8 in the channel order a model takes ('rgb', or 'bgr' for models exported with
    export_onnx.py --input-order bgr). The letterbox and pad value are the same either way."""
    img = cv2.imread(str(path))  # BGR
    if img is None:
        raise FileNotFoundError(path)
    return img if order == "bgr" else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def letterbox(img_rgb, dst_w, dst_h, pad=PAD_VALUE):
    """Resize `img_rgb` (HxWx3 uint8) into a dst_h x dst_w canvas. Returns (canvas, scale, x_pad, y_pad)."""
    src_h, src_w = img_rgb.shape[:2]
    scale_w = np.float32(dst_w) / np.float32(src_w)
    scale_h = np.float32(dst_h) / np.float32(src_h)
    resize_w, resize_h = dst_w, dst_h
    if scale_w < scale_h:
        scale = scale_w
        resize_h = int(np.float32(src_h) * scale)
    else:
        scale = scale_h
        resize_w = int(np.float32(src_w) * scale)
    resize_w -= resize_w % 4
    resize_h -= resize_h % 2
    x_pad = y_pad = 0
    if scale_w < scale_h:
        y_pad = (dst_h - resize_h) // 2
        y_pad -= y_pad % 2
    else:
        x_pad = (dst_w - resize_w) // 2
        x_pad -= x_pad % 2
    canvas = np.full((dst_h, dst_w, 3), pad, np.uint8)
    canvas[y_pad : y_pad + resize_h, x_pad : x_pad + resize_w] = cv2.resize(
        img_rgb, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR
    )
    return canvas, float(scale), x_pad, y_pad


STRIDES = (8, 16, 32)


def anchor_grid(h, w, strides=STRIDES):
    """Anchor points (2, N) in grid units (cell centres, x then y) and strides (N,), in the order the head concatenates
    its three levels. Identical to the constants baked into the exported graph."""
    pts, st = [], []
    for s in strides:
        gh, gw = h // s, w // s
        ys, xs = np.divmod(np.arange(gh * gw), gw)
        pts.append(np.stack([xs + 0.5, ys + 0.5]).astype(np.float32))
        st.append(np.full(gh * gw, s, np.float32))
    return np.concatenate(pts, 1), np.concatenate(st)


def ltrb_to_boxes(dist, h, w):
    """(4, N) left/top/right/bottom distances in grid units -> (4, N) x1, y1, x2, y2 in input pixels. This is exactly the
    arithmetic the exported graph does after the cut point (Sub, Add, Mul by stride), in float32."""
    pts, stride = anchor_grid(h, w)
    return np.concatenate([pts - dist[:2], pts + dist[2:]], 0) * stride


# Mirrors YOLO26_SUSPECT_RUN in src/yolo26/yolo26_postprocess.h: a class score that sits in a run of this many or more
# identical values along the anchors is ignored. Guards against an RK3588 runtime fault that overwrites 16-element
# chunks of a class plane with the INT8 value 0 (score ~0.5). 0 disables.
SUSPECT_RUN = 8


def constant_runs(scores, min_run=SUSPECT_RUN):
    """(nc, N) bool mask of the scores that sit in a run of at least `min_run` identical consecutive values within
    their class plane."""
    nc, n = scores.shape
    change = np.ones((nc, n), bool)
    change[:, 1:] = scores[:, 1:] != scores[:, :-1]  # column 0 always starts a run, so runs never cross planes
    run_id = np.cumsum(change.ravel()) - 1
    return (np.bincount(run_id)[run_id] >= min_run).reshape(nc, n)


def decode_yolo26(scores, boxes, conf, scale, x_pad, y_pad, model_w, model_h, max_det=300, as_int=False,
                  min_run=SUSPECT_RUN):
    """Decode the two truncated YOLO26 outputs for ONE image.

    scores: (nc, N) class probabilities (sigmoid already applied in the graph)
    boxes:  (4, N) x1, y1, x2, y2 in model-input pixels (decoded in the graph)
    Returns an (K, 6) float array of x1, y1, x2, y2 (original-image pixels), score, class, sorted by score.
    There is no NMS: the model's one-to-one head already emits one box per object, so this is a threshold, a sort
    and a truncation (the same thing the TopK removed from the graph would do). Scores in constant runs (see
    SUSPECT_RUN) are ignored; the anchor falls back to its best remaining class.
    """
    if min_run > 0:
        scores = np.where(constant_runs(scores, min_run), np.float32(-np.inf), scores)
    best = scores.max(0)
    cls = scores.argmax(0)
    keep = np.nonzero(best > conf)[0]
    keep = keep[np.argsort(-best[keep], kind="stable")][:max_det]
    b = boxes[:, keep].astype(np.float32)
    x1 = np.clip(b[0] - x_pad, 0, model_w) / scale
    y1 = np.clip(b[1] - y_pad, 0, model_h) / scale
    x2 = np.clip(b[2] - x_pad, 0, model_w) / scale
    y2 = np.clip(b[3] - y_pad, 0, model_h) / scale
    out = np.stack([x1, y1, x2, y2, best[keep], cls[keep]], 1).astype(np.float32)
    if as_int:  # the C++ stores integer pixel coordinates, rounded to the nearest pixel
        out[:, :4] = np.floor(out[:, :4] + np.float32(0.5))
    return out
