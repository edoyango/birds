# YOLO26 training and RKNN deployment

Trains a YOLO26 detector on an Intel Arc GPU (A770, no FP64), converts it for the RK3588 NPU (INT8), and runs it
through the `birdector` inference code. It replaced the YOLOv7 model the project ran before.

```
train_xpu.py ──► best.pt ──► export_onnx.py ──► .onnx ──► convert_rknn.py ──► .rknn ──► board (src/yolo26 + birdector)
 (XPU, fp16)               (NMS-free head,     + .json    (rknn-toolkit2 2.5.0,        (batch size baked in)
                            attention patch)              INT8, ltrb cut)
```

## Why YOLO26

Its one-to-one head emits one box per object, so there is **no NMS**. On the board the CPU does a threshold, a partial
sort and the letterbox undo, about 3 ms per 24-frame batch. Everything else in the network runs on the NPU.

Measured on the Orange Pi 5 Pro against the YOLOv7 model it replaced (details in [Deploy](#5-deploy)):

| | YOLOv7 864x864 INT8 | YOLO26s 768x448 INT8 |
|---|---|---|
| wall time per 24-frame batch | 1341 ms | 350-400 ms (3.4-3.8x faster) |
| process CPU per 24-frame batch | 172 ms | 85 ms (about half) |

The CPU work does not need several cores: averaged over a batch it keeps about a quarter of a core busy.

## Environments

Two Python 3.12 virtualenvs, because rknn-toolkit2 2.5.0 supports Python up to 3.12 and pins `torch<=2.4.0`,
`numpy<=1.26.4` while the Arc GPU needs a recent `torch+xpu`.

| env | file | used for |
|---|---|---|
| `.venv-ultra` | `requirements-ultralytics.txt` | training, ONNX export, `eval_map.py`, `check_*.py` |
| `.venv-rknn` | `requirements-rknn.txt` | `convert_rknn.py`, `eval_detections.py`, `make_calib.py` |

The host needs the Intel compute runtime for the GPU (Ubuntu: `libze-intel-gpu1 libze1 intel-opencl-icd`).

## 1. Train (Arc GPU)

```
python training/train_xpu.py cfg=training/cfg-yolo26.yaml data_root=/path/to/birds-3-57 \
    epochs=120 project=runs/detect name=yolo26s_704 val=False
```

All numbers in this README come from a 25-epoch run of this command (`epochs=25`); a longer run has not been measured.

- `device=xpu`, fp16 AMP. `torch.amp.GradScaler` uses float64 internally, which the A770 does not have, so
  `xpu_compat.py` swaps in a scaler that does the same arithmetic in float32 (the loss scale is a power of two, so the
  float32 reciprocal is exact). Ultralytics itself needed no other change.
- `deterministic: false` in `cfg-yolo26.yaml`. The Ultralytics default is about 40% slower per step on this GPU.
- `val=False` is worth passing. Ultralytics validates with the NMS head; on this machine that took 3.3 min per epoch
  (about as long as training one) and logged `NMS time limit exceeded` on nearly every batch, which truncates NMS and so
  makes the reported mAP unreliable. I did not isolate whether the XPU NMS or the early-training candidate counts were to
  blame. Evaluate with `eval_detections.py` + `eval_map.py` instead (below).
- Speed: YOLO26s, 704 px, batch 32: about 4.2 min per epoch (9650 images), 10.4 GB of GPU memory.
- On an identical batch, the fp32 loss on the XPU matches the CPU to 1e-6 (relative) and the gradient norm to 1e-5. fp16
  deviates from fp32 by a similar amount on the CPU as on the XPU, so that gap comes from fp16, not from the GPU.

## 2. Pick an input size, export to ONNX

```
python training/export_onnx.py runs/detect/yolo26s_704/weights/best.pt --imgsz 448 768   # H W
```

- Exports with `nms=False` (the one-to-one head). Recent Ultralytics exports the NMS head by default, so Rockchip's
  `yolo export ... format=onnx` for YOLO26 no longer gives an NMS-free graph.
- Patches `Attention.forward` to compute `softmax((q^T k) * scale)` instead of Ultralytics' `softmax((q * scale)^T k)`.
  The results are the same (0.0 difference over 8 images on the s model; the scale is an exact power of two there,
  so on other sizes expect float rounding), but rknn-toolkit2 2.5.0 crashes in its SDPA fusion rule on the second
  form, and the workaround of disabling that rule produces a model whose softmax fails on the NPU
  (`failed to submit! ... exSoftmax13`, RKNPU driver 0.9.6). With the patched export the toolkit fuses attention to an
  NPU `exSDPAttention` op and the model runs. `--keep-q-scaling` turns the patch off.
- The model is fully convolutional, so a model trained at 704x704 can be exported at another size.
- `--input-order bgr` reverses the input channels of the first conv (the only layer that sees the image), so the model
  takes OpenCV's BGR frames directly. The json and the `.rknn` record the order (`custom_string="input_order=bgr"`);
  conversion then calibrates on BGR (`quant_img_RGB2BGR`), and the eval scripts and `src/yolo26` feed BGR. Untagged
  models are treated as RGB. See [Colour order](#colour-order) for whether it is worth it.
- The metadata json records the two tensors the converter cuts at, found by graph structure.

### Input size: 768x448 for the 864x486 camera frames

FP32 (ONNX) accuracy of the same 704-trained checkpoint, full validation set (2164 images):

| input | anchors N | mAP50 | mAP50-95 | letterbox from 864x486 |
|---|---|---|---|---|
| 704x704 | 10164 | 0.954 | 0.752 | downscale 0.81, 44% of the canvas is padding |
| **768x448** | **7056** | **0.952** | **0.748** | **downscale 0.89, 4% padding** |
| 640x640 | 8400 | 0.951 | 0.745 | downscale 0.74 |
| 896x512 | 9408 | 0.949 | 0.742 | upscale 1.04 |
| 1024x576 | 12096 | 0.944 | 0.727 | upscale 1.19 |

768x448 is within 0.4 points of 704x704 with 31% fewer pixels, and it has two properties that matter on the board:

- **It downscales.** RGA, which does the letterbox resize on the board, shifted the image content by about one pixel
  when *upscaling* (checked by comparing the RGA output with a cv2 resize) and not when downscaling. At 896x512 the
  board scored 0.714 mAP50-95 on 720 frames against 0.753 for FP32, a 4-point gap of which INT8 explains about 1.5. At
  768x448 the board matches the INT8 simulator (0.754 vs 0.753, table below).
- **Its anchor count is a multiple of 16** (N = (H/8)(W/8) + (H/16)(W/16) + (H/32)(W/32)). On the Orange Pi 5 Pro a
  704x704 nano model (N = 10164) returned unstable values in a few box outputs at the row edges on real frames, in 3 of 3
  runs, while 640x640, 896x512 and 768x448 models did not. This is an observation on this board, not a documented rule.
  `export_onnx.py` warns when N is not a multiple of 16.

## 3. Calibration set and conversion

```
python training/make_calib.py --data-root /path/to/birds-3-57 --out calib448x768 --n 200 --imgsz 448 768
python training/convert_rknn.py model.onnx --dtype i8 --calib calib448x768/calib.txt --batch 24 --out yolo26s-768x448-i8-bs24.rknn
```

- The calibration images are class-stratified (about equal per class, 10% backgrounds) and letterboxed exactly as the
  board does, so the toolkit calibrates on what the NPU will see. The old 25-image list had 5 non-bird images.
- `--batch` is baked into the model and must equal the batch size the inference container uses (24 in
  `docker-compose.yml`).
- **The graph is cut at the raw left/top/right/bottom distances (grid units), not at decoded boxes.** Rockchip's YOLO26
  example cuts after the anchor/stride arithmetic, where box values span the whole image and INT8 resolves them to a few
  pixels per step. On the bird data at 768x448 that cut gives mAP50-95 0.593 in INT8; cutting at the distances gives
  0.732 (FP32 0.748). The distances span about -0.3 to 9 grid units, which 8 bits resolve to about 0.4 px at stride 8.
  The CPU applies the anchor and stride only to the few detections that survive the threshold. `--cut boxes` exists for
  comparison experiments; `src/yolo26` decodes the `ltrb` cut only.
- FP16 (`--dtype fp`) is as accurate as FP32 but not usable on this board: see the timing table.

## 4. Verify

| script | checks |
|---|---|
| `check_decode.py` | the cut graph plus `preproc.py` decode reproduces the model's own end-to-end `(1,300,6)` output. All detections match with identical scores; the only differences are second-best-class duplicates of the same box, which this decode (one class per anchor, like Rockchip's example) does not emit. |
| `check_cpp_decode.py` | `src/yolo26` decodes identically to the Python reference, float and INT8, 0 px difference, including the 128-detection cap and injected output faults (see [the guard](#runtime-output-fault-and-the-guard)). Needs a C++17 compiler (`CXX`, or `pip install ziglang` and `--cxx "python -m ziglang c++"`). |
| `eval_detections.py` + `eval_map.py` | mAP of the ONNX FP32 reference vs the toolkit's FP16 and INT8 simulators on the validation set, with identical preprocessing and decode. `eval_map.py` prints the F1-optimal score threshold. |

```
python training/eval_detections.py m.onnx --backend onnx --data-root /path/to/birds-3-57 --out d-fp32.npz
python training/eval_detections.py m.onnx --backend i8 --calib calib448x768/calib.txt --data-root ... --out d-i8.npz
python training/eval_map.py d-fp32.npz d-i8.npz --names-from /path/to/birds-3-57/data.yaml
```

The simulator emulates the NPU numerics on x86. It is not bit-exact with the board, and it can only run a model built in
the same process, which is why `eval_detections.py` builds instead of loading an `.rknn`.

### Accuracy, 768x448, yolo26s trained 25 epochs at 704 (`runs/detect/yolo26s_704_3-57_25`)

| | images | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| FP32 ONNX | 2164 (all) | 0.952 | 0.911 | 0.952 | 0.748 |
| FP16 simulator | 2164 | 0.952 | 0.909 | 0.952 | 0.748 |
| INT8 simulator, ltrb cut | 2164 | 0.932 | 0.914 | 0.948 | 0.732 |
| INT8 simulator, boxes cut (Rockchip's) | 2164 | 0.930 | 0.912 | 0.944 | 0.593 |
| FP32 ONNX | 720 | 0.960 | 0.931 | 0.969 | 0.768 |
| INT8 simulator, ltrb cut | 720 | 0.946 | 0.934 | 0.965 | 0.753 |
| **board** (RGA letterbox, NPU, C++ decode, BGR frames in) | 720 | 0.943 | 0.927 | 0.963 | **0.754** |

The 720 frames are evenly spaced through the validation set (the board run needs a multiple of 24 and a copy of the
images on the board). P and R are at each row's F1-optimal threshold.

## 5. Deploy

- `Dockerfile` pins `librknnrt.so` to `RKNN_RUNTIME_VERSION=2.5.0`. Keep it equal to the toolkit that converted the
  model.
- `docker-compose.yml` mounts `models/yolo26s-768x448-i8-bs24.rknn` (stored with Git LFS) at `/app/models/birbs.rknn`.
  To deploy another model, copy it into `models/` and change the mount; the batch size it was converted with must match
  the last argument of the service's `command`.
- `--conf` / `BIRDS_CONF` default to 0.25. Pooled over all classes on the 720 board frames (1083 labelled birds):

  | conf | precision | recall | F1 | false detections per frame |
  |---|---|---|---|---|
  | 0.15 | 0.926 | 0.928 | 0.927 | 0.111 |
  | 0.20 | 0.940 | 0.918 | 0.929 | 0.087 |
  | **0.25** | **0.952** | **0.905** | **0.928** | **0.068** |
  | 0.30 | 0.962 | 0.900 | 0.930 | 0.053 |
  | 0.40 | 0.969 | 0.875 | 0.920 | 0.042 |
  | 0.50 | 0.983 | 0.838 | 0.905 | 0.022 |

  INT8 scores run lower than FP32 (FP32's F1 peaks near 0.4), so a threshold tuned on the ONNX model is too high for
  the board.
- Letterbox padding is 114, as in training.
- Box coordinates are rounded to the nearest pixel. Against unrounded FP32 boxes, truncating to integers cost 0.6 points
  of mAP50-95 and rounding 0.3.
- Set `YOLO26_TIMING=1` to print per-stage wall and process-CPU milliseconds, and the guard count, for every batch.
- `YOLO26_INT8_RAW=1` reads the raw INT8 outputs and decodes on the integers. On 768x448 it gave the same detections
  as the default float path, but no measurable CPU saving, so there is no reason to turn it on.

### Colour order

YOLO26 is trained on RGB and `cv2.VideoCapture` yields BGR; feeding BGR cut mAP50 from 0.88 to 0.45 (300 validation
images, early checkpoint). `birdector.detector.model.inference` passes the frames as `IMAGE_FORMAT_BGR888`
and the RGA letterbox writes the model input as RGB888, so the swap happens in the same hardware pass as the resize. The
caller's frames are not modified, so drawing still uses them as BGR.

- On the Pi the canvases are bit-identical to swapping with `cv2.cvtColor` first and then letterboxing (24 of 24 real
  frames), and the 720-frame board evaluation gives identical detections. None of the letterbox calls fell back to the
  CPU.
- Swapping in Python instead cost 40-120 ms of process CPU per 24-frame batch (`cv2.cvtColor` spreads it over several
  threads; a numpy channel flip was worse).
- The CPU fallback of the letterbox (`convert_image_cpu`, used when a width is not a multiple of 16 or RGA fails) also
  swaps, and gives the same canvases.
- **Reordering the model instead (`export_onnx.py --input-order bgr`) measured no faster.** The letterbox still runs on
  RGA (with the same CPU fallback); only the conversion inside it goes away, and that conversion is free: 57.3 vs 57.5 ms
  median wall and 33.8 vs 34.7 ms CPU per 24 frames, 30 interleaved repeats. Whole batches: 345-396 ms (RGB model) vs
  352-403 ms (BGR model), CPU 85-113 vs 85-87 ms. Accuracy is the same within quantization noise: FP32 0.748 vs 0.748
  (identical metrics), INT8 simulator 0.732 vs 0.730, board 0.754 vs 0.753 (720 frames). Both kinds of model work with
  the same library, so either can be deployed.

### Measured on an Orange Pi 5 Pro

RK3588, kernel 6.1.43, NPU driver 0.9.6. 24 real 864x486 frames per batch, NPU cores 0,1,2, through the Python binding
(`bench.py`, 10 iterations, inside the production image).

| model | runtime | wall per batch | per frame | process CPU per batch |
|---|---|---|---|---|
| previous YOLOv7, 864x864, INT8 | 2.3.2 | 1341 ms | 55.9 ms | 172 ms |
| YOLO26s, 768x448, INT8 | 2.5.0 | 350-400 ms | 14.7-16.8 ms | 83-87 ms |
| YOLO26s, 768x448, FP16 | 2.5.0 | 928 ms | 38.7 ms | 716 ms |

Stages of the YOLO26s INT8 batch, wall/CPU ms: letterbox (RGA) 44/33, inputs_set 3/3, NPU run 250-330/18-28 (the
runtime waits without spinning a core), outputs to float 30/30, decode 3/3. Before the BGR to RGB swap moved into the
RGA letterbox it cost another 40 ms of CPU per batch (127 ms in total).

One run with the host's own `librknnrt` 2.3.0 instead of 2.5.0 spent 115-175 ms of CPU inside `rknn_run` per batch (vs
18-28), so use the pinned runtime.

## Runtime output fault, and the guard

On this board the INT8 YOLO26 models sporadically return a **16-element chunk of one class plane of the scores output
set to the raw value 0** instead of the background value -128, in some slots of a batch. 0 dequantises to a score of
0.49-0.50, so each chunk becomes a row of up to 16 false detections of one class (for example 32 boxes at 0.49
on a frame where the clean slots have none). The same frame in the other 23 slots, or in the next run, is clean.

- Seen with librknnrt 2.3.0 (host), 2.3.2 (production image) and 2.5.0, on all three cores and on a single core. The
  production YOLOv7 model on 2.3.2 showed no differences between slots in the same
  test. I could not test another NPU driver (0.9.6 here; updating it needs root). The root cause is unknown; it is worth
  reporting to Rockchip (airockchip/rknn-toolkit2 issues) with `slot_test`-style evidence.
- **Guard:** the decoder ignores a class score that sits in a run of 8 or more identical values along consecutive
  anchors, and the anchor falls back to its next best class (`YOLO26_SUSPECT_RUN` in `src/yolo26/yolo26_postprocess.h`,
  `SUSPECT_RUN` in `preproc.py`). A real detection never looks like this: the one-to-one head fires on one anchor per
  object, and on the 2164 validation images the guard fired 0 times on the FP32 outputs (even at conf 0.001) and
  0 times on the INT8 simulator's.
- Effect on the board (24 copies of each of 8 frames per batch, 3 runs): before, some slots differed (up to 32 extra
  detections); with the guard all 24 slots agree in every run (9 runs: runtimes 2.3.0 and 2.5.0, float and raw INT8 decode), while
  the guard drops 0-176 bad scores per run. On the
  720-frame evaluation, the guarded run is identical to the unguarded one on every frame except the one that had 16
  false detections, and precision at the F1-optimal threshold rises from 0.931 to 0.943.
- The library prints a one-time notice when the guard first fires; `YOLO26_TIMING=1` prints a per-batch count.
- The guard costs nothing measurable (decode stays at about 3 ms per batch). It cannot recover a real score that the
  fault overwrote, which would need the whole chunk to land on the one anchor of a real bird.

## Things worth knowing

- **Letterbox padding is filled on the CPU, padding only.** RGA's `imfill` fails with `Invalid argument` on this Pi
  (dmesg: `rga_policy: invalid function policy`). On the RK3588 colour fill only runs on the RGA2 core, which cannot
  address memory above 4 GiB, and on an 8 GB board `malloc`'d canvases land above it: the fill worked on a buffer from
  the low-memory `dma_heap/reserved` (0xea700000) and failed on `malloc`'d ones at 0x1858dc000, in RGB and RGBA alike.
  The resize runs on the RGA3 cores, which reach that memory. So `convert_image_rga` no longer calls `imfill` for
  packed formats it can write: after the RGA resize it `memset`s only the padding bands. This is byte-identical to the
  old whole-canvas fill (12 of 12 frames for 768x448, 864x864, a pillarbox and the CPU path; same detections for the
  YOLOv7 and YOLO26 models), removes the librga error spam, and costs 5 ms less CPU per 24 frames (30 vs 35 ms for the
  letterbox). `imfill` is still used for YUV and for buffers the CPU cannot reach.
- `src/utils/image_utils.c` warns on stderr whenever the letterbox falls back to the CPU: a width that is not a
  multiple of 16 (RGA is not tried), an RGA error (the frame is resized on the CPU), or a failed `imfill` (only YUV or
  fd-only buffers now). Each warning repeats on its 10th, 100th, 1000th, ... occurrence with the count, so it stays
  visible without flooding the log. The padding fill above is by design and does not warn.
- Nothing here was run on a board other than the Orange Pi 5 Pro.
