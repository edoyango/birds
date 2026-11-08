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
    epochs=80 project=runs/detect name=yolo26s_704 val=False
```

Most numbers in this README come from a 25-epoch run of this command (`epochs=25`); the 120-epoch run is in
[INT8 after long training](#int8-after-long-training-the-head-clips). 80 epochs is enough: validated every epoch, the
120-epoch run (`runs/detect/yolo26s_704_3-57_120_fast`) levelled out at mAP50-95 0.774-0.780 from about epoch 60 (peak
0.780 at epoch 73, 0.774 at 80, 0.772 at 120). Keep `epochs` at 67 or more: with `optimizer=auto` (the default)
Ultralytics picks MuSGD (lr 0.01) only above 10000 optimizer iterations (151 per epoch at 64 images per step), and
AdamW below that. Ultralytics validates after the last epoch even
with `val=False`. On this 15 GB machine `workers=8` ran out of memory: stock Ultralytics gives validation twice the
training workers (8 + 12 processes). `train_xpu.py` and `cfg-yolo26.yaml` now use 6 + 6 (`workers`, `val_workers`);
resume a killed run with `resume=True`, which passes the same worker counts.

- `device=xpu`, fp16 AMP. `torch.amp.GradScaler` uses float64 internally, which the A770 does not have, so
  `xpu_compat.py` swaps in a scaler that does the same arithmetic in float32 (the loss scale is a power of two, so the
  float32 reciprocal is exact). Ultralytics itself needed no other change.
- `deterministic: false` in `cfg-yolo26.yaml`. The Ultralytics default is about 40% slower per step on this GPU.
- Validation: stock Ultralytics sends XPU boxes through a Python NMS loop (~80 ms per image) that trips its own time
  limit (`NMS time limit exceeded`) and silently drops detections, so mAP it reports on XPU is wrong (the final eval
  of one run said mAP50-95 0.408 for weights that score 0.644). `train_xpu.py` fixes this by default (see
  [Training speed](#training-speed-xpu_speedupspy)): validation then takes ~7 s and `val=True` is affordable. Runs
  made with `speedups=false`, or before this change, should pass `val=False` and be evaluated with
  `eval_detections.py` + `eval_map.py` (below).
- Speed: YOLO26s, 704 px, batch 32, 9650 images: about 1.7 min per epoch with the default speedups (~93 s of
  training + ~7 s of validation; 4.6 min with `speedups=false`), ~7 GB of GPU memory.
- On an identical batch, the fp32 loss on the XPU matches the CPU to 1e-6 (relative) and the gradient norm to 1e-5. fp16
  deviates from fp32 by a similar amount on the CPU as on the XPU, so that gap comes from fp16, not from the GPU.

### Next training run: 640 px, batch 64

Decided 2026-11-09, not run yet: train the next model at 640 and move production to 640x384 (see
[Input size](#input-size-640x384-was-768x448)). At 640 px, batch 64 fits on the A770 and is the fastest setting
measured:

```
PYTORCH_ALLOC_CONF=expandable_segments:True UR_L0_DISABLE_USM_ALLOCATOR=1 \
python training/train_xpu.py cfg=training/cfg-yolo26.yaml data_root=/path/to/birds-3-57 \
    epochs=80 imgsz=640 batch=64 project=runs/detect name=yolo26s_640
```

| yolo26s, `bench_step.py` fixed batches, all speedups | images/s | GPU memory held by the process |
|---|---|---|
| 704 px, batch 32 (the recipe so far) | 108 | 9.6 GiB |
| 640 px, batch 32 | 125 | |
| **640 px, batch 64** | **139-142** | **14.1 GiB with the two settings below, 15.5 without** |
| 672 px, batch 64, settings on, desktop off | 127 | 15.6 GiB |
| 704 px, batch 64, settings on, desktop off | 91 (spills to host RAM) | 16.3 GiB |

- About 30% faster than the 704 recipe: the training part of an epoch should drop from ~90 s to ~70 s.
- Once the process holds more than about 15.5-15.8 GiB, the i915 driver moves GPU buffers out to host RAM and back
  every step, and the step slows by 2-3x (see the batch size note below). The process holds more than PyTorch reports:
  on top of PyTorch's reserve, Level Zero keeps ~2.7 GiB, mostly a pool of command lists. Per-process figures are in
  `/proc/<pid>/fdinfo/*` (`drm-total-local0`, `drm-resident-local0`).
- `PYTORCH_ALLOC_CONF=expandable_segments:True` lets PyTorch's caching allocator grow segments instead of keeping
  rounded-up blocks (0.5-1 GiB less reserved). `UR_L0_DISABLE_USM_ALLOCATOR=1` turns off Level Zero's own memory pool
  under PyTorch's (~0.7 GiB). Neither measured slower (704 px, batch 32: 107 vs 108 images/s). Set them in the
  environment before Python starts. Together they give 640 px / batch 64 about 1.4 GiB of margin.
- Keep the `UR_L0_COMMANDLISTS_CLEANUP_THRESHOLD=-1` that `train_xpu.py` sets: a threshold of 20 frees another ~1 GiB,
  but every step is ~6% slower. A GPU synchronize in the middle of the step does not shrink the command-list pool.
- The GNOME desktop on the A770's HDMI output holds ~0.4 GiB (more with a browser open). 640 px / batch 64 fit with it
  running; stopping it (`sudo systemctl stop gdm3`) adds margin.
- Recipe: with Ultralytics' `nbs=64` an optimizer step is still 64 images (64 x 1 instead of 32 x 2 accumulated), with
  the same weight decay and MuSGD lr. BatchNorm averages over 64 images instead of 32, and validation runs at batch
  128 (twice the training batch).
- Not checked yet: validation at batch 128, the real data loader, and memory over a whole run. Start with
  `stop_epochs=3 val=True` and watch `grep Shmem /proc/meminfo`: it stays flat when everything fits, and swings by
  ~1 GB every step when the driver is moving memory.
- Then the board files at 640x384: `export_onnx.py ... --imgsz 384 640`, `make_calib.py ... --imgsz 384 640`
  (`calib384x640`), `convert_rknn.py`. The 704-trained 120-epoch model's 640x384 files, to compare against, are in
  `runs/final120/r640/`.

### Training speed (`xpu_speedups.py`)

A stock YOLO26 training step on the A770 was limited by the CPU, not the GPU or the data loader: ~4700 kernel launches
per step, host syncs that drained the GPU queue, and a GPU idle while Python caught up. `train_xpu.py` applies
`xpu_speedups.py` unless `speedups=false`:

| patch | what | effect on results |
|---|---|---|
| launch settings | `UR_L0_USE_IMMEDIATE_COMMANDLISTS=0`: the driver's default immediate command lists cost ~22 us of CPU per kernel launch (~157 us while the GPU is busy); batched lists cost ~5-36 us. With `fused_bn` also `UR_L0_COMMANDLISTS_CLEANUP_THRESHOLD=-1`: otherwise every launch scans the command lists still in flight, so launches get slower (~10 -> ~100 us) the further the CPU runs ahead of the GPU | bit-identical |
| `channels_last` | NHWC weights and input on XPU (Ultralytics only does this on CUDA). Removes oneDNN's NCHW<->blocked reorders around every conv (158 ms/step of GPU time) and uses faster BatchNorm kernels | summation order only: fp32 grads within 1e-4 of NCHW. Checkpoints are still saved NCHW |
| sync-free loss | removes the host syncs in the detection loss: strides read back from the GPU in `make_anchors`, `unique`/`max` in target preprocessing, two `nonzero` calls, and 6 host->device copies of constants (`torch.tensor(shape)`, a Python-list index, `torch.tensor(stride)`), each of which waited for the whole GPU queue | gradients bit-identical; loss within 1.4e-7 (summation order) |
| validation NMS | `torchvision.ops.nms` on XPU instead of Ultralytics' Python loop | identical kept boxes (checked on 30 validation batches) |
| MuSGD | queues MuSGD's momentum + Newton-Schulz work before GradScaler's inf check (a host sync that waits for the backward pass) instead of after it | bit-identical parameters and optimizer state |
| `fused_bn` | `xpu_fused_bn.py`: Triton kernels for training-mode BatchNorm + SiLU (3 kernels forward, 3 backward, 8 passes over the activation instead of 13; SiLU recomputed in backward instead of storing BN's output). Training mode only: eval, validation and export use the stock modules | not bit-identical: fp32 arithmetic inside, so fp16 gradients are closer to fp32 than stock (cosine 0.93 vs 0.87 on one batch). See the accuracy note below |
| `half_glue` | the input image goes uint8 -> fp16 channels_last in one kernel, and nearest upsampling runs in fp16 (autocast ran it, and the Concat after it, in fp32 and cast back) | forward bit-identical; gradients within 3e-5 |
| `fast_topk` | the task-aligned assigner's `topk` (3-15 ms per call on XPU) as k argmax passes | same loss; ties between equal metrics may pick other anchors (`topk` leaves tie order unspecified) |
| `fast_val` | validation NMS and metrics on the CPU: one device->host copy of the raw predictions per batch instead of ~12k tiny copies per pass (each a queue drain), and Ultralytics' own NMS on host tensors (0.2 ms/image for the ~20 candidates a trained model leaves, vs 1.7 ms on the XPU; images with >2000 boxes, e.g. early in training, still use the XPU kernel). Validation 15.2 -> 6.6 s (2164 images; 41 -> 27 s with plots) | identical NMS output, metrics, plots and confusion matrices |
| `fast_cast` | all conv weights cast to fp16 in one multi-tensor kernel per forward (and their fp16 grads back in one), instead of autocast's ~280 per-weight cast kernels | bit-identical (same roundings) |
| `fast_cat` | C2f/C3k2, C3, SPPF, C2PSA and Concat: the split after `cv1` is written by the fused BN kernel as two contiguous tensors (no channel-slice views for the next conv to copy), and concatenations use `xpu_cat.py` (Triton; 2-4x faster than `torch.cat` on NHWC, and its backward gives each input a contiguous gradient instead of a slice that consumers copy or add with a slow strided kernel) | same values; gradients move by ~1e-3 relative because other kernels now sum in another order, 200x below fp16's own error (fp16 vs fp32 gradient error 0.2094 with and without) |
| `fast_attn` | attention (C2PSA, PSABlock) computed as out^T = attn @ v^T with an fp16 softmax and a channels_last output: no transposed copy of the (B, heads, N, N) gradient in softmax's backward, no fp32 copy of the scores. 8.2 -> 4.7 ms per block, forward + backward | forward bit-identical; gradient error vs fp32 unchanged |

Measured with `bench_step.py` (yolo26s, 704 px, batch 32, fp16, MuSGD; ms per training step, 60-100 steps):

| | reused batches | real data loader |
|---|---|---|
| stock Ultralytics | 849 | 885 |
| + launch setting (immediate command lists off) | 675 | |
| + `channels_last` | 500 | |
| + sync-free loss (first version) | 462 | 473 |
| + MuSGD change | 451 | 461 (69 images/s) |
| + `fused_bn` | 397 (peak memory 10.2 -> 7.5 GB) | |
| + constant copies removed, `half_glue`, `fast_topk`, cleanup threshold | 332 | 337 (95 images/s) |
| + `fast_cast`, `fast_cat`, `fast_attn` (same-session baseline 335) | 301 (peak memory 7.3 -> 6.9 GB) | 307 (104 images/s) |

In real training (`train_xpu.py` defaults, 6 + 6 workers, epochs 2-3 of a `stop_epochs=3` run on the 120-epoch
schedule): ~101 s per epoch (~93 s of training + ~7 s of validation). The same run with
`speedups_off=fast_val,fast_cast,fast_cat,fast_attn` took ~125 s (~105 + ~19 s), and stock Ultralytics 273 s per epoch
without validation. The final validation after training (plots, best.pt) took 30 s instead of 41 s.

Accuracy checks, one 11-epoch run per variant on the 120-epoch schedule, compared at `epoch10.pt`:

| | val mAP50 / 50-95 (704, XPU) | FP32 ONNX 768x448 | RK3588 INT8 768x448 |
|---|---|---|---|
| stock run | 0.917 / 0.652 | | |
| speedups before `fused_bn` | 0.923 / 0.644 | 0.880 / 0.614 | 0.881 / 0.613 |
| with `fused_bn` | 0.909 / 0.629 | 0.847 / 0.578 | 0.854 / 0.580 |

The `fused_bn` checkpoint is lower, but in that run alone consecutive epochs moved by 1.5-1.8 points (0.625 -> 0.643 ->
0.629), so one checkpoint per run does not separate the two; a full 120-epoch run is the real check (`fused_bn=false`
turns it off). Export and RKNN are unaffected: the ONNX graphs of the two checkpoints have the same 455 nodes and 210
initializers (names, shapes, dtypes), and INT8 on the board matches FP32 for both.

Notes:
- The data loader is not a limit: 8 workers produce ~510 images/s; the training loop waits ~0.2 ms per batch.
- Batch size (re-measured with all the patches, `bench_step` fixed batches): 32 runs at 108 images/s with a 6.9 GB
  peak. 48 fits (10.2 GB) but runs at 100 images/s, and with Ultralytics' `nbs=64` it changes the optimizer step from
  64 images (32 x 2 accumulated) to 48. 64 now fits (13.5 GB of tensors; PyTorch reserves 14.7 of the card's 15.1 GiB)
  but runs at 49 images/s. Its kernels take 8% less time per image than at 32, but the card is over-full: every step the
  i915 driver moves about 1.8 GB of GPU buffers out to host RAM and back (system `Shmem` swings 1.2 -> 3.0 GB; nothing
  is swapped), so ~0.8 s of each 1.3-1.4 s step runs no kernel. Before the patches 64 did not fit at all (the A770
  driver can segfault instead of raising an out-of-memory error). With the memory settings and the desktop stopped it
  still spills (91 images/s). At 704 px stay at 32; for 640 px see
  [Next training run](#next-training-run-640-px-batch-64).
- `fused_bn` autotunes its tile sizes on first use per layer shape (the first epoch of the first run is slower); Triton
  caches the results on disk.
- The cleanup-threshold setting made the stock BatchNorm path slower and erratic (434 -> 444-510 ms/step), so it is
  only set with `fused_bn`.
- MuSGD is what `optimizer=auto` picks for long runs (>10k iterations); short runs get AdamW, which is cheaper per step
  on this GPU. Force `optimizer=MuSGD` when a short run should behave like a long one.
- Tried and not used: `torch.compile` (slower, 591 vs 500 ms/step, though 3 GB less memory; needs `libze-dev` and
  `intel-ocloc`), an Inductor-compiled BatchNorm + SiLU (same speed as stock), XPU graphs (the A770 driver has no
  graph record/replay), other Level Zero settings (copy engine, v2 adapter, batch size, driver in-order lists),
  `F.scaled_dot_product_attention` (no fused XPU kernel in torch 2.15: math fallback, no faster), a Triton weight
  gradient for the depthwise 3x3 convs (1.5-2.3x slower than oneDNN's), the head's 4/7-channel 1x1 convs as matmuls
  (slower in backward), padding the 3-channel input to 4 (first conv 13% faster, ~0.6 ms/step, not worth a 4-channel
  image), re-tiling the fused BatchNorm reductions (already at ~85-100% of what a plain Triton read kernel reaches).
- What is left per step (~300 ms, the GPU is now busy nearly all of it): convolutions ~120 ms (oneDNN `jit:ir`,
  17-26 TFLOPS: forward 26, backward-data 25, backward-weights 17), fused BatchNorm ~73, the detection loss and
  assigner ~23, MuSGD's Newton-Schulz ~14 (hundreds of tiny bf16 matmuls at ~50 us each), autograd's gradient sums
  at branch points ~7, the image upload ~5.
- The GPU clock: this card's power limit is 190 W (`/sys/class/drm/card1/device/hwmon/hwmon*/power1_max`). Under
  convolutions it holds 2000-2400 MHz (pure fp16 matmul: 1750-1800 MHz at 190 W, 88 TFLOPS); memory-bound kernels
  run at 2400 MHz. A higher limit (root, and only if the card and PSU are rated for it) would mostly speed up the
  convolutions. Not changed here.
- `bench_step.py` reproduces all of this: `mode=fixed|real|loader|opt` for timing, `stages=true` for a per-stage split,
  `profile=<dir>` for a torch.profiler trace, `mode=parity|lossparity|optparity` for the numerical checks (use
  `batch=16` for parity: with `launch_env=true` at batch 32 it ran out of device memory; parity
  includes `fused_bn`, `half_glue`, `fast_topk` and `fast_cast` variants against fp32). Each patch can be switched on
  separately (`launch_env= channels_last= sync_free= fast_musgd= fused_bn= half_glue= fast_topk= fast_cast=
  fast_cat= fast_attn=`). In `train_xpu.py`, `speedups_off=fast_cat,fast_attn` (any of the names) turns patches off.

## 2. Pick an input size, export to ONNX

```
python training/export_onnx.py runs/detect/yolo26s_704/weights/last.pt --imgsz 448 768 --data-root /path/to/birds-3-57
```

- `--imgsz` is H W. `--data-root` is needed for the head clips (next bullet and
  [INT8 after long training](#int8-after-long-training-the-head-clips)); the export reads 200 training images from it.
- Adds clips to the head so that INT8 conversion survives long training, and checks on the training images that they
  change no detection (the export stops if they do). The bounds are in the json (`clip_*`).
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

### Input size: 640x384 (was 768x448)

The detector runs on frames as the cameras deliver them: `extract_birbs.py` asks for 1280x720. 864x486 is only the
size of the saved videos. The validation set is about half 864x486 and half 1280x720 frames, and birds are about the
same size in native pixels in both (shorter side: 5th percentile ~28 px, median ~60 px), so the 1280x720 frames are
the harder case once letterboxed.

**Larger is not better.** FP32 (ONNX), the 25-epoch 704-trained checkpoint, full validation set (2164 images):

| input | anchors N | mAP50 | mAP50-95 | letterbox scale, 864x486 / 1280x720 frames |
|---|---|---|---|---|
| 704x704 | 10164 | 0.954 | 0.752 | 0.81 / 0.55, 44% of the canvas is padding |
| **768x448** | **7056** | **0.952** | **0.748** | **0.89 / 0.60, 4% padding** |
| 640x640 | 8400 | 0.951 | 0.745 | 0.74 / 0.50 |
| 896x512 | 9408 | 0.949 | 0.742 | 1.04 (upscale) / 0.70 |
| 1024x576 | 12096 | 0.944 | 0.727 | 1.19 (upscale) / 0.80 |

768x448 is within 0.4 points of 704x704 with 31% fewer pixels, and it has two properties that matter on the board:

- **It downscales.** RGA, which does the letterbox resize on the board, shifted the image content by about one pixel
  when *upscaling* (checked by comparing the RGA output with a cv2 resize) and not when downscaling. At 896x512 the
  board scored 0.714 mAP50-95 on 720 864x486 frames against 0.753 for FP32, a 4-point gap of which INT8 explains about
  1.5. At 768x448 the board matches the INT8 simulator (0.754 vs 0.753, table below).
- **Its anchor count is a multiple of 16** (N = (H/8)(W/8) + (H/16)(W/16) + (H/32)(W/32)). On the Orange Pi 5 Pro a
  704x704 nano model (N = 10164) returned unstable values in a few box outputs at the row edges on real frames, in 3 of 3
  runs, while 640x640, 896x512 and 768x448 models did not. This is an observation on this board, not a documented rule.
  `export_onnx.py` warns when N is not a multiple of 16. The 16:9 sizes that pass: 448x256, 512x288, 640x384, 768x448,
  896x512 (576x352 and 704x416 do not).

**Smaller is not better either** (measured 2026-11-09 with the 120-epoch model). The floor comes from the architecture:

- yolo26s and yolo26m share the geometry that matters: strides 8/16/32, no DFL, and the same task-aligned assigner
  (box sides under 16 px are widened to 16 px when picking anchors). m is wider, not differently strided, so the same
  floor applies to it (not measured on m).
- At 768x448 the trained one-to-one head finds 94% of birds on the stride-16 level and none on stride 32. Birds whose
  shorter side is below ~20-24 px *in the network input* fall to the stride-8 level, which recalls them 0.4-0.75 at
  conf 0.5, against 0.88-0.96 for bigger birds. The receptive field is not the limit (~550 px).
- So the rule is to keep birds at ~20 px or more in the input. Share of validation birds below 20 px: 768x448 5.5%
  (8% of those in 1280x720 frames), 640x384 11% (17%), 512x288 24% (33%), 448x256 33% (43%).

Change against 768x448, full validation set, F1 at conf 0.5 / mAP50-95:

| input | 120-epoch model, FP32 | same pixels upscaled and run at 768x448 (image detail only) | fine-tuned 10 epochs at that size, FP32 | board INT8 |
|---|---|---|---|---|
| 896x512 to 704x416 | within ±0.005 / ±0.008 | | | |
| 640x384 | -0.002 / -0.005 | +0.002 / -0.001 | -0.003 / -0.007 | -0.006 / -0.016 |
| 576x352 | -0.008 / -0.019 | +0.001 / -0.002 | | |
| 512x288 | -0.014 / -0.042 | 0.000 / -0.005 | -0.015 / -0.036 | |
| 448x256 | -0.038 / -0.090 | -0.003 / -0.013 | -0.030 / -0.068 | |
| 320x192 | -0.231 / -0.317 | -0.021 / -0.050 | | |

- The detail-only column shows that a 512-wide frame still carries the information. The loss from running at 512 comes
  from the birds becoming small for the strides, and training at that size wins back only about a quarter of it
  (fine-tunes from the 120-epoch model, compared with an identical fine-tune at 704 evaluated at 768x448;
  `runs/detect/minsize_ft*`). Training at 512 and running at 640 was no better than training at 704.
- The losses are recall; precision stays at ~0.95. On the 1280x720 frames alone they are about 1.5x larger.
- Board, 640x384 against 768x448, both INT8, all 2164 frames (`runs/final120/r640/`): 252 vs 323 ms per 24-frame batch
  (22% less), F1 0.928 vs 0.934, mAP50-95 0.737 vs 0.754. INT8 costs more at 640x384 (FP32 to board: -0.021
  mAP50-95, against -0.009 at 768x448), mostly in the boxes of the smallest birds.
- Training below 704 used to save little time on the A770 (3.6 vs 3.3 it/s at 512 vs 704, before the GPU-side
  speedups). The step is now GPU-bound: at batch 32, 640 runs at 125 images/s against 108 at 704, and 640 also fits
  batch 64 (139-142 images/s).

**Decision: 640x384** (2026-11-09; until then 768x448). 640x384 is the smallest usable input. With the 704-trained
120-epoch model it costs about 0.6 points of F1 and 1.6 of mAP50-95 on the board for 22% less inference time (at
768x448 the detector took ~13.5 ms per frame). Training at 640 is ~30% faster on the A770. The next model is trained at
640 and exported at 640x384 ([Next training run](#next-training-run-640-px-batch-64)). Below 640x384, retraining does
not fix the loss. Until that model exists, the export, calibration, accuracy and board sections of this README
describe the 768x448 models.

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

### INT8 after long training: the head clips

A 120-epoch yolo26s (`runs/detect/yolo26s_704_3-57_120`) is better than the 25-epoch one in FP32 but, exported plainly,
fell apart in INT8. INT8 gives each tensor one scale, and in the head the positions with no bird drifted to huge values
during training, which then set that scale:

1. **The stride-32 level never fires** (no detection on 200 training images in either model) and drifted furthest:
   logits down to -4e7, box distances -1091..5854 (real detections: logits below 3.5, distances 0.1-8.5 grid units).
   Through the Concats that join the levels, every real box distance rounded to 0: board mAP50 **0.000**.
2. **The class features that feed each level's final class conv** reached 228-306 at background positions while
   detections use at most 7. At an INT8 step of ~1.2, the SiLU's small negative outputs (-0.28..0) all became 0. The
   120-epoch final conv has about 3x larger weights than the 25-epoch one, many of them negative on exactly those
   features, so every bird's logit fell by ~2.3 (score 0.96 -> 0.70). Float weights on the INT8 features reproduce the
   simulator's logits to 0.03, so it is the features' quantization, not the weights'. The board lost more than the
   simulator. RKNN's per-layer `accuracy_analysis` cosines did not show it: these tensors are almost all background.

`export_onnx.py` therefore clips, per level: the class features to `[-1, B]` before the final class conv, the logits to
`[-8, 8]` and the distances to `[-1, 16]` before the Concats.

- **The bound B is measured, not fixed** (`--clip-cls-feat auto`, the default). On 200 training images it finds the
  largest class feature at detection anchors (best score > 0.05) and sets B to twice that, rounded up, but only when the
  background reaches more than 4 times it. The 25-epoch model's birds use features up to 194, so it gets no class-feature
  clip; a fixed bound would have broken it. The 120-epoch model gets B = 14.
- **The export checks the clips** on the same images: every score above 0.001 and every detection's distances must stay
  within 1e-3 of the unclipped graph, or it stops. Both models: change 0. FP32 on the full validation set is identical
  with and without the clips.

Board, all 2164 validation images (P/R at conf 0.5):

| yolo26s | FP32 mAP50 / mAP50-95 | board mAP50 / mAP50-95 | board P / R |
|---|---|---|---|
| 25 epochs, no clips | 0.952 / 0.748 | 0.947 / 0.736 | 0.973 / 0.819 |
| 25 epochs, clips (B: none) | 0.952 / 0.748 | 0.947 / 0.736 | 0.973 / 0.819 |
| 120 epochs, no clips | 0.952 / 0.763 | 0.000 / 0.000 | - |
| 120 epochs, logit + distance clips only | 0.952 / 0.763 | 0.939 / 0.735 | 0.985 / 0.584 |
| **120 epochs, clips (B = 14)** | 0.952 / 0.763 | **0.951 / 0.754** | **0.953 / 0.916** |

With the clips, the 120-epoch model loses 0.1 / 0.9 points to INT8 (the 25-epoch model 0.5 / 1.2), and its board
precision and recall at 0.5 match FP32 (0.953 / 0.915).

## 5. Deploy

- `Dockerfile` pins `librknnrt.so` to `RKNN_RUNTIME_VERSION=2.5.0`. Keep it equal to the toolkit that converted the
  model.
- `docker-compose.yml` mounts `models/yolo26s-768x448-i8-bs24.rknn` (stored with Git LFS) at `/app/models/birbs.rknn`.
  It is the 120-epoch yolo26s (`runs/detect/yolo26s_704_3-57_120`, `last.pt`) exported with the head clips (B = 14).
  To deploy another model, copy it into `models/` and change the mount; the batch size it was converted with must match
  the last argument of the service's `command`.
- `--conf` / `BIRDS_CONF` default to 0.5. For the model in `models/`, pooled over all classes on the board, all 2164
  validation images (3582 labelled birds):

  | conf | precision | recall | F1 | false detections per frame |
  |---|---|---|---|---|
  | 0.25 | 0.928 | 0.933 | 0.930 | 0.121 |
  | 0.30 | 0.934 | 0.930 | 0.932 | 0.109 |
  | 0.40 | 0.944 | 0.924 | 0.934 | 0.091 |
  | **0.50** | **0.953** | **0.916** | **0.934** | **0.074** |
  | 0.60 | 0.959 | 0.906 | 0.932 | 0.064 |
  | 0.70 | 0.967 | 0.891 | 0.928 | 0.050 |

  With the head clips, the board's scores match FP32's (the F1-optimal threshold is 0.48 on the board, 0.47 in FP32).
  The 25-epoch model's INT8 scores ran lower than FP32's (F1-optimal 0.26 vs 0.42), which is why the default used to be
  0.25. The cause is the class-feature quantization in
  [INT8 after long training](#int8-after-long-training-the-head-clips), at a smaller scale (about 0.5 logit).
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
