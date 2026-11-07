// Post-processing for YOLO26 models converted with training/convert_rknn.py.
//
// The RKNN graph is cut at two tensors of the head, so the NPU runs the whole network except the last bit of box
// arithmetic and the top-k:
//     output 0  scores  [B, OBJ_CLASS_NUM, N]  class probabilities (the sigmoid is in the graph)
//     output 1  dist    [B, 4, N]              left, top, right, bottom distances from the anchor point, in grid units
// where N = (H/8)*(W/8) + (H/16)*(W/16) + (H/32)*(W/32) for an HxW input (9408 for 896x512).
//
// The distances are cut before the anchor/stride arithmetic on purpose. Cutting after it (at absolute pixel
// coordinates, as Rockchip's yolo26 example does) leaves INT8 only ~3.8 px per step at 704x704 and cost 11 points of
// mAP50-95 on the bird data; the distances span a few grid units, so the same 8 bits resolve ~0.3 px at stride 8.
// The CPU turns a distance into a box only for the few detections that survive the threshold:
//     x1 = (gx + 0.5 - l) * stride    y1 = (gy + 0.5 - t) * stride
//     x2 = (gx + 0.5 + r) * stride    y2 = (gy + 0.5 + b) * stride
//
// YOLO26's one-to-one head is trained to emit a single box per object, so there is no NMS: the CPU does a threshold,
// a partial sort and the letterbox undo. Reference implementation: training/preproc.py (ltrb_to_boxes + decode_yolo26).
#ifndef _BIRDS_YOLO26_POSTPROCESS_H_
#define _BIRDS_YOLO26_POSTPROCESS_H_

#include "yolov5.h" // rknn_app_context_t, object_detect_result_list, letterbox_t

static_assert(OBJ_CLASS_NUM != 4, "the dist tensor is told apart from the scores tensor by its channel count (4)");

// Guard against a runtime fault. On an RK3588 (Orange Pi 5 Pro, RKNPU driver 0.9.6, librknnrt 2.3.2 and 2.5.0) the
// INT8 YOLO26 models sporadically come back with a 16-element chunk of one class plane of the scores output set to the
// raw value 0 instead of the background value -128, in some slots of a batch. 0 dequantises to a score of about 0.5,
// so each chunk turns into a burst of false detections of one class along a row of anchors. The same frame in other
// slots, or in another run, is clean. A real score above the threshold is never repeated bit for bit by this many
// neighbouring anchors (the one-to-one head fires on one anchor per object; 0 such runs on the validation set with the
// FP32 reference or the INT8 simulator), so a class score inside a run of YOLO26_SUSPECT_RUN or more identical values is
// ignored, and the anchor falls back to its next best class. Set to 0 to disable.
#ifndef YOLO26_SUSPECT_RUN
#define YOLO26_SUSPECT_RUN 8
#endif

// Pure decode of one image. `scores` is (num_classes, num_anchors) and `dist` is (4, num_anchors), both row-major
// float32. num_anchors must match the model input size (-1 is returned otherwise). At most OBJ_NUMB_MAX_SIZE
// detections are returned, best first, one class per anchor. If `suppressed` is given, the number of class scores
// ignored by the YOLO26_SUSPECT_RUN guard is added to it.
int decode_yolo26(const float *scores, const float *dist, int num_classes, int num_anchors,
                  const letterbox_t *letter_box, int model_in_w, int model_in_h, float conf_threshold,
                  object_detect_result_list *od_results, int *suppressed = nullptr);

// Same decode for the raw INT8 outputs of a quantized model. The threshold and the ranking are done on the integer
// scores (the dequantisation (q - zp) * scale is monotonic), and only the detections that survive are converted to
// float. Gives the same detections as decode_yolo26 applied to the dequantised tensors.
int decode_yolo26_i8(const int8_t *scores, const int8_t *dist, int num_classes, int num_anchors, int32_t scores_zp,
                     float scores_scale, int32_t dist_zp, float dist_scale, const letterbox_t *letter_box,
                     int model_in_w, int model_in_h, float conf_threshold, object_detect_result_list *od_results,
                     int *suppressed = nullptr);

// Finds the scores and dist tensors among the RKNN outputs (by shape) and decodes one image. `outputs` is an
// rknn_output[] whose buffers hold the data for ONE image: float32 if `outputs_are_float` (rknn_outputs_get with
// want_float = 1), otherwise the raw INT8 of a quantized model.
int post_process_yolo26(rknn_app_context_t *app_ctx, void *outputs, letterbox_t *letter_box, float conf_threshold,
                        object_detect_result_list *od_results, bool outputs_are_float, int *suppressed = nullptr);

#endif // _BIRDS_YOLO26_POSTPROCESS_H_
