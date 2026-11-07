#ifndef _BIRDS_YOLO26_H_
#define _BIRDS_YOLO26_H_

#include "yolo26_postprocess.h"

// Model loading and release are architecture independent: use init_yolov5_model / release_yolov5_model from yolov5.h.

// Run a YOLO26 model built by training/convert_rknn.py (outputs: scores, boxes). `img` holds `batch_size` RGB888 frames
// of identical size laid out back to back; `od_results` must have room for `batch_size` entries. `batch_size` must equal
// the rknn_batch_size the model was converted with. Detections with a score above `conf_threshold` are returned.
extern "C" int inference_yolo26_model(rknn_app_context_t *app_ctx, image_buffer_t *img,
                                      object_detect_result_list *od_results, const int batch_size,
                                      const float conf_threshold, const bool verbose);

#endif // _BIRDS_YOLO26_H_
