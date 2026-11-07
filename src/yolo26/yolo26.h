#ifndef _BIRDS_YOLO26_H_
#define _BIRDS_YOLO26_H_

#include "yolo26_postprocess.h"

// Load and release the model with init_model / release_model (model.h).

// Run a YOLO26 model built by training/convert_rknn.py (outputs: scores, dist). `img` holds `batch_size` RGB888 or
// BGR888 frames of identical size laid out back to back; `od_results` must have room for `batch_size` entries.
// `batch_size` must equal the rknn_batch_size the model was converted with. Detections with a score above
// `conf_threshold` are returned.
extern "C" int inference_yolo26_model(rknn_app_context_t *app_ctx, image_buffer_t *img,
                                      object_detect_result_list *od_results, const int batch_size,
                                      const float conf_threshold, const bool verbose);

#endif // _BIRDS_YOLO26_H_
