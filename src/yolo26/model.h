// Copyright (c) 2023 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Model loading and the detection result types. birdector/detector/_interface.py mirrors these structs and constants.
#ifndef _BIRDS_MODEL_H_
#define _BIRDS_MODEL_H_

#include <stdint.h>

#include "common.h"
#include "image_utils.h"
#include "rknn_api.h"

#define OBJ_NUMB_MAX_SIZE 128 // detections returned per image
#define OBJ_CLASS_NUM 7       // classes the model was trained on (birdector/config/classes.toml)

typedef struct {
    rknn_context rknn_ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr* input_attrs;
    rknn_tensor_attr* output_attrs;
    int model_channel;
    int model_width;
    int model_height;
    bool is_quant;
} rknn_app_context_t;

typedef struct {
    image_rect_t box;
    float prop;
    int cls_id;
} object_detect_result;

typedef struct {
    int id;
    int count;
    object_detect_result results[OBJ_NUMB_MAX_SIZE];
} object_detect_result_list;

extern "C" int init_model(const char* model_path, rknn_app_context_t* app_ctx, _rknn_core_mask core_mask = RKNN_NPU_CORE_AUTO);

extern "C" int release_model(rknn_app_context_t* app_ctx);

#endif // _BIRDS_MODEL_H_
