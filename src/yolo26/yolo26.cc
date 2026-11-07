#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "common.h"
#include "image_utils.h"
#include "yolo26.h"

// Ultralytics trains and validates with gray (114) letterbox padding, so use the same at inference.
static const int LETTERBOX_PAD_COLOR = 114;

// Channel order the model takes. training/convert_rknn.py records it in the model's custom string ("input_order=bgr"
// for models exported with export_onnx.py --input-order bgr, whose first conv was reordered); models without it are
// RGB. The letterbox writes the model input in this order, converting from the caller's format if they differ (RGA
// does that in the same pass as the resize).
static image_format_t model_input_format(rknn_context ctx)
{
    rknn_custom_string cs;
    memset(&cs, 0, sizeof(cs));
    if (rknn_query(ctx, RKNN_QUERY_CUSTOM_STRING, &cs, sizeof(cs)) == RKNN_SUCC &&
        strstr(cs.string, "input_order=bgr") != NULL)
        return IMAGE_FORMAT_BGR888;
    return IMAGE_FORMAT_RGB888;
}

// Per-stage timing, printed when `verbose` is set or the environment variable YOLO26_TIMING is non-empty. Reports wall
// time and the CPU time the whole process burned in the same window; a stage whose CPU time is close to its wall time
// is keeping a core busy (for example a runtime that spins while waiting for the NPU).
struct Stamp
{
    double wall_ms, cpu_ms;
};
static Stamp stamp()
{
    timespec w, c;
    clock_gettime(CLOCK_MONOTONIC, &w);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &c);
    return {w.tv_sec * 1e3 + w.tv_nsec / 1e6, c.tv_sec * 1e3 + c.tv_nsec / 1e6};
}

extern "C" int inference_yolo26_model(rknn_app_context_t *app_ctx, image_buffer_t *img,
                                      object_detect_result_list *od_results, const int batch_size,
                                      const float conf_threshold, const bool verbose)
{
    if (!app_ctx || !img || !od_results || batch_size < 1)
        return -1;

    int ret = 0;
    image_buffer_t dst_img;
    letterbox_t letter_box[batch_size];
    rknn_input inputs[app_ctx->io_num.n_input];
    rknn_output outputs[app_ctx->io_num.n_output];
    bool outputs_held = false;
    const char *timing_env = getenv("YOLO26_TIMING");
    const bool timing = verbose || (timing_env && *timing_env);
    // Outputs are requested as float32, which the runtime produces by dequantizing on the CPU (about 1.3 ms per frame of
    // CPU for this model). Setting YOLO26_INT8_RAW takes the raw INT8 outputs of a quantized model instead and decodes
    // them on the integers (decode_yolo26_i8), skipping that conversion. On the 768x448 model it gave the same
    // detections as the float path but no measurable CPU saving per batch, so it is opt-in.
    const char *raw_env = getenv("YOLO26_INT8_RAW");
    const bool use_float = !app_ctx->is_quant || !(raw_env && *raw_env);
    Stamp t0 = stamp(), t1 = t0, t2 = t0, t3 = t0, t4 = t0, t5 = t0;
    int suppressed = 0; // class scores ignored by the output-fault guard (YOLO26_SUSPECT_RUN)

    memset(od_results, 0, sizeof(*od_results) * batch_size);
    memset(letter_box, 0, sizeof(letter_box));
    memset(&dst_img, 0, sizeof(image_buffer_t));
    memset(inputs, 0, sizeof(inputs));
    memset(outputs, 0, sizeof(outputs));

    // Pre process: letterbox each frame into the model input
    dst_img.width = app_ctx->model_width;
    dst_img.height = app_ctx->model_height;
    dst_img.format = model_input_format(app_ctx->rknn_ctx);
    dst_img.size = get_image_size(&dst_img);
    dst_img.virt_addr = (unsigned char *)malloc((size_t)dst_img.size * batch_size);
    if (dst_img.virt_addr == NULL)
    {
        printf("malloc buffer size:%d fail!\n", dst_img.size);
        return -1;
    }

    image_buffer_t tmp_img = *img, tmp_dst_img = dst_img;
    for (int i = 0; i < batch_size; ++i)
    {
        ret = convert_image_with_letterbox(&tmp_img, &tmp_dst_img, &letter_box[i], LETTERBOX_PAD_COLOR, (int)verbose);
        if (ret < 0)
        {
            printf("convert_image_with_letterbox fail! ret=%d\n", ret);
            goto out;
        }
        tmp_img.virt_addr += img->size;
        tmp_dst_img.virt_addr += dst_img.size;
    }

    t1 = stamp();

    // Set input data
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].size = app_ctx->model_width * app_ctx->model_height * app_ctx->model_channel * batch_size;
    inputs[0].buf = dst_img.virt_addr;
    ret = rknn_inputs_set(app_ctx->rknn_ctx, app_ctx->io_num.n_input, inputs);
    if (ret < 0)
    {
        printf("rknn_input_set fail! ret=%d\n", ret);
        goto out;
    }

    t2 = stamp();

    // Run
    if (verbose)
        printf("rknn_run\n");
    ret = rknn_run(app_ctx->rknn_ctx, nullptr);
    if (ret < 0)
    {
        printf("rknn_run fail! ret=%d\n", ret);
        goto out;
    }

    t3 = stamp();

    // Get outputs, as float32 or raw INT8 (see use_float above)
    for (uint32_t i = 0; i < app_ctx->io_num.n_output; i++)
    {
        outputs[i].index = i;
        outputs[i].want_float = use_float ? 1 : 0;
    }
    ret = rknn_outputs_get(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs, NULL);
    if (ret < 0)
    {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        goto out;
    }
    outputs_held = true;
    t4 = stamp();

    // Post process: each output holds the whole batch; hand post_process_yolo26 one image at a time
    {
        void *base[app_ctx->io_num.n_output];
        size_t per_image[app_ctx->io_num.n_output];
        for (uint32_t d = 0; d < app_ctx->io_num.n_output; ++d)
        {
            base[d] = outputs[d].buf;
            per_image[d] = outputs[d].size / batch_size;
            outputs[d].size = per_image[d];
        }
        for (int i = 0; i < batch_size; ++i)
        {
            ret = post_process_yolo26(app_ctx, outputs, &letter_box[i], conf_threshold, &od_results[i], use_float,
                                      &suppressed);
            if (ret < 0)
                break;
            for (uint32_t d = 0; d < app_ctx->io_num.n_output; ++d)
                outputs[d].buf = (char *)outputs[d].buf + per_image[d];
        }
        for (uint32_t d = 0; d < app_ctx->io_num.n_output; ++d)
        {
            outputs[d].buf = base[d];
            outputs[d].size = per_image[d] * batch_size;
        }
    }

    t5 = stamp();
    if (timing && ret >= 0)
        printf("yolo26 batch=%d wall/cpu ms: letterbox %.2f/%.2f  inputs_set %.2f/%.2f  run %.2f/%.2f  "
               "outputs_get %.2f/%.2f  postprocess %.2f/%.2f  suppressed %d\n",
               batch_size, t1.wall_ms - t0.wall_ms, t1.cpu_ms - t0.cpu_ms, t2.wall_ms - t1.wall_ms,
               t2.cpu_ms - t1.cpu_ms, t3.wall_ms - t2.wall_ms, t3.cpu_ms - t2.cpu_ms, t4.wall_ms - t3.wall_ms,
               t4.cpu_ms - t3.cpu_ms, t5.wall_ms - t4.wall_ms, t5.cpu_ms - t4.cpu_ms, suppressed);
    // Say once per process that the output-fault guard fired (see YOLO26_SUSPECT_RUN in yolo26_postprocess.h), so the
    // fault does not go unnoticed; YOLO26_TIMING reports the count for every batch.
    {
        static bool reported = false;
        if (suppressed > 0 && !reported)
        {
            reported = true;
            printf("yolo26: ignored %d class scores that came back as a run of identical values (a known NPU runtime "
                   "output fault, see yolo26_postprocess.h); not reported again\n",
                   suppressed);
        }
    }

out:
    // remember to release rknn outputs
    if (outputs_held)
        rknn_outputs_release(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs);
    if (dst_img.virt_addr != NULL)
        free(dst_img.virt_addr);
    return ret;
}
