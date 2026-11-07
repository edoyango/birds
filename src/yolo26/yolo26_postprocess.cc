#include "yolo26_postprocess.h"

#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <vector>

namespace
{
inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

// Anchor point (grid units, cell centre) and stride of every anchor for an HxW input, in the order the head
// concatenates its three levels (stride 8, 16, 32; each row-major). Rebuilt only when the input size changes.
struct AnchorGrid
{
    int h = 0, w = 0;
    std::vector<float> x, y, stride;
};

const AnchorGrid &anchor_grid(int h, int w)
{
    static thread_local AnchorGrid g;
    if (g.h != h || g.w != w)
    {
        g.h = h;
        g.w = w;
        g.x.clear();
        g.y.clear();
        g.stride.clear();
        for (int s : {8, 16, 32})
        {
            const int gh = h / s, gw = w / s;
            for (int j = 0; j < gh; ++j)
                for (int i = 0; i < gw; ++i)
                {
                    g.x.push_back(i + 0.5f);
                    g.y.push_back(j + 0.5f);
                    g.stride.push_back((float)s);
                }
        }
    }
    return g;
}

// Write detection k from the anchor's distances (already dequantised) and undo the letterbox.
inline void write_result(object_detect_result_list *out, int k, const AnchorGrid &g, int anchor, float l, float t,
                         float r, float b, float score, int cls, const letterbox_t *lb, int model_w, int model_h)
{
    const float st = g.stride[anchor];
    const float x1 = (g.x[anchor] - l) * st - lb->x_pad;
    const float y1 = (g.y[anchor] - t) * st - lb->y_pad;
    const float x2 = (g.x[anchor] + r) * st - lb->x_pad;
    const float y2 = (g.y[anchor] + b) * st - lb->y_pad;

    // image_rect_t holds integers. Round to the nearest pixel rather than truncate: truncation shifts every edge by half a
    // pixel on average, which cost 0.6 points of mAP50-95 on the validation set (rounding costs 0.3). The clamped value
    // is never negative, so adding 0.5 and truncating rounds.
    object_detect_result &res = out->results[k];
    res.box.left = (int)(clampf(x1, 0, (float)model_w) / lb->scale + 0.5f);
    res.box.top = (int)(clampf(y1, 0, (float)model_h) / lb->scale + 0.5f);
    res.box.right = (int)(clampf(x2, 0, (float)model_w) / lb->scale + 0.5f);
    res.box.bottom = (int)(clampf(y2, 0, (float)model_h) / lb->scale + 0.5f);
    res.prop = score;
    res.cls_id = cls;
}

template <typename Cand>
inline bool by_score_then_anchor(const Cand &a, const Cand &b)
{
    return a.score > b.score || (a.score == b.score && a.anchor < b.anchor);
}

static_assert(OBJ_CLASS_NUM <= 32, "best_class() keeps the excluded classes in a 32-bit mask");

// True if plane[i] belongs to a run of at least YOLO26_SUSPECT_RUN consecutive anchors with exactly the same value.
template <typename T>
inline bool in_constant_run(const T *plane, int n, int i)
{
    const T v = plane[i];
    int lo = i, hi = i;
    while (lo > 0 && hi - lo + 1 < YOLO26_SUSPECT_RUN && plane[lo - 1] == v)
        --lo;
    while (hi + 1 < n && hi - lo + 1 < YOLO26_SUSPECT_RUN && plane[hi + 1] == v)
        ++hi;
    return hi - lo + 1 >= YOLO26_SUSPECT_RUN;
}

// Best class of anchor i, ignoring class scores that sit in a constant run (see YOLO26_SUSPECT_RUN in the header).
// Returns true and sets best/best_cls if the best remaining score is above thr. The run check costs nothing for the
// vast majority of anchors, which are below the threshold. Equivalent to: mask every score that is in a constant run,
// then take the argmax (first class on ties) and threshold it, which is what training/preproc.py does.
template <typename T, typename Thr>
inline bool best_class(const T *scores, int nc, int n, int i, Thr thr, T &best, int &best_cls, int *suppressed)
{
    uint32_t excluded = 0;
    for (;;)
    {
        best_cls = -1;
        for (int c = 0; c < nc; ++c)
        {
            if (excluded >> c & 1u)
                continue;
            const T s = scores[(size_t)c * n + i];
            if (best_cls < 0 || s > best)
            {
                best = s;
                best_cls = c;
            }
        }
        if (best_cls < 0 || !(best > thr))
            return false;
        if (YOLO26_SUSPECT_RUN <= 0 || !in_constant_run(scores + (size_t)best_cls * n, n, i))
            return true;
        excluded |= 1u << best_cls;
        if (suppressed)
            ++*suppressed;
    }
}
} // namespace

int decode_yolo26(const float *scores, const float *dist, int num_classes, int num_anchors,
                  const letterbox_t *letter_box, int model_in_w, int model_in_h, float conf_threshold,
                  object_detect_result_list *od_results, int *suppressed)
{
    memset(od_results, 0, sizeof(object_detect_result_list));
    if (num_classes > 32)
        return -1;
    const AnchorGrid &g = anchor_grid(model_in_h, model_in_w);
    if ((int)g.x.size() != num_anchors)
    {
        printf("decode_yolo26: model has %d anchors but a %dx%d input has %d (input must be a multiple of 32)\n",
               num_anchors, model_in_w, model_in_h, (int)g.x.size());
        return -1;
    }

    // best class per anchor, kept if above the threshold. Few anchors survive, so the vector stays small.
    struct Candidate
    {
        float score;
        int anchor;
        int cls;
    };
    std::vector<Candidate> cands;
    cands.reserve(256);
    for (int i = 0; i < num_anchors; ++i)
    {
        float best;
        int best_cls;
        if (best_class(scores, num_classes, num_anchors, i, conf_threshold, best, best_cls, suppressed))
            cands.push_back({best, i, best_cls});
    }

    const int keep = std::min<int>((int)cands.size(), OBJ_NUMB_MAX_SIZE);
    if (keep == 0)
        return 0;
    // higher score first; equal scores keep anchor order, which makes the result deterministic
    std::partial_sort(cands.begin(), cands.begin() + keep, cands.end(), by_score_then_anchor<Candidate>);

    for (int k = 0; k < keep; ++k)
    {
        const int i = cands[k].anchor;
        write_result(od_results, k, g, i, dist[0 * num_anchors + i], dist[1 * num_anchors + i],
                     dist[2 * num_anchors + i], dist[3 * num_anchors + i], cands[k].score, cands[k].cls, letter_box,
                     model_in_w, model_in_h);
    }
    od_results->count = keep;
    return 0;
}

int decode_yolo26_i8(const int8_t *scores, const int8_t *dist, int num_classes, int num_anchors, int32_t scores_zp,
                     float scores_scale, int32_t dist_zp, float dist_scale, const letterbox_t *letter_box,
                     int model_in_w, int model_in_h, float conf_threshold, object_detect_result_list *od_results,
                     int *suppressed)
{
    memset(od_results, 0, sizeof(object_detect_result_list));
    if (num_classes > 32)
        return -1;
    const AnchorGrid &g = anchor_grid(model_in_h, model_in_w);
    if ((int)g.x.size() != num_anchors)
    {
        printf("decode_yolo26_i8: model has %d anchors but a %dx%d input has %d (input must be a multiple of 32)\n",
               num_anchors, model_in_w, model_in_h, (int)g.x.size());
        return -1;
    }

    // float score = (q - zp) * scale > conf  <=>  q > floor(conf / scale + zp)
    const float q_thr_f = floorf(conf_threshold / scores_scale + (float)scores_zp);
    if (q_thr_f >= 127.f)
        return 0; // nothing can exceed the threshold
    const int q_thr = q_thr_f < -128.f ? -129 : (int)q_thr_f;

    struct CandidateI8
    {
        int score;
        int anchor;
        int cls;
    };
    std::vector<CandidateI8> cands;
    cands.reserve(256);
    for (int i = 0; i < num_anchors; ++i)
    {
        int8_t best;
        int best_cls;
        if (best_class(scores, num_classes, num_anchors, i, q_thr, best, best_cls, suppressed))
            cands.push_back({(int)best, i, best_cls});
    }

    const int keep = std::min<int>((int)cands.size(), OBJ_NUMB_MAX_SIZE);
    if (keep == 0)
        return 0;
    std::partial_sort(cands.begin(), cands.begin() + keep, cands.end(), by_score_then_anchor<CandidateI8>);

    auto deq = [&](int channel, int anchor) {
        return ((float)dist[channel * num_anchors + anchor] - (float)dist_zp) * dist_scale;
    };
    for (int k = 0; k < keep; ++k)
    {
        const int i = cands[k].anchor;
        write_result(od_results, k, g, i, deq(0, i), deq(1, i), deq(2, i), deq(3, i),
                     ((float)cands[k].score - (float)scores_zp) * scores_scale, cands[k].cls, letter_box, model_in_w,
                     model_in_h);
    }
    od_results->count = keep;
    return 0;
}

int post_process_yolo26(rknn_app_context_t *app_ctx, void *outputs, letterbox_t *letter_box, float conf_threshold,
                        object_detect_result_list *od_results, bool outputs_are_float, int *suppressed)
{
    rknn_output *_outputs = (rknn_output *)outputs;

    int scores_idx = -1, dist_idx = -1;
    for (uint32_t i = 0; i < app_ctx->io_num.n_output; ++i)
    {
        const rknn_tensor_attr &a = app_ctx->output_attrs[i];
        if (a.n_dims != 3)
            continue;
        if (a.dims[1] == OBJ_CLASS_NUM)
            scores_idx = (int)i;
        else if (a.dims[1] == 4)
            dist_idx = (int)i;
    }
    if (scores_idx < 0 || dist_idx < 0)
    {
        printf("post_process_yolo26: expected a [B,%d,N] scores output and a [B,4,N] dist output, found %u outputs "
               "(is this a YOLO26 model with OBJ_CLASS_NUM=%d?)\n",
               OBJ_CLASS_NUM, app_ctx->io_num.n_output, OBJ_CLASS_NUM);
        memset(od_results, 0, sizeof(object_detect_result_list));
        return -1;
    }
    const int num_anchors = (int)app_ctx->output_attrs[scores_idx].dims[2];

    if (outputs_are_float)
        return decode_yolo26((const float *)_outputs[scores_idx].buf, (const float *)_outputs[dist_idx].buf,
                             OBJ_CLASS_NUM, num_anchors, letter_box, app_ctx->model_width, app_ctx->model_height,
                             conf_threshold, od_results, suppressed);

    const rknn_tensor_attr &sa = app_ctx->output_attrs[scores_idx];
    const rknn_tensor_attr &da = app_ctx->output_attrs[dist_idx];
    return decode_yolo26_i8((const int8_t *)_outputs[scores_idx].buf, (const int8_t *)_outputs[dist_idx].buf,
                            OBJ_CLASS_NUM, num_anchors, sa.zp, sa.scale, da.zp, da.scale, letter_box,
                            app_ctx->model_width, app_ctx->model_height, conf_threshold, od_results, suppressed);
}
