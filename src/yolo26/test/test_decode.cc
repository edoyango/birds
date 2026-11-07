// Host-side test harness for decode_yolo26(). Not part of the library build.
// Driven by training/check_cpp_decode.py, which writes the cases and compares the output with the Python reference.
//
// Input file (little endian):
//   int32 ncases
//   per case: int32 num_classes, num_anchors, model_w, model_h, x_pad, y_pad; float32 scale, conf;
//             int32 quantized; float32 scores_scale, boxes_scale; int32 scores_zp, boxes_zp;
//             then scores[num_classes * num_anchors] and boxes[4 * num_anchors], float32 or, if quantized, int8
// Output: per case "case <i> <count>" then one "<left> <top> <right> <bottom> <score> <cls>" line per detection.
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "yolo26_postprocess.h"

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "usage: %s <cases.bin>\n", argv[0]);
        return 2;
    }
    FILE *f = fopen(argv[1], "rb");
    if (!f)
    {
        perror("fopen");
        return 2;
    }
    int ncases = 0;
    if (fread(&ncases, 4, 1, f) != 1)
        return 2;

    for (int c = 0; c < ncases; ++c)
    {
        int h[6];
        float fh[2], qs[2];
        int quantized, qz[2];
        if (fread(h, 4, 6, f) != 6 || fread(fh, 4, 2, f) != 2 || fread(&quantized, 4, 1, f) != 1 ||
            fread(qs, 4, 2, f) != 2 || fread(qz, 4, 2, f) != 2)
            return 2;
        const int nc = h[0], n = h[1];
        if (nc != OBJ_CLASS_NUM)
        {
            fprintf(stderr, "case has %d classes but the library is built for OBJ_CLASS_NUM=%d\n", nc, OBJ_CLASS_NUM);
            return 3;
        }
        std::vector<float> scores, boxes;
        std::vector<int8_t> scores_i8, boxes_i8;
        if (quantized)
        {
            scores_i8.resize((size_t)nc * n);
            boxes_i8.resize((size_t)4 * n);
            if (fread(scores_i8.data(), 1, scores_i8.size(), f) != scores_i8.size() ||
                fread(boxes_i8.data(), 1, boxes_i8.size(), f) != boxes_i8.size())
                return 2;
        }
        else
        {
            scores.resize((size_t)nc * n);
            boxes.resize((size_t)4 * n);
            if (fread(scores.data(), 4, scores.size(), f) != scores.size() ||
                fread(boxes.data(), 4, boxes.size(), f) != boxes.size())
                return 2;
        }

        letterbox_t lb;
        lb.x_pad = h[4];
        lb.y_pad = h[5];
        lb.scale = fh[0];
        static object_detect_result_list res; // large, keep off the stack
        if (quantized)
            decode_yolo26_i8(scores_i8.data(), boxes_i8.data(), nc, n, qz[0], qs[0], qz[1], qs[1], &lb, h[2], h[3],
                             fh[1], &res);
        else
            decode_yolo26(scores.data(), boxes.data(), nc, n, &lb, h[2], h[3], fh[1], &res);

        printf("case %d %d\n", c, res.count);
        for (int i = 0; i < res.count; ++i)
        {
            const object_detect_result &r = res.results[i];
            printf("%d %d %d %d %.9g %d\n", r.box.left, r.box.top, r.box.right, r.box.bottom, r.prop, r.cls_id);
        }
    }
    fclose(f);
    return 0;
}
