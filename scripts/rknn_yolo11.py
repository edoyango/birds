import cv2
from copy import copy
import numpy as np
import json
import os
import torch

from rknn.api import RKNN

_PAD_COLOUR = (0, 0, 0)
_NMS_THRESH = 0.45

CLASSES = ("Blackbird", "Butcherbird", "Currawong", "Dove", "Lorikeet", "Myna", "Sparrow", "Starling", "Wattlebird")
coco_id_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]

class Letter_Box_Info():
    def __init__(self, shape, new_shape, w_ratio, h_ratio, dw, dh, pad_color) -> None:
        self.origin_shape = shape
        self.new_shape = new_shape
        self.w_ratio = w_ratio
        self.h_ratio = h_ratio
        self.dw = dw 
        self.dh = dh
        self.pad_color = pad_color

class COCO_test_helper():
    def __init__(self, enable_letter_box = False) -> None:
        self.record_list = []
        self.enable_ltter_box = enable_letter_box
        if self.enable_ltter_box is True:
            self.letter_box_info_list = []
        else:
            self.letter_box_info_list = None

    def letter_box(self, im, new_shape, pad_color=(0,0,0), info_need=False):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        ratio = r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)  # add border
        
        if self.enable_ltter_box is True:
            self.letter_box_info_list.append(Letter_Box_Info(shape, new_shape, ratio, ratio, dw, dh, pad_color))
        if info_need is True:
            return im, ratio, (dw, dh)
        else:
            return im

    def direct_resize(self, im, new_shape, info_need=False):
        shape = im.shape[:2]
        h_ratio = new_shape[0]/ shape[0]
        w_ratio = new_shape[1]/ shape[1]
        if self.enable_ltter_box is True:
            self.letter_box_info_list.append(Letter_Box_Info(shape, new_shape, w_ratio, h_ratio, 0, 0, (0,0,0)))
        im = cv2.resize(im, (new_shape[1], new_shape[0]))
        return im

    def get_real_box(self, box, in_format='xyxy'):
        bbox = copy(box)
        if self.enable_ltter_box == True:
        # unletter_box result
            if in_format=='xyxy':
                bbox[:,0] -= self.letter_box_info_list[-1].dw
                bbox[:,0] /= self.letter_box_info_list[-1].w_ratio
                bbox[:,0] = np.clip(bbox[:,0], 0, self.letter_box_info_list[-1].origin_shape[1])

                bbox[:,1] -= self.letter_box_info_list[-1].dh
                bbox[:,1] /= self.letter_box_info_list[-1].h_ratio
                bbox[:,1] = np.clip(bbox[:,1], 0, self.letter_box_info_list[-1].origin_shape[0])

                bbox[:,2] -= self.letter_box_info_list[-1].dw
                bbox[:,2] /= self.letter_box_info_list[-1].w_ratio
                bbox[:,2] = np.clip(bbox[:,2], 0, self.letter_box_info_list[-1].origin_shape[1])

                bbox[:,3] -= self.letter_box_info_list[-1].dh
                bbox[:,3] /= self.letter_box_info_list[-1].h_ratio
                bbox[:,3] = np.clip(bbox[:,3], 0, self.letter_box_info_list[-1].origin_shape[0])
        return bbox

    def get_real_seg(self, seg):
        #! fix side effect
        dh = int(self.letter_box_info_list[-1].dh)
        dw = int(self.letter_box_info_list[-1].dw)
        origin_shape = self.letter_box_info_list[-1].origin_shape
        new_shape = self.letter_box_info_list[-1].new_shape
        if (dh == 0) and (dw == 0) and origin_shape == new_shape:
            return seg
        elif dh == 0 and dw != 0:
            seg = seg[:, :, dw:-dw] # a[0:-0] = []
        elif dw == 0 and dh != 0 : 
            seg = seg[:, dh:-dh, :]
        seg = np.where(seg, 1, 0).astype(np.uint8).transpose(1,2,0)
        seg = cv2.resize(seg, (origin_shape[1], origin_shape[0]), interpolation=cv2.INTER_LINEAR)
        if len(seg.shape) < 3:
            return seg[None,:,:]
        else:
            return seg.transpose(2,0,1)

    def add_single_record(self, image_id, category_id, bbox, score, in_format='xyxy', pred_masks = None):
        if self.enable_ltter_box == True:
        # unletter_box result
            if in_format=='xyxy':
                bbox[0] -= self.letter_box_info_list[-1].dw
                bbox[0] /= self.letter_box_info_list[-1].w_ratio

                bbox[1] -= self.letter_box_info_list[-1].dh
                bbox[1] /= self.letter_box_info_list[-1].h_ratio

                bbox[2] -= self.letter_box_info_list[-1].dw
                bbox[2] /= self.letter_box_info_list[-1].w_ratio

                bbox[3] -= self.letter_box_info_list[-1].dh
                bbox[3] /= self.letter_box_info_list[-1].h_ratio
                # bbox = [value/self.letter_box_info_list[-1].ratio for value in bbox]

        if in_format=='xyxy':
        # change xyxy to xywh
            bbox[2] = bbox[2] - bbox[0]
            bbox[3] = bbox[3] - bbox[1]
        else:
            assert False, "now only support xyxy format, please add code to support others format"
        
        def single_encode(x):
            from pycocotools.mask import encode
            rle = encode(np.asarray(x[:, :, None], order="F", dtype="uint8"))[0]
            rle["counts"] = rle["counts"].decode("utf-8")
            return rle

        if pred_masks is None:
            self.record_list.append({"image_id": image_id,
                                    "category_id": category_id,
                                    "bbox":[round(x, 3) for x in bbox],
                                    'score': round(score, 5),
                                    })
        else:
            rles = single_encode(pred_masks)
            self.record_list.append({"image_id": image_id,
                                    "category_id": category_id,
                                    "bbox":[round(x, 3) for x in bbox],
                                    'score': round(score, 5),
                                    'segmentation': rles,
                                    })
    
    def export_to_json(self, path):
        with open(path, 'w') as f:
            json.dump(self.record_list, f)

class RKNN_YOLO11_RESULTS():
    def __init__(self, boxes, classes, scores, img):
        self.boxes = boxes
        self.classes = classes
        self.scores = scores
        self.img = img
        self.names = CLASSES
    def draw(self):
        img_copy = self.img.copy()
        if self.boxes is not None and self.scores is not None and self.classes is not None:
            for box, score, cl in zip(self.boxes, self.scores, self.classes):
                top, left, right, bottom = [int(_b) for _b in box]
                print("%s @ (%d %d %d %d) %.3f" % (CLASSES[cl], top, left, right, bottom, score))
                cv2.rectangle(img_copy, (top, left), (right, bottom), (255, 0, 0), 2)
                cv2.putText(img_copy, '{0} {1:.2f}'.format(CLASSES[cl], score),
                            (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return img_copy
    def save(self, path):
        if self.boxes is not None:
            self.draw()
            os.makedirs(os.dirname(path))
            cv2.imwrite(path, self.img)

class RKNN_YOLO11():

    def _filter_boxes(self, boxes, box_confidences, box_class_probs, conf=0.6):
        """Filter boxes with object threshold.
        """
        box_confidences = box_confidences.reshape(-1)
        candidate, class_num = box_class_probs.shape

        class_max_score = np.max(box_class_probs, axis=-1)
        classes = np.argmax(box_class_probs, axis=-1)

        _class_pos = np.where(class_max_score* box_confidences >= conf)
        scores = (class_max_score* box_confidences)[_class_pos]

        boxes = boxes[_class_pos]
        classes = classes[_class_pos]

        return boxes, classes, scores

    def _nms_boxes(self, boxes, scores):
        """Suppress non-maximal boxes.
        # Returns
            keep: ndarray, index of effective boxes.
        """
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]

        areas = w * h
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

            w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
            h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
            inter = w1 * h1

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= _NMS_THRESH)[0]
            order = order[inds + 1]
        keep = np.array(keep)
        return keep

    def _dfl(self, position):
        # Distribution Focal Loss (DFL)
        x = torch.tensor(position)
        n,c,h,w = x.shape
        p_num = 4
        mc = c//p_num
        y = x.reshape(n,p_num,mc,h,w)
        y = y.softmax(2)
        acc_metrix = torch.tensor(range(mc)).float().reshape(1,1,mc,1,1)
        y = (y*acc_metrix).sum(2)
        return y.numpy()

    def _box_process(self, position, imgsz=(640, 640)):
        grid_h, grid_w = position.shape[2:4]
        col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
        col = col.reshape(1, 1, grid_h, grid_w)
        row = row.reshape(1, 1, grid_h, grid_w)
        grid = np.concatenate((col, row), axis=1)
        stride = np.array([imgsz[1]//grid_h, imgsz[0]//grid_w]).reshape(1,2,1,1)

        position = self._dfl(position)
        box_xy  = grid +0.5 -position[:,0:2,:,:]
        box_xy2 = grid +0.5 +position[:,2:4,:,:]
        xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1)

        return xyxy

    def _post_process(self, input_data, imgsz=(640, 640), conf=0.6):
        boxes, scores, classes_conf = [], [], []
        defualt_branch=3
        pair_per_branch = len(input_data)//defualt_branch
        # Python 忽略 score_sum 输出
        for i in range(defualt_branch):
            boxes.append(self._box_process(input_data[pair_per_branch*i]))
            classes_conf.append(input_data[pair_per_branch*i+1])
            scores.append(np.ones_like(input_data[pair_per_branch*i+1][:,:1,:,:], dtype=np.float32))

        def sp_flatten(_in):
            ch = _in.shape[1]
            _in = _in.transpose(0,2,3,1)
            return _in.reshape(-1, ch)

        boxes = [sp_flatten(_v) for _v in boxes]
        classes_conf = [sp_flatten(_v) for _v in classes_conf]
        scores = [sp_flatten(_v) for _v in scores]

        boxes = np.concatenate(boxes)
        classes_conf = np.concatenate(classes_conf)
        scores = np.concatenate(scores)

        # filter according to threshold
        boxes, classes, scores = self._filter_boxes(boxes, scores, classes_conf, conf)

        # nms
        nboxes, nclasses, nscores = [], [], []
        for c in set(classes):
            inds = np.where(classes == c)
            b = boxes[inds]
            c = classes[inds]
            s = scores[inds]
            keep = self._nms_boxes(b, s)

            if len(keep) != 0:
                nboxes.append(b[keep])
                nclasses.append(c[keep])
                nscores.append(s[keep])

        if not nclasses and not nscores:
            return None, None, None

        boxes = np.concatenate(nboxes)
        classes = np.concatenate(nclasses)
        scores = np.concatenate(nscores)

        return boxes, classes, scores

    def __init__(self, model_path, target=None, device_id=None) -> None:
        self.rknn = RKNN()

        # Direct Load RKNN Model
        self.rknn.load_rknn(model_path)

        print('--> Init runtime environment')
        if target==None:
            ret = self.rknn.init_runtime()
        else:
            ret = self.rknn.init_runtime(target=target, device_id=device_id)
        if ret != 0:
            raise RuntimeError('Init runtime environment failed')
        
        self.co_helper = COCO_test_helper(enable_letter_box=True)
        

    def __call__(self, input, imgsz=(640, 640), conf=0.6):
        if self.rknn is None:
            print("ERROR: rknn has been released")
            return []
        
        input_pp = self.co_helper.letter_box(im=input.copy(), new_shape=(imgsz[1], imgsz[0]), pad_color=(0,0,0))
        input_pp = cv2.cvtColor(input_pp, cv2.COLOR_BGR2RGB)

        outputs = self.rknn.inference(inputs=[input_pp])
        boxes, classes, scores = self._post_process(outputs, imgsz, conf)
    
        return RKNN_YOLO11_RESULTS(boxes, classes, scores, input.copy())

    def release(self):
        self.rknn.release()
        self.rknn = None
