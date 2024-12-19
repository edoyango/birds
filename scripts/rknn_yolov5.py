import numpy as np
import cv2
from copy import copy
from rknn.api import RKNN

class RKNN_model():
    def __init__(self, model_path, target=None, device_id=None) -> None:

        assert model_path.endswith('.rknn'), f"{model_path} is not rknn/pytorch/onnx model"

        rknn = RKNN()

        # Direct Load RKNN Model
        rknn.load_rknn(model_path)

        print('--> Init runtime environment')
        if target==None:
            ret = rknn.init_runtime()
        else:
            ret = rknn.init_runtime(target=target, device_id=device_id)
        if ret != 0:
            print('Init runtime environment failed')
            exit(ret)
        print('done')
        
        self.rknn = rknn

    # def __del__(self):
    #     self.release()

    def run(self, inputs):
        if self.rknn is None:
            print("ERROR: rknn has been released")
            return []

        if isinstance(inputs, list) or isinstance(inputs, tuple):
            pass
        else:
            inputs = [inputs]

        result = self.rknn.inference(inputs=inputs)
    
        return result

    @staticmethod
    def post_process(input_data, anchors, imgsz=(640, 640), nms_thresh=0.7):
        boxes, scores, classes_conf = [], [], []
        # 1*255*h*w -> 3*85*h*w
        input_data = [_in.reshape([len(anchors[0]),-1]+list(_in.shape[-2:])) for _in in input_data]
        for i in range(len(input_data)):
            boxes.append(box_process(input_data[i][:,:4,:,:], anchors[i], imgsz=imgsz))
            scores.append(input_data[i][:,4:5,:,:])
            classes_conf.append(input_data[i][:,5:,:,:])

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
        boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

        # nms
        nboxes, nclasses, nscores = [], [], []

        for c in set(classes):
            inds = np.where(classes == c)
            b = boxes[inds]
            c = classes[inds]
            s = scores[inds]
            keep = nms_boxes(b, s, nms_thresh)

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
    
    def infer(self, inputs, anchors, imgsz, nms_thresh):
        # create a letterbox helper to store original dimensions
        letterbox_helper = COCO_test_helper()
        # letter box image
        letterboxed_image = cv2.cvtColor(letterbox_helper.letter_box(inputs, imgsz), cv2.COLOR_BGR2RGB)
        # inference
        boxes, classes, scores = self.post_process(self.run([letterboxed_image]), anchors, imgsz, nms_thresh)
        # return results, with unletterboxed boxes
        if boxes is not None:
            boxes = letterbox_helper.get_real_box(boxes)
        return boxes, classes, scores

    def release(self):
        self.rknn.release()
        self.rknn = None

def nms_boxes(boxes, scores, nms_thresh=0.7):
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
        inds = np.where(ovr <= nms_thresh)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def box_process(position, anchors, imgsz=(640, 640)):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([imgsz[1]//grid_h, imgsz[0]//grid_w]).reshape(1,2,1,1)

    col = col.repeat(len(anchors), axis=0)
    row = row.repeat(len(anchors), axis=0)
    anchors = np.array(anchors)
    anchors = anchors.reshape(*anchors.shape, 1, 1)

    box_xy = position[:,:2,:,:]*2 - 0.5
    box_wh = pow(position[:,2:4,:,:]*2, 2) * anchors

    box_xy += grid
    box_xy *= stride
    box = np.concatenate((box_xy, box_wh), axis=1)

    # Convert [c_x, c_y, w, h] to [x1, y1, x2, y2]
    xyxy = np.copy(box)
    xyxy[:, 0, :, :] = box[:, 0, :, :] - box[:, 2, :, :]/ 2  # top left x
    xyxy[:, 1, :, :] = box[:, 1, :, :] - box[:, 3, :, :]/ 2  # top left y
    xyxy[:, 2, :, :] = box[:, 0, :, :] + box[:, 2, :, :]/ 2  # bottom right x
    xyxy[:, 3, :, :] = box[:, 1, :, :] + box[:, 3, :, :]/ 2  # bottom right y

    return xyxy


def filter_boxes(boxes, box_confidences, box_class_probs, obj_thresh=0.5):
    """Filter boxes with object threshold.
    """
    box_confidences = box_confidences.reshape(-1)
    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score* box_confidences >= obj_thresh)
    scores = (class_max_score* box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]

    return boxes, classes, scores

def draw(image, boxes, scores, classes, model_classes):
    img_copy = image.copy()
    if boxes is not None:
        for box, score, cl in zip(boxes, scores, classes):
            top, left, right, bottom = [int(_b) for _b in box]
            cv2.rectangle(img_copy, (top, left), (right, bottom), (255, 0, 0), 2)
            cv2.putText(img_copy, '{0} {1:.2f}'.format(model_classes[cl], score),
                        (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return img_copy

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
    def __init__(self) -> None:
        self.record_list = []
        self.letter_box_info = None

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
        
        self.letter_box_info = Letter_Box_Info(shape, new_shape, ratio, ratio, dw, dh, pad_color)
        if info_need is True:
            return im, ratio, (dw, dh)
        else:
            return im

    def get_real_box(self, box, in_format='xyxy'):
        bbox = copy(box)
        # unletter_box result
        if in_format=='xyxy':
            bbox[:,0] -= self.letter_box_info.dw
            bbox[:,0] /= self.letter_box_info.w_ratio
            bbox[:,0] = np.clip(bbox[:,0], 0, self.letter_box_info.origin_shape[1])

            bbox[:,1] -= self.letter_box_info.dh
            bbox[:,1] /= self.letter_box_info.h_ratio
            bbox[:,1] = np.clip(bbox[:,1], 0, self.letter_box_info.origin_shape[0])

            bbox[:,2] -= self.letter_box_info.dw
            bbox[:,2] /= self.letter_box_info.w_ratio
            bbox[:,2] = np.clip(bbox[:,2], 0, self.letter_box_info.origin_shape[1])

            bbox[:,3] -= self.letter_box_info.dh
            bbox[:,3] /= self.letter_box_info.h_ratio
            bbox[:,3] = np.clip(bbox[:,3], 0, self.letter_box_info.origin_shape[0])
        return bbox