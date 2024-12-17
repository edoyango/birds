import os
import cv2
import sys
import argparse
from copy import copy
import csv

import numpy as np
import multiprocessing as mp

from rknn.api import RKNN


OBJ_THRESH = 0.45
NMS_THRESH = 0.45

# The follew two param is for map test
# OBJ_THRESH = 0.001
# NMS_THRESH = 0.65

IMG_SIZE = (704, 704)  # (width, height), such as (1280, 736)

CLASSES = ("Blackbird", "Butcherbird", "Currawong", "Dove", "Lorikeet", "Myna", "Sparrow", "Starling", "Wattlebird")

class RKNN_model_container():
    def __init__(self, model_path, target=None, device_id=None) -> None:
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

    def release(self):
        self.rknn.release()
        self.rknn = None

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

def open_video(vname):

    cap = cv2.VideoCapture(vname)

    if (cap.isOpened() == False):  
        raise RuntimeError("Error reading video file")

    return cap


def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with object threshold.
    """
    box_confidences = box_confidences.reshape(-1)
    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score* box_confidences >= OBJ_THRESH)
    scores = (class_max_score* box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]

    return boxes, classes, scores

def nms_boxes(boxes, scores):
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
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def box_process(position, anchors):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1]//grid_h, IMG_SIZE[0]//grid_w]).reshape(1,2,1,1)

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

def post_process(input_data, anchors):
    boxes, scores, classes_conf = [], [], []
    # 1*255*h*w -> 3*85*h*w
    input_data = [_in.reshape([len(anchors[0]),-1]+list(_in.shape[-2:])) for _in in input_data]
    for i in range(len(input_data)):
        boxes.append(box_process(input_data[i][:,:4,:,:], anchors[i]))
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
        keep = nms_boxes(b, s)

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


def draw(image, boxes, scores, classes):
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = [int(_b) for _b in box]
        print("%s @ (%d %d %d %d) %.3f" % (CLASSES[cl], top, left, right, bottom, score))
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

def setup_model(args):
    model_path = args.model_path
    if model_path.endswith('.rknn'):
        platform = 'rknn' 
        model = RKNN_model_container(args.model_path, args.target, args.device_id)
    else:
        assert False, "{} is not rknn/pytorch/onnx model".format(model_path)
    print('Model-{} is {} model, starting val'.format(model_path, platform))
    return model, platform

def img_check(path):
    img_type = ['.jpg', '.jpeg', '.png', '.bmp']
    for _type in img_type:
        if path.endswith(_type) or path.endswith(_type.upper()):
            return True
    return False

def count_detections(classes):

    counts = {}
    for i in classes:
        ii = int(i)
        if counts.get(ii):
            counts[ii] += 1
        else:
            counts[ii] = 1
    
    return {CLASSES[k]: v for k, v in counts.items()}

def video_writer_worker(queue, w, h):
    cap = cv2.VideoWriter(
        "result/output_video.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        10,
        (w, h)
    )
    total_class_count = {c: 0 for c in CLASSES}
    if not cap.isOpened(): raise RuntimeError(f"Couldn't create video file result/output_video.mp4")
    nframes = 0
    while True:
        res = queue.get()
        if res is None:
            break
        else:
            frame, classes = res
            cap.write(frame)
            classes_count = {} if classes is None else count_detections(classes)
            nframes += 1
        for k, v in classes_count.items():
            total_class_count[k] += v
    total_instances = sum([v for v in total_class_count.values()])
    header = ["reference video path", "start time", "original video path", "original first frame path", "trigger video path", "trigger first frame path", "nframes", "average ninstance per frame"]
    row = ["result/output_video.mp4", "00-00-00", "test_original_video_path", "test_original_first_frame_path", "test_trigger_video_path", "test_trigger_first_frame_path", str(nframes), str(total_instances/nframes)]
    with open("result/meta.csv", "a") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(header)  
        csvwriter.writerow(row)
    cap.release()

    # delete video if too short
    # if nframes <= minframes + 1*fps:
    #     os.remove(path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    # basic params
    parser.add_argument('--model_path', type=str, required= True, help='model path, could be .pt or .rknn file')
    parser.add_argument('--target', type=str, default='rk3566', help='target RKNPU platform')
    parser.add_argument('--device_id', type=str, default=None, help='device id')
    
    parser.add_argument('--img_save', action='store_true', default=False, help='save the result')

    # data params
    parser.add_argument('--video', '-v', type=str, default='0', help='Video to watch. Can be either video device index, e.g. 0, or video file e.g. video.mkv.')
    parser.add_argument('--anchors', type=str, default='../model/anchors_yolov5.txt', help='target to anchor file, only yolov5, yolov7 need this param')

    args = parser.parse_args()

    # load anchor
    with open(args.anchors, 'r') as f:
        values = [float(_v) for _v in f.readlines()]
        anchors = np.array(values).reshape(3,-1,2).tolist()
    print("use anchors from '{}', which is {}".format(args.anchors, anchors))
    
    # init model
    model, platform = setup_model(args)

    co_helper = COCO_test_helper()

    # open video
    cap = open_video(args.video)
    fps=10.0 # temporary fix
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wait_limit = 50
    print(f"""INPUT VIDEO: {args.video}
    Resolution: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}x{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}
    FPS:        {fps}
OUTPUT DIRECTORY: result/{{originals,triggers}}
DETECTION MODEL: {args.model_path}
WAIT LIMIT: {wait_limit} frames

Beginning processing...
""")
    
    # initiate video writer
    frame_queue = mp.Queue()
    worker = mp.Process(
        target=video_writer_worker,
        args=(frame_queue, w, h)
    )
    worker.start()

    # run test
    suc = True
    iframe = 0
    while suc and iframe < total_frames:
        print('infer {}/{}'.format(iframe, total_frames), end='\r')

        suc, frame = cap.read()

        # letterbox video as per training
        pad_color = (0,0,0)
        img = co_helper.letter_box(im= frame.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0,0,0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # inference
        outputs = model.run([img])
        boxes, classes, scores = post_process(outputs, anchors)

        # draw boxes on top of frame
        img_p = frame.copy()
        if boxes is not None:
            draw(img_p, co_helper.get_real_box(boxes), scores, classes)

        # send frame and classes to video worker
        frame_queue.put((img_p, classes))

        # check video writer worker is still alive
        if worker.exitcode:
            raise RuntimeError(f"Worker has died with error code {worker.exitcode}")

        iframe +=1
    
    frame_queue.put(None)

    # release
    model.release()
    cap.release()
