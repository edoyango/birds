import numpy as np
import cv2
from copy import copy
from pathlib import Path
from typing import Tuple, List
from rknn.api import RKNN

# a mostly random collection of colours
CLASS_COLOURS = (
    (154, 64, 128),
    (169, 249, 117),
    (244, 86, 228),
    (15, 16, 210),
    (108, 184, 191),
    (221, 112, 37),
    (152, 107, 164),
    (255, 0, 0),
    (0, 0, 255),
    (80, 130, 185),
    (43, 178, 17),
    (252, 143, 62),
    (208, 23, 73),
    (255, 255, 0),
)


class inference_result:
    """
    A class to represent the result of an inference.

    Attributes
    ----------
    boxes : list
        A list of bounding boxes for detected objects.
    classes : list
        A list of class indices for detected objects.
    scores : list
        A list of confidence scores for detected objects.
    img : numpy.ndarray
        The image on which inference was performed.

    Methods
    -------
    draw(model_classes, conf=True):
        Draws bounding boxes and class labels on the image.
    """

    def __init__(
        self,
        boxes: np.ndarray,
        classes: np.ndarray,
        scores: np.ndarray,
        img: np.ndarray,
    ) -> None:
        """
        Initialize the object with bounding boxes, class labels, confidence scores, and an image.

        Args:
            boxes (list): A list of bounding boxes.
            classes (list): A list of class labels corresponding to the bounding boxes.
            scores (list): A list of confidence scores corresponding to the bounding boxes.
            img (numpy.ndarray): The image on which the detections were made.

        Attributes:
            boxes (list): Stores the bounding boxes.
            classes (list): Stores the class labels.
            scores (list): Stores the confidence scores.
            img (numpy.ndarray): A copy of the input image.
        """
        self.boxes = boxes
        self.classes = classes
        self.scores = scores
        self.img = img.copy()

    def draw(self, model_classes: np.ndarray, conf: bool = True) -> np.ndarray:
        """
        Draws bounding boxes and class labels on the image.

        Args:
            model_classes (list): List of class names corresponding to the detected objects.
            conf (bool, optional): If True, display the confidence score along with the class label. Defaults to True.

        Returns:
            numpy.ndarray: The image with bounding boxes and class labels drawn.
        """
        img_copy = self.img.copy()
        if self.boxes is not None:
            for box, score, cl in zip(self.boxes, self.scores, self.classes):
                top, left, right, bottom = [int(_b) for _b in box]
                cv2.rectangle(
                    img_copy, (top, left), (right, bottom), CLASS_COLOURS[cl], 2
                )
                if conf:
                    cv2.putText(
                        img_copy,
                        "{0} {1:.2f}".format(model_classes[cl], score),
                        (top, left - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        CLASS_COLOURS[cl],
                        2,
                    )
                else:
                    cv2.putText(
                        img_copy,
                        model_classes[cl],
                        (top, left - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        CLASS_COLOURS[cl],
                        2,
                    )
        return img_copy


class RKNN_model:
    """
    RKNN_model class for handling RKNN model loading, inference, and post-processing.
    Methods:
        __init__(model_path, target=None, device_id=None):
        run(inputs):
            Run inference on the given inputs using the loaded RKNN model.
        post_process(input_data, anchors, imgsz=(640, 640), nms_thresh=0.7):
        infer(inputs, anchors, imgsz, nms_thresh):
            Perform inference on the input image and return the results with bounding boxes, classes, and scores.
        release():
            Release the RKNN model and free up resources.
    """

    def __init__(
        self, model_path: Path, target: str = None, device_id: int = None
    ) -> None:
        """
        Initialize the RKNN model and runtime environment.
        Args:
            model_path (str): Path to the RKNN model file. Must end with '.rknn'.
            target (str, optional): Target device for the runtime environment. Defaults to None.
            device_id (str, optional): Device ID for the target device. Defaults to None.
        Raises:
            AssertionError: If the model_path does not end with '.rknn'.
            SystemExit: If initializing the runtime environment fails.
        """

        assert model_path.endswith(
            ".rknn"
        ), f"{model_path} is not rknn/pytorch/onnx model"

        rknn = RKNN()

        # Direct Load RKNN Model
        rknn.load_rknn(model_path)

        print("--> Init runtime environment")
        if target == None:
            ret = rknn.init_runtime()
        else:
            ret = rknn.init_runtime(target=target, device_id=device_id)
        if ret != 0:
            print("Init runtime environment failed")
            exit(ret)
        print("done")

        self.rknn = rknn

    # def __del__(self):
    #     self.release()

    def run(self, inputs: np.ndarray) -> List:
        """
        Run inference on the given inputs using the RKNN model.
        Args:
            inputs (numpy.ndarray): The input image to run inference on. If a single input is provided,
                                    it will be converted to a list.
        Returns:
            list: The inference results from the RKNN model. If the RKNN model has been released,
                  an empty list is returned.
        """
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
    def post_process(
        input_data: np.ndarray,
        anchors: np.ndarray,
        imgsz: Tuple[int, int] = (640, 640),
        nms_thresh: float = 0.7,
    ):
        """
        Post-process the RKNN inference output to get bounding boxes, classes, and scores.
        Args:
            input_data (list of np.ndarray): The output from the model inference.
            anchors (list of list of tuples): Anchor boxes used in the model.
            imgsz (tuple, optional): The size of the input image. Defaults to (640, 640).
            nms_thresh (float, optional): Non-max suppression threshold. Defaults to 0.7.

        Returns:
            tuple: A tuple containing:
                - boxes (np.ndarray): Array of bounding boxes.
                - classes (np.ndarray): Array of class indices for each box.
                - scores (np.ndarray): Array of confidence scores for each box.
        """
        boxes, scores, classes_conf = [], [], []
        # 1*255*h*w -> 3*85*h*w
        input_data = [
            _in.reshape([len(anchors[0]), -1] + list(_in.shape[-2:]))
            for _in in input_data
        ]
        for i in range(len(input_data)):
            boxes.append(
                box_process(input_data[i][:, :4, :, :], anchors[i], imgsz=imgsz)
            )
            scores.append(input_data[i][:, 4:5, :, :])
            classes_conf.append(input_data[i][:, 5:, :, :])

        def sp_flatten(_in):
            ch = _in.shape[1]
            _in = _in.transpose(0, 2, 3, 1)
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

    def infer(
        self,
        inputs: np.ndarray,
        anchors: np.ndarray,
        imgsz: Tuple[int, int],
        nms_thresh: float,
    ) -> inference_result:
        """
        Perform inference on the input image using the YOLOv5 model.

        Args:
            inputs (numpy.ndarray): The input image in numpy array format.
            anchors (list): List of anchor boxes used by the YOLOv5 model.
            imgsz (tuple): The size to which the input image should be resized.
            nms_thresh (float): The threshold for non-maximum suppression.

        Returns:
            inference_result: An object containing the detected bounding boxes,
                      classes, scores, and the original input image.
        """
        # create a letterbox helper to store original dimensions
        letterbox_helper = COCO_test_helper()
        # letter box image
        letterboxed_image = cv2.cvtColor(
            letterbox_helper.letter_box(inputs, imgsz), cv2.COLOR_BGR2RGB
        )
        # inference
        boxes, classes, scores = self.post_process(
            self.run([letterboxed_image]), anchors, imgsz, nms_thresh
        )
        # return results, with unletterboxed boxes
        if boxes is not None:
            boxes = letterbox_helper.get_real_box(boxes)
        return inference_result(boxes, classes, scores, inputs)

    def release(self) -> None:
        self.rknn.release()
        self.rknn = None


def nms_boxes(
    boxes: np.ndarray, scores: np.ndarray, nms_thresh: float = 0.7
) -> np.ndarray:
    """
    Suppress non-maximal boxes using Non-Maximum Suppression (NMS).
    Args:
        boxes (numpy.ndarray): Array of bounding boxes with shape (N, 4), where N is the number of boxes.
                            Each box is represented by four coordinates [x1, y1, x2, y2].
        scores (numpy.ndarray): Array of scores for each bounding box with shape (N,).
        nms_thresh (float): Threshold for the IoU (Intersection over Union) to suppress overlapping boxes. Default is 0.7.

    Returns:
        numpy.ndarray: Array of indices of the boxes that are kept after NMS.
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


def box_process(
    position: np.array, anchors: np.ndarray, imgsz: Tuple = (640, 640)
) -> np.ndarray:
    """
    Processes bounding box predictions from a neural network to convert them into
    a more usable format.
    Args:
        position (np.ndarray): The raw bounding box predictions from the network.
            Expected shape is (batch_size, 4, grid_h, grid_w).
        anchors (list or np.ndarray): The anchor boxes used in the network. Expected
            shape is (num_anchors, 2).
        imgsz (tuple, optional): The size of the input image. Default is (640, 640).

    Returns:
        np.ndarray: The processed bounding boxes in the format [x1, y1, x2, y2].
            Shape is (batch_size, 4, grid_h, grid_w).
    """
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([imgsz[1] // grid_h, imgsz[0] // grid_w]).reshape(1, 2, 1, 1)

    col = col.repeat(len(anchors), axis=0)
    row = row.repeat(len(anchors), axis=0)
    anchors = np.array(anchors)
    anchors = anchors.reshape(*anchors.shape, 1, 1)

    box_xy = position[:, :2, :, :] * 2 - 0.5
    box_wh = pow(position[:, 2:4, :, :] * 2, 2) * anchors

    box_xy += grid
    box_xy *= stride
    box = np.concatenate((box_xy, box_wh), axis=1)

    # Convert [c_x, c_y, w, h] to [x1, y1, x2, y2]
    xyxy = np.copy(box)
    xyxy[:, 0, :, :] = box[:, 0, :, :] - box[:, 2, :, :] / 2  # top left x
    xyxy[:, 1, :, :] = box[:, 1, :, :] - box[:, 3, :, :] / 2  # top left y
    xyxy[:, 2, :, :] = box[:, 0, :, :] + box[:, 2, :, :] / 2  # bottom right x
    xyxy[:, 3, :, :] = box[:, 1, :, :] + box[:, 3, :, :] / 2  # bottom right y

    return xyxy


def filter_boxes(
    boxes: np.ndarray,
    box_confidences: np.ndarray,
    box_class_probs: np.ndarray,
    obj_thresh: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter boxes with object threshold.
    Args:
        boxes (numpy.ndarray): Array of bounding boxes.
        box_confidences (numpy.ndarray): Array of confidences for each box.
        box_class_probs (numpy.ndarray): Array of class probabilities for each box.
        obj_thresh (float): Object confidence threshold. Default is 0.5.

    Returns:
        tuple: A tuple containing:
            - boxes (numpy.ndarray): Filtered bounding boxes.
            - classes (numpy.ndarray): Class indices for the filtered boxes.
            - scores (numpy.ndarray): Scores for the filtered boxes.
    """
    box_confidences = box_confidences.reshape(-1)
    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score * box_confidences >= obj_thresh)
    scores = (class_max_score * box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]

    return boxes, classes, scores


class Letter_Box_Info:
    """
    A class to store information about letterbox resizing.

    Attributes:
        origin_shape (tuple): The original shape of the image.
        new_shape (tuple): The new shape of the image after resizing.
        w_ratio (float): The width ratio between the new shape and the original shape.
        h_ratio (float): The height ratio between the new shape and the original shape.
        dw (int): The width padding added to the image.
        dh (int): The height padding added to the image.
        pad_color (tuple): The color used for padding.
    """

    def __init__(
        self,
        shape: Tuple[int, int],
        new_shape: Tuple[int, int],
        w_ratio: float,
        h_ratio: float,
        dw: int,
        dh: int,
        pad_color: tuple[int, int, int],
    ) -> None:
        """
        Initializes the parameters for resizing and padding an image.

        Args:
            shape (tuple): The original shape of the image (height, width).
            new_shape (tuple): The new shape of the image (height, width).
            w_ratio (float): The width ratio for resizing.
            h_ratio (float): The height ratio for resizing.
            dw (int): The padding added to the width.
            dh (int): The padding added to the height.
            pad_color (tuple): The color used for padding (R, G, B).
        """
        self.origin_shape = shape
        self.new_shape = new_shape
        self.w_ratio = w_ratio
        self.h_ratio = h_ratio
        self.dw = dw
        self.dh = dh
        self.pad_color = pad_color


class COCO_test_helper:
    """
    A helper class for handling COCO dataset related operations, specifically for resizing and padding images
    while maintaining aspect ratio and for converting bounding box coordinates.
    Attributes:
        letter_box_info (Letter_Box_Info): Stores information about the letterbox transformation.
    Methods:
        __init__():
            Initializes the COCO_test_helper instance.
        letter_box(im, new_shape, pad_color=(0,0,0), info_need=False):
            Resizes and pads the input image to the new shape while maintaining aspect ratio.
            Optionally returns the transformation ratio and padding information.
        get_real_box(box, in_format='xyxy'):
            Converts the bounding box coordinates from the letterboxed image back to the original image coordinates.
    """

    def __init__(self) -> None:
        """
        Initializes the instance variables for the class.

        Attributes:
            letter_box_info (None): Placeholder for letter box information, initialized to None.
        """
        self.letter_box_info = None

    def letter_box(
        self,
        im: np.ndarray,
        new_shape: Tuple[int, int, int],
        pad_color: Tuple[int, int, int] = (0, 0, 0),
        info_need: bool = False,
    ) -> np.ndarray:
        """
        Resize and pad an image to fit a new shape while maintaining aspect ratio.
        Args:
            im (numpy.ndarray): The input image to be resized and padded.
            new_shape (int or tuple): The desired shape (height, width) to resize the image to. If an integer is provided, it will be used for both dimensions.
            pad_color (tuple, optional): The color of the padding to be added to the image. Default is (0, 0, 0) which is black.
            info_need (bool, optional): If True, returns additional information about the resizing and padding. Default is False.
        Returns:
            numpy.ndarray: The resized and padded image.
            tuple (optional): If info_need is True, returns a tuple containing:
                - ratio (float): The scaling ratio used for resizing.
                - (dw, dh) (tuple): The padding added to the width and height respectively.
        """
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
        im = cv2.copyMakeBorder(
            im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color
        )  # add border

        self.letter_box_info = Letter_Box_Info(
            shape, new_shape, ratio, ratio, dw, dh, pad_color
        )
        if info_need is True:
            return im, ratio, (dw, dh)
        else:
            return im

    def get_real_box(self, box: np.ndarray, in_format: str = "xyxy") -> np.ndarray:
        """
        Transforms bounding box coordinates from letterboxed image back to original image coordinates.

        Args:
            box (numpy.ndarray): Array of shape (N, 4) containing bounding box coordinates
                in the letterboxed image space, where N is the number of boxes.
            in_format (str, optional): Format of input box coordinates. Currently only supports 'xyxy'.
                Defaults to 'xyxy'.

        Returns:
            numpy.ndarray: Array of shape (N, 4) containing transformed bounding box coordinates
                in the original image space. Coordinates are clipped to image boundaries.

        Notes:
            - For 'xyxy' format, box coordinates are in order: [x1, y1, x2, y2]
            - The method removes letterboxing padding and scales coordinates according to
              the letterboxing ratios stored in self.letter_box_info
            - Coordinates are clipped to ensure they fall within original image dimensions
        """
        bbox = copy(box)
        # unletter_box result
        if in_format == "xyxy":
            bbox[:, 0] -= self.letter_box_info.dw
            bbox[:, 0] /= self.letter_box_info.w_ratio
            bbox[:, 0] = np.clip(bbox[:, 0], 0, self.letter_box_info.origin_shape[1])

            bbox[:, 1] -= self.letter_box_info.dh
            bbox[:, 1] /= self.letter_box_info.h_ratio
            bbox[:, 1] = np.clip(bbox[:, 1], 0, self.letter_box_info.origin_shape[0])

            bbox[:, 2] -= self.letter_box_info.dw
            bbox[:, 2] /= self.letter_box_info.w_ratio
            bbox[:, 2] = np.clip(bbox[:, 2], 0, self.letter_box_info.origin_shape[1])

            bbox[:, 3] -= self.letter_box_info.dh
            bbox[:, 3] /= self.letter_box_info.h_ratio
            bbox[:, 3] = np.clip(bbox[:, 3], 0, self.letter_box_info.origin_shape[0])
        return bbox
