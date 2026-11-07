from pathlib import Path
import ctypes
from ._interface import _rknn_core_mask, _rknn_app_context_t, _object_detect_result_list, object_detect_result, numpy_to_image_buffer_t, _image_format_t, CLASS_NAMES
import numpy as np
import functools, operator

# check that we're running on aarch64, Linux
import platform
if platform.system() != "Linux" or platform.machine() != "aarch64":
    raise ImportError("This module is only supported on aarch64 Linux systems")

_pwd = Path(__file__).absolute().parent

_rknn_yolov5_lib = ctypes.CDLL(_pwd / "librknn_yolov5.so")

_CORE_MASK_MAP = {
    0: _rknn_core_mask.RKNN_NPU_CORE_0,
    1: _rknn_core_mask.RKNN_NPU_CORE_1,
    2: _rknn_core_mask.RKNN_NPU_CORE_2,
}

ARCHITECTURES = ("yolov5", "yolo26")  # "yolov5" covers YOLOv5 and YOLOv7 (anchor-based); "yolo26" is NMS-free

class model():
    def __init__(self, model_path: Path, anchors: np.ndarray = None, npu_cores=[-1], arch: str = "yolov5"):
        """
        Args:
            model_path: compiled .rknn model.
            anchors: 3x6 anchor array. Required for arch="yolov5", unused for arch="yolo26".
            npu_cores: NPU core indices to use, or [-1] to let the runtime choose.
            arch: "yolov5" (YOLOv5/YOLOv7, anchors + NMS on CPU) or "yolo26" (graph cut before box decoding, no NMS).
        """
        assert arch in ARCHITECTURES, f"arch must be one of {ARCHITECTURES}, but was {arch!r}"
        self.arch = arch
        self.model_path = model_path
        self._model_ctx = _rknn_app_context_t()

        # calculate NPU core mask
        assert all(type(n) == int for n in npu_cores), "Failed to init rknn model: all elements in npu_cores must be int"
        assert any(-1 <= n <= 2 for n in npu_cores), "Failed to init rknn model: npu_cores must contain elements between -1 and 2 (inclusive)"
        if -1 in npu_cores:
            self._core_mask = _rknn_core_mask.RKNN_NPU_CORE_AUTO
        else:
            core_maps = [_CORE_MASK_MAP[n] for n in npu_cores]
            self._core_mask = functools.reduce(operator.or_, core_maps)
        
        # call C lib to init rknn model
        ret = _rknn_yolov5_lib.init_yolov5_model(str(model_path).encode(), ctypes.byref(self._model_ctx), self._core_mask)
        assert ret==0, f"Failed to initialize RKNN model at {model_path}"
        self.anchors = anchors
        if arch == "yolov5":
            assert isinstance(anchors, np.ndarray), "anchors must be a numpy array"
            assert anchors.shape == (3, 6), f"anchors must be a 3x6 numpy array, but were {anchors.shape}"
            self._anchors = np.ctypeslib.as_ctypes(anchors.astype(np.int32)) # should convert to a 3x6 raw int array

    def release(self):
        ret = _rknn_yolov5_lib.release_yolov5_model(self._model_ctx)
        assert ret==0, f"Failed to release RKNN model at {self.model_path}"

    def inference(self, img: np.ndarray, verbose: bool = False, conf: float = 0.25) -> object_detect_result:
        """Run detection on one frame (HxWx3) or a batch (BxHxWx3) of uint8 pixels as OpenCV reads them (BGR).

        conf is the score threshold for arch="yolo26". For arch="yolov5" the threshold is compiled in (BOX_THRESH)
        and conf is ignored.

        arch="yolo26" models are trained on RGB (measured: feeding them BGR cuts mAP50 from 0.88 to 0.45). The frames
        are passed as BGR888 and the RGA letterbox writes the model input as RGB888, so the channel swap costs no CPU
        and `img` is not modified (the CPU fallback of the letterbox swaps too). arch="yolov5" passes the pixels
        through exactly as before.
        """
        fmt = _image_format_t.IMAGE_FORMAT_BGR888 if self.arch == "yolo26" else _image_format_t.IMAGE_FORMAT_RGB888
        # numpy_to_image_buffer_t copies a non-contiguous array into a temporary that it does not keep alive, so make
        # the contiguous array here and hold it until the call returns (a no-op for the usual contiguous frames)
        model_input = np.ascontiguousarray(img)
        img_buf = numpy_to_image_buffer_t(model_input, fmt)
        bs = 1 if img.ndim == 3 else img.shape[0]
        res = (_object_detect_result_list*bs)()
        if self.arch == "yolo26":
            ret = _rknn_yolov5_lib.inference_yolo26_model(
                ctypes.byref(self._model_ctx),
                ctypes.byref(img_buf),
                res,
                ctypes.c_int32(bs),
                ctypes.c_float(conf),
                ctypes.c_bool(verbose)
            )
        else:
            ret = _rknn_yolov5_lib.inference_yolov5_model(
                ctypes.byref(self._model_ctx),
                ctypes.byref(img_buf),
                res,
                ctypes.byref(self._anchors),
                ctypes.c_int32(bs),
                ctypes.c_bool(verbose)
            )
        assert ret==0, f"Failed to perform inference on RKNN model at {self.model_path}"
        return [object_detect_result(res[i], img[i]) for i in range(bs)]
