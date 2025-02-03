from pathlib import Path
import ctypes
from ._interface import _rknn_app_context_t, _object_detect_result_list, object_detect_result, numpy_to_image_buffer_t, _image_format_t
import numpy as np

# check that we're running on aarch64, Linux
import platform
if platform.system() != "Linux" or platform.machine() != "aarch64":
    raise ImportError("This module is only supported on aarch64 Linux systems")

_pwd = Path(__file__).absolute().parent

_rknn_yolov5_lib = ctypes.CDLL(_pwd / "librknn_yolov5.so")

class model():
    def __init__(self, model_path: Path, anchors: np.ndarray):
        self.model_path = model_path
        self._model_ctx = _rknn_app_context_t()
        ret = _rknn_yolov5_lib.init_yolov5_model(str(model_path).encode(), ctypes.byref(self._model_ctx))
        assert ret==0, f"Failed to initialize RKNN model at {model_path}"
        self.anchors = anchors
        assert isinstance(anchors, np.ndarray), "anchors must be a numpy array"
        assert anchors.shape == (3, 6), "anchors must be a 3x6 numpy array"
        self._anchors = np.ctypeslib.as_ctypes(anchors.astype(np.int32)) # should convert to a 3x6 raw int array

    def release(self):
        ret = _rknn_yolov5_lib.release_yolov5_model(self._model_ctx)
        assert ret==0, f"Failed to release RKNN model at {self.model_path}"

    def inference(self, img: np.ndarray, verbose: bool = False) -> object_detect_result:
        img_buf = numpy_to_image_buffer_t(img, _image_format_t.IMAGE_FORMAT_RGB888)
        res = _object_detect_result_list()
        ret = _rknn_yolov5_lib.inference_yolov5_model(
            ctypes.byref(self._model_ctx), 
            ctypes.byref(img_buf), 
            ctypes.byref(res), 
            ctypes.byref(self._anchors), 
            ctypes.c_bool(verbose)
        )
        assert ret==0, f"Failed to perform inference on RKNN model at {self.model_path}"
        return object_detect_result(res, img)