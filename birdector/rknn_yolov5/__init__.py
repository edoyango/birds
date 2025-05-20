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

class model():
    def __init__(self, model_path: Path, anchors: np.ndarray, npu_cores=[-1]):
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
        assert isinstance(anchors, np.ndarray), "anchors must be a numpy array"
        assert anchors.shape == (3, 6), f"anchors must be a 3x6 numpy array, but were {anchors.shape}"
        self._anchors = np.ctypeslib.as_ctypes(anchors.astype(np.int32)) # should convert to a 3x6 raw int array

    def release(self):
        ret = _rknn_yolov5_lib.release_yolov5_model(self._model_ctx)
        assert ret==0, f"Failed to release RKNN model at {self.model_path}"

    def inference(self, img: np.ndarray, verbose: bool = False) -> object_detect_result:
        img_buf = numpy_to_image_buffer_t(img, _image_format_t.IMAGE_FORMAT_RGB888)
        bs = 1 if img.ndim == 3 else img.shape[0]
        res = (_object_detect_result_list*bs)()
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
