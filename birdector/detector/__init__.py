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

_lib = ctypes.CDLL(_pwd / "libbirds_detector.so")

_CORE_MASK_MAP = {
    0: _rknn_core_mask.RKNN_NPU_CORE_0,
    1: _rknn_core_mask.RKNN_NPU_CORE_1,
    2: _rknn_core_mask.RKNN_NPU_CORE_2,
}

class model():
    def __init__(self, model_path: Path, npu_cores=[-1]):
        """
        Args:
            model_path: YOLO26 .rknn model built by training/convert_rknn.py (graph cut before box decoding, no NMS).
            npu_cores: NPU core indices to use, or [-1] to let the runtime choose.
        """
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
        ret = _lib.init_model(str(model_path).encode(), ctypes.byref(self._model_ctx), self._core_mask)
        assert ret==0, f"Failed to initialize RKNN model at {model_path}"

    def release(self):
        ret = _lib.release_model(self._model_ctx)
        assert ret==0, f"Failed to release RKNN model at {self.model_path}"

    def inference(self, img: np.ndarray, verbose: bool = False, conf: float = 0.25) -> object_detect_result:
        """Run detection on one frame (HxWx3) or a batch (BxHxWx3) of uint8 pixels as OpenCV reads them (BGR).

        Detections scoring above conf are returned.

        The model is trained on RGB (measured: feeding it BGR cuts mAP50 from 0.88 to 0.45). The frames are passed as
        BGR888 and the RGA letterbox writes the model input in the model's order, so the channel swap costs no CPU and
        `img` is not modified (the CPU fallback of the letterbox swaps too).
        """
        # numpy_to_image_buffer_t copies a non-contiguous array into a temporary that it does not keep alive, so make
        # the contiguous array here and hold it until the call returns (a no-op for the usual contiguous frames)
        model_input = np.ascontiguousarray(img)
        img_buf = numpy_to_image_buffer_t(model_input, _image_format_t.IMAGE_FORMAT_BGR888)
        bs = 1 if img.ndim == 3 else img.shape[0]
        res = (_object_detect_result_list*bs)()
        ret = _lib.inference_yolo26_model(
            ctypes.byref(self._model_ctx),
            ctypes.byref(img_buf),
            res,
            ctypes.c_int32(bs),
            ctypes.c_float(conf),
            ctypes.c_bool(verbose)
        )
        assert ret==0, f"Failed to perform inference on RKNN model at {self.model_path}"
        frames = img if img.ndim == 4 else img[None]
        return [object_detect_result(res[i], frames[i]) for i in range(bs)]
