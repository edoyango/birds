import ctypes as _ctypes
import numpy as _np
import cv2 as _cv2

_RKNN_MAX_DIMS = 16      # src/3rdparty/include/rknpu2/rknn_api.h
_RKNN_MAX_NAME_LEN = 256 # src/3rdparty/include/rknpu2/rknn_api.h
_OBJ_NUMB_MAX_SIZE = 128 # src/yolov5/yolov5.h

# move this to a user cfg file
_CLASS_COLOURS = (
    (208, 224, 64), # Blackbird
    (255,   0, 255), # Butcherbird
    (  0, 215, 255), # Currawong
    (255,   0,   0), # Dove
    (  0, 165, 255), # Lorikeet
    (255, 255,   0), # Myna
    (128,   0, 128), # Sparrow
    (  0, 255, 255), # Starling
    (  0,   0, 255), # Wattlebird
)

# move this to a user cfg file
CLASS_NAMES = (
    "Blackbird",
    "Butcherbird",
    "Currawong",
    "Dove",
    "Lorikeet",
    "Myna",
    "Sparrow",
    "Starling",
    "Wattlebird",
)

class _image_format_t(_ctypes.c_int):
    IMAGE_FORMAT_GRAY8 = 0
    IMAGE_FORMAT_RGB888 = 1
    IMAGE_FORMAT_RGBA8888 = 2
    IMAGE_FORMAT_YUV420SP_NV21 = 3
    IMAGE_FORMAT_YUV420SP_NV12 = 4

class _image_buffer_t(_ctypes.Structure):
    _fields_ = [
        ("width", _ctypes.c_int),
        ("height", _ctypes.c_int),
        ("width_stride", _ctypes.c_int),
        ("height_stride", _ctypes.c_int),
        ("format", _image_format_t),
        ("virt_addr", _ctypes.POINTER(_ctypes.c_ubyte)),  # unsigned char*
        ("size", _ctypes.c_int),
        ("fd", _ctypes.c_int),
    ]

class _rknn_input_output_num(_ctypes.Structure):
    _fields_ = [
        ("n_input", _ctypes.c_uint32),
        ("n_output", _ctypes.c_uint32),
    ]

class _rknn_tensor_format(_ctypes.c_int):
    RKNN_TENSOR_NCHW=0
    RKNN_TENSOR_NHWC=1
    RKNN_TENSOR_NC1HWC2=2
    RKNN_TENSOR_UNDEFINED=3
    RKNN_TENSOR_FORMAT_MAX=4

class _rknn_tensor_type(_ctypes.c_int):
    RKNN_TENSOR_FLOAT32=0 # data type is float32.
    RKNN_TENSOR_FLOAT16=1 # data type is float16.
    RKNN_TENSOR_INT8=2    # data type is int8.
    RKNN_TENSOR_UINT8=3   # data type is uint8.
    RKNN_TENSOR_INT16=4   # data type is int16.
    RKNN_TENSOR_UINT16=5  # data type is uint16.
    RKNN_TENSOR_INT32=6   # data type is int32.
    RKNN_TENSOR_UINT32=7  # data type is uint32.
    RKNN_TENSOR_INT64=8   # data type is int64.
    RKNN_TENSOR_BOOL=9
    RKNN_TENSOR_INT4=10
    RKNN_TENSOR_BFLOAT16=11

class _rknn_tensor_qnt_type(_ctypes.c_int):
    RKNN_TENSOR_QNT_NONE=0              # none.
    RKNN_TENSOR_QNT_DFP=1               # dynamic fixed point.
    RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC=2 # asymmetric affine.
    RKNN_TENSOR_QNT_MAX=3

class _rknn_core_mask(_ctypes.c_int):
    RKNN_NPU_CORE_AUTO=0
    RKNN_NPU_CORE_0=1
    RKNN_NPU_CORE_1=2
    RKNN_NPU_CORE_2=4
    RKNN_NPU_CORE_ALL=0xfff

class _rknn_tensor_attr(_ctypes.Structure):
    _fields_ = [
        ("index", _ctypes.c_uint32),
        ("n_dims", _ctypes.c_uint32),
        ("dims", _ctypes.c_uint32*_RKNN_MAX_DIMS),
        ("name", _ctypes.c_char*_RKNN_MAX_NAME_LEN),
        ("n_elems", _ctypes.c_uint32),
        ("size", _ctypes.c_uint32),
        ("fmt", _ctypes.c_int),
        ("type", _ctypes.c_int),
        ("qnt_type", _ctypes.c_int),
        ("fl", _ctypes.c_int8),
        ("zp", _ctypes.c_int32),
        ("scale", _ctypes.c_float),
        ("w_stride", _ctypes.c_uint32),
        ("size_with_stride", _ctypes.c_uint32),
        ("pass_through", _ctypes.c_uint8),
        ("h_stride", _ctypes.c_uint32)
    ]

class _rknn_app_context_t(_ctypes.Structure):
    _fields_ = [
        ('rknn_ctx', _ctypes.c_uint32),
        ('io_num', _rknn_input_output_num),
        ('input_attrs', _ctypes.POINTER(_rknn_tensor_attr)),
        ('output_attrs', _ctypes.POINTER(_rknn_tensor_attr)),
        ("model_channel", _ctypes.c_int32),
        ("model_width", _ctypes.c_int32),
        ("model_height", _ctypes.c_int32),
        ("is_quant", _ctypes.c_bool),
    ]

class _image_rect_t(_ctypes.Structure):
    _fields_ = [
        ("left", _ctypes.c_int),
        ("top", _ctypes.c_int),
        ("right", _ctypes.c_int),
        ("bottom", _ctypes.c_int),
    ]

class _object_detect_result(_ctypes.Structure):
    _fields_ = [
        ("box", _image_rect_t),
        ("prop", _ctypes.c_float),
        ("cls_id", _ctypes.c_int),
    ]

class _object_detect_result_list(_ctypes.Structure):
    _fields_ = [
        ("id", _ctypes.c_int),
        ("count", _ctypes.c_int),
        ("results", _object_detect_result*_OBJ_NUMB_MAX_SIZE),
    ]

def numpy_to_image_buffer_t(arr: _np.ndarray, format: _image_format_t, fd: int = -1) -> _image_buffer_t:
    """
    Convert a NumPy array to an image_buffer_t struct.

    Args:
        arr: A NumPy array of dtype uint8.
        format: The image format (e.g., IMAGE_FORMAT_RGB, IMAGE_FORMAT_BGR).
        fd: The file descriptor (default is -1, meaning no file descriptor).

    Returns:
        An image_buffer_t struct.
    """
    if arr.dtype != _np.uint8:
        raise ValueError("The NumPy array must have dtype uint8.")

    height, width = arr.shape[:2]  # Assumes shape is (height, width, channels) for multi-channel images
    # channels = 1 if arr.ndim == 2 else arr.shape[2]  # Handle grayscale (2D) and color (3D) images

    # Calculate strides (currently unused)
    width_stride = arr.strides[1]  # Stride along the width dimension
    height_stride = arr.strides[0]  # Stride along the height dimension

    # Ensure the array is contiguous
    if not arr.flags['C_CONTIGUOUS']:
        arr = _np.ascontiguousarray(arr)

    # Create an image_buffer_t struct
    image_buffer = _image_buffer_t()
    image_buffer.width = width
    image_buffer.height = height
    image_buffer.width_stride = width_stride
    image_buffer.height_stride = height_stride
    image_buffer.format = format
    image_buffer.virt_addr = arr.ctypes.data_as(_ctypes.POINTER(_ctypes.c_ubyte))  # Pointer to the array data
    image_buffer.size = 0 #arr.nbytes  # Total size in bytes
    image_buffer.fd = fd  # File descriptor (if applicable)

    return image_buffer

class box():
    def __init__(self, cresult: _image_rect_t):
        self.left = cresult.left
        self.top = cresult.top
        self.right = cresult.right
        self.bottom = cresult.bottom

class detection():
    def __init__(self, cresult: _object_detect_result):
        self.box = box(cresult.box)
        self.score = cresult.prop
        self.class_id = cresult.cls_id

class object_detect_result():
    def __init__(self, cresult: _object_detect_result_list, input_img: _np.ndarray):
        self.detections = [detection(cresult.results[i]) for i in range(cresult.count)]
        self.src_img = input_img
        self.class_names = CLASS_NAMES
    def draw(self, conf: bool = True) -> _np.ndarray:
        img_copy = self.src_img.copy()
        for d in self.detections:
            _cv2.rectangle(
                img_copy, (d.box.left, d.box.top), (d.box.right, d.box.bottom), _CLASS_COLOURS[d.class_id], 2
            )
            label = "{0} {1:.2f}".format(CLASS_NAMES[d.class_id], d.score) if conf else f"{CLASS_NAMES[d.class_id]}"
            _cv2.putText(
                img_copy,
                label,
                (d.box.left, d.box.top - 6),
                _cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                _CLASS_COLOURS[d.class_id],
                2,
            )
        return img_copy
