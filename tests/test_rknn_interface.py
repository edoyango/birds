"""Tests for birdector.rknn_yolov5._interface - the ctypes struct
definitions and Python<->C data marshalling that sit directly on the
boundary with the compiled librknn_yolov5.so / src/yolov5 C++ code.

_interface.py itself has no platform gate or compiled-library dependency
(only birdector.rknn_yolov5's __init__.py does, to load the actual .so),
so it's loaded here via importlib directly from its file path - this
exercises the real module on any platform without needing to fake out
ctypes.CDLL or the aarch64/Linux check that its parent package enforces.
"""

import ctypes
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_INTERFACE_PATH = (
    Path(__file__).resolve().parents[1] / "birdector" / "rknn_yolov5" / "_interface.py"
)


@pytest.fixture(scope="module")
def interface():
    spec = importlib.util.spec_from_file_location(
        "birdector.rknn_yolov5._interface", _INTERFACE_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
        yield mod
    finally:
        sys.modules.pop(spec.name, None)


def test_module_loads_without_the_compiled_library_or_platform_gate(interface):
    # the parent package (birdector.rknn_yolov5) is what asserts aarch64
    # Linux and loads librknn_yolov5.so - importing _interface directly
    # must not have triggered any of that
    assert "birdector.rknn_yolov5" not in sys.modules
    assert len(interface.CLASS_NAMES) == len(interface._CLASS_COLOURS)
    assert len(interface.CLASS_NAMES) > 0


def test_numpy_to_image_buffer_t_rejects_non_uint8(interface):
    arr = np.zeros((4, 4, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        interface.numpy_to_image_buffer_t(arr, interface._image_format_t.IMAGE_FORMAT_RGB888)


def test_numpy_to_image_buffer_t_dimensions_and_format(interface):
    h, w, c = 6, 10, 3
    arr = np.arange(h * w * c, dtype=np.uint8).reshape(h, w, c)

    buf = interface.numpy_to_image_buffer_t(arr, interface._image_format_t.IMAGE_FORMAT_RGB888)

    assert buf.width == w
    assert buf.height == h
    # _image_buffer_t.format is a _image_format_t (a c_int subclass) field;
    # ctypes only auto-unboxes *plain* c_int fields to a Python int, so a
    # struct field typed as a c_int subclass stays a _image_format_t
    # instance - .value gets the underlying int
    assert buf.format.value == interface._image_format_t.IMAGE_FORMAT_RGB888
    assert buf.size == w * h * 3
    assert buf.fd == -1


def test_numpy_to_image_buffer_t_virt_addr_points_at_the_real_pixel_data(interface):
    h, w, c = 3, 4, 3
    arr = np.arange(h * w * c, dtype=np.uint8).reshape(h, w, c)

    buf = interface.numpy_to_image_buffer_t(arr, interface._image_format_t.IMAGE_FORMAT_RGB888)

    # read the bytes back out through the ctypes pointer the C side would
    # actually dereference, and confirm they match the source array exactly
    nbytes = h * w * c
    raw = ctypes.string_at(buf.virt_addr, nbytes)
    assert raw == arr.tobytes()


def test_numpy_to_image_buffer_t_handles_non_contiguous_input(interface):
    # a transposed view is a classic source of a non-contiguous array;
    # numpy_to_image_buffer_t is documented to copy it to a contiguous
    # buffer first rather than handing the C side a strided pointer
    base = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(6, 4, 3)
    arr = np.transpose(base, (1, 0, 2))  # now (4, 6, 3), non-contiguous
    assert not arr.flags["C_CONTIGUOUS"]

    buf = interface.numpy_to_image_buffer_t(arr, interface._image_format_t.IMAGE_FORMAT_RGB888)

    h, w = arr.shape[:2]
    raw = ctypes.string_at(buf.virt_addr, h * w * 3)
    assert raw == np.ascontiguousarray(arr).tobytes()


def _make_result_list(interface, boxes):
    """boxes: list of (left, top, right, bottom, prop, cls_id)."""
    results = interface._object_detect_result_list()
    results.count = len(boxes)
    for i, (left, top, right, bottom, prop, cls_id) in enumerate(boxes):
        r = results.results[i]
        r.box.left, r.box.top, r.box.right, r.box.bottom = left, top, right, bottom
        r.prop = prop
        r.cls_id = cls_id
    return results


def test_object_detect_result_wraps_ctypes_results_into_python_objects(interface):
    src_img = np.zeros((20, 20, 3), dtype=np.uint8)
    raw = _make_result_list(interface, [(1, 2, 3, 4, 0.9, 0), (5, 6, 7, 8, 0.5, 2)])

    wrapped = interface.object_detect_result(raw, src_img)

    assert len(wrapped.detections) == 2
    d0 = wrapped.detections[0]
    assert (d0.box.left, d0.box.top, d0.box.right, d0.box.bottom) == (1, 2, 3, 4)
    assert d0.score == pytest.approx(0.9)
    assert d0.class_id == 0
    assert wrapped.detections[1].class_id == 2
    assert wrapped.src_img is src_img


def test_object_detect_result_count_limits_which_raw_entries_are_read(interface):
    # only `count` entries should be turned into detections, even though
    # the underlying ctypes array is fixed-size and may hold stale data
    # beyond that point
    src_img = np.zeros((10, 10, 3), dtype=np.uint8)
    raw = _make_result_list(interface, [(0, 0, 1, 1, 0.1, 0)])
    raw.results[1].cls_id = 99  # would be garbage if ever read

    wrapped = interface.object_detect_result(raw, src_img)

    assert len(wrapped.detections) == 1


def test_draw_renders_without_mutating_the_source_image(interface):
    src_img = np.zeros((20, 20, 3), dtype=np.uint8)
    raw = _make_result_list(interface, [(2, 2, 10, 10, 0.9, 0)])
    wrapped = interface.object_detect_result(raw, src_img)

    drawn = wrapped.draw(conf=False)

    assert not np.array_equal(drawn, src_img), "draw() should have painted a box"
    assert np.array_equal(src_img, np.zeros((20, 20, 3), dtype=np.uint8)), "source image must be untouched"
