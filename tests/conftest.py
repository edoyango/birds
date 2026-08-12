"""Shared pytest fixtures.

birdector.rknn_yolov5 hard-requires aarch64 Linux: it asserts the platform
at import time and then loads a compiled librknn_yolov5.so that's only
built (via CMake, targeting rk3588) as part of the Docker image. Most
dev machines and CI runners are neither, so tests that need
birdector.extract_birbs (or anything else importing birdector.rknn_yolov5)
use the fake_rknn_yolov5 fixture below to install a lightweight stand-in
in sys.modules before that import happens.
"""

import sys
import types

import numpy as np
import pytest


class FakeDetection:
    def __init__(self, class_id: int):
        self.class_id = class_id


class FakeObjectDetectResult:
    """Stands in for birdector.rknn_yolov5.object_detect_result."""

    def __init__(self, detections=(), src_img=None, class_names=None):
        self.detections = list(detections)
        self.src_img = src_img
        self.class_names = class_names

    def draw(self, conf: bool = True):
        return self.src_img


class NullModel:
    """A fake rknn_yolov5.model that reports zero detections for every frame."""

    def __init__(self, model_path, anchors, npu_cores=None):
        self.model_path = model_path
        self.anchors = anchors
        self.npu_cores = npu_cores

    def inference(self, frames):
        batch = frames if frames.ndim == 4 else frames[None]
        return [
            FakeObjectDetectResult(class_names=CLASS_NAMES, src_img=batch[i])
            for i in range(batch.shape[0])
        ]

    def release(self):
        pass


CLASS_NAMES = ("Blackbird", "Currawong", "Dove", "Myna", "Sparrow", "Starling", "Wattlebird")


def make_fake_rknn_yolov5(model_cls=NullModel) -> types.ModuleType:
    """Build a fresh fake birdector.rknn_yolov5 module."""
    fake = types.ModuleType("birdector.rknn_yolov5")
    fake.CLASS_NAMES = CLASS_NAMES
    fake.object_detect_result = FakeObjectDetectResult
    fake.model = model_cls
    return fake


@pytest.fixture
def fake_rknn_yolov5():
    """Install a fake birdector.rknn_yolov5 for the duration of one test.

    Also drops any already-imported birdector.extract_birbs so that
    `from . import rknn_yolov5` re-binds to this fake on next import -
    mutating the returned module's `.model` attribute (before spawning any
    process that uses it) is enough to control what inference_worker sees,
    without needing to re-import anything.
    """
    fake = make_fake_rknn_yolov5()
    previous_rknn = sys.modules.get("birdector.rknn_yolov5")
    previous_eb = sys.modules.get("birdector.extract_birbs")
    sys.modules["birdector.rknn_yolov5"] = fake
    sys.modules.pop("birdector.extract_birbs", None)
    try:
        yield fake
    finally:
        if previous_rknn is not None:
            sys.modules["birdector.rknn_yolov5"] = previous_rknn
        else:
            sys.modules.pop("birdector.rknn_yolov5", None)
        if previous_eb is not None:
            sys.modules["birdector.extract_birbs"] = previous_eb
        else:
            sys.modules.pop("birdector.extract_birbs", None)


@pytest.fixture
def extract_birbs(fake_rknn_yolov5):
    """Import birdector.extract_birbs with the fake rknn_yolov5 in place."""
    import birdector.extract_birbs as eb

    return eb
