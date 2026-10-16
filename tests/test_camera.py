"""Tests for IMX500 tensor parsing and label handling (M1), using fakes."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from pi_ai_camera.camera import LABELS_FALLBACK, IMX500Detector, filter_labels
from pi_ai_camera.config import AppConfig


class FakeIMX500:
    """Stand-in for picamera2.devices.IMX500 recording converted coords."""

    def __init__(self, outputs: Any, input_size=(320, 320)) -> None:
        self.outputs = outputs
        self.input_size = input_size
        self.coords: list[tuple] = []

    def get_outputs(self, metadata, add_batch=False):
        return self.outputs

    def get_input_size(self):
        return self.input_size

    def convert_inference_coords(self, coords, metadata, picam2):
        self.coords.append(tuple(round(float(c), 4) for c in coords))
        y0, x0, y1, x1 = coords
        return (x0 * 100, y0 * 100, (x1 - x0) * 100, (y1 - y0) * 100)


def _detector(outputs, *, norm=False, order="yx", labels=("person", "car")) -> IMX500Detector:
    det = object.__new__(IMX500Detector)  # bypass hardware __init__
    det.cfg = AppConfig()
    det.imx500 = FakeIMX500(outputs)
    det.picam2 = None
    det.intrinsics = SimpleNamespace(bbox_normalization=norm, bbox_order=order)
    det.labels = list(labels)
    return det


def _outputs(boxes, scores, classes):
    # add_batch=True -> leading batch dimension on every tensor.
    return [np.array([boxes], float), np.array([scores], float), np.array([classes], float), np.array([[len(scores)]])]


def test_no_tensor_returns_none() -> None:
    assert _detector(None).parse({}) is None


def test_yx_boxes_threshold_and_labels() -> None:
    det = _detector(_outputs([[0.1, 0.2, 0.3, 0.6], [0, 0, 1, 1]], [0.9, 0.1], [1, 0]))
    dets = det.parse({})
    assert len(dets) == 1 and dets[0].label == "car"
    assert dets[0].box == (20, 10, 40, 20)


def test_xy_order_and_normalization() -> None:
    det = _detector(_outputs([[64, 32, 192, 96]], [0.8], [0]), norm=True, order="xy")
    det.parse({})
    # 320-px input: (x0,y0,x1,y1)=(64,32,192,96)/320 -> reordered to y0,x0,y1,x1
    assert det.imx500.coords == [(0.1, 0.2, 0.3, 0.6)]


def test_unknown_class_id_falls_back_to_number() -> None:
    dets = _detector(_outputs([[0, 0, 1, 1]], [0.9], [42])).parse({})
    assert dets[0].label == "42"


def test_bundled_coco_labels_layout() -> None:
    labels = LABELS_FALLBACK.read_text().splitlines()
    assert labels[0] == "person" and "-" in labels  # 91-id layout with placeholders
    assert len(filter_labels(labels, True)) == 80
    assert filter_labels(labels, False) == labels


def test_stop_is_idempotent_without_camera() -> None:
    det = _detector(None)
    det._started = False
    det.stop()
    det.stop()
