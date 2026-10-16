"""IMX500 (Raspberry Pi AI Camera) object detection backend.

Inference runs on the sensor's own NPU; the Pi only parses output tensors
delivered in each frame's metadata. Based on the picamera2
``examples/imx500/imx500_object_detection_demo.py`` flow.
"""
from __future__ import annotations

from typing import Iterator, Optional

import numpy as np

from .config import AppConfig
from .detection import Detection


class IMX500Detector:
    def __init__(self, cfg: AppConfig):
        # Import lazily so the package can be imported off-device.
        from picamera2 import Picamera2
        from picamera2.devices import IMX500
        from picamera2.devices.imx500 import NetworkIntrinsics

        self.cfg = cfg
        # IMX500 must be created before Picamera2 so the network firmware is loaded.
        self.imx500 = IMX500(cfg.imx500.model)
        intrinsics = self.imx500.network_intrinsics
        if not intrinsics:
            intrinsics = NetworkIntrinsics()
            intrinsics.task = "object detection"
        elif intrinsics.task != "object detection":
            raise ValueError(f"Model is not an object detection network (task={intrinsics.task!r})")

        if cfg.imx500.bbox_normalization is not None:
            intrinsics.bbox_normalization = cfg.imx500.bbox_normalization
        if cfg.imx500.bbox_order is not None:
            intrinsics.bbox_order = cfg.imx500.bbox_order
        if cfg.imx500.inference_rate is not None:
            intrinsics.inference_rate = cfg.imx500.inference_rate
        labels = cfg.load_labels()
        if labels:
            intrinsics.labels = labels
        intrinsics.update_with_defaults()  # fills COCO labels / inference_rate if unset
        self.intrinsics = intrinsics

        self.picam2 = Picamera2(self.imx500.camera_num)
        self.config = self.picam2.create_preview_configuration(
            controls={"FrameRate": intrinsics.inference_rate}, buffer_count=12
        )

    def start(self, preview: bool = False) -> None:
        self.imx500.show_network_fw_progress_bar()
        self.picam2.start(self.config, show_preview=preview)
        if self.intrinsics.preserve_aspect_ratio:
            self.imx500.set_auto_aspect_ratio()

    def stop(self) -> None:
        self.picam2.stop()
        self.picam2.close()

    def _labels(self) -> list[str]:
        labels = self.intrinsics.labels or []
        if self.intrinsics.ignore_dash_labels:
            labels = [l for l in labels if l and l != "-"]
        return labels

    def parse(self, metadata: dict) -> Optional[list[Detection]]:
        """Return detections for one frame, or None if no tensor was attached."""
        np_outputs = self.imx500.get_outputs(metadata, add_batch=True)
        if np_outputs is None:
            return None
        _, input_h = self.imx500.get_input_size()
        threshold = self.cfg.detection.threshold

        # Post-processed (_pp) SSD models output boxes, scores, classes, count.
        boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
        if self.intrinsics.bbox_normalization:
            boxes = boxes / input_h
        if self.intrinsics.bbox_order == "xy":
            boxes = boxes[:, [1, 0, 3, 2]]

        labels = self._labels()
        results: list[Detection] = []
        for box, score, cls in zip(boxes, scores, classes):
            if score < threshold:
                continue
            x, y, w, h = self.imx500.convert_inference_coords(tuple(box), metadata, self.picam2)
            idx = int(cls)
            label = labels[idx] if 0 <= idx < len(labels) else str(idx)
            results.append(Detection(label, float(score), (int(x), int(y), int(w), int(h))))
        results.sort(key=lambda d: d.score, reverse=True)
        return results[: self.cfg.detection.max_detections]

    def detections(self) -> Iterator[list[Detection]]:
        while True:
            metadata = self.picam2.capture_metadata()
            dets = self.parse(metadata)
            if dets is not None:
                yield dets
