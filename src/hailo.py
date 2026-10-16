"""Hailo AI HAT+ backend (optional).

Frames come from the camera (the AI Camera works as a normal camera here)
and inference runs on the Hailo NPU over PCIe using ``picamera2.devices.Hailo``.
Expects a YOLO HEF compiled with on-chip NMS, e.g. the ``hailo-models``
package's ``/usr/share/hailo-models/yolov8s_h8l.hef``.
"""
from __future__ import annotations

from typing import Iterator

from .config import AppConfig
from .detection import Detection

COCO_LABELS = (
    "person bicycle car motorcycle airplane bus train truck boat traffic_light fire_hydrant "
    "stop_sign parking_meter bench bird cat dog horse sheep cow elephant bear zebra giraffe "
    "backpack umbrella handbag tie suitcase frisbee skis snowboard sports_ball kite "
    "baseball_bat baseball_glove skateboard surfboard tennis_racket bottle wine_glass cup "
    "fork knife spoon bowl banana apple sandwich orange broccoli carrot hot_dog pizza donut "
    "cake chair couch potted_plant bed dining_table toilet tv laptop mouse remote keyboard "
    "cell_phone microwave oven toaster sink refrigerator book clock vase scissors teddy_bear "
    "hair_drier toothbrush"
).split()


class HailoDetector:
    def __init__(self, cfg: AppConfig):
        from picamera2 import Picamera2
        from picamera2.devices import Hailo

        self.cfg = cfg
        self.hailo = Hailo(cfg.hailo.model)
        model_h, model_w, _ = self.hailo.get_input_shape()
        self.model_size = (model_w, model_h)
        self.main_size = tuple(cfg.hailo.main_size)
        self.labels = cfg.load_labels() or list(COCO_LABELS)

        self.picam2 = Picamera2(cfg.hailo.camera_num)
        self.config = self.picam2.create_preview_configuration(
            main={"size": self.main_size, "format": "XRGB8888"},
            lores={"size": self.model_size, "format": "RGB888"},
            controls={"FrameRate": 30},
        )

    def start(self, preview: bool = False) -> None:
        self.picam2.start(self.config, show_preview=preview)

    def stop(self) -> None:
        self.picam2.stop()
        self.picam2.close()
        self.hailo.close()

    def parse(self, results) -> list[Detection]:
        """HEF NMS output: one array per class of [y0, x0, y1, x1, score] (normalised)."""
        w, h = self.main_size
        dets: list[Detection] = []
        for class_id, class_dets in enumerate(results):
            for d in class_dets:
                y0, x0, y1, x1, score = (float(v) for v in d[:5])
                if score < self.cfg.detection.threshold:
                    continue
                label = self.labels[class_id] if class_id < len(self.labels) else str(class_id)
                box = (int(x0 * w), int(y0 * h), int((x1 - x0) * w), int((y1 - y0) * h))
                dets.append(Detection(label, score, box))
        dets.sort(key=lambda d: d.score, reverse=True)
        return dets[: self.cfg.detection.max_detections]

    def detections(self) -> Iterator[list[Detection]]:
        while True:
            frame = self.picam2.capture_array("lores")
            yield self.parse(self.hailo.run(frame))
