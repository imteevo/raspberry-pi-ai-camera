"""Hailo AI HAT+ object detection backend (optional).

How this backend works
----------------------
Frames come from a camera via Picamera2 (the AI Camera works fine as a
plain camera here) and are sent over PCIe to the Hailo NPU for inference
using :class:`picamera2.devices.Hailo`, a thin wrapper over HailoRT.

The camera is configured with two streams:

* ``main``: full-resolution XRGB8888, for preview/recording and the
  coordinate space that reported boxes are scaled to.
* ``lores``: RGB888 at exactly the network input size, produced by the
  Pi 5 ISP in hardware, so no CPU resize is needed before inference.

The expected model is a YOLO ``.hef`` compiled with on-chip NMS, such as
``/usr/share/hailo-models/yolov8s_h8l.hef`` from the ``hailo-models``
package that ``hailo-all`` installs.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator, Sequence

from .config import AppConfig
from .detection import Detection

if TYPE_CHECKING:  # pragma: no cover - typing only; not installed off-device
    import numpy as np
    from picamera2 import Picamera2
    from picamera2.devices import Hailo

#: The 80 COCO class names, in model class-id order. Used when no labels
#: file is configured; the stock Hailo YOLO models are COCO-trained.
#: Multi-word names use underscores to keep this a whitespace-split list.
COCO_LABELS: tuple[str, ...] = tuple(
    (
        "person bicycle car motorcycle airplane bus train truck boat traffic_light fire_hydrant "
        "stop_sign parking_meter bench bird cat dog horse sheep cow elephant bear zebra giraffe "
        "backpack umbrella handbag tie suitcase frisbee skis snowboard sports_ball kite "
        "baseball_bat baseball_glove skateboard surfboard tennis_racket bottle wine_glass cup "
        "fork knife spoon bowl banana apple sandwich orange broccoli carrot hot_dog pizza donut "
        "cake chair couch potted_plant bed dining_table toilet tv laptop mouse remote keyboard "
        "cell_phone microwave oven toaster sink refrigerator book clock vase scissors teddy_bear "
        "hair_drier toothbrush"
    ).split()
)


class HailoDetector:
    """Run object detection on a Hailo NPU attached over PCIe.

    Lifecycle: construct, call :meth:`start`, iterate :meth:`detections`, and
    finally call :meth:`stop` (ideally in a ``finally`` block).
    """

    def __init__(self, cfg: AppConfig) -> None:
        """Load the HEF onto the NPU and build a dual-stream camera config.

        Args:
            cfg: Application config; ``cfg.hailo`` and ``cfg.detection`` are used.
        """
        # Lazy imports: keep the package importable without Picamera2/HailoRT.
        from picamera2 import Picamera2
        from picamera2.devices import Hailo

        self.cfg = cfg

        # Opening Hailo() configures the device and loads the network. The
        # input shape comes from the HEF itself, so the camera always matches
        # whichever model is configured.
        self.hailo: Hailo = Hailo(cfg.hailo.model)
        model_h, model_w, _ = self.hailo.get_input_shape()  # (H, W, C)
        self.model_size: tuple[int, int] = (model_w, model_h)  # Picamera2 wants (W, H)
        self.main_size: tuple[int, int] = tuple(cfg.hailo.main_size)
        self.labels: list[str] = cfg.load_labels() or list(COCO_LABELS)

        self.picam2: Picamera2 = Picamera2(cfg.hailo.camera_num)
        self.config: dict[str, Any] = self.picam2.create_preview_configuration(
            main={"size": self.main_size, "format": "XRGB8888"},
            # The ISP scales the lores stream to the network size in hardware.
            # RGB888 is 3-channel, matching the HEF's expected input.
            lores={"size": self.model_size, "format": "RGB888"},
            controls={"FrameRate": 30},
        )

    def start(self, preview: bool = False) -> None:
        """Start the camera.

        Args:
            preview: Open a Picamera2 preview window (shows the main stream).
        """
        self.picam2.start(self.config, show_preview=preview)

    def stop(self) -> None:
        """Stop the camera and release both the camera and the NPU."""
        self.picam2.stop()
        self.picam2.close()
        # Releasing the VDevice frees the NPU for other processes.
        self.hailo.close()

    def parse(self, results: Sequence[Sequence["np.ndarray"]]) -> list[Detection]:
        """Convert HEF NMS output into detections.

        Args:
            results: HailoRT NMS output: one sequence per class id, each
                holding rows of ``[y0, x0, y1, x1, score]`` normalised to
                ``[0, 1]``.

        Returns:
            Detections above the threshold, in main-stream pixels, sorted by
            descending score and capped at ``max_detections``.
        """
        w, h = self.main_size
        dets: list[Detection] = []
        for class_id, class_dets in enumerate(results):
            for d in class_dets:
                y0, x0, y1, x1, score = (float(v) for v in d[:5])
                if score < self.cfg.detection.threshold:
                    continue
                label = self.labels[class_id] if class_id < len(self.labels) else str(class_id)
                # Normalised corners -> main-stream (x, y, w, h). lores and main
                # share the same field of view, so plain scaling is correct.
                box = (int(x0 * w), int(y0 * h), int((x1 - x0) * w), int((y1 - y0) * h))
                dets.append(Detection(label, score, box))
        dets.sort(key=lambda d: d.score, reverse=True)
        return dets[: self.cfg.detection.max_detections]

    def detections(self) -> Iterator[list[Detection]]:
        """Yield the detection list for every captured frame.

        Each iteration blocks on the next lores frame, then runs synchronous
        inference on the NPU. Runs until the caller stops iterating.
        """
        while True:
            frame = self.picam2.capture_array("lores")
            yield self.parse(self.hailo.run(frame))
