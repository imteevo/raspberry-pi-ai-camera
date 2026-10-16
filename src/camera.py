"""IMX500 (Raspberry Pi AI Camera) object detection backend.

How the AI Camera works
-----------------------
The Sony IMX500 has a small NPU and memory stacked under the pixel array.
A compiled network (``.rpk``) is uploaded to the sensor when the camera
starts; every frame is then inferred *on the sensor*, and the output tensors
are delivered to the Pi inside the frame's libcamera metadata. The Pi's CPU
only has to decode those tensors, which is why this backend needs almost no
host compute.

This module follows the flow of Picamera2's official
``examples/imx500/imx500_object_detection_demo.py``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator, Optional

from .config import AppConfig
from .detection import Detection

if TYPE_CHECKING:  # pragma: no cover - typing only; not installed off-device
    from picamera2 import Picamera2
    from picamera2.devices import IMX500


class IMX500Detector:
    """Run object detection on the IMX500's on-sensor NPU.

    Lifecycle: construct, call :meth:`start`, iterate :meth:`detections`, and
    finally call :meth:`stop` (ideally in a ``finally`` block).
    """

    def __init__(self, cfg: AppConfig) -> None:
        """Load the network onto the sensor and prepare a camera config.

        Args:
            cfg: Application config; ``cfg.imx500`` and ``cfg.detection`` are used.

        Raises:
            ValueError: If the ``.rpk`` declares a task other than object detection.
        """
        # Import Picamera2 lazily so this package can still be imported (for
        # tests, linting, --help) on machines without libcamera.
        from picamera2 import Picamera2
        from picamera2.devices import IMX500
        from picamera2.devices.imx500 import NetworkIntrinsics

        self.cfg = cfg

        # IMX500 must be created *before* Picamera2: it selects the sensor and
        # stages the network firmware that libcamera uploads when the camera
        # is opened.
        self.imx500: IMX500 = IMX500(cfg.imx500.model)

        # Network intrinsics describe how to interpret the output tensors
        # (task, labels, bbox format, rate). Newer .rpk files embed them;
        # older ones don't, so fall back to a blank object-detection set.
        intrinsics = self.imx500.network_intrinsics
        if not intrinsics:
            intrinsics = NetworkIntrinsics()
            intrinsics.task = "object detection"
        elif intrinsics.task != "object detection":
            raise ValueError(f"Model is not an object detection network (task={intrinsics.task!r})")

        # Apply explicit overrides from config.yaml, only when set, so the
        # model's embedded metadata wins by default.
        if cfg.imx500.bbox_normalization is not None:
            intrinsics.bbox_normalization = cfg.imx500.bbox_normalization
        if cfg.imx500.bbox_order is not None:
            intrinsics.bbox_order = cfg.imx500.bbox_order
        if cfg.imx500.inference_rate is not None:
            intrinsics.inference_rate = cfg.imx500.inference_rate
        labels = cfg.load_labels()
        if labels:
            intrinsics.labels = labels

        # Fill anything still unset (e.g. COCO labels, default inference rate).
        intrinsics.update_with_defaults()
        self.intrinsics = intrinsics

        # Open the specific camera the IMX500 helper resolved, so this works
        # even when multiple cameras are attached to the Pi 5's two CSI ports.
        self.picam2: Picamera2 = Picamera2(self.imx500.camera_num)

        # Run the sensor at the network's inference rate; extra frames would be
        # produced without tensors. 12 buffers keep the pipeline from stalling
        # while metadata is processed in Python.
        self.config: dict[str, Any] = self.picam2.create_preview_configuration(
            controls={"FrameRate": intrinsics.inference_rate}, buffer_count=12
        )

    def start(self, preview: bool = False) -> None:
        """Start streaming; the first start uploads the network to the sensor.

        Args:
            preview: Open a Picamera2 preview window.
        """
        # Network upload over I2C/SPI can take tens of seconds on first run,
        # so show progress rather than appearing hung.
        self.imx500.show_network_fw_progress_bar()
        self.picam2.start(self.config, show_preview=preview)

        # Some networks expect the full field of view letterboxed rather than
        # stretched; this sets the sensor's input crop to match.
        if self.intrinsics.preserve_aspect_ratio:
            self.imx500.set_auto_aspect_ratio()

    def stop(self) -> None:
        """Stop streaming and release the camera."""
        self.picam2.stop()
        self.picam2.close()

    def _labels(self) -> list[str]:
        """Return the active label list.

        Some label files use ``"-"`` placeholders for unused COCO ids; when
        the model's intrinsics say so, drop them so indices line up.
        """
        labels = self.intrinsics.labels or []
        if self.intrinsics.ignore_dash_labels:
            labels = [label for label in labels if label and label != "-"]
        return labels

    def parse(self, metadata: dict[str, Any]) -> Optional[list[Detection]]:
        """Decode one frame's output tensors into detections.

        Args:
            metadata: libcamera metadata from :meth:`Picamera2.capture_metadata`.

        Returns:
            Detections sorted by descending score (capped at
            ``max_detections``), or ``None`` if this frame carried no tensor,
            which happens while the network is still loading.
        """
        np_outputs = self.imx500.get_outputs(metadata, add_batch=True)
        if np_outputs is None:
            return None

        # Only the height is needed: normalised models divide by it.
        _, input_h = self.imx500.get_input_size()
        threshold = self.cfg.detection.threshold

        # "_pp" SSD models emit four tensors: boxes, scores, classes, count.
        # [0] strips the batch dimension added by add_batch=True.
        boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]

        if self.intrinsics.bbox_normalization:
            boxes = boxes / input_h
        if self.intrinsics.bbox_order == "xy":
            # Reorder to the (y0, x0, y1, x1) layout convert_inference_coords expects.
            boxes = boxes[:, [1, 0, 3, 2]]

        labels = self._labels()
        results: list[Detection] = []
        for box, score, cls in zip(boxes, scores, classes):
            if score < threshold:
                continue
            # Map from network-input space to main-stream pixels, accounting
            # for the sensor crop/ISP scaling in effect for this frame.
            x, y, w, h = self.imx500.convert_inference_coords(tuple(box), metadata, self.picam2)
            idx = int(cls)
            label = labels[idx] if 0 <= idx < len(labels) else str(idx)
            results.append(Detection(label, float(score), (int(x), int(y), int(w), int(h))))

        results.sort(key=lambda d: d.score, reverse=True)
        return results[: self.cfg.detection.max_detections]

    def detections(self) -> Iterator[list[Detection]]:
        """Yield the detection list for each frame that carries a tensor.

        Blocks on each frame. Frames without tensors are skipped, so the
        caller only ever sees real inference results. Runs until the caller
        stops iterating.
        """
        while True:
            # Metadata only: we don't need pixel data, which saves a copy.
            metadata = self.picam2.capture_metadata()
            dets = self.parse(metadata)
            if dets is not None:
                yield dets
