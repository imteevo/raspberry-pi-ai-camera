"""IMX500 (Raspberry Pi AI Camera) object detection backend.

How the AI Camera works
-----------------------
The Sony IMX500 has a small NPU and memory stacked under the pixel array.
A compiled network (``.rpk``) is uploaded to the sensor when the camera
starts; every frame is then inferred *on the sensor*, and the output tensors
are delivered to the Pi inside the frame's libcamera metadata. The Pi's CPU
only has to decode those tensors, which is why this backend needs almost no
host compute.

This module follows the flow of the official
``picamera2-examples/examples/imx500/imx500_object_detection_demo.py``.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Optional

from .config import AppConfig
from .detection import Detection

if TYPE_CHECKING:  # pragma: no cover - typing only; not installed off-device
    from picamera2 import Picamera2
    from picamera2.devices import IMX500

#: COCO labels bundled with the package (copied from picamera2-examples).
#: Uses the 91-id COCO layout with ``-`` placeholders for unused ids, which
#: is what the stock SSD ``.rpk`` emits. Used only when neither the model nor
#: ``detection.labels_file`` provides labels; ``NetworkIntrinsics
#: .update_with_defaults()`` does *not* supply labels (only inference_rate).
LABELS_FALLBACK: Path = Path(__file__).resolve().parent / "assets" / "coco_labels.txt"


def filter_labels(labels: list[str], ignore_dash_labels: bool) -> list[str]:
    """Drop ``"-"`` placeholder labels when the model's intrinsics ask for it.

    Some label files pad unused COCO ids with ``"-"``; models trained on the
    compacted 80-class list set ``ignore_dash_labels`` so indices line up.
    """
    if ignore_dash_labels:
        return [label for label in labels if label and label != "-"]
    return list(labels)


class IMX500Detector:
    """Run object detection on the IMX500's on-sensor NPU.

    Lifecycle: construct, call :meth:`start`, iterate :meth:`detections`, and
    finally call :meth:`stop` (idempotent; safe after a failed ``start``).
    """

    def __init__(self, cfg: AppConfig) -> None:
        """Load the network onto the sensor and prepare a camera config.

        Args:
            cfg: Application config; ``cfg.imx500`` and ``cfg.detection`` are used.

        Raises:
            FileNotFoundError: If the ``.rpk`` model doesn't exist.
            ValueError: If the ``.rpk`` declares a task other than object detection.
            NotImplementedError: For post-processing types this parser can't decode.
        """
        # Import Picamera2 lazily so this package can still be imported (for
        # tests, linting, --help) on machines without libcamera.
        from picamera2 import Picamera2
        from picamera2.devices import IMX500
        from picamera2.devices.imx500 import NetworkIntrinsics

        self.cfg = cfg
        self._started = False
        self.picam2: Optional[Picamera2] = None

        if not Path(cfg.imx500.model).is_file():
            raise FileNotFoundError(f"IMX500 model not found: {cfg.imx500.model} (install imx500-all)")

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

        # Only the "_pp" (on-sensor NMS) tensor layout is decoded here. The
        # official demo also has a NanoDet branch; fail loudly instead of
        # silently producing garbage boxes.
        if getattr(intrinsics, "postprocess", None) == "nanodet":
            raise NotImplementedError("nanodet post-processing is not supported yet")

        # Apply explicit overrides from config.yaml, only when set, so the
        # model's embedded metadata wins by default.
        if cfg.imx500.bbox_normalization is not None:
            intrinsics.bbox_normalization = cfg.imx500.bbox_normalization
        if cfg.imx500.bbox_order is not None:
            intrinsics.bbox_order = cfg.imx500.bbox_order
        if cfg.imx500.inference_rate is not None:
            intrinsics.inference_rate = cfg.imx500.inference_rate

        # Label precedence: labels_file > labels embedded in the .rpk > the
        # bundled COCO list (same fallback as the official demo).
        labels = cfg.load_labels()
        if labels:
            intrinsics.labels = labels
        elif intrinsics.labels is None:
            intrinsics.labels = LABELS_FALLBACK.read_text().splitlines()

        # Fills only what's still unset - in practice inference_rate (30 fps).
        intrinsics.update_with_defaults()
        self.intrinsics = intrinsics

        # Resolve the label list once rather than on every frame.
        self.labels: list[str] = filter_labels(intrinsics.labels or [], bool(intrinsics.ignore_dash_labels))

        # Open the specific camera the IMX500 helper resolved, so this works
        # even when cameras are attached to both of the Pi 5's CSI ports.
        self.picam2 = Picamera2(self.imx500.camera_num)

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
        assert self.picam2 is not None
        # Network upload can take tens of seconds on first run, so show
        # progress rather than appearing hung.
        self.imx500.show_network_fw_progress_bar()
        # Mark as started *before* start() so a failure/Ctrl+C mid-start still
        # gets picam2.stop() in stop().
        self._started = True
        self.picam2.start(self.config, show_preview=preview)

        # Some networks expect the full field of view letterboxed rather than
        # stretched; this sets the sensor's input crop to match.
        if self.intrinsics.preserve_aspect_ratio:
            self.imx500.set_auto_aspect_ratio()

    def stop(self) -> None:
        """Stop streaming and release the camera. Idempotent."""
        picam2, self.picam2 = self.picam2, None
        if picam2 is None:
            return
        try:
            if self._started:
                picam2.stop()
        finally:
            self._started = False
            # close() must run even if stop() raised, or the camera stays busy.
            picam2.close()

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

        # Normalised models divide by the input height for *both* axes, as
        # the upstream demo does. That is only exact for square inputs (all
        # stock SSD models are 320x320); revisit for non-square networks.
        _, input_h = self.imx500.get_input_size()
        threshold = self.cfg.detection.threshold

        # "_pp" SSD models emit four tensors: boxes, scores, classes, count.
        # [0] strips the batch dimension added by add_batch=True. Padding
        # slots beyond `count` have ~0 scores and are removed by the threshold.
        boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]

        if self.intrinsics.bbox_normalization:
            boxes = boxes / input_h
        if self.intrinsics.bbox_order == "xy":
            # Reorder to the (y0, x0, y1, x1) layout convert_inference_coords expects.
            boxes = boxes[:, [1, 0, 3, 2]]

        results: list[Detection] = []
        for box, score, cls in zip(boxes, scores, classes):
            if score < threshold:
                continue
            # Map from network-input space to main-stream pixels, accounting
            # for the sensor crop/ISP scaling in effect for this frame.
            x, y, w, h = self.imx500.convert_inference_coords(tuple(box), metadata, self.picam2)
            idx = int(cls)
            label = self.labels[idx] if 0 <= idx < len(self.labels) else str(idx)
            results.append(Detection(label, float(score), (int(x), int(y), int(w), int(h))))

        results.sort(key=lambda d: d.score, reverse=True)
        return results[: self.cfg.detection.max_detections]

    def detections(self) -> Iterator[list[Detection]]:
        """Yield the detection list for each frame that carries a tensor.

        Blocks on each frame. Frames without tensors are skipped, so the
        caller only ever sees real inference results.
        """
        assert self.picam2 is not None
        while True:
            # Metadata only: we don't need pixel data, which saves a copy.
            metadata = self.picam2.capture_metadata()
            dets = self.parse(metadata)
            if dets is not None:
                yield dets
