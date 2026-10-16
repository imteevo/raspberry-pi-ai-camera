"""Hailo AI HAT+ / AI HAT+ 2 object detection backend.

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

The model must be a single-output YOLO ``.hef`` compiled with on-chip NMS.
With ``hailo.model: auto`` the HEF matching the fitted NPU is chosen from
:data:`DEFAULT_HEF` (files installed by ``hailo-all`` / ``hailo-h10-all``),
mirroring the official ``picamera2-examples`` ``hailo/detect.py``.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Optional, Sequence

from .config import AUTO_MODEL, AppConfig
from .detection import Detection

if TYPE_CHECKING:  # pragma: no cover - typing only; not installed off-device
    import numpy as np
    from picamera2 import Picamera2
    from picamera2.devices import Hailo

#: Directory where the hailo-models package installs its HEFs.
HAILO_MODELS_DIR: Path = Path("/usr/share/hailo-models")

#: Default HEF per NPU architecture string from ``hailo_architecture()``.
#: HEFs are compiled per architecture: an H8L HEF won't load on a Hailo-10H,
#: and a Hailo-8 can run the larger/faster h8 build.
DEFAULT_HEF: dict[str, Path] = {
    "HAILO10H": HAILO_MODELS_DIR / "yolov8m_h10.hef",
    "HAILO8": HAILO_MODELS_DIR / "yolov8s_h8.hef",
    "HAILO8L": HAILO_MODELS_DIR / "yolov8s_h8l.hef",
}

#: The 80 COCO class names in YOLO class-id order. Used when no labels file
#: is configured; the stock Hailo YOLO models are COCO-trained (80 classes,
#: unlike the 91-id layout the IMX500 SSD uses).
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


def resolve_hef(model: Optional[str], arch: Optional[str]) -> Path:
    """Pick the HEF to load.

    Args:
        model: Configured path, or ``"auto"``/``None`` to choose by ``arch``.
        arch: NPU architecture from ``hailo_architecture()`` (e.g. ``"HAILO8L"``);
            unknown/``None`` falls back to the Hailo-8L model, which is the
            most conservative choice.

    Returns:
        Path to an existing HEF file.

    Raises:
        FileNotFoundError: If the resolved file doesn't exist; the message
            lists the HEFs that are installed to help the user pick one.
    """
    if model in (None, "", AUTO_MODEL):
        path = DEFAULT_HEF.get((arch or "").upper(), DEFAULT_HEF["HAILO8L"])
    else:
        path = Path(model).expanduser()
    if not path.is_file():
        installed = sorted(p.name for p in HAILO_MODELS_DIR.glob("*.hef")) if HAILO_MODELS_DIR.is_dir() else []
        raise FileNotFoundError(f"HEF not found: {path} (installed in {HAILO_MODELS_DIR}: {installed or 'none'})")
    return path


def _clip01(v: float) -> float:
    """Clamp a normalised coordinate to [0, 1]; NMS can overshoot slightly."""
    return min(max(v, 0.0), 1.0)


def parse_nms(
    results: Any,
    main_size: tuple[int, int],
    labels: Sequence[str],
    threshold: float,
    max_detections: int,
) -> list[Detection]:
    """Convert HailoRT NMS output into detections.

    Args:
        results: Output of ``Hailo.run()`` for a single-output NMS HEF: one
            sequence per class id, each holding rows of
            ``[y0, x0, y1, x1, score]`` normalised to ``[0, 1]``.
        main_size: ``(width, height)`` to scale boxes to.
        labels: Class names indexed by class id.
        threshold: Minimum score to keep.
        max_detections: Maximum detections to return.

    Returns:
        Detections sorted by descending score.

    Raises:
        RuntimeError: If ``results`` is a dict, i.e. the HEF has several
            output tensors and is not an NMS model this parser understands.
    """
    if isinstance(results, dict):
        raise RuntimeError(f"Expected a single NMS-output HEF, got outputs {list(results)}")
    w, h = main_size
    dets: list[Detection] = []
    for class_id, class_dets in enumerate(results):
        for d in class_dets:
            y0, x0, y1, x1, score = (float(v) for v in d[:5])
            if score < threshold:
                continue
            y0, x0, y1, x1 = (_clip01(v) for v in (y0, x0, y1, x1))
            label = labels[class_id] if class_id < len(labels) else str(class_id)
            # Normalised corners -> main-stream (x, y, w, h). lores and main
            # share the same field of view, so plain scaling is correct.
            box = (int(x0 * w), int(y0 * h), int((x1 - x0) * w), int((y1 - y0) * h))
            dets.append(Detection(label, score, box))
    dets.sort(key=lambda d: d.score, reverse=True)
    return dets[:max_detections]


class HailoDetector:
    """Run object detection on a Hailo NPU attached over PCIe.

    Lifecycle: construct, call :meth:`start`, iterate :meth:`detections`, and
    finally call :meth:`stop` (idempotent; safe after a failed ``start``).
    """

    def __init__(self, cfg: AppConfig) -> None:
        """Load the HEF onto the NPU and build a dual-stream camera config.

        Args:
            cfg: Application config; ``cfg.hailo`` and ``cfg.detection`` are used.

        Raises:
            FileNotFoundError: If the HEF can't be found.
            RuntimeError: If the HEF isn't a single-output NMS model.
        """
        # Lazy imports: keep the package importable without Picamera2/HailoRT.
        from picamera2 import Picamera2
        from picamera2.devices import Hailo, hailo_architecture

        self.cfg = cfg
        self._started = False
        self.hailo: Optional[Hailo] = None
        self.picam2: Optional[Picamera2] = None

        try:
            self.model_path = resolve_hef(cfg.hailo.model, hailo_architecture())

            # Opening Hailo() configures the device and loads the network. The
            # input shape comes from the HEF itself, so the camera always
            # matches whichever model is configured.
            self.hailo = Hailo(str(self.model_path))
            model_h, model_w, _ = self.hailo.get_input_shape()  # (H, W, C)
            self.model_size: tuple[int, int] = (int(model_w), int(model_h))  # Picamera2 wants (W, H)
            self.main_size: tuple[int, int] = (int(cfg.hailo.main_size[0]), int(cfg.hailo.main_size[1]))
            self.labels: list[str] = cfg.load_labels() or list(COCO_LABELS)

            self.picam2 = Picamera2(cfg.hailo.camera_num)
            self.config: dict[str, Any] = self.picam2.create_preview_configuration(
                main={"size": self.main_size, "format": "XRGB8888"},
                # The ISP scales lores to the network size in hardware. Note:
                # Picamera2 "RGB888" is B,G,R in memory - same as upstream
                # detect.py; verify accuracy on hardware.
                lores={"size": self.model_size, "format": "RGB888"},
                controls={"FrameRate": 30},
            )
        except BaseException:
            # Don't leak the NPU VDevice / camera if any step above failed.
            self.stop()
            raise

    def start(self, preview: bool = False) -> None:
        """Start the camera.

        Args:
            preview: Open a Picamera2 preview window (shows the main stream).
        """
        assert self.picam2 is not None
        self._started = True  # set first so a mid-start failure is cleaned up
        self.picam2.start(self.config, show_preview=preview)

    def stop(self) -> None:
        """Stop the camera and release both the camera and the NPU. Idempotent.

        Each release step runs even if an earlier one raises, so the NPU is
        never left claimed by a half-closed process.
        """
        picam2, self.picam2 = self.picam2, None
        hailo, self.hailo = self.hailo, None
        try:
            if picam2 is not None:
                try:
                    if self._started:
                        picam2.stop()
                finally:
                    picam2.close()
        finally:
            self._started = False
            if hailo is not None:
                # Releasing the VDevice frees the NPU for other processes.
                hailo.close()

    def parse(self, results: Sequence[Sequence["np.ndarray"]]) -> list[Detection]:
        """Convert one inference result into detections (see :func:`parse_nms`)."""
        return parse_nms(
            results, self.main_size, self.labels,
            self.cfg.detection.threshold, self.cfg.detection.max_detections,
        )

    def detections(self) -> Iterator[list[Detection]]:
        """Yield the detection list for every captured frame.

        Each iteration blocks on the next lores frame, then runs synchronous
        inference on the NPU.
        """
        assert self.picam2 is not None and self.hailo is not None
        while True:
            frame = self.picam2.capture_array("lores")
            yield self.parse(self.hailo.run(frame))
