"""YAML-driven configuration loader.

Configuration is resolved in three layers, lowest priority first:

1. :data:`DEFAULTS` built into this module (so the app runs with no file),
2. ``config.yaml`` at the repo root, or a file passed via ``--config``,
3. CLI flags, applied by :mod:`src.main` after loading.

The merged mapping is converted into typed dataclasses so the rest of the
code base gets attribute access, IDE completion and a single place where
types are coerced and validated.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import yaml

#: ``config.yaml`` in the repository root (one level above this package).
DEFAULT_CONFIG_PATH: Path = Path(__file__).resolve().parent.parent / "config.yaml"

#: Backends accepted for the ``backend`` key / ``--backend`` flag.
VALID_BACKENDS: tuple[str, ...] = ("imx500", "hailo")

#: Built-in defaults. These mirror ``config.yaml`` so a missing or partial
#: YAML file still yields a complete, working configuration.
DEFAULTS: dict[str, Any] = {
    "backend": "imx500",
    "preview": False,
    "max_frames": 0,
    "detection": {"threshold": 0.55, "max_detections": 10, "labels_file": None},
    "imx500": {
        # SSD MobileNetV2 FPN-Lite with on-sensor post-processing ("_pp"),
        # shipped by the imx500-models package. "_pp" means NMS runs on the
        # sensor, so outputs arrive as ready-made boxes/scores/classes.
        "model": "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk",
        "bbox_normalization": None,
        "bbox_order": None,
        "inference_rate": None,
    },
    "hailo": {
        # YOLOv8s compiled for Hailo-8L (13 TOPS), shipped by hailo-models.
        # Hailo-8 (26 TOPS) boards should use the matching *_h8.hef file.
        "model": "/usr/share/hailo-models/yolov8s_h8l.hef",
        "camera_num": 0,
        "main_size": [1280, 960],
    },
}


@dataclass
class DetectionConfig:
    """Settings shared by all backends for filtering detections.

    Attributes:
        threshold: Minimum confidence score for a detection to be reported.
        max_detections: Cap on detections returned per frame (highest first).
        labels_file: Optional newline-separated labels file overriding the
            model's built-in or default COCO labels.
    """

    threshold: float = 0.55
    max_detections: int = 10
    labels_file: Optional[str] = None


@dataclass
class IMX500Config:
    """IMX500 (AI Camera) backend settings.

    The ``None``-valued fields are overrides for models whose ``.rpk`` lacks
    embedded network intrinsics; leave them ``None`` to use the model's
    metadata or Picamera2's defaults.

    Attributes:
        model: Path to the ``.rpk`` network package to load onto the sensor.
        bbox_normalization: ``True`` if the model emits boxes in input-pixel
            units that must be divided by the input height.
        bbox_order: ``"yx"`` (default) or ``"xy"`` coordinate ordering.
        inference_rate: Frame rate to run the camera (and network) at, in fps.
    """

    model: str = DEFAULTS["imx500"]["model"]
    bbox_normalization: Optional[bool] = None
    bbox_order: Optional[str] = None
    inference_rate: Optional[float] = None


@dataclass
class HailoConfig:
    """Hailo (AI HAT+) backend settings.

    Attributes:
        model: Path to a compiled Hailo ``.hef`` with on-chip NMS.
        camera_num: libcamera index of the camera feeding frames to the NPU.
        main_size: ``(width, height)`` of the main stream; detection boxes are
            scaled to this resolution.
    """

    model: str = DEFAULTS["hailo"]["model"]
    camera_num: int = 0
    main_size: tuple[int, int] = (1280, 960)


@dataclass
class AppConfig:
    """Top-level application configuration.

    Attributes:
        backend: ``"imx500"`` or ``"hailo"``.
        preview: Show a Picamera2 preview window (needs a display).
        max_frames: Stop after this many result frames; ``0`` runs forever.
        detection: Shared detection filtering settings.
        imx500: IMX500 backend settings.
        hailo: Hailo backend settings.
    """

    backend: str = "imx500"
    preview: bool = False
    max_frames: int = 0
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    imx500: IMX500Config = field(default_factory=IMX500Config)
    hailo: HailoConfig = field(default_factory=HailoConfig)

    def load_labels(self) -> Optional[list[str]]:
        """Read the configured labels file, if any.

        Returns:
            A list of non-empty, stripped label lines, or ``None`` when no
            ``labels_file`` is configured (callers then fall back to the
            model's own labels).

        Raises:
            OSError: If the configured file cannot be read.
        """
        if not self.detection.labels_file:
            return None
        text = Path(self.detection.labels_file).expanduser().read_text()
        return [line.strip() for line in text.splitlines() if line.strip()]


def _deep_merge(base: dict[str, Any], override: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Recursively merge ``override`` onto a deep copy of ``base``.

    Nested dicts are merged key by key so a YAML file can override a single
    nested value (e.g. ``detection.threshold``) without restating siblings.
    Non-dict values in ``override`` replace those in ``base`` wholesale.
    ``base`` is never mutated.
    """
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: Optional[Union[str, Path]] = None) -> AppConfig:
    """Load configuration from YAML merged over built-in defaults.

    Args:
        path: Explicit config file. When omitted, :data:`DEFAULT_CONFIG_PATH`
            is used if it exists; otherwise defaults alone are used.

    Returns:
        A fully populated :class:`AppConfig`.

    Raises:
        FileNotFoundError: If an explicit ``path`` was given but doesn't exist.
            (A missing *default* file is tolerated.)
        ValueError: If ``backend`` is not one of :data:`VALID_BACKENDS`.
        TypeError: If a section contains keys the dataclasses don't define,
            which usually indicates a typo in the YAML.
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH

    data: dict[str, Any] = {}
    if cfg_path.exists():
        with cfg_path.open() as fh:
            # safe_load: never construct arbitrary Python objects from YAML.
            # An empty file yields None, hence the `or {}`.
            data = yaml.safe_load(fh) or {}
    elif path:
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    merged = _deep_merge(DEFAULTS, data)

    backend = str(merged["backend"]).lower()
    if backend not in VALID_BACKENDS:
        raise ValueError(f"backend must be 'imx500' or 'hailo', got {backend!r}")

    hailo = merged["hailo"]
    return AppConfig(
        backend=backend,
        preview=bool(merged["preview"]),
        max_frames=int(merged["max_frames"]),
        detection=DetectionConfig(**merged["detection"]),
        imx500=IMX500Config(**merged["imx500"]),
        hailo=HailoConfig(
            model=hailo["model"],
            camera_num=int(hailo["camera_num"]),
            # YAML gives a list; normalise to an immutable (w, h) tuple.
            main_size=tuple(hailo["main_size"]),
        ),
    )
