"""YAML-driven configuration loader with validation.

Configuration is resolved in three layers, lowest priority first:

1. :data:`DEFAULTS` built into this module (so the app runs with no file),
2. a YAML file - either the one passed via ``--config`` or the first that
   exists in :func:`default_config_paths`,
3. CLI flags, applied by :mod:`pi_ai_camera.main` after loading.

The merged mapping is validated and converted into typed dataclasses. All
validation problems are raised as :class:`ConfigError` (a ``ValueError``)
with a message naming the offending key, so a typo in YAML produces an
actionable error rather than a ``TypeError`` from deep inside a dataclass.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional, Union

import yaml

#: Repository checkout root (``src/pi_ai_camera`` -> repo). Only meaningful
#: when running from a clone; an installed package won't have config.yaml here.
_REPO_ROOT: Path = Path(__file__).resolve().parents[2]

#: Backends accepted for the ``backend`` key / ``--backend`` flag.
VALID_BACKENDS: tuple[str, ...] = ("imx500", "hailo")

#: Special value for ``hailo.model``: pick the HEF matching the detected NPU.
AUTO_MODEL: str = "auto"

#: Built-in defaults. These mirror ``config.yaml`` so a missing or partial
#: YAML file still yields a complete, working configuration.
DEFAULTS: dict[str, Any] = {
    "backend": "imx500",
    "preview": False,
    "max_frames": 0,
    "detection": {"threshold": 0.55, "max_detections": 10, "labels_file": None},
    "imx500": {
        # SSD MobileNetV2 FPN-Lite with on-sensor post-processing ("_pp"),
        # shipped by imx500-models. "_pp" means NMS runs on the sensor, so
        # outputs arrive as ready-made boxes/scores/classes.
        "model": "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk",
        "bbox_normalization": None,
        "bbox_order": None,
        "inference_rate": None,
    },
    "hailo": {
        # "auto" selects yolov8s_h8l / yolov8s_h8 / yolov8m_h10 according to
        # the NPU actually fitted (see pi_ai_camera.hailo.DEFAULT_HEF).
        "model": AUTO_MODEL,
        "camera_num": 0,
        "main_size": [1280, 960],
    },
}


class ConfigError(ValueError):
    """Raised when configuration is malformed or out of range."""


def default_config_paths() -> list[Path]:
    """Return candidate config file locations, highest priority first.

    1. ``./config.yaml`` in the current working directory,
    2. ``$XDG_CONFIG_HOME/pi-ai-camera/config.yaml`` (``~/.config`` default),
    3. ``/etc/pi-ai-camera/config.yaml`` for system-wide deployments,
    4. ``config.yaml`` at the repository root (development checkouts).
    """
    xdg = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return [
        Path.cwd() / "config.yaml",
        xdg / "pi-ai-camera" / "config.yaml",
        Path("/etc/pi-ai-camera/config.yaml"),
        _REPO_ROOT / "config.yaml",
    ]


@dataclass
class DetectionConfig:
    """Settings shared by all backends for filtering detections.

    Attributes:
        threshold: Minimum confidence score in ``[0, 1]`` to report a detection.
        max_detections: Cap (>= 1) on detections returned per frame.
        labels_file: Optional newline-separated labels file overriding the
            model's own labels / the bundled COCO fallback.
    """

    threshold: float = 0.55
    max_detections: int = 10
    labels_file: Optional[str] = None


@dataclass
class IMX500Config:
    """IMX500 (AI Camera) backend settings.

    The ``None``-valued fields override intrinsics for models whose ``.rpk``
    lacks embedded metadata; leave them ``None`` to use the model's values.

    Attributes:
        model: Path to the ``.rpk`` network package to load onto the sensor.
        bbox_normalization: ``True`` if boxes are in input-pixel units and must
            be divided by the input size.
        bbox_order: ``"yx"`` (default) or ``"xy"`` coordinate ordering.
        inference_rate: Camera (and network) frame rate in fps; must be > 0.
    """

    model: str = DEFAULTS["imx500"]["model"]
    bbox_normalization: Optional[bool] = None
    bbox_order: Optional[str] = None
    inference_rate: Optional[float] = None


@dataclass
class HailoConfig:
    """Hailo (AI HAT+) backend settings.

    Attributes:
        model: Path to a compiled ``.hef`` with on-chip NMS, or ``"auto"``.
        camera_num: libcamera index of the camera feeding frames to the NPU.
        main_size: ``(width, height)`` of the main stream; detection boxes are
            scaled to this resolution.
    """

    model: str = AUTO_MODEL
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
        source: File the config was loaded from (``None`` = defaults only).
    """

    backend: str = "imx500"
    preview: bool = False
    max_frames: int = 0
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    imx500: IMX500Config = field(default_factory=IMX500Config)
    hailo: HailoConfig = field(default_factory=HailoConfig)
    source: Optional[Path] = None

    def load_labels(self) -> Optional[list[str]]:
        """Read the configured labels file, if any.

        Returns:
            Stripped, non-empty label lines, or ``None`` when no
            ``labels_file`` is configured (callers then fall back).

        Raises:
            OSError: If the configured file cannot be read.
        """
        if not self.detection.labels_file:
            return None
        text = Path(self.detection.labels_file).expanduser().read_text()
        return [line.strip() for line in text.splitlines() if line.strip()]

    def validate(self) -> "AppConfig":
        """Check value ranges; also called after CLI overrides are applied.

        Returns:
            ``self``, for chaining.

        Raises:
            ConfigError: On the first invalid value found.
        """
        if self.backend not in VALID_BACKENDS:
            raise ConfigError(f"backend must be one of {VALID_BACKENDS}, got {self.backend!r}")
        if self.max_frames < 0:
            raise ConfigError("max_frames must be >= 0")
        if not 0.0 <= self.detection.threshold <= 1.0:
            raise ConfigError("detection.threshold must be within [0, 1]")
        if self.detection.max_detections < 1:
            raise ConfigError("detection.max_detections must be >= 1")
        if self.imx500.bbox_order not in (None, "yx", "xy"):
            raise ConfigError("imx500.bbox_order must be 'yx' or 'xy'")
        if self.imx500.inference_rate is not None and self.imx500.inference_rate <= 0:
            raise ConfigError("imx500.inference_rate must be > 0")
        w, h = self.hailo.main_size
        if w <= 0 or h <= 0:
            raise ConfigError("hailo.main_size values must be positive")
        if self.hailo.camera_num < 0:
            raise ConfigError("hailo.camera_num must be >= 0")
        return self


def _deep_merge(base: dict[str, Any], override: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Recursively merge ``override`` onto a deep copy of ``base``.

    Nested dicts merge key by key so YAML can override one nested value (e.g.
    ``detection.threshold``) without restating siblings. A ``None`` override
    for a section (``detection:`` with nothing under it) keeps the defaults
    rather than wiping the section. ``base`` is never mutated.
    """
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(out.get(key), dict) and value is None:
            continue  # empty YAML section -> keep defaults
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _section(merged: dict[str, Any], key: str, cls: type) -> dict[str, Any]:
    """Fetch section ``key``, ensuring it's a mapping with only known keys."""
    val = merged.get(key)
    if not isinstance(val, dict):
        raise ConfigError(f"'{key}' must be a mapping, got {type(val).__name__}")
    unknown = set(val) - {f.name for f in fields(cls)}
    if unknown:
        raise ConfigError(f"Unknown keys in '{key}': {sorted(unknown)}")
    return val


def _as(value: Any, typ: type, name: str, optional: bool = False) -> Any:
    """Coerce ``value`` to ``typ`` with a readable error.

    ``bool`` is handled strictly: YAML already yields real booleans, and
    ``bool("false")`` would be ``True``, so strings are rejected.
    """
    if value is None and optional:
        return None
    if typ is bool:
        if isinstance(value, bool):
            return value
        raise ConfigError(f"{name} must be true/false, got {value!r}")
    if typ in (int, float) and isinstance(value, bool):
        raise ConfigError(f"{name} must be a number, got {value!r}")
    if typ is str and not isinstance(value, str):
        raise ConfigError(f"{name} must be a string, got {value!r}")
    try:
        return typ(value)
    except (TypeError, ValueError):
        raise ConfigError(f"{name} must be {typ.__name__}, got {value!r}") from None


def _find_config(path: Optional[Union[str, Path]]) -> Optional[Path]:
    """Resolve which config file to load (or ``None`` for defaults only)."""
    if path:
        p = Path(path).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"Config file not found: {p}")
        return p
    return next((p for p in default_config_paths() if p.is_file()), None)


def load_config(path: Optional[Union[str, Path]] = None) -> AppConfig:
    """Load and validate configuration.

    Args:
        path: Explicit config file. When omitted, the first existing file from
            :func:`default_config_paths` is used, else defaults alone.

    Returns:
        A fully populated, validated :class:`AppConfig`.

    Raises:
        FileNotFoundError: If an explicit ``path`` doesn't exist.
        ConfigError: If the YAML is malformed, has unknown keys, wrong types
            or out-of-range values.
    """
    cfg_path = _find_config(path)

    data: Any = {}
    if cfg_path is not None:
        try:
            with cfg_path.open() as fh:
                # safe_load: never construct arbitrary Python objects. An empty
                # file yields None, hence the `or {}`.
                data = yaml.safe_load(fh) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"Invalid YAML in {cfg_path}: {exc}") from None
    if not isinstance(data, dict):
        raise ConfigError(f"Top level of {cfg_path} must be a mapping")

    # Reject unknown top-level keys, consistent with the per-section check.
    unknown = set(data) - set(DEFAULTS)
    if unknown:
        raise ConfigError(f"Unknown config keys: {sorted(unknown)}")

    merged = _deep_merge(DEFAULTS, data)
    det = _section(merged, "detection", DetectionConfig)
    imx = _section(merged, "imx500", IMX500Config)
    hai = _section(merged, "hailo", HailoConfig)

    size = hai["main_size"]
    if not (isinstance(size, (list, tuple)) and len(size) == 2):
        raise ConfigError("hailo.main_size must be [width, height]")

    cfg = AppConfig(
        backend=_as(merged["backend"], str, "backend").lower(),
        preview=_as(merged["preview"], bool, "preview"),
        max_frames=_as(merged["max_frames"], int, "max_frames"),
        detection=DetectionConfig(
            threshold=_as(det["threshold"], float, "detection.threshold"),
            max_detections=_as(det["max_detections"], int, "detection.max_detections"),
            labels_file=_as(det["labels_file"], str, "detection.labels_file", optional=True),
        ),
        imx500=IMX500Config(
            model=_as(imx["model"], str, "imx500.model"),
            bbox_normalization=_as(imx["bbox_normalization"], bool, "imx500.bbox_normalization", optional=True),
            bbox_order=_as(imx["bbox_order"], str, "imx500.bbox_order", optional=True),
            inference_rate=_as(imx["inference_rate"], float, "imx500.inference_rate", optional=True),
        ),
        hailo=HailoConfig(
            model=_as(hai["model"], str, "hailo.model"),
            camera_num=_as(hai["camera_num"], int, "hailo.camera_num"),
            # YAML gives a list; normalise to an immutable (w, h) tuple of ints.
            main_size=(_as(size[0], int, "hailo.main_size[0]"), _as(size[1], int, "hailo.main_size[1]")),
        ),
        source=cfg_path,
    )
    return cfg.validate()
