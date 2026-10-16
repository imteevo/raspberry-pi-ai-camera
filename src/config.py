"""YAML-driven configuration loader."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

DEFAULTS: dict[str, Any] = {
    "backend": "imx500",
    "preview": False,
    "max_frames": 0,
    "detection": {"threshold": 0.55, "max_detections": 10, "labels_file": None},
    "imx500": {
        "model": "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk",
        "bbox_normalization": None,
        "bbox_order": None,
        "inference_rate": None,
    },
    "hailo": {
        "model": "/usr/share/hailo-models/yolov8s_h8l.hef",
        "camera_num": 0,
        "main_size": [1280, 960],
    },
}


@dataclass
class DetectionConfig:
    threshold: float = 0.55
    max_detections: int = 10
    labels_file: Optional[str] = None


@dataclass
class IMX500Config:
    model: str = DEFAULTS["imx500"]["model"]
    bbox_normalization: Optional[bool] = None
    bbox_order: Optional[str] = None
    inference_rate: Optional[float] = None


@dataclass
class HailoConfig:
    model: str = DEFAULTS["hailo"]["model"]
    camera_num: int = 0
    main_size: tuple[int, int] = (1280, 960)


@dataclass
class AppConfig:
    backend: str = "imx500"
    preview: bool = False
    max_frames: int = 0
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    imx500: IMX500Config = field(default_factory=IMX500Config)
    hailo: HailoConfig = field(default_factory=HailoConfig)

    def load_labels(self) -> Optional[list[str]]:
        if not self.detection.labels_file:
            return None
        text = Path(self.detection.labels_file).expanduser().read_text()
        return [line.strip() for line in text.splitlines() if line.strip()]


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: Optional[str | Path] = None) -> AppConfig:
    """Load config.yaml (or ``path``) merged over built-in defaults."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    data: dict[str, Any] = {}
    if cfg_path.exists():
        with cfg_path.open() as fh:
            data = yaml.safe_load(fh) or {}
    elif path:
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    merged = _deep_merge(DEFAULTS, data)
    backend = str(merged["backend"]).lower()
    if backend not in ("imx500", "hailo"):
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
            main_size=tuple(hailo["main_size"]),
        ),
    )
