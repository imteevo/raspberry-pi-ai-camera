"""Tests for configuration loading, merging and validation (M8)."""
from __future__ import annotations

import pytest

from pi_ai_camera.config import AUTO_MODEL, ConfigError, load_config


def test_defaults_from_repo_config(isolated_config) -> None:
    cfg = load_config()
    assert cfg.backend == "imx500"
    assert cfg.hailo.model == AUTO_MODEL
    assert cfg.hailo.main_size == (1280, 960)


def test_partial_override_keeps_siblings(write_cfg) -> None:
    cfg = load_config(write_cfg("detection:\n  threshold: 0.8\n"))
    assert cfg.detection.threshold == 0.8
    assert cfg.detection.max_detections == 10


def test_empty_section_keeps_defaults(write_cfg) -> None:
    cfg = load_config(write_cfg("detection:\n"))
    assert cfg.detection.threshold == 0.55


def test_empty_file_is_defaults(write_cfg) -> None:
    assert load_config(write_cfg("")).backend == "imx500"


def test_cwd_config_is_picked_up(isolated_config) -> None:
    (isolated_config / "config.yaml").write_text("backend: hailo\n")
    cfg = load_config()
    assert cfg.backend == "hailo"
    assert cfg.source == isolated_config / "config.yaml"


def test_missing_explicit_file(isolated_config) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(isolated_config / "nope.yaml")


@pytest.mark.parametrize(
    "yaml_text, match",
    [
        ("backend: tpu\n", "backend"),
        ("bogus: 1\n", "Unknown config keys"),
        ("detection:\n  thresh: 0.5\n", "Unknown keys in 'detection'"),
        ("detection: 5\n", "must be a mapping"),
        ("detection:\n  threshold: 1.5\n", "threshold"),
        ("detection:\n  threshold: high\n", "threshold"),
        ("detection:\n  max_detections: 0\n", "max_detections"),
        ("imx500:\n  bbox_order: zz\n", "bbox_order"),
        ("imx500:\n  inference_rate: 0\n", "inference_rate"),
        ("hailo:\n  main_size: 1280\n", "main_size"),
        ("hailo:\n  main_size: [1, 2, 3]\n", "main_size"),
        ("hailo:\n  main_size: [0, 960]\n", "main_size"),
        ("preview: 'false'\n", "preview"),
        ("max_frames: -1\n", "max_frames"),
        ("- a\n- b\n", "Top level"),
        ("detection: [\n", "Invalid YAML"),
    ],
)
def test_validation_errors(write_cfg, yaml_text: str, match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        load_config(write_cfg(yaml_text))


def test_load_labels(write_cfg, isolated_config) -> None:
    labels = isolated_config / "labels.txt"
    labels.write_text("cat\n\n dog \n")
    cfg = load_config(write_cfg(f"detection:\n  labels_file: {labels}\n"))
    assert cfg.load_labels() == ["cat", "dog"]
