"""Tests for Hailo HEF selection (M2) and NMS output parsing."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pi_ai_camera import hailo
from pi_ai_camera.hailo import parse_nms, resolve_hef


@pytest.fixture
def models_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the module at a temp hailo-models dir with all default HEFs."""
    monkeypatch.setattr(hailo, "HAILO_MODELS_DIR", tmp_path)
    mapping = {}
    for arch, p in hailo.DEFAULT_HEF.items():
        (tmp_path / p.name).write_bytes(b"")
        mapping[arch] = tmp_path / p.name
    monkeypatch.setattr(hailo, "DEFAULT_HEF", mapping)
    return tmp_path


@pytest.mark.parametrize(
    "arch, expected",
    [("HAILO8L", "yolov8s_h8l.hef"), ("HAILO8", "yolov8s_h8.hef"),
     ("HAILO10H", "yolov8m_h10.hef"), (None, "yolov8s_h8l.hef"), ("WEIRD", "yolov8s_h8l.hef")],
)
def test_auto_selects_by_arch(models_dir: Path, arch, expected: str) -> None:
    assert resolve_hef("auto", arch).name == expected


def test_explicit_path(models_dir: Path) -> None:
    custom = models_dir / "custom.hef"
    custom.write_bytes(b"")
    assert resolve_hef(str(custom), "HAILO8") == custom


def test_missing_hef_lists_installed(models_dir: Path) -> None:
    with pytest.raises(FileNotFoundError, match="yolov8s_h8l.hef"):
        resolve_hef(str(models_dir / "missing.hef"), None)


def test_parse_nms_scales_filters_sorts_and_clips() -> None:
    results = [
        np.array([[0.1, 0.25, 0.5, 0.75, 0.9], [0.0, 0.0, 0.1, 0.1, 0.2]]),  # class 0
        np.empty((0, 5)),                                                  # class 1
        np.array([[-0.05, 0.5, 0.5, 1.2, 0.7]]),                           # class 2, out of range
    ]
    dets = parse_nms(results, (1000, 500), ["person", "bicycle", "car"], 0.5, 10)
    assert [d.label for d in dets] == ["person", "car"]
    assert dets[0].box == (250, 50, 500, 200)
    assert dets[1].box == (500, 0, 500, 250)  # clipped to [0, 1]


def test_parse_nms_max_detections_and_unknown_label() -> None:
    results = [np.array([[0, 0, 1, 1, s] for s in (0.6, 0.9, 0.8)])]
    dets = parse_nms(results, (10, 10), [], 0.5, 2)
    assert [d.score for d in dets] == [0.9, 0.8]
    assert dets[0].label == "0"


def test_parse_nms_rejects_multi_output() -> None:
    with pytest.raises(RuntimeError, match="single NMS-output"):
        parse_nms({"a": 1, "b": 2}, (10, 10), [], 0.5, 10)
