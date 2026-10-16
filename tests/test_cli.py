"""Tests for CLI override handling and error exit codes."""
from __future__ import annotations

import pytest

from pi_ai_camera.config import ConfigError, load_config
from pi_ai_camera.main import apply_overrides, build_parser, main


def _cfg(*argv: str):
    return apply_overrides(load_config(), build_parser().parse_args(list(argv)))


def test_model_applies_to_backend_selected_on_cli(isolated_config) -> None:
    cfg = _cfg("-b", "hailo", "--model", "x.hef", "--headless", "-n", "3")
    assert (cfg.backend, cfg.hailo.model, cfg.preview, cfg.max_frames) == ("hailo", "x.hef", False, 3)
    assert cfg.imx500.model.endswith(".rpk")  # untouched


def test_preview_none_keeps_config(isolated_config) -> None:
    assert _cfg().preview is False
    assert _cfg("--preview").preview is True


def test_cli_values_are_validated(isolated_config) -> None:
    with pytest.raises(ConfigError):
        _cfg("--threshold", "2")


def test_main_config_error_exit_code(isolated_config, tmp_path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("backend: nope\n")
    assert main(["--config", str(bad)]) == 2


def test_main_missing_model_exit_code(isolated_config) -> None:
    # picamera2 isn't installed off-device, so initialisation fails cleanly
    # (ImportError or FileNotFoundError) with exit code 2 - not a traceback.
    assert main(["-b", "imx500", "--model", "/nonexistent.rpk", "-n", "1"]) == 2
