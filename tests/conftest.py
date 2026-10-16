"""Shared pytest fixtures.

The tests never touch hardware: backends are exercised through their pure
parsing helpers, or via instances created with ``object.__new__`` whose
hardware attributes are replaced by fakes.
"""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run from an empty temp dir with no XDG config, so only explicit files
    or built-in defaults apply (the repo's config.yaml is still a fallback)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    return tmp_path


@pytest.fixture
def write_cfg(isolated_config: Path):
    """Return a helper that writes YAML text to a temp config file."""
    def _write(text: str) -> Path:
        p = isolated_config / "test-config.yaml"
        p.write_text(text)
        return p
    return _write
