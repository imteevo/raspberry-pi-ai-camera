"""Backend-agnostic detection result type."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    box: tuple[int, int, int, int]  # (x, y, w, h) in main-stream pixel coordinates

    def __str__(self) -> str:
        x, y, w, h = self.box
        return f"{self.label:<15} {self.score:5.2f}  box=({x}, {y}, {w}, {h})"
