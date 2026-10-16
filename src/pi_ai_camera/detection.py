"""Backend-agnostic detection result type and detector interface.

Both backends convert their native output tensors into :class:`Detection`
objects and implement :class:`Detector`, so the CLI (and future consumers,
e.g. MQTT publishers or overlays) never needs to know which accelerator
produced a result.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol


@dataclass(frozen=True)
class Detection:
    """A single detected object.

    Frozen (immutable) so detections can be safely shared between threads or
    used as dict keys/set members.

    Attributes:
        label: Human-readable class name (e.g. ``"person"``). Falls back to
            the numeric class id as a string when no label is available.
        score: Model confidence in ``[0.0, 1.0]``.
        box: Bounding box as ``(x, y, w, h)`` in pixel coordinates of the
            camera's *main* output stream (not the network input), so it can
            be drawn directly on preview/recorded frames.
    """

    label: str
    score: float
    box: tuple[int, int, int, int]

    def __str__(self) -> str:
        """Return a fixed-width, single-line summary suitable for logs."""
        x, y, w, h = self.box
        return f"{self.label:<15} {self.score:5.2f}  box=({x}, {y}, {w}, {h})"


class Detector(Protocol):
    """Structural interface implemented by every inference backend.

    Lifecycle: construct (acquires hardware), :meth:`start`, iterate
    :meth:`detections`, then :meth:`stop`. ``stop`` must be idempotent and
    safe to call even if ``start`` failed or was never called.
    """

    def start(self, preview: bool = False) -> None: ...

    def stop(self) -> None: ...

    def detections(self) -> Iterator[list[Detection]]: ...
