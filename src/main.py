"""Command-line entry point.

Usage::

    python -m src.main --backend imx500|hailo [--preview | --headless]
                       [--model PATH] [--threshold T] [--frames N]
                       [--config FILE] [--verbose]

Loads :mod:`config <src.config>`, applies CLI overrides, builds the chosen
backend and prints detections for each frame to stdout until ``--frames`` is
reached or the user presses Ctrl+C.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Optional, Sequence

from .config import AppConfig, load_config

log = logging.getLogger("ai-camera")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Every option defaults to ``None`` so :func:`apply_overrides` can tell an
    explicit flag apart from "not given" and leave config.yaml values alone.
    """
    p = argparse.ArgumentParser(description="Object detection on Raspberry Pi 5 AI Camera / AI HAT+")
    p.add_argument("--config", "-c", help="path to config.yaml (default: repo config.yaml)")
    p.add_argument("--backend", "-b", choices=["imx500", "hailo"], help="inference backend")
    p.add_argument("--model", help="model path (.rpk for imx500, .hef for hailo)")
    p.add_argument("--threshold", "-t", type=float, help="detection score threshold")
    p.add_argument("--frames", "-n", type=int, help="stop after N result frames (0 = forever)")

    # --preview and --headless write the same destination; default=None means
    # "use config.yaml" when neither is passed.
    view = p.add_mutually_exclusive_group()
    view.add_argument("--preview", dest="preview", action="store_true", default=None, help="show preview window")
    view.add_argument("--headless", dest="preview", action="store_false", help="no preview window")

    p.add_argument("--verbose", "-v", action="store_true")
    return p


def apply_overrides(cfg: AppConfig, args: argparse.Namespace) -> AppConfig:
    """Apply explicitly passed CLI flags on top of the loaded config.

    Args:
        cfg: Config loaded from YAML/defaults. Modified in place.
        args: Parsed CLI arguments.

    Returns:
        The same ``cfg`` object, for convenience.
    """
    if args.backend:
        cfg.backend = args.backend
    if args.preview is not None:
        cfg.preview = args.preview
    if args.threshold is not None:
        cfg.detection.threshold = args.threshold
    if args.frames is not None:
        cfg.max_frames = args.frames
    if args.model:
        # --model applies to whichever backend is active *after* --backend
        # was applied, which is why this override comes last.
        if cfg.backend == "imx500":
            cfg.imx500.model = args.model
        else:
            cfg.hailo.model = args.model
    return cfg


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the detection loop.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``); injectable for tests.

    Returns:
        Process exit code (``0`` on normal exit or Ctrl+C).
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    cfg = apply_overrides(load_config(args.config), args)

    # Import only the selected backend, so a machine with just one
    # accelerator's software installed can still run the other.
    if cfg.backend == "imx500":
        from .camera import IMX500Detector as Detector
    else:
        from .hailo import HailoDetector as Detector

    log.info("backend=%s preview=%s threshold=%.2f", cfg.backend, cfg.preview, cfg.detection.threshold)
    detector = Detector(cfg)
    detector.start(preview=cfg.preview)

    frames = 0
    t0 = time.monotonic()  # monotonic: immune to wall-clock/NTP jumps
    try:
        for dets in detector.detections():
            frames += 1
            # Average FPS since start; guard against a zero divisor.
            fps = frames / max(time.monotonic() - t0, 1e-6)
            print(f"[frame {frames} | {fps:4.1f} fps] {len(dets)} detection(s)")
            for d in dets:
                print(f"    {d}")
            # Flush so output appears promptly when piped (e.g. into tee or journald).
            sys.stdout.flush()
            if cfg.max_frames and frames >= cfg.max_frames:
                break
    except KeyboardInterrupt:
        # Ctrl+C is the normal way to stop; exit cleanly without a traceback.
        pass
    finally:
        # Always release the camera/NPU, or the next run fails with "device busy".
        detector.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
