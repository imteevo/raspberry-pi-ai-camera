"""Command-line entry point.

Usage::

    pi-ai-camera --backend imx500|hailo [--preview | --headless]
                 [--model PATH|auto] [--threshold T] [--frames N]
                 [--config FILE] [--verbose]

(``python -m pi_ai_camera ...`` is equivalent.)

Loads :mod:`config <pi_ai_camera.config>`, applies CLI overrides, builds the
chosen backend and prints detections for each frame to stdout until
``--frames`` is reached or the user presses Ctrl+C.

Exit codes: ``0`` success/Ctrl+C, ``2`` configuration or initialisation
error, ``1`` unexpected runtime error.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Optional, Sequence

from . import __version__
from .config import AppConfig, ConfigError, load_config
from .detection import Detector

log = logging.getLogger("pi_ai_camera")

#: Smoothing factor for the FPS exponential moving average. 0.1 weights
#: roughly the last ~10 frames, so the figure reacts quickly but doesn't
#: jitter, and isn't skewed by the one-off IMX500 firmware upload.
FPS_EMA_ALPHA: float = 0.1


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Every option defaults to ``None`` so :func:`apply_overrides` can tell an
    explicit flag apart from "not given" and leave config values alone.
    """
    p = argparse.ArgumentParser(
        prog="pi-ai-camera",
        description="Object detection on Raspberry Pi 5 AI Camera / AI HAT+",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("--config", "-c", help="path to config.yaml (default: search ./, ~/.config, /etc, repo)")
    p.add_argument("--backend", "-b", choices=["imx500", "hailo"], help="inference backend")
    p.add_argument("--model", help="model path (.rpk for imx500, .hef or 'auto' for hailo)")
    p.add_argument("--threshold", "-t", type=float, help="detection score threshold [0-1]")
    p.add_argument("--frames", "-n", type=int, help="stop after N result frames (0 = forever)")

    # --preview and --headless write the same destination; default=None means
    # "use the config file" when neither is passed.
    view = p.add_mutually_exclusive_group()
    view.add_argument("--preview", dest="preview", action="store_true", default=None, help="show preview window")
    view.add_argument("--headless", dest="preview", action="store_false", help="no preview window")

    p.add_argument("--verbose", "-v", action="store_true", help="debug logging and full tracebacks")
    return p


def apply_overrides(cfg: AppConfig, args: argparse.Namespace) -> AppConfig:
    """Apply explicitly passed CLI flags on top of the loaded config.

    Args:
        cfg: Config loaded from YAML/defaults. Modified in place.
        args: Parsed CLI arguments.

    Returns:
        The same ``cfg``, re-validated so CLI values get the same range checks
        as YAML values.

    Raises:
        ConfigError: If an override is out of range (e.g. ``--threshold 2``).
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
    return cfg.validate()


def create_detector(cfg: AppConfig) -> Detector:
    """Instantiate the configured backend.

    Only the selected backend's module is imported, so a machine with just
    one accelerator's software installed can still run the other.
    """
    if cfg.backend == "imx500":
        from .camera import IMX500Detector

        return IMX500Detector(cfg)
    from .hailo import HailoDetector

    return HailoDetector(cfg)


def run(detector: Detector, cfg: AppConfig) -> None:
    """Start ``detector`` and print detections until done.

    The caller owns ``detector`` and must call ``stop()`` afterwards; this
    function deliberately doesn't, so cleanup lives in one ``finally``.
    """
    detector.start(preview=cfg.preview)
    frames = 0
    fps: Optional[float] = None
    last = time.monotonic()  # monotonic: immune to wall-clock/NTP jumps
    for dets in detector.detections():
        frames += 1
        now = time.monotonic()
        inst = 1.0 / max(now - last, 1e-6)  # guard against a zero interval
        last = now
        fps = inst if fps is None else FPS_EMA_ALPHA * inst + (1 - FPS_EMA_ALPHA) * fps
        print(f"[frame {frames} | {fps:4.1f} fps] {len(dets)} detection(s)")
        for d in dets:
            print(f"    {d}")
        # Flush so output appears promptly when piped (tee, journald, ...).
        sys.stdout.flush()
        if cfg.max_frames and frames >= cfg.max_frames:
            break


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``); injectable for tests.

    Returns:
        Process exit code (see module docstring).
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    try:
        cfg = apply_overrides(load_config(args.config), args)
    except (ConfigError, FileNotFoundError) as exc:
        log.error("Configuration error: %s", exc)
        return 2
    log.info("config=%s backend=%s preview=%s threshold=%.2f",
             cfg.source or "<defaults>", cfg.backend, cfg.preview, cfg.detection.threshold)

    # Construction acquires hardware. Backends clean up after themselves if
    # construction fails, so there is nothing to stop() here.
    try:
        detector = create_detector(cfg)
    except KeyboardInterrupt:
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, ImportError, NotImplementedError) as exc:
        log.error("Failed to initialise %s backend: %s", cfg.backend, exc, exc_info=args.verbose)
        return 2

    try:
        run(detector, cfg)
    except KeyboardInterrupt:
        # Ctrl+C is the normal way to stop (including during the IMX500
        # firmware upload); exit cleanly without a traceback.
        pass
    except Exception as exc:  # noqa: BLE001 - top-level boundary: report, don't crash
        log.error("Runtime error: %s", exc, exc_info=args.verbose)
        return 1
    finally:
        # Always release the camera/NPU, or the next run fails "device busy".
        detector.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
