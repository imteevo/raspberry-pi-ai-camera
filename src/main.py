"""CLI entry point: python -m src.main --backend imx500|hailo [--preview|--headless]"""
from __future__ import annotations

import argparse
import logging
import sys
import time

from .config import load_config

log = logging.getLogger("ai-camera")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Object detection on Raspberry Pi 5 AI Camera / AI HAT+")
    p.add_argument("--config", "-c", help="path to config.yaml (default: repo config.yaml)")
    p.add_argument("--backend", "-b", choices=["imx500", "hailo"], help="inference backend")
    p.add_argument("--model", help="model path (.rpk for imx500, .hef for hailo)")
    p.add_argument("--threshold", "-t", type=float, help="detection score threshold")
    p.add_argument("--frames", "-n", type=int, help="stop after N result frames (0 = forever)")
    view = p.add_mutually_exclusive_group()
    view.add_argument("--preview", dest="preview", action="store_true", default=None, help="show preview window")
    view.add_argument("--headless", dest="preview", action="store_false", help="no preview window")
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    if args.backend:
        cfg.backend = args.backend
    if args.preview is not None:
        cfg.preview = args.preview
    if args.threshold is not None:
        cfg.detection.threshold = args.threshold
    if args.frames is not None:
        cfg.max_frames = args.frames
    if args.model:
        if cfg.backend == "imx500":
            cfg.imx500.model = args.model
        else:
            cfg.hailo.model = args.model

    if cfg.backend == "imx500":
        from .camera import IMX500Detector as Detector
    else:
        from .hailo import HailoDetector as Detector

    log.info("backend=%s preview=%s threshold=%.2f", cfg.backend, cfg.preview, cfg.detection.threshold)
    detector = Detector(cfg)
    detector.start(preview=cfg.preview)
    frames, t0 = 0, time.monotonic()
    try:
        for dets in detector.detections():
            frames += 1
            fps = frames / max(time.monotonic() - t0, 1e-6)
            print(f"[frame {frames} | {fps:4.1f} fps] {len(dets)} detection(s)")
            for d in dets:
                print(f"    {d}")
            sys.stdout.flush()
            if cfg.max_frames and frames >= cfg.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        detector.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
