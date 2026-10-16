"""Raspberry Pi 5 AI camera object detection.

This package runs real-time object detection on a Raspberry Pi 5 using one of
two interchangeable inference backends:

* ``imx500`` - the Raspberry Pi AI Camera (Sony IMX500). The neural network
  runs *on the image sensor itself*; the Pi only receives output tensors in
  each frame's metadata, so CPU load is minimal.
* ``hailo`` - the Raspberry Pi AI HAT+ (Hailo-8L/Hailo-8 NPU on PCIe). Frames
  are captured on the Pi and pushed to the NPU for inference.

Modules:
    config:     YAML configuration loading and typed config dataclasses.
    detection:  Backend-agnostic :class:`~src.detection.Detection` result.
    camera:     IMX500 backend (:class:`~src.camera.IMX500Detector`).
    hailo:      Hailo backend (:class:`~src.hailo.HailoDetector`).
    main:       Command-line entry point (``python -m src.main``).
"""

__version__ = "0.1.0"
