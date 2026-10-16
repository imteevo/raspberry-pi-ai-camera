"""Raspberry Pi 5 AI camera object detection.

This package runs real-time object detection on a Raspberry Pi 5 using one of
two interchangeable inference backends:

* ``imx500`` - the Raspberry Pi AI Camera (Sony IMX500). The neural network
  runs *on the image sensor itself*; the Pi only receives output tensors in
  each frame's metadata, so CPU load is minimal.
* ``hailo`` - the Raspberry Pi AI HAT+ / AI HAT+ 2 (Hailo-8L, Hailo-8 or
  Hailo-10H NPU on PCIe). Frames are captured on the Pi and pushed to the NPU.

Modules:
    config:     YAML configuration loading, validation and typed dataclasses.
    detection:  :class:`Detection` result type and :class:`Detector` protocol.
    camera:     IMX500 backend (:class:`~pi_ai_camera.camera.IMX500Detector`).
    hailo:      Hailo backend (:class:`~pi_ai_camera.hailo.HailoDetector`).
    main:       Command-line entry point (``pi-ai-camera`` /
                ``python -m pi_ai_camera``).
"""

__version__ = "0.2.0"
