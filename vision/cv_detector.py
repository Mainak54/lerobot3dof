"""
cv_detector.py — OpenCV HSV threshold detector, the `opencv` backend from the
6-DOF project, extracted as the only backend.

WHAT IT KEYS ON
---------------
Bright and UNSATURATED: V >= v_min, S <= s_max. That is "white", which is the
cube. It rejects the yellow arm and the blue drop cylinder on saturation
alone, without a neural net. This is not a hue band — hue is meaningless for a
white object, and thresholding it would key on whatever colour cast the
lighting happens to give.

Measured on synthetic frames in the 6-DOF project: 300/300 detected, 0.25 px
median centre error, 0/200 false positives. Near-perfect in sim, brittle on a
real camera — which is what `randomize_thresholds` is for.

OUTPUT CONTRACT — the same 4 numbers as the analytic and YOLO backends
-----------------------------------------------------------------------
    [0] u        centroid x, normalised to [-1, +1], +1 = right edge
    [1] v        centroid y, normalised to [-1, +1], +1 = UP
                 (note the sign: v is +UP, so it is 1 - 2*row/h, NOT the raw
                 pixel row. Every overlay and every policy trained in the
                 6-DOF project assumes this.)
    [2] size     sqrt(bbox area / image area) — the depth cue
    [3] visible  1.0 or 0.0

    All four are exactly zero when nothing is detected. u=v=0 also means "dead
    centre", so consumers MUST read the visible flag rather than treating zero
    as a sentinel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import cv2
except ImportError as e:  # pragma: no cover
    raise ImportError("opencv-python is required: pip install opencv-python") from e


@dataclass
class DetectorConfig:
    """OpenCV HSV ranges: H in [0,179], S in [0,255], V in [0,255]."""
    v_min: int = 140            # brightness floor
    s_max: int = 90             # saturation ceiling
    min_area_px: int = 25
    randomize_thresholds: bool = True
    width: int = 640
    height: int = 480


class CubeDetector:
    """Stateless threshold detector. Thresholds may be jittered per episode."""

    def __init__(self, cfg: DetectorConfig | dict | None = None):
        if isinstance(cfg, dict):
            cfg = DetectorConfig(**cfg)
        self.cfg = cfg or DetectorConfig()
        self.v_min = int(self.cfg.v_min)
        self.s_max = int(self.cfg.s_max)
        self._k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # ------------------------------------------------------------------
    def randomize(self, rng):
        """Jitter the operating point for one episode.

        Without this the policy can come to depend on one tuned threshold
        pair, and any real camera whose exposure differs pushes it off that
        point entirely. Skipped when randomize_thresholds is false.
        """
        if not self.cfg.randomize_thresholds:
            self.v_min, self.s_max = int(self.cfg.v_min), int(self.cfg.s_max)
            return
        self.v_min = int(np.clip(self.cfg.v_min + rng.normal(0, 15), 60, 240))
        self.s_max = int(np.clip(self.cfg.s_max + rng.normal(0, 20), 30, 200))

    def mask(self, rgb: np.ndarray) -> np.ndarray:
        """RGB uint8 HxWx3 -> binary mask. Exposed separately for tuning."""
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        m = cv2.inRange(hsv, (0, 0, self.v_min), (179, self.s_max, 255))
        return cv2.morphologyEx(m, cv2.MORPH_OPEN, self._k)

    # ------------------------------------------------------------------
    def detect(self, rgb: np.ndarray) -> np.ndarray:
        """RGB uint8 HxWx3 -> np.float32([u, v, size, visible])."""
        h, w = rgb.shape[:2]
        contours, _ = cv2.findContours(self.mask(rgb), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        best, best_area = None, 0.0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= self.cfg.min_area_px and area > best_area:
                best, best_area = cnt, area
        if best is None:
            return np.zeros(4, dtype=np.float32)

        M = cv2.moments(best)
        if M["m00"] <= 0:
            return np.zeros(4, dtype=np.float32)
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        bx, by, bw, bh = cv2.boundingRect(best)

        u = 2.0 * cx / w - 1.0
        v = 1.0 - 2.0 * cy / h              # +v is UP, see the module docstring
        size = float(np.sqrt((bw * bh) / float(w * h)))
        return np.array([u, v, size, 1.0], dtype=np.float32)


# ----------------------------------------------------------------------
class MujocoCameraSource:
    """
    Renders the wrist camera. Separate from CubeDetector so the detector has
    no MuJoCo dependency and can be imported unchanged by the robot script.

    Offscreen rendering dominates this env's cost. Set MUJOCO_GL=egl for
    headless GPU rendering; on CPU (osmesa) expect roughly an order of
    magnitude less throughput.
    """

    def __init__(self, model, camera: str = "wrist_cam",
                 width: int = 640, height: int = 480):
        import mujoco
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.camera = camera
        self.width, self.height = width, height

    def frame(self, data) -> np.ndarray:
        self.renderer.update_scene(data, camera=self.camera)
        return self.renderer.render()

    def close(self):
        try:
            self.renderer.close()
        except Exception:
            pass