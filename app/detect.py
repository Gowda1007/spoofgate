import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import threading
from dataclasses import dataclass
from typing import Optional

import cv2
cv2.setNumThreads(1)
import numpy as np

from . import config


@dataclass
class Face:
    box: tuple                    # x, y, w, h (ints, inside the image)
    row: Optional[np.ndarray]     # raw YuNet row (15 values) or None for Haar
    score: float

    @property
    def landmarks(self):
        return None if self.row is None else self.row[4:14].reshape(5, 2)


def prep_frame(img):
    """Downscale big frames so every stage is fast and consistent."""
    h, w = img.shape[:2]
    if w > config.MAX_WIDTH:
        s = config.MAX_WIDTH / w
        img = cv2.resize(img, (config.MAX_WIDTH, int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def _clip_box(x, y, w, h, W, H):
    x = max(0, min(int(x), W - 2))
    y = max(0, min(int(y), H - 2))
    w = max(2, min(int(w), W - x))
    h = max(2, min(int(h), H - y))
    return (x, y, w, h)


def crop_box(img, box, expand=0.0):
    x, y, w, h = box
    H, W = img.shape[:2]
    dx, dy = int(w * expand), int(h * expand)
    x0, y0 = max(0, x - dx), max(0, y - dy)
    x1, y1 = min(W, x + w + dx), min(H, y + h + dy)
    return img[y0:y1, x0:x1]


class FaceDetector:
    """YuNet (OpenCV Zoo) if the ONNX file exists, otherwise OpenCV's built-in Haar cascade."""

    def __init__(self):
        self.lock = threading.Lock()
        self.yunet = None
        if config.YUNET_PATH.exists():
            try:
                self.yunet = cv2.FaceDetectorYN.create(str(config.YUNET_PATH), "", (320, 320), 0.7, 0.3, 5000)
            except Exception as e:  # noqa
                print("YuNet failed to load, falling back to Haar:", e)
        self.haar = None
        if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
            try:
                self.haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            except Exception:
                self.haar = None
        self.name = "yunet" if self.yunet is not None else ("haar" if self.haar is not None else "none")

    def detect(self, img) -> Optional[Face]:
        H, W = img.shape[:2]
        if self.yunet is not None:
            with self.lock:
                self.yunet.setInputSize((W, H))
                _, faces = self.yunet.detect(img)
            if faces is None or len(faces) == 0:
                return None
            row = max(faces, key=lambda r: r[2] * r[3])
            box = _clip_box(row[0], row[1], row[2], row[3], W, H)
            return Face(box, np.array(row, dtype=np.float32), float(row[14]))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        s = min(1.0, 320.0 / W)                      # Haar is slow on big frames, so search a small copy
        small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1.0 else gray
        boxes = self.haar.detectMultiScale(small, 1.1, 5, minSize=(max(20, int(60 * s)),) * 2)
        if len(boxes) == 0:
            return None
        x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
        return Face(_clip_box(x / s, y / s, w / s, h / s, W, H), None, 1.0)


def yaw_signed(face: Face) -> float:
    lm = face.landmarks
    if lm is None:
        return 0.0
    re, le, nose = lm[0], lm[1], lm[2]
    eye_mid = (re + le) / 2.0
    eye_d = float(np.linalg.norm(re - le)) + 1e-6
    return float((nose[0] - eye_mid[0]) / eye_d)


def quality(img, face: Face) -> dict:
    x, y, w, h = face.box
    crop = crop_box(img, face.box)
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g128 = cv2.resize(g, (128, 128), interpolation=cv2.INTER_AREA)
    sharp = float(cv2.Laplacian(g128, cv2.CV_64F).var())
    bright = float(g.mean())
    yaw = abs(yaw_signed(face))
    reasons = []
    if min(w, h) < config.MIN_FACE_PX:
        reasons.append("face_too_small")
    if bright < config.MIN_BRIGHT:
        reasons.append("too_dark")
    if bright > config.MAX_BRIGHT:
        reasons.append("too_bright")
    if sharp < config.MIN_SHARP:
        reasons.append("blurry")
    if yaw > config.MAX_YAW:
        reasons.append("head_turned")
    return {"ok": not reasons, "reasons": reasons, "face_px": int(min(w, h)),
            "brightness": round(bright, 1), "sharpness": round(sharp, 1), "yaw": round(yaw, 2)}
