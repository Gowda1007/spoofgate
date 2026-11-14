import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
LOG_FILE = DATA_DIR / "logs" / "attempts.jsonl"
STORE_DIR = DATA_DIR / "store"

YUNET_PATH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_PATH = MODEL_DIR / "face_recognition_sface_2021dec.onnx"
ANTISPOOF_PATH = MODEL_DIR / "antispoof.onnx"      # optional MiniFASNet-style ONNX
LIVENESS_PATH = MODEL_DIR / "liveness_clf.joblib"  # written by: python tools.py train


def _f(name, default):
    return float(os.getenv(name, default))


def _i(name, default):
    return int(os.getenv(name, default))


# frame / quality gate
MAX_WIDTH = _i("MAX_WIDTH", 640)
MIN_FACE_PX = _i("MIN_FACE_PX", 80)
MIN_BRIGHT = _f("MIN_BRIGHT", 40)
MAX_BRIGHT = _f("MAX_BRIGHT", 225)
MIN_SHARP = _f("MIN_SHARP", 15)
MAX_YAW = _f("MAX_YAW", 0.6)
MIN_GOOD_FRAMES = _i("MIN_GOOD_FRAMES", 2)

# decision thresholds (liveness thresholds are overridden by the trained model file unless set here)
T_DENY = os.getenv("T_DENY")
T_ALLOW = os.getenv("T_ALLOW")
T_DENY_DEFAULT = 0.30
T_ALLOW_DEFAULT = 0.70
ID_MATCH = _f("ID_MATCH", 0.363)   # SFace cosine threshold from the OpenCV docs; tune on your data
ID_UNSURE = _f("ID_UNSURE", 0.25)

# abuse protection
LOCK_FAILS = _i("LOCK_FAILS", 5)
LOCK_WINDOW_S = _i("LOCK_WINDOW_S", 180)
LOCK_SECONDS = _i("LOCK_SECONDS", 300)
MAX_RETRIES = _i("MAX_RETRIES", 3)

# optional anti-spoof ONNX details (these differ between ports; check yours)
ANTISPOOF_SCALE = _f("ANTISPOOF_SCALE", 2.7)
ANTISPOOF_NORM = os.getenv("ANTISPOOF_NORM", "raw")   # raw = 0..255, unit = 0..1
ANTISPOOF_REAL_INDEX = _i("ANTISPOOF_REAL_INDEX", 1)
