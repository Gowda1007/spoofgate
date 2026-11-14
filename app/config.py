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

# Security & Keys
KEYS_DIR = Path(os.getenv("KEYS_DIR", str(ROOT.parent / ".keys")))
KEYS_DIR.mkdir(parents=True, exist_ok=True)
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "hid-demo-secret-token-2026")
DEMO_MODE = os.getenv("DEMO_MODE", "1") == "1"

# CORS Configuration (Restricted in production; override via CORS_ORIGINS)
_cors_env = os.getenv("CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000,http://localhost:3000,http://127.0.0.1:3000")
CORS_ORIGINS = [o.strip() for o in _cors_env.split(",") if o.strip()]
if DEMO_MODE and "*" not in CORS_ORIGINS:
    CORS_ORIGINS.append("*")

# Hybrid & AWS Rekognition Configuration
VERIFICATION_MODE = os.getenv("VERIFICATION_MODE", "hybrid")  # "hybrid" | "local" | "aws_strict"
ENABLE_LOCAL_PREFILTER = os.getenv("ENABLE_LOCAL_PREFILTER", "1") == "1"  # Cost-Optimization Filter
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")  # Mumbai default
AWS_LIVENESS_THRESHOLD = _f("AWS_LIVENESS_THRESHOLD", 80.0)
AWS_FACE_MATCH_THRESHOLD = _f("AWS_FACE_MATCH_THRESHOLD", 95.0)

# Request Limits
MAX_FRAMES_PER_REQUEST = _i("MAX_FRAMES_PER_REQUEST", 10)
MAX_FRAME_BYTES = _i("MAX_FRAME_BYTES", 2_000_000)  # 2MB max per frame

# optional anti-spoof ONNX details (these differ between ports; check yours)
ANTISPOOF_SCALE = _f("ANTISPOOF_SCALE", 2.7)
ANTISPOOF_NORM = os.getenv("ANTISPOOF_NORM", "raw")   # raw = 0..255, unit = 0..1
ANTISPOOF_REAL_INDEX = _i("ANTISPOOF_REAL_INDEX", 1)
