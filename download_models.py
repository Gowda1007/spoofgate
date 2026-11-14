"""Downloads the two open-source OpenCV Zoo models (YuNet face detector, SFace face recogniser)."""
import sys
import urllib.request
from pathlib import Path

DEST = Path(__file__).parent / "models"
BASE = "https://github.com/opencv/opencv_zoo/raw/main/models/"
FILES = {
    "face_detection_yunet_2023mar.onnx": BASE + "face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx": BASE + "face_recognition_sface/face_recognition_sface_2021dec.onnx",
}

DEST.mkdir(exist_ok=True)
ok = True
for name, url in FILES.items():
    out = DEST / name
    if out.exists() and out.stat().st_size > 100_000:
        print("already have", name)
        continue
    try:
        print("downloading", name)
        urllib.request.urlretrieve(url, out)
        if out.stat().st_size < 100_000:   # a git-LFS pointer file is only ~130 bytes
            raise RuntimeError("got a tiny file (LFS pointer?)")
    except Exception as e:
        ok = False
        out.unlink(missing_ok=True)
        print(f"  FAILED: {e}\n  Download it in a browser from:\n  {url}\n  and put it in {DEST}")
print("done" if ok else "some downloads failed (see above)")
sys.exit(0 if ok else 1)
