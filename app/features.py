"""Hand-made, CPU-cheap anti-spoof features + an optional MiniFASNet-style ONNX score.

The classifier in liveness.py is trained on YOUR collected data, so these features only need to be
informative, not perfectly calibrated."""
import cv2
import numpy as np

from . import config
from .detect import crop_box

_R = None
_WIN = np.outer(np.hanning(128), np.hanning(128)).astype(np.float32)
STD_KEYS = {"lap_log", "fft_hf", "sat_mean"}


def _radius(n=128):
    global _R
    if _R is None:
        yy, xx = np.mgrid[-n // 2:n // 2, -n // 2:n // 2]
        _R = np.sqrt(yy ** 2 + xx ** 2) / n
    return _R


def frame_features(img, face):
    crop = crop_box(img, face.box, 0.1)
    g_full = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g_full, (128, 128), interpolation=cv2.INTER_AREA)
    f = {}

    # sharpness / texture
    f["lap_log"] = float(np.log1p(cv2.Laplacian(g, cv2.CV_64F).var()))
    f["edge_density"] = float(cv2.Canny(g, 50, 150).mean() / 255.0)
    hist = np.bincount((g // 8).ravel(), minlength=32) / g.size
    hist = hist[hist > 0]
    f["entropy"] = float(-(hist * np.log2(hist)).sum())

    # frequency content: screens add moire / pixel-grid energy, prints lose fine detail
    spec = np.fft.fftshift(np.fft.fft2((g.astype(np.float32) - g.mean()) * _WIN))
    P = np.abs(spec) ** 2 + 1e-9
    R = _radius()
    tot = P.sum()
    f["fft_mid"] = float(np.log10(P[(R > 0.10) & (R <= 0.25)].sum() / tot + 1e-9))
    f["fft_hf"] = float(np.log10(P[R > 0.25].sum() / tot + 1e-9))
    mag = np.log1p(np.sqrt(P))
    band = mag[R > 0.15]
    f["fft_peak"] = float((band.max() - np.median(band)) / (band.std() + 1e-6))

    # ENHANCED MOIRE & SCREEN ARTIFACT DETECTORS
    # 1. Directional Harmonic Moire (Screens emit repetitive vertical/horizontal pixel-grid lines)
    horiz_band = mag[58:70, :]
    vert_band = mag[:, 58:70]
    f["moire_horiz_spike"] = float(horiz_band.max() / (horiz_band.mean() + 1e-6))
    f["moire_vert_spike"] = float(vert_band.max() / (vert_band.mean() + 1e-6))

    # 2. Local Binary Pattern (LBP) Texture Uniformity (Real skin has micro-pores; screens/prints have flat or grid grain)
    g_pad = np.pad(g, 1, mode='edge')
    lbp = ((g_pad[1:-1, 1:-1] >= g_pad[:-2, 1:-1]) * 1 +
           (g_pad[1:-1, 1:-1] >= g_pad[2:, 1:-1]) * 2 +
           (g_pad[1:-1, 1:-1] >= g_pad[1:-1, :-2]) * 4 +
           (g_pad[1:-1, 1:-1] >= g_pad[1:-1, 2:]) * 8 +
           (g_pad[1:-1, 1:-1] >= g_pad[:-2, :-2]) * 16 +
           (g_pad[1:-1, 1:-1] >= g_pad[2:, 2:]) * 32)
    lbp_hist = np.bincount(lbp.ravel(), minlength=64) / lbp.size
    lbp_hist = lbp_hist[lbp_hist > 0]
    f["lbp_entropy"] = float(-(lbp_hist * np.log2(lbp_hist)).sum())

    # 3. Specular Screen Reflection & Chromatic Aberration
    # Phone OLED/LCD displays have RGB subpixel shifts visible at close range
    b, g_ch, r = cv2.split(crop)
    f["chroma_diff_rg"] = float(np.abs(r.astype(np.float32) - g_ch.astype(np.float32)).mean() / 255.0)
    f["chroma_diff_bg"] = float(np.abs(b.astype(np.float32) - g_ch.astype(np.float32)).mean() / 255.0)

    # colour / reflectance
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    f["sat_mean"] = float(hsv[..., 1].mean() / 255.0)
    f["sat_std"] = float(hsv[..., 1].std() / 255.0)
    f["val_std"] = float(hsv[..., 2].std() / 255.0)
    ycc = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
    f["cr_mean"] = float(ycc[..., 1].mean() / 255.0)
    f["cb_mean"] = float(ycc[..., 2].mean() / 255.0)
    f["cr_std"] = float(ycc[..., 1].std() / 255.0)
    f["cb_std"] = float(ycc[..., 2].std() / 255.0)
    f["glare"] = float((g_full > 245).mean())

    # context: straight long edges around the face = phone / paper / screen border
    ctx = crop_box(img, face.box, 0.9)
    cg = cv2.cvtColor(ctx, cv2.COLOR_BGR2GRAY)
    ce = cv2.Canny(cg, 60, 160)
    minlen = max(20, int(0.6 * face.box[2]))
    lines = cv2.HoughLinesP(ce, 1, np.pi / 180, threshold=40, minLineLength=minlen, maxLineGap=6)
    n, length = 0, 0.0
    if lines is not None and len(lines) > 0:
        for seg in np.array(lines).reshape(-1, 4):
            x1, y1, x2, y2 = seg
            ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180
            if min(ang, abs(ang - 90), abs(ang - 180)) < 10:
                n += 1
                length += float(np.hypot(x2 - x1, y2 - y1))
    f["ctx_lines"] = float(min(n, 20))
    f["ctx_line_len"] = float(length / (ctx.shape[0] + ctx.shape[1] + 1e-6))
    return f


class AntiSpoof:
    """Optional: any MiniFASNet-style ONNX (e.g. from Silent-Face-Anti-Spoofing ports).
    Port details differ (input size, scaling, class order), so the trained classifier consumes the raw
    score as one feature and calibrates it. Verify on one real face and one print before trusting it."""

    def __init__(self, path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        shp = list(inp.shape)
        self.name = inp.name
        self.nhwc = len(shp) == 4 and shp[-1] == 3
        hw = shp[1:3] if self.nhwc else shp[2:4]
        self.H = hw[0] if isinstance(hw[0], int) else 80
        self.W = hw[1] if isinstance(hw[1], int) else 80

    def real_prob(self, img, face):
        x, y, w, h = face.box
        side = max(w, h) * config.ANTISPOOF_SCALE
        cx, cy = x + w / 2, y + h / 2
        x0, y0, x1, y1 = int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2)
        pad = max(0, -x0, -y0, x1 - img.shape[1], y1 - img.shape[0])
        if pad > 0:
            img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
            x0, y0, x1, y1 = x0 + pad, y0 + pad, x1 + pad, y1 + pad
        crop = cv2.resize(img[y0:y1, x0:x1], (self.W, self.H)).astype(np.float32)
        if config.ANTISPOOF_NORM == "unit":
            crop /= 255.0
        arr = crop[None] if self.nhwc else crop.transpose(2, 0, 1)[None]
        out = self.sess.run(None, {self.name: arr})[0][0]
        e = np.exp(out - out.max())
        p = e / e.sum()
        return float(p[min(config.ANTISPOOF_REAL_INDEX, len(p) - 1)])


def burst_features(frames, faces, antispoof=None):
    per, smalls, centers, widths = [], [], [], []
    for img, fc in zip(frames, faces):
        per.append(frame_features(img, fc))
        small = cv2.resize(cv2.cvtColor(crop_box(img, fc.box), cv2.COLOR_BGR2GRAY), (64, 64))
        smalls.append(small.astype(np.float32))
        x, y, w, h = fc.box
        centers.append((x + w / 2, y + h / 2))
        widths.append(w)

    out = {}
    for k in per[0]:
        v = np.array([p[k] for p in per])
        out[k + "_mean"] = float(v.mean())
        if k in STD_KEYS:
            out[k + "_std"] = float(v.std())

    # temporal cues: flat photos move rigidly, replays have different frame-to-frame change
    if len(smalls) > 1:
        diffs = [float(np.abs(smalls[i + 1] - smalls[i]).mean() / 255.0) for i in range(len(smalls) - 1)]
        shifts = [float(np.hypot(centers[i + 1][0] - centers[i][0], centers[i + 1][1] - centers[i][1]) / widths[i])
                  for i in range(len(smalls) - 1)]
        out["t_diff_mean"], out["t_diff_std"] = float(np.mean(diffs)), float(np.std(diffs))
        out["t_shift_mean"], out["t_shift_std"] = float(np.mean(shifts)), float(np.std(shifts))
        out["t_size_std"] = float(np.std(widths) / (np.mean(widths) + 1e-6))
    else:
        for k in ("t_diff_mean", "t_diff_std", "t_shift_mean", "t_shift_std", "t_size_std"):
            out[k] = 0.0

    if antispoof is not None:
        probs = [antispoof.real_prob(img, fc) for img, fc in zip(frames, faces)]
        out["antispoof_mean"], out["antispoof_min"] = float(np.mean(probs)), float(np.min(probs))

    # TASK 6: High-Glare Auto-Compensation (> 210 EV / outdoor sunlight washout)
    face_brightness = [cv2.cvtColor(crop_box(img, fc.box), cv2.COLOR_BGR2GRAY).mean() for img, fc in zip(frames, faces)]
    mean_luminance = float(np.mean(face_brightness))
    out["mean_luminance"] = mean_luminance
    out["glare_compensation_active"] = float(mean_luminance > 210.0)
    return out
