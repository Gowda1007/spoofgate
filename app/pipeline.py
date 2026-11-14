import base64
import json
import secrets
import threading
import time
import uuid
from collections import Counter, defaultdict

import cv2
import numpy as np

from . import config
from .detect import FaceDetector, prep_frame, quality, yaw_signed
from .features import AntiSpoof, burst_features
from .identity import Embedder, IdentityStore
from .liveness import LivenessModel
from .decision_engine import DecisionEngine
from .aws_service import AWSRekognitionService


class ModelMissing(RuntimeError):
    pass


def ms(t0):
    return round((time.perf_counter() - t0) * 1000.0, 1)


def sigmoid(x):
    return float(1.0 / (1.0 + np.exp(-x)))


def decode_frame(s):
    raw = base64.b64decode(s.split(",")[-1])
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode image")
    return prep_frame(img)


class Gate:
    def __init__(self):
        self.det = FaceDetector()
        self.emb = Embedder()
        self.store = IdentityStore()
        self.live = LivenessModel()
        self.decision_engine = DecisionEngine()
        self.aws = AWSRekognitionService()
        self.anti = None
        if config.ANTISPOOF_PATH.exists():
            try:
                self.anti = AntiSpoof(config.ANTISPOOF_PATH)
            except Exception as e:  # noqa
                print("anti-spoof ONNX failed to load, continuing without it:", e)
        self.sessions = {}
        self.fails = defaultdict(list)
        self.locked = {}
        self.lock = threading.Lock()
        config.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- status
    def status(self, include_sensitive: bool = False):
        self.store.reload()
        base = {
            "status": "healthy",
            "detector": self.det.name,
            "identity_model": self.emb.ready,
            "antispoof_onnx": self.anti is not None,
            "liveness_trained": self.live.trained,
            "aws_rekognition_available": self.aws.available,
            "verification_mode": config.VERIFICATION_MODE,
            "cost_prefilter_active": config.ENABLE_LOCAL_PREFILTER,
            "t_deny": self.live.t_deny,
            "t_allow": self.live.t_allow,
            "id_match": config.ID_MATCH
        }
        if include_sensitive:
            base["enrolled_users"] = sorted(self.store.tpl)
        return base

    # ------------------------------------------------------------- analysis
    def analyze(self, frames, need_identity=True):
        """Detect -> quality gate -> liveness score -> (identity embedding)."""
        t = {}
        t0 = time.perf_counter()
        faces = [self.det.detect(f) for f in frames]
        t["detect"] = ms(t0)
        res = {"timings": t}
        pairs = [(f, fc) for f, fc in zip(frames, faces) if fc is not None]
        if not pairs:
            res.update(ok=False, reasons=["no_face"])
            return res

        yaws = [yaw_signed(fc) for _, fc in pairs if fc.row is not None]
        res["yaw_range"] = float(max(yaws) - min(yaws)) if len(yaws) > 1 else 0.0
        cx = [fc.box[0] + fc.box[2] / 2 for _, fc in pairs]
        wmean = float(np.mean([fc.box[2] for _, fc in pairs]))
        res["shift"] = float((max(cx) - min(cx)) / wmean) if len(cx) > 1 else 0.0

        quals = [quality(f, fc) for f, fc in pairs]
        good = [(p, q) for p, q in zip(pairs, quals) if q["ok"]]
        if len(good) < min(config.MIN_GOOD_FRAMES, len(frames)):
            reasons = [r for r, _ in Counter(r for q in quals for r in q["reasons"]).most_common(2)]
            res.update(ok=False, reasons=reasons or ["low_quality"], quality=quals[0])
            return res

        t1 = time.perf_counter()
        gframes = [p[0] for p, _ in good]
        gfaces = [p[1] for p, _ in good]
        res["feats"] = burst_features(gframes, gfaces, self.anti)
        res["p_live"] = self.live.score(res["feats"])
        t["liveness"] = ms(t1)

        if need_identity:
            if not self.emb.ready:
                raise ModelMissing("SFace model missing: run `python download_models.py`")
            t2 = time.perf_counter()
            best = sorted(good, key=lambda g: -g[1]["sharpness"])[:3]
            e = np.mean([self.emb.embed(p[0], p[1]) for p, _ in best], axis=0)
            res["emb"] = e / (np.linalg.norm(e) + 1e-9)
            t["identity"] = ms(t2)
        res.update(ok=True, quality=good[0][1])
        return res

    # --------------------------------------------------------------- enroll
    def enroll_images(self, user_id, images):
        if not self.emb.ready:
            raise ModelMissing("SFace model missing: run `python download_models.py`")
        cands = []
        for img in images:
            fc = self.det.detect(img)
            if fc is None:
                continue
            q = quality(img, fc)
            if not q["ok"]:
                continue
            # Security Rule: Verify Liveness during enrollment (no enrolling printed photos or screens!)
            if self.live.trained:
                feats = burst_features([img], [fc], self.anti)
                p_live = self.live.score(feats)
                if p_live < self.live.t_deny:
                    continue  # Reject spoofed enrollment frames
            cands.append((q["sharpness"], img, fc))
        if not cands:
            raise ValueError("no usable live face found for enrollment (check lighting, or spoof attempt rejected)")
        cands.sort(key=lambda c: -c[0])
        embs = [self.emb.embed(img, fc) for _, img, fc in cands[:8]]
        self.store.enroll(user_id, embs)
        return len(embs)

    def enroll(self, user_id, frames_b64):
        return self.enroll_images(user_id, [decode_frame(s) for s in frames_b64])

    # --------------------------------------------------------------- verify
    def _resp(self, decision, reason, conf, next_step=None, p_live=None, ident=None, **extra):
        r = {"decision": decision, "reason_code": reason, "confidence": round(float(conf), 2),
             "liveness_score": None if p_live is None else round(float(p_live), 3),
             "identity": ident or {"matched": False, "user_id": None, "similarity": None},
             "next_step": next_step, "explanation": None}
        r.update(extra)
        return r

    def verify(self, frames_b64, **kw):
        t0 = time.perf_counter()
        frames = [decode_frame(s) for s in frames_b64]
        return self.verify_images(frames, decode_ms=ms(t0), **kw)

    def verify_images(self, frames, claimed_id=None, device_id="device", session_id=None,
                      scenario_label=None, capture_ms=None, decode_ms=0.0, log=True):
        t_start = time.perf_counter()
        sid = session_id or uuid.uuid4().hex[:12]
        with self.lock:
            # TTL Session Eviction: Prevent Memory Leaks across long-running kiosks
            if len(self.sessions) > 500:
                oldest_keys = list(self.sessions.keys())[:100]
                for k in oldest_keys:
                    self.sessions.pop(k, None)
            sess = self.sessions.setdefault(sid, {"attempts": 0, "otp": None, "otp_fails": 0,
                                                  "claimed_id": claimed_id, "created_at": time.time()})
            sess["attempts"] += 1
        timings = {"decode": decode_ms}
        p_live, sim, uid = None, None, None

        if self._locked(device_id):
            r = self._resp("DENY", "LOCKED_OUT", 0.99)
        else:
            a = self.analyze(frames)
            timings.update(a["timings"])
            if not a["ok"]:
                if sess["attempts"] >= config.MAX_RETRIES:
                    r = self._resp("DENY", "TOO_MANY_RETRIES", 0.9, reasons=a["reasons"])
                else:
                    r = self._resp("REQUEST_VERIFICATION", "LOW_QUALITY", 0.0, "RECAPTURE",
                                   reasons=a["reasons"], quality=a.get("quality"))
            else:
                p_live = a["p_live"]
                uid, sim = self.store.identify(a["emb"], claimed_id)
                id_score = sigmoid((sim - config.ID_MATCH) * 12) if uid else 0.0
                ident = {"matched": bool(uid and sim >= config.ID_MATCH), "user_id": uid,
                         "similarity": round(sim, 3)}
                if p_live < self.live.t_deny:
                    r = self._resp("DENY", "SPOOF_DETECTED", 1 - p_live, p_live=p_live, ident=ident)
                elif p_live >= self.live.t_allow:
                    if ident["matched"]:
                        r = self._resp("ALLOW", "LIVE_MATCH", min(p_live, id_score), p_live=p_live, ident=ident)
                    elif uid is None or sim < config.ID_UNSURE:
                        r = self._resp("DENY", "NO_MATCH", 1 - id_score, p_live=p_live, ident=ident)
                    else:
                        sess["state"] = "AWAITING_STEP_UP"
                        sess["created_at"] = time.time()
                        sess["user_id"] = uid
                        sess["similarity"] = sim
                        r = self._otp_response(sess, sid, p_live, ident, id_score)
                else:
                    sess["state"] = "AWAITING_STEP_UP"
                    sess["created_at"] = time.time()
                    sess["user_id"] = uid
                    sess["similarity"] = sim
                    r = self._resp("REQUEST_VERIFICATION", "LOW_CONFIDENCE", min(p_live, id_score), "ACTION",
                                   p_live=p_live, ident=ident)
                r["quality"] = a.get("quality")
            if r["decision"] == "DENY" and r["reason_code"] in ("SPOOF_DETECTED", "NO_MATCH"):
                self._fail(device_id)

        timings["total_server"] = round(ms(t_start) + decode_ms, 1)
        r.update(session_id=sid, scenario_label=scenario_label, timings_ms=timings,
                 model={"detector": self.det.name, "liveness_trained": self.live.trained})
        if log:
            self._log({"ts": time.time(), "device": device_id, "session_id": sid,
                       "scenario_label": scenario_label, "decision": r["decision"],
                       "reason": r["reason_code"], "p_live": r["liveness_score"],
                       "sim": r["identity"]["similarity"], "user": r["identity"]["user_id"],
                       "total_ms": timings["total_server"], "capture_ms": capture_ms})
        return r

    def _otp_response(self, sess, sid, p_live, ident, id_score):
        otp = f"{secrets.randbelow(10 ** 6):06d}"
        sess["otp"], sess["otp_fails"] = otp, 0
        sess["state"] = "AWAITING_STEP_UP"
        sess["created_at"] = time.time()
        # Security hardening: Raw OTP is never written to public stdout or production logs
        return self._resp("REQUEST_VERIFICATION", "LOW_CONFIDENCE", min(p_live, id_score), "OTP",
                          p_live=p_live, ident=ident)

    # ------------------------------------------------------------ step-up
    def verify_step(self, session_id, step, otp=None, frames_b64=None, device_id="device"):
        t_start = time.perf_counter()
        now = time.time()
        with self.lock:
            sess = self.sessions.get(session_id)
            if not sess:
                return self._resp("DENY", "SESSION_EXPIRED_OR_INVALID", 1.0, session_id=session_id)
            
            # TASK 2: Enforce 60-second TTL and strictly bind to AWAITING_STEP_UP state
            if sess.get("state") != "AWAITING_STEP_UP" or (now - sess.get("created_at", now) > 60.0):
                self.sessions.pop(session_id, None)
                return self._resp("DENY", "SESSION_EXPIRED_OR_INVALID", 1.0, session_id=session_id)
            
            sess["attempts"] = sess.get("attempts", 0) + 1

        timings = {}
        if step == "OTP":
            if not sess.get("otp"):
                with self.lock:
                    self.sessions.pop(session_id, None)
                return self._resp("DENY", "SESSION_EXPIRED_OR_INVALID", 1.0, session_id=session_id)
            
            if otp == sess["otp"]:
                uid = sess.get("user_id")
                stored_sim = sess.get("similarity", 1.0)
                ident = {"matched": bool(uid), "user_id": uid, "similarity": stored_sim} if uid else None
                with self.lock:
                    sess["state"] = "COMPLETED"
                    sess["otp"] = None
                r = self._resp("ALLOW", "STEP_UP_VERIFIED", 0.9, ident=ident)
            else:
                if sess["attempts"] >= 3:
                    with self.lock:
                        self.sessions.pop(session_id, None)
                    r = self._resp("DENY", "STEP_UP_MAX_ATTEMPTS_EXCEEDED", 1.0)
                    self._fail(device_id)
                else:
                    r = self._resp("REQUEST_VERIFICATION", "OTP_WRONG", 0.0, "OTP")
        elif step == "ACTION":
            if not frames_b64:
                return self._resp("REQUEST_VERIFICATION", "LOW_QUALITY", 0.0, "ACTION")
            frames = [decode_frame(s) for s in frames_b64]
            a = self.analyze(frames)
            timings.update(a["timings"])
            if not a["ok"]:
                r = self._resp("REQUEST_VERIFICATION", "LOW_QUALITY", 0.0, "ACTION", reasons=a["reasons"])
            else:
                uid, sim = self.store.identify(a["emb"], sess.get("claimed_id"))
                action_ok = a["yaw_range"] >= 0.25 or a["shift"] >= 0.15
                mid = (self.live.t_deny + self.live.t_allow) / 2
                ident = {"matched": bool(uid and sim >= config.ID_MATCH), "user_id": uid,
                         "similarity": round(sim, 3)}
                if action_ok and a["p_live"] >= mid and uid and sim >= config.ID_UNSURE:
                    with self.lock:
                        sess["state"] = "COMPLETED"
                    r = self._resp("ALLOW", "STEP_UP_VERIFIED", a["p_live"], p_live=a["p_live"], ident=ident)
                else:
                    with self.lock:
                        self.sessions.pop(session_id, None)
                    r = self._resp("DENY", "STEP_UP_MAX_ATTEMPTS_EXCEEDED", 1 - a["p_live"], p_live=a["p_live"], ident=ident)
                    self._fail(device_id)
        else:
            raise ValueError("step must be OTP or ACTION")
        timings["total_server"] = ms(t_start)
        r.update(session_id=session_id, timings_ms=timings, scenario_label=None)
        return r

    def otp_for(self, session_id):
        s = self.sessions.get(session_id)
        return s.get("otp") if s else None

    # ------------------------------------------------------------- lockout
    def _locked(self, dev):
        return self.locked.get(dev, 0) > time.time()

    def _fail(self, dev):
        now = time.time()
        lst = [x for x in self.fails[dev] if now - x <= config.LOCK_WINDOW_S] + [now]
        self.fails[dev] = lst
        if len(lst) >= config.LOCK_FAILS:
            self.locked[dev] = now + config.LOCK_SECONDS
            self.fails[dev] = []

    def _log(self, rec):
        with self.lock:
            with open(config.LOG_FILE, "a") as f:
                f.write(json.dumps(rec) + "\n")
