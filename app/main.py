import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import secrets
import time
from collections import OrderedDict
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ConfigDict

from . import config
from .metrics import read_log, summarize
from .pipeline import Gate, ModelMissing
from .policy import ContextPolicyEngine

app = FastAPI(title="SpoofGate")
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"]
)
gate = Gate()
policy_engine = ContextPolicyEngine()

# Task 1: Bounded LRU event_id cache for idempotent offline sync (max 5,000)
PROCESSED_EVENTS = OrderedDict()


class ContextModel(BaseModel):
    zone: Optional[str] = "campus_gate"
    time_str: Optional[str] = None
    demo_admin_token: Optional[str] = None


class OfflineEventItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event_id: str
    session_id: Optional[str] = None
    zone: Optional[str] = "campus_gate"
    decision: str
    reason: Optional[str] = None
    timestamp: str


class OfflineSyncBatch(BaseModel):
    events: List[OfflineEventItem]


class EnrollReq(BaseModel):
    user_id: str
    frames: List[str] = Field(..., max_length=10)


class VerifyReq(BaseModel):
    frames: List[str] = Field(..., max_length=10)
    session_id: Optional[str] = None       # new id per person; reuse it only for RECAPTURE / step-up
    scenario_label: Optional[str] = None   # genuine | print | screen_photo | video_replay | mask
    claimed_id: Optional[str] = None       # 1:1 if given, otherwise 1:N search
    device_id: Optional[str] = None
    capture_ms: Optional[float] = None
    context: Optional[ContextModel] = None


class StepReq(BaseModel):
    session_id: str
    step: str                              # OTP | ACTION
    otp: Optional[str] = None
    frames: Optional[List[str]] = Field(None, max_length=10)
    device_id: Optional[str] = None
    zone: Optional[str] = "campus_gate"


def _verify_admin(auth_header: Optional[str]):
    if not auth_header or auth_header != f"Bearer {config.ADMIN_TOKEN}":
        raise HTTPException(401, "Admin authorization token required")


def _device(req_device, request: Request):
    return req_device or (request.client.host if request.client else "unknown")


class AWSLivenessSessionReq(BaseModel):
    claimed_id: Optional[str] = None
    prefilter_frames: Optional[List[str]] = None  # Optional client burst to run zero-cost local filter


class AWSVerifyReq(BaseModel):
    session_id: str
    claimed_id: Optional[str] = None
    zone: Optional[str] = "campus_gate"
    time_str: Optional[str] = None
    demo_admin_token: Optional[str] = None


@app.get("/api/health")
def health():
    return gate.status(include_sensitive=config.DEMO_MODE)


@app.get("/api/admin/diagnostics")
def admin_diagnostics(authorization: Optional[str] = Header(None)):
    _verify_admin(authorization)
    return gate.status(include_sensitive=True)


@app.post("/api/aws/liveness/session")
def aws_create_liveness_session(req: AWSLivenessSessionReq):
    """
    Cost-Optimized Endpoint:
    If prefilter_frames are provided, runs local zero-cost edge verification.
    If the face is blurry, missing, multiple faces, or obvious flat photo, rejects for $0.00!
    Only creates an AWS Face Liveness session ($0.015) when pre-check passes.
    """
    if req.prefilter_frames:
        try:
            frames = [gate.decode_frame(f) if hasattr(gate, 'decode_frame') else __import__('app.pipeline', fromlist=['decode_frame']).decode_frame(f) for f in req.prefilter_frames]
            faces = [gate.det.detect(f) for f in frames]
            pairs = [(f, fc) for f, fc in zip(frames, faces) if fc is not None]
            if not pairs:
                raise HTTPException(400, "Local pre-filter reject: No face detected. Zero-cost fast reject.")
            quals = [__import__('app.detect', fromlist=['quality']).quality(f, fc) for f, fc in pairs]
            should_proceed, reason = gate.aws.pre_filter_check(quals[0])
            if not should_proceed:
                raise HTTPException(400, f"Local pre-filter reject: {reason}. Saved $0.015 AWS fee.")
        except HTTPException:
            raise
        except Exception as e:
            pass  # Fall through to standard creation if prefilter decoding errors

    try:
        session_info = gate.aws.create_liveness_session()
        return session_info
    except RuntimeError as e:
        raise HTTPException(503, str(e))


@app.post("/api/aws/liveness/verify")
def aws_verify_liveness(req: AWSVerifyReq):
    """
    Evaluates AWS Face Liveness session results with Liveness-First Hard Gate.
    If liveness succeeds, extracts reference image, runs identity matching,
    and executes the centralized DecisionEngine.
    """
    try:
        results = gate.aws.get_liveness_session_results(req.session_id)
    except RuntimeError as e:
        raise HTTPException(503, str(e))

    liveness_pass = results.get("liveness_pass", False)
    liveness_conf = results.get("confidence", 0.0) / 100.0  # normalize to 0..1
    ref_bytes = results.get("reference_image_bytes")

    # Hard Gate 1: If liveness fails, reject immediately. Never call CompareFaces ($0.001 saved).
    if not liveness_pass or not ref_bytes:
        return gate.decision_engine.evaluate(
            quality_ok=True,
            quality_reasons=[],
            liveness_score=liveness_conf,
            liveness_pass=False,
            liveness_provider="aws_rekognition",
            identity_match=False,
            user_id=req.claimed_id,
            similarity=0.0,
            id_score=0.0,
            zone=req.zone or "campus_gate"
        )

    # Hard Gate 2: Liveness passed. Now perform Identity verification
    # Try SFace cancelable template matching using decoded reference image
    import cv2
    import numpy as np
    nparr = np.frombuffer(ref_bytes, np.uint8)
    ref_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    sim, matched, uid = 0.0, False, req.claimed_id
    if ref_img is not None:
        fc = gate.det.detect(ref_img)
        if fc is not None and gate.emb.ready:
            emb = gate.emb.embed(ref_img, fc)
            uid, sim = gate.store.identify(emb, req.claimed_id)
            matched = bool(uid and sim >= config.ID_MATCH)

    id_score = float(sim)

    # Evaluate complete security protocol through DecisionEngine
    sim_time = None
    if req.time_str and config.DEMO_MODE and req.demo_admin_token == config.ADMIN_TOKEN:
        sim_time = req.time_str

    final_decision = gate.decision_engine.evaluate(
        quality_ok=True,
        quality_reasons=[],
        liveness_score=liveness_conf,
        liveness_pass=True,
        liveness_provider="aws_rekognition",
        identity_match=matched,
        user_id=uid,
        similarity=sim,
        id_score=id_score,
        zone=req.zone or "campus_gate",
        time_str=sim_time
    )
    final_decision["aws_session_id"] = req.session_id
    return final_decision



@app.post("/api/enroll")
def enroll(req: EnrollReq):
    try:
        return {"ok": True, "user_id": req.user_id, "frames_used": gate.enroll(req.user_id, req.frames)}
    except ModelMissing as e:
        raise HTTPException(503, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/verify")
def verify(req: VerifyReq, request: Request):
    try:
        raw_res = gate.verify(req.frames, claimed_id=req.claimed_id, device_id=_device(req.device_id, request),
                              session_id=req.session_id, scenario_label=req.scenario_label,
                              capture_ms=req.capture_ms)

        # Context-Aware Policy Evaluation (Surprise Challenge Principle 01)
        if raw_res.get("decision") == "ALLOW":
            user_id = raw_res.get("identity", {}).get("user_id")
            zone = req.context.zone if req.context and req.context.zone else "campus_gate"
            
            # Security Rule: Only accept time_str if DEMO_MODE=1 AND valid demo_admin_token is supplied
            sim_time = None
            if req.context and req.context.time_str:
                if config.DEMO_MODE and req.context.demo_admin_token == config.ADMIN_TOKEN:
                    sim_time = req.context.time_str
                # Otherwise ignored; server clock is authoritative

            p_dec, p_reason, p_meta = policy_engine.evaluate_access(user_id, zone=zone, time_str=sim_time)
            raw_res["decision"] = p_dec
            raw_res["reason_code"] = p_reason
            raw_res["context_eval"] = p_meta
            
            # TASK 2: Hardened Step-Up State Machine initialization
            sid = raw_res.get("session_id")
            if p_dec == "REQUEST_VERIFICATION":
                raw_res["next_step"] = p_meta.get("next_step", "OTP")
                with gate.lock:
                    if sid not in gate.sessions:
                        gate.sessions[sid] = {}
                    gate.sessions[sid].update({
                        "state": "AWAITING_STEP_UP",
                        "user_id": user_id,
                        "zone": zone,
                        "created_at": time.time(),
                        "attempts": 0,
                        "otp": f"{secrets.randbelow(10 ** 6):06d}"
                    })
        return raw_res
    except ModelMissing as e:
        raise HTTPException(503, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/offline-sync")
def offline_sync(batch: OfflineSyncBatch, authorization: Optional[str] = Header(None)):
    """TASK 1: Strict Zero-Biometric Schema & Idempotent Offline Sync."""
    _verify_admin(authorization)
    # Sort incoming events chronologically by timestamp
    sorted_events = sorted(batch.events, key=lambda x: x.timestamp)
    new_count = 0
    dup_count = 0

    for item in sorted_events:
        ev_id = item.event_id
        if ev_id in PROCESSED_EVENTS:
            dup_count += 1
            continue
        
        # Bounded LRU (cap at 5,000 processed IDs)
        if len(PROCESSED_EVENTS) >= 5000:
            PROCESSED_EVENTS.popitem(last=False)
        PROCESSED_EVENTS[ev_id] = True

        gate._log(item.model_dump())
        new_count += 1

    return {
        "status": "synced",
        "synced_count": new_count,
        "duplicates_skipped": dup_count,
        "zero_biometrics_verified": True
    }


@app.post("/api/verify-step")
def verify_step(req: StepReq, request: Request):
    try:
        step_res = gate.verify_step(req.session_id, req.step.upper(), req.otp, req.frames,
                                    _device(req.device_id, request))
        
        # Security Fix: Run the policy engine on step-up success! No bypassing the schedule via head-turn/OTP!
        if step_res.get("decision") == "ALLOW":
            sess = gate.sessions.get(req.session_id, {})
            user_id = step_res.get("identity", {}).get("user_id") or sess.get("user_id")
            zone = req.zone or sess.get("zone", "campus_gate")
            p_dec, p_reason, p_meta = policy_engine.evaluate_access(user_id, zone=zone)
            if p_dec != "ALLOW":
                step_res["decision"] = p_dec
                step_res["reason_code"] = p_reason
                step_res["context_eval"] = p_meta
        return step_res
    except KeyError:
        raise HTTPException(404, "unknown session_id")
    except ModelMissing as e:
        raise HTTPException(503, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/otp/{session_id}")
def otp(session_id: str, authorization: Optional[str] = Header(None)):
    """Demo only: guarded by admin token to prevent unauthorized PIN scraping."""
    _verify_admin(authorization)
    return {"otp": gate.otp_for(session_id)}


@app.delete("/api/users/{user_id}")
def revoke(user_id: str, authorization: Optional[str] = Header(None)):
    _verify_admin(authorization)
    gate.store.revoke(user_id)
    return {"ok": True, "note": "template and key deleted; enroll again to issue a new cancelable template"}


@app.get("/api/stats")
def stats():
    return {"rows": summarize(read_log()), "status": gate.status()}


if (config.ROOT / "frontend").exists():
    app.mount("/", StaticFiles(directory=str(config.ROOT / "frontend"), html=True), name="frontend")
