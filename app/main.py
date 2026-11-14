from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config
from .metrics import read_log, summarize
from .pipeline import Gate, ModelMissing
from .policy import ContextPolicyEngine

app = FastAPI(title="SpoofGate")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
gate = Gate()
policy_engine = ContextPolicyEngine()


class ContextModel(BaseModel):
    zone: Optional[str] = "campus_gate"
    time_str: Optional[str] = None
    demo_admin_token: Optional[str] = None
    is_offline: Optional[bool] = False


class OfflineSyncBatch(BaseModel):
    events: List[dict]


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


@app.get("/api/health")
def health():
    return gate.status()


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
            if p_dec == "REQUEST_VERIFICATION" and p_meta.get("next_step") == "OTP":
                raw_res["next_step"] = "OTP"
                # Fix: Generate the actual pending OTP on session so /api/verify-step works!
                sid = raw_res.get("session_id")
                if sid in gate.sessions:
                    gate.sessions[sid]["otp"] = f"{secrets.randbelow(10 ** 6):06d}"
                    gate.sessions[sid]["otp_fails"] = 0
                    gate.sessions[sid]["user_id"] = user_id
                    gate.sessions[sid]["zone"] = zone
        return raw_res
    except ModelMissing as e:
        raise HTTPException(503, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/offline-sync")
def offline_sync(batch: OfflineSyncBatch, authorization: Optional[str] = Header(None)):
    """Surprise Challenge Principle 02: Ingest queued offline events with admin/kiosk authorization."""
    _verify_admin(authorization)
    for ev in batch.events:
        gate._log(ev)
    return {"ok": True, "synced_events": len(batch.events)}


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
