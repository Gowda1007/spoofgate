"""Centralized Risk-Based Decision Engine for SpoofGate.

Enforces hard security gates:
  1. Quality Gate (reject zero faces, multiple faces, blur, extremes)
  2. Presentation Attack / Liveness Gate (AWS Rekognition Face Liveness / Local Hybrid)
  3. Biometric Identity Matching Gate (AWS CompareFaces / Local SFace cancelable template)
  4. Context-Aware Policy Gate (Campus access schedules, zones, account status)
  5. State-Preserving Step-Up Authorization (OTP/Action cannot bypass biometric identity)
"""

from typing import Dict, Any, Optional
from . import config


class DecisionEngine:
    def __init__(self, policy_engine=None):
        from .policy import ContextPolicyEngine
        self.policy = policy_engine or ContextPolicyEngine()

    def evaluate(
        self,
        quality_ok: bool,
        quality_reasons: list,
        liveness_score: Optional[float],
        liveness_pass: bool,
        liveness_provider: str,
        identity_match: bool,
        user_id: Optional[str],
        similarity: Optional[float],
        id_score: float,
        zone: str = "campus_gate",
        time_str: Optional[str] = None,
        is_step_up: bool = False,
        step_up_type: Optional[str] = None,
        step_up_success: bool = False,
        quality_meta: Optional[dict] = None
    ) -> Dict[str, Any]:
        """
        Unified authorization logic across all modalities.
        Produces consistent response schema: decision, reason_code, confidence, liveness_score, identity, next_step, context_eval.
        """
        ident = {
            "matched": bool(identity_match and user_id),
            "user_id": user_id,
            "similarity": round(similarity, 3) if similarity is not None else None
        }

        # ------------------------------------------------------------- 1. Quality Gate
        if not quality_ok:
            return {
                "decision": "REQUEST_VERIFICATION",
                "reason_code": "LOW_QUALITY",
                "confidence": 0.0,
                "liveness_score": liveness_score,
                "liveness_provider": liveness_provider,
                "identity": ident,
                "next_step": "RECAPTURE",
                "reasons": quality_reasons or ["low_quality"],
                "quality": quality_meta
            }

        # ------------------------------------------------------------- 2. Liveness Gate (HARD GATE)
        # Rule: A photo/video spoof MUST NEVER be approved, even if face match is 100%.
        if not liveness_pass:
            conf = 1.0 - (liveness_score if liveness_score is not None else 0.0)
            return {
                "decision": "DENY",
                "reason_code": "SPOOF_DETECTED",
                "confidence": round(float(conf), 2),
                "liveness_score": liveness_score,
                "liveness_provider": liveness_provider,
                "identity": ident,
                "next_step": None,
                "explanation": "Presentation attack detected. Biometric verification failed liveness check."
            }

        # ------------------------------------------------------------- 3. Identity Gate
        # Rule: For primary verification or step-up, identity must be matched
        if not is_step_up:
            if not ident["matched"]:
                return {
                    "decision": "DENY",
                    "reason_code": "NO_MATCH",
                    "confidence": round(1.0 - id_score, 2),
                    "liveness_score": liveness_score,
                    "liveness_provider": liveness_provider,
                    "identity": ident,
                    "next_step": None,
                    "explanation": "Face not enrolled or identity similarity below threshold."
                }
        else:
            # In a step-up (OTP or Action), verify that the step-up challenge succeeded
            # AND the session still maintains verified identity constraints.
            if not step_up_success:
                return {
                    "decision": "DENY",
                    "reason_code": f"STEP_UP_{step_up_type or 'CHALLENGE'}_FAILED",
                    "confidence": 1.0,
                    "liveness_score": liveness_score,
                    "liveness_provider": liveness_provider,
                    "identity": ident,
                    "next_step": None
                }
            if not ident["matched"] and not user_id:
                return {
                    "decision": "DENY",
                    "reason_code": "UNAUTHORIZED_IDENTITY",
                    "confidence": 1.0,
                    "liveness_score": liveness_score,
                    "liveness_provider": liveness_provider,
                    "identity": ident,
                    "next_step": None
                }

        # ------------------------------------------------------------- 4. Context & Policy Gate
        # Real person + matched identity -> Apply zone and schedule policy
        pol_dec, pol_reason, pol_meta = self.policy.evaluate_access(
            user_id=user_id,
            zone=zone,
            time_str=time_str
        )

        overall_conf = round(min(liveness_score if liveness_score is not None else 1.0, id_score if id_score else 0.95), 2)

        if pol_dec == "ALLOW":
            return {
                "decision": "ALLOW",
                "reason_code": pol_reason or ("STEP_UP_VERIFIED" if is_step_up else "LIVE_MATCH"),
                "confidence": overall_conf,
                "liveness_score": liveness_score,
                "liveness_provider": liveness_provider,
                "identity": ident,
                "next_step": None,
                "context_eval": pol_meta
            }
        elif pol_dec == "REQUEST_VERIFICATION":
            return {
                "decision": "REQUEST_VERIFICATION",
                "reason_code": pol_reason or "AFTER_HOURS_POLICY_TRIGGER",
                "confidence": overall_conf,
                "liveness_score": liveness_score,
                "liveness_provider": liveness_provider,
                "identity": ident,
                "next_step": pol_meta.get("next_step", "OTP"),
                "context_eval": pol_meta
            }
        else:
            return {
                "decision": "DENY",
                "reason_code": pol_reason or "POLICY_DENIED",
                "confidence": 1.0,
                "liveness_score": liveness_score,
                "liveness_provider": liveness_provider,
                "identity": ident,
                "next_step": None,
                "context_eval": pol_meta
            }
