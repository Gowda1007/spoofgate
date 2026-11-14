import datetime
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Tuple

POLICY_FILE = Path(__file__).parent.parent / "data" / "policy" / "campus_policy.json"


class ContextPolicyEngine:
    """Evaluates context (Access Zone, Campus Time Schedule, User Role) post-biometrics."""

    def __init__(self):
        self._load()

    def _load(self):
        if POLICY_FILE.exists():
            try:
                self.policy = json.loads(POLICY_FILE.read_text(encoding="utf-8"))
            except Exception:
                self.policy = {"users": {}, "zones": {}}
        else:
            self.policy = {"users": {}, "zones": {}}

    def evaluate_access(self, user_id: str, zone: str = "campus_gate", time_str: str = None) -> Tuple[str, str, Dict[str, Any]]:
        """
        Returns: (Decision: "ALLOW" | "STEP_UP" | "DENY", ReasonCode: str, ContextMetadata: dict)
        """
        self._load()
        users = self.policy.get("users", {})
        zones = self.policy.get("zones", {})

        # Default time if not simulated
        if not time_str:
            time_str = datetime.datetime.now().strftime("%H:%M")

        user_info = users.get(user_id)
        if not user_info:
            return "DENY", "USER_NOT_FOUND_IN_POLICY", {"schedule_ok": False, "zone_ok": False}

        if user_info.get("status") != "ACTIVE":
            return "DENY", "ACCOUNT_REVOKED_OR_SUSPENDED", {"status": user_info.get("status")}

        # 1. Zone Clearance Check
        allowed_zones = user_info.get("allowed_zones", [])
        zone_info = zones.get(zone, {})
        if zone not in allowed_zones:
            return "DENY", "UNAUTHORIZED_ZONE", {
                "user_role": user_info.get("role"),
                "zone_requested": zone,
                "reason": f"Role '{user_info.get('role')}' does not have clearance for '{zone}'"
            }

        if zone_info.get("requires_faculty") and user_info.get("role") != "faculty":
            return "DENY", "RESTRICTED_FACULTY_ZONE", {"zone": zone}

        # 2. Campus Schedule Check
        if zone_info.get("allow_24_7"):
            return "ALLOW", "AUTHORIZED_24_7_CAMPUS_ACCESS", {"zone": zone, "schedule": "24_7"}

        sched = user_info.get("schedule", {"start": "08:00", "end": "18:00"})
        start_h, start_m = map(int, sched.get("start", "08:00").split(":"))
        end_h, end_m = map(int, sched.get("end", "18:00").split(":"))
        cur_h, cur_m = map(int, time_str.split(":"))

        cur_mins = cur_h * 60 + cur_m
        start_mins = start_h * 60 + start_m
        end_mins = end_h * 60 + end_m

        # If inside normal hours -> ALLOW
        if start_mins <= cur_mins <= end_mins:
            return "ALLOW", "SCHEDULE_AUTHORIZED", {
                "window": f"{sched['start']} - {sched['end']}",
                "simulated_time": time_str
            }

        # Outside hours -> STEP_UP VERIFICATION (The core of the surprise challenge!)
        return "STEP_UP", "AFTER_HOURS_POLICY_TRIGGER", {
            "window": f"{sched['start']} - {sched['end']}",
            "simulated_time": time_str,
            "next_step": "SUPERVISOR_OTP"
        }
