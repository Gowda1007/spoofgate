"""AWS Rekognition Service with Built-in Cost Optimization for SpoofGate.

Features:
  - AWS Rekognition Face Liveness (CreateFaceLivenessSession & GetFaceLivenessSessionResults)
  - AWS Rekognition CompareFaces (Face verification against reference image)
  - Edge Pre-Filtering & Fast-Reject Gate: Saves $0.015-$0.020 per call by rejecting invalid,
    low-quality, or obvious 2D spoof attacks on the local CPU before hitting AWS APIs.
  - Liveness-First Hard Gate: Never invokes CompareFaces ($0.001) if Liveness fails.
  - Zero-Cost Local Mode: Gracefully falls back when AWS credentials or boto3 are not present.
"""

import logging
from typing import Dict, Any, Optional, Tuple
from . import config

logger = logging.getLogger("spoofgate.aws")

try:
    import boto3
    from botocore.exceptions import ClientError, BotoCoreError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    ClientError = Exception
    BotoCoreError = Exception


class AWSRekognitionService:
    def __init__(self, region: str = config.AWS_REGION):
        self.region = region
        self.client = None
        self.available = False
        self._init_client()

    def _init_client(self):
        if not BOTO3_AVAILABLE:
            logger.info("boto3 not installed; running in local zero-cost mode.")
            return
        try:
            # Respects standard AWS credential chain (IAM roles on EC2, env vars, ~/.aws)
            self.client = boto3.client("rekognition", region_name=self.region)
            self.available = True
            logger.info("AWS Rekognition client initialized successfully in region %s", self.region)
        except Exception as e:
            logger.warning("AWS Rekognition client could not be initialized (%s); fallback to local.", e)
            self.available = False

    # ------------------------------------------------------------- Cost Optimization Gates
    def pre_filter_check(self, quality: dict, feats: Optional[dict] = None) -> Tuple[bool, str]:
        """
        Cost Optimization Gate:
        Evaluates whether an attempt should be sent to AWS Rekognition ($0.015/call)
        or rejected locally for $0.00 cost.
        Returns: (should_proceed, reason_if_rejected)
        """
        if not config.ENABLE_LOCAL_PREFILTER:
            return True, "PREFILTER_DISABLED"

        # 1. Quality check on edge: No face or multiple faces detected
        if not quality.get("ok", False):
            reasons = quality.get("reasons", ["low_quality"])
            return False, f"LOCAL_QUALITY_REJECT:{','.join(reasons)}"

        # 2. Obvious flat 2D presentation attack (static frozen image)
        if feats and feats.get("is_frozen_static", 0.0) == 1.0:
            return False, "LOCAL_SPOOF_REJECT:frozen_static_photo"

        return True, "LOCAL_PREFILTER_PASSED"

    # ------------------------------------------------------------- Face Liveness API
    def create_liveness_session(self, settings: Optional[dict] = None) -> Dict[str, Any]:
        """
        Creates an AWS Face Liveness streaming session.
        Returns session ID and parameters required by Amplify FaceLivenessDetector.
        """
        if not self.available or not self.client:
            raise RuntimeError("AWS Rekognition is not available (check boto3 and AWS credentials).")

        req_params = {}
        if settings:
            req_params["Settings"] = settings

        try:
            res = self.client.create_face_liveness_session(**req_params)
            return {
                "session_id": res["SessionId"],
                "status": "CREATED",
                "region": self.region
            }
        except (ClientError, BotoCoreError) as e:
            logger.error("AWS CreateFaceLivenessSession error: %s", e)
            raise RuntimeError(f"AWS Rekognition error: {str(e)}")

    def get_liveness_session_results(self, session_id: str) -> Dict[str, Any]:
        """
        Retrieves the evaluation results of an AWS Face Liveness session.
        Returns confidence, status, and raw reference image bytes (if successful).
        """
        if not self.available or not self.client:
            raise RuntimeError("AWS Rekognition is not available.")

        try:
            res = self.client.get_face_liveness_session_results(SessionId=session_id)
            status = res.get("Status")
            confidence = float(res.get("Confidence", 0.0))
            
            ref_image_bytes = None
            ref_image = res.get("ReferenceImage", {})
            if ref_image and "Bytes" in ref_image:
                ref_image_bytes = ref_image["Bytes"]

            liveness_pass = (status == "SUCCEEDED" and confidence >= config.AWS_LIVENESS_THRESHOLD)

            return {
                "session_id": session_id,
                "status": status,
                "confidence": confidence,
                "liveness_pass": liveness_pass,
                "reference_image_bytes": ref_image_bytes
            }
        except (ClientError, BotoCoreError) as e:
            logger.error("AWS GetFaceLivenessSessionResults error: %s", e)
            raise RuntimeError(f"AWS Rekognition error: {str(e)}")

    # ------------------------------------------------------------- CompareFaces API
    def compare_faces(
        self,
        source_bytes: bytes,
        target_bytes: bytes,
        threshold: float = config.AWS_FACE_MATCH_THRESHOLD
    ) -> Dict[str, Any]:
        """
        Cost Optimization: Only call when Liveness has already PASSED.
        Compares reference face from liveness session against enrolled identity image.
        """
        if not self.available or not self.client:
            raise RuntimeError("AWS Rekognition is not available.")

        try:
            res = self.client.compare_faces(
                SourceImage={"Bytes": source_bytes},
                TargetImage={"Bytes": target_bytes},
                SimilarityThreshold=threshold
            )
            matches = res.get("FaceMatches", [])
            if matches:
                top_match = matches[0]
                sim = float(top_match.get("Similarity", 0.0))
                return {
                    "matched": True,
                    "similarity": sim,
                    "confidence": float(top_match.get("Face", {}).get("Confidence", 100.0))
                }
            return {
                "matched": False,
                "similarity": 0.0,
                "confidence": 0.0
            }
        except (ClientError, BotoCoreError) as e:
            logger.error("AWS CompareFaces error: %s", e)
            raise RuntimeError(f"AWS Rekognition error: {str(e)}")
