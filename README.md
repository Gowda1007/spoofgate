# SpoofGate: Hybrid Presentation Attack Detection & Biometric Access Control

High-assurance, cost-optimized biometric access control system combining **AWS Rekognition (Face Liveness & CompareFaces)** with a **zero-cost local CPU edge filter** (YuNet & SFace cancelable biometrics).

Features:
- **Hybrid Cloud + Edge Architecture:** Edge pre-filtering rejects low-quality frames, missing faces, multi-face presentations, and static 2D planar photos on the local CPU at **$0.00 cost**, saving ~$0.015 - $0.020 per call before hitting AWS Rekognition.
- **Centralized Risk-Based Decision Engine:** Strict hard gates: Liveness Gate $\rightarrow$ Identity Gate $\rightarrow$ Context/Schedule Policy. A spoofed photo or video replay is **never** approved, regardless of face similarity.
- **Cancelable Biometric Cryptography:** $Q_k$ random orthogonal projection matrices store zero raw biometric embeddings at rest.
- **PAD Metric Benchmarks:** Evaluates Attack Presentation Classification Error Rate (APCER) and Bona Fide Presentation Classification Error Rate (BPCER) following ISO/IEC 30107-3 methodology.

---

## ⚡ Quick Start

### 1. Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download Open-Source Vision Models
Downloads YuNet (Face Detection) and SFace (128-d Biometric Embedder) from OpenCV Zoo:
```bash
python download_models.py
```

### 3. Run the FastAPI Server
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```
API Documentation & Swagger UI: `http://localhost:8000/docs`

---

## 🏛️ Architecture & Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/verify` | `POST` | Core multi-signal burst verification (< 150ms). Decides `ALLOW`, `STEP_UP`, or `DENY`. |
| `/api/verify-step` | `POST` | Secondary challenge handler (Action head-turn or Supervisor OTP). |
| `/api/aws/liveness/session` | `POST` | Cost-optimized AWS Face Liveness session creation with local edge pre-filter. |
| `/api/aws/liveness/verify` | `POST` | Ingests AWS Face Liveness results, runs CompareFaces, and evaluates Decision Engine. |
| `/api/enroll` | `POST` | Extracts 128-d vector and generates a cancelable orthogonal template ($Q_k$). |
| `/api/offline-sync` | `POST` | Ingests cached offline audit records (zero raw biometrics stored). |
| `/api/health` | `GET` | Health status and operational mode. |
| `/api/admin/diagnostics` | `GET` | Authenticated system telemetry and enrolled identities. |
| `/api/stats` | `GET` | Empirical confusion matrix & latency telemetry. |
| `/api/users/{id}` | `DELETE` | Permanently revokes a cancelable template and key. |

---

## 🔬 Training & Evaluation Loop

```bash
# 1. Collect bursts with your webcam
python tools.py collect --person <name> --scenario enroll
python tools.py collect --person <name> --scenario genuine --round train
python tools.py collect --person <name> --scenario screen_photo --round train

# 2. Enroll & Train
python tools.py enroll
python tools.py train --model lr

# 3. Evaluate on unseen test round
python tools.py evaluate
```
