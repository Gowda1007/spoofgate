# SpoofGate Backend: CPU-Only Liveness & Biometric Access Control

Open-source, CPU-only presentation attack detection (PAD) and biometric identity verification system for unattended HID entry kiosks.

Compliant with **evaluated using ISO/IEC 30107-3 metrics (APCER/BPCER)**, integrating **Context-Aware Schedule Policies** and an **Offline Fail-Safe Gate** (never fails open, zero raw biometrics stored).

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
| `/api/enroll` | `POST` | Extracts 128-d vector and generates a cancelable orthogonal template ($Q_k$). |
| `/api/offline-sync` | `POST` | Ingests cached offline audit records (zero raw biometrics stored). |
| `/api/health` | `GET` | Health status, loaded models, and enrolled user IDs. |
| `/api/stats` | `GET` | Empirical Red-Team confusion matrix & latency telemetry. |
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
