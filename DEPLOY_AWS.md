# 🚀 AWS Sandbox Deployment Runbook for Teammate 1 (Role A)

> **Target:** AWS Sandbox (t3.medium / t3.large or c6i.large EC2 Instance, Ubuntu 22.04 / 24.04 LTS)  
> **Pre-requisites:** Access code provided at `tinyurl.com/hid-hackathon`

---

## 📋 Option 1: Automated 2-Minute Setup on EC2 (Recommended)

When launching the EC2 instance in the AWS Console, paste this into the **User Data** field, or SSH in and run:

```bash
#!/bin/bash
# 1. Update system & install dependencies
sudo apt-get update -y
sudo apt-get install -y python3-pip python3-venv git libgl1 libglib2.0-0 curl

# 2. Clone the repository
git clone https://github.com/Gowda1007/spoofgate.git
cd spoofgate

# 3. Setup Virtual Environment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 4. Download Open-Source Vision Models (YuNet + SFace)
python3 download_models.py

# 5. Start SpoofGate in background on port 8000
nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 > app.log 2>&1 &
echo "SpoofGate running successfully on port 8000!"
```

---

## 🐳 Option 2: Docker Deployment

If Docker is available on the sandbox EC2:

```bash
git clone https://github.com/Gowda1007/spoofgate.git
cd spoofgate
docker build -t spoofgate-backend .
docker run -d -p 8000:8000 --name spoofgate spoofgate-backend
```

---

## 🔒 Security & Cost-Optimization Hardening

### 1. IAM Instance Profile (No hardcoded credentials)
Attach an IAM Role to your EC2 instance with the following permissions instead of storing static access keys:
- `rekognition:CreateFaceLivenessSession`
- `rekognition:GetFaceLivenessSessionResults`
- `rekognition:CompareFaces`

### 2. Built-in Cost Optimization
SpoofGate includes an edge pre-filter gate (`ENABLE_LOCAL_PREFILTER=1`).
- **Cost Savings:** AWS Face Liveness costs ~$0.015 - $0.020 per session.
- Low-quality frames, missing faces, multi-face frames, and static 2D planar photos are rejected locally on the CPU at **$0.00 cost** before making any AWS API calls.
- `CompareFaces` ($0.001) is strictly guarded behind a **Liveness-First hard gate** and is never called if liveness fails.

### 3. Production HTTPS Reverse Proxy (Nginx)
Do NOT expose port 8000 directly to `0.0.0.0/0` in production. Run behind an Nginx reverse proxy with HTTPS (Let's Encrypt / AWS Certificate Manager):
```nginx
server {
    listen 443 ssl http2;
    server_name gate.yourcampus.edu;

    ssl_certificate /etc/letsencrypt/live/gate.yourcampus.edu/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/gate.yourcampus.edu/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

---

## 🌐 Connecting Kiosks to EC2 with HTTPS

> **Important:** Web browsers strictly block webcam streams over raw HTTP (e.g., `http://ec2-xx-xx-xx-xx.compute.amazonaws.com:8000`).

To enable HTTPS in 10 seconds on the EC2 instance:
```bash
# Install and run cloudflared tunnel
curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared.deb
cloudflared tunnel --url http://localhost:8000
```
It will output an official public link:
`https://your-tunnel-name.trycloudflare.com`

Share that HTTPS link with judges and teammates—the webcam and all API endpoints work instantly!
