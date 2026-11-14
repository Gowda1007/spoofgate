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

## 🔒 Security Group Configuration (Crucial!)
In the AWS EC2 Management Console:
1. Go to **Security Groups** attached to your instance.
2. Under **Inbound Rules**, click **Edit inbound rules**.
3. Add a rule:
   - **Type:** Custom TCP
   - **Port Range:** `8000`
   - **Source:** `0.0.0.0/0` (Anywhere)
4. (Optional) Allow port `22` (SSH) and `443` (HTTPS).

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
