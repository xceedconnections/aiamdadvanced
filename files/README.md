# OpenAMD Advanced — Hybrid AI AMD Server (Heuristic + Silero)

Installs the **OpenAMD Advanced AI AMD server + web portal** on Ubuntu 24.04.

**Engine:** Hybrid — OpenAMD heuristic acoustic rules **+** Silero VAD (ONNX).

For ViciBox / VICIdial dialer setup, use:  
https://github.com/xceedconnections/vicidialaiamd

GitHub installer repo: https://github.com/xceedconnections/aiamdadvanced

## One-command install (new Ubuntu 24.04 server)

Run as **root**:

```bash
curl -fsSL https://raw.githubusercontent.com/xceedconnections/aiamdadvanced/main/remote-install.sh | bash
```

### Alternative (clone first)

```bash
git clone https://github.com/xceedconnections/aiamdadvanced.git /root/aiamdadvanced
bash /root/aiamdadvanced/install.sh
```

## What gets installed

| Phase | Script | Purpose |
|-------|--------|---------|
| 1 | `install1.sh` | Ubuntu packages, Redis, PostgreSQL, `/opt/openamd` dirs |
| 2 | `install2.sh` | PostgreSQL database `openamd` + user |
| 3 | `install3.sh` | Python venv + pip requirements + Silero ONNX model |
| 4 | `install4.sh` | Deploy backend + portal to `/opt/openamd` |
| 5 | `install5.sh` | systemd `openamd` + daily recording cleanup timer |
| 6 | `install6.sh` | Nginx reverse proxy (port 80 → 8000) |
| 7 | `install7.sh` | Health check + final verification |

## AMD engine

- **Heuristic:** beep / SIT / energy-burst rules (outbound VICIdial tuned)
- **Silero VAD:** ONNX speech structure (speech ratio, segments, longest speech)
- **Fusion:** strong heuristic signals win; Silero refines uncertain HUMAN/MACHINE calls
- **Fallback:** if Silero model is unavailable, heuristic-only continues to work

Portal shows: `OpenAMD Hybrid (Heuristic + Silero)` / engine `4.0.0`

## Portal features

- Login captcha (math challenge)
- Live Calls filters (VICIdial server, Human / Machine / ALL)
- CDR page with search by called number or caller ID
- Cron Job page — set recording retention days (auto-delete daily ~02:15)
- Minimum HUMAN confidence gate

## Open portal

```text
http://NEW_SERVER_IP/
Username: admin
Password: Openaccount@123
```

## After install — connect ViciBox

1. Portal → **VICIdial Servers** → add server → **Generate API Key**
2. On the dialer:

```bash
curl -fsSL https://raw.githubusercontent.com/xceedconnections/vicidialaiamd/main/remote-install.sh | bash -s -- YOUR_AI_AMD_IP oam_YOUR_API_KEY
```

## Paths

```text
/opt/openamd/backend/     FastAPI app + portal
/opt/openamd/models/      Silero VAD ONNX weights
/opt/openamd/recordings/  WAV files from VICIdial
/opt/openamd/logs/
/opt/openamd/venv/        Python virtualenv
```

## Service commands

```bash
systemctl status openamd
systemctl restart openamd
journalctl -u openamd -f
curl http://127.0.0.1:8000/api/health
```

## Re-run / upgrade

```bash
rm -rf /root/aiamdadvanced
git clone https://github.com/xceedconnections/aiamdadvanced.git /root/aiamdadvanced
bash /root/aiamdadvanced/install4.sh
bash /root/aiamdadvanced/install3.sh
systemctl restart openamd
```

Or run full `install.sh` again (safe on an existing server).

## Local pack / scp (optional)

```powershell
cd c:\xampp\htdocs\AIAMD
powershell -ExecutionPolicy Bypass -File install\pack.ps1
scp openamd-install.tar.gz root@NEW_SERVER_IP:/root/
```

On server:

```bash
tar -xzf /root/openamd-install.tar.gz -C /root/
bash /root/install/install.sh
```
