# OpenAMD Advanced — Hybrid AI AMD Server

Installs the **OpenAMD Advanced AI AMD server + web portal** on a new Ubuntu 24.04 machine.

**Engine:** Hybrid — OpenAMD heuristic acoustic rules **+** Silero VAD (ONNX Runtime).

For ViciBox / VICIdial extension **8399**, use:  
https://github.com/xceedconnections/vicidialaiamd

## One-command install (new Ubuntu 24.04 server)

Run as **root**:

```bash
curl -fsSL https://raw.githubusercontent.com/xceedconnections/aiamdadvanced/main/remote-install.sh | bash
```

That downloads this repo and runs the full installer.

### Alternative (clone first)

```bash
git clone https://github.com/xceedconnections/aiamdadvanced.git /root/aiamdadvanced
bash /root/aiamdadvanced/install.sh
```

## After install

Open:

```text
http://YOUR_SERVER_IP/
```

Default login:

```text
Username: admin
Password: Openaccount@123
```

Portal AMD engine label:

```text
OpenAMD Hybrid (Heuristic + Silero)  v4.0.0
```

## What gets installed

| Phase | Script | Purpose |
|-------|--------|---------|
| 1 | `install1.sh` | Ubuntu packages, Redis, PostgreSQL, `/opt/openamd` |
| 2 | `install2.sh` | PostgreSQL database `openamd` + user |
| 3 | `install3.sh` | Python venv + pip requirements + Silero ONNX model |
| 4 | `install4.sh` | Deploy backend + portal to `/opt/openamd` |
| 5 | `install5.sh` | systemd `openamd` + daily recording cleanup timer |
| 6 | `install6.sh` | Nginx reverse proxy (port 80 → 8000) |
| 7 | `install7.sh` | Health check + verification |

## Hybrid engine notes

- Heuristic detects VM beeps, SIT tones, and long scripted greetings
- Silero VAD scores speech continuity / segments on 8 kHz telephony audio
- Fusion prefers HUMAN when uncertain (agent connect rate)
- If Silero weights cannot download, heuristic-only mode still serves calls

Model path: `/opt/openamd/models/silero_vad.onnx`

## Portal highlights

- Math captcha on login
- Live Calls filters (VICIdial / Human / Machine / ALL)
- CDR search (called number or caller ID)
- Cron Job page for recording retention (auto-delete)
- Minimum HUMAN confidence gate

## Paths

```text
/opt/openamd/backend/     FastAPI app + portal
/opt/openamd/models/      Silero VAD ONNX
/opt/openamd/recordings/  WAV files from VICIdial
/opt/openamd/logs/
/opt/openamd/venv/
```

## Service commands

```bash
systemctl status openamd
systemctl restart openamd
journalctl -u openamd -f
curl http://127.0.0.1:8000/api/health
```

## Connect a ViciBox dialer

1. Portal → **VICIdial Servers** → add server → **Generate API Key**
2. On the dialer:

```bash
curl -fsSL https://raw.githubusercontent.com/xceedconnections/vicidialaiamd/main/remote-install.sh | bash -s -- YOUR_AI_AMD_IP oam_YOUR_API_KEY
```

## Private repo note

If this repository is private, either:

- make it public for the one-liner above, or
- on the server, configure git credentials / a deploy key, then run the clone commands instead of `curl | bash`.

## Re-run / upgrade

```bash
rm -rf /root/aiamdadvanced
git clone https://github.com/xceedconnections/aiamdadvanced.git /root/aiamdadvanced
bash /root/aiamdadvanced/install4.sh
bash /root/aiamdadvanced/install3.sh
systemctl restart openamd
```

Or run full `install.sh` again (safe on an existing server).
