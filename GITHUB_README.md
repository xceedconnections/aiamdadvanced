# OpenAMD Advanced — Hybrid AI AMD Server

Installs the **OpenAMD Advanced AI AMD server + web portal** on a new Ubuntu 24.04 machine.

**Engine:** Hybrid — OpenAMD heuristic acoustic rules **+** Silero VAD (ONNX Runtime).

Repo: https://github.com/xceedconnections/aiamdadvanced

---

## One-command install (new Ubuntu 24.04 server)

Run as **root**:

```bash
curl -fsSL https://raw.githubusercontent.com/xceedconnections/aiamdadvanced/main/remote-install.sh | bash
```

That downloads this repo and runs the full installer in one go.

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

## Upgrade existing server

```bash
rm -rf /root/aiamdadvanced
git clone https://github.com/xceedconnections/aiamdadvanced.git /root/aiamdadvanced
find /root/aiamdadvanced -type f -name '*.sh' -exec sed -i 's/\r$//' {} +
bash /root/aiamdadvanced/install4.sh
bash /root/aiamdadvanced/install6.sh
systemctl restart openamd nginx
```

Portal = port **80**. VICIdial AMD API = port **2130** (firewall dialer IPs only).

Dialer URL example: `http://YOUR_HOST:2130` in `vicibox_install.sh` / `openamd.conf`.

Or run full `install.sh` again (safe on an existing server).

---

# OpenAMD for ViciBox / VICIdial (extensions 8369 / 8398 / 8399)

Installs OpenAMD **AGI + AMD routing extensions** on a ViciBox / VICIdial dialer (Asterisk).

AI AMD Advanced server install (Ubuntu portal):  
https://github.com/xceedconnections/aiamdadvanced

## Prerequisites

1. AI AMD server already installed
2. Portal reachable: `http://YOUR_AI_AMD_IP/` or `https://your-domain/`
3. Portal → **VICIdial Servers** → add server → **Generate API Key**

## One-command install (on ViciBox as root)

Pass IP, domain, or full `http://` / `https://` URL:

```bash
curl -fsSL https://raw.githubusercontent.com/xceedconnections/vicidialaiamd/main/remote-install.sh | bash -s -- https://aiamd.xceedconnections.com oam_YOUR_API_KEY
```

```bash
curl -fsSL https://raw.githubusercontent.com/xceedconnections/vicidialaiamd/main/remote-install.sh | bash -s -- http://aiamd.xceedconnections.com oam_YOUR_API_KEY
```

```bash
curl -fsSL https://raw.githubusercontent.com/xceedconnections/vicidialaiamd/main/remote-install.sh | bash -s -- 204.168.200.221 oam_xxxxxxxx
```

Bare IP/domain defaults to **http://**. Prefer **https://your-domain** when TLS is configured.

### Alternative (clone first)

```bash
git clone https://github.com/xceedconnections/vicidialaiamd.git /root/vicidialaiamd
bash /root/vicidialaiamd/vicibox_install.sh https://aiamd.xceedconnections.com oam_YOUR_API_KEY
bash /root/vicidialaiamd/vicibox_install.sh 204.168.200.221 oam_YOUR_API_KEY
```

## What this installs

1. `/etc/asterisk/openamd.conf` (URL + API key)
2. `openamd.agi` (Caller ID / Called Number aware)
3. `VD_amd_human_only.agi` + extensions **8398** (human-only) and **8399** (AIAMD final)
4. Local-presence metadata store for **any** VICIdial dial prefix  
   (`_8.` `_9.` `_87899.` `_94455.` `_94556.` `_6.` etc.)  
   Called Number is taken from that pattern's `Dial(...${EXTEN:N}...)`  
   (works with CURL / RAND Caller ID — does **not** change `Dial()` destinations)

## Campaign setting

- **8369** = stock Vicidial AMD (untouched)
- **8398** = human-only stock AMD (HUMAN/PERSON → agent; else hangup; no scam)
- **8399** = stock MACHINE→hangup; HUMAN/PERSON/NOTSURE→AIAMD; AIAMD HUMAN→agent; AIAMD down → 8398 hangup

Leave AMD Agent Route Options blank for 8398/8399. CPS/offline on 8399 = 8398-style hangup (not Goto 8369). Install strips old 8389/8397.

## Folder contents

```text
vicibox_install.sh              # main installer
remote-install.sh               # GitHub one-liner bootstrap
fix_caller_called.sh            # re-apply metadata only
agi/
  openamd.agi
  VD_amd_human_only.agi
  fix_local_presence_metadata.sh
  restore_vicidial_dialplan.sh  # recovery only
```

## After VICIdial rebuilds dialplan

Re-run the same one-liner, or:

```bash
bash /root/vicidialaiamd/vicibox_install.sh https://aiamd.xceedconnections.com oam_YOUR_API_KEY
```

Metadata only:

```bash
bash /root/vicidialaiamd/fix_caller_called.sh
```

## Verify

```bash
asterisk -rx "dialplan show 8369@default"
asterisk -rx "dialplan show 8398@default"
asterisk -rx "dialplan show 8399@default"
asterisk -rx "dialplan show openamd-detect"
curl -sS https://aiamd.xceedconnections.com/api/health
# or: curl -sS http://YOUR_AI_AMD_IP/api/health
```

ViciBox dialer installer repo: https://github.com/xceedconnections/vicidialaiamd
