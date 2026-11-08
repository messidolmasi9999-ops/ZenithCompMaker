#!/usr/bin/env bash
# Ubuntu 22.04/24.04 (VM veya RunPod/Vast gibi Docker pod): bash deploy.sh
# Alan adı + HTTPS için: DOMAIN=comp.senin-site.com bash deploy.sh
set -e
cd "$(dirname "$0")"; DIR="$(pwd)"; SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO=sudo
HASGPU=0; command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1 && HASGPU=1
$SUDO apt-get update -qq
$SUDO apt-get install -y -qq python3-venv python3-pip ffmpeg aria2 libgl1 libglib2.0-0 curl xz-utils
if [ $HASGPU = 1 ] && ! ffmpeg -hide_banner -encoders 2>/dev/null | grep -q h264_nvenc; then
  echo "NVENC destekli ffmpeg indiriliyor..."
  curl -L -o /tmp/ff.tar.xz https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz
  tar -xJf /tmp/ff.tar.xz -C /tmp && $SUDO cp /tmp/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg /tmp/ffmpeg-master-latest-linux64-gpl/bin/ffprobe /usr/local/bin/ && hash -r
fi
python3 -m venv venv && . venv/bin/activate
pip -q install --upgrade pip
if [ $HASGPU = 1 ]; then
  pip -q install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  pip -q install -r requirements.txt
else
  echo "GPU yok: CPU sürümü kuruluyor"
  pip -q install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  grep -v onnxruntime-gpu requirements.txt > /tmp/req-cpu.txt; echo onnxruntime >> /tmp/req-cpu.txt
  pip -q install -r /tmp/req-cpu.txt
fi
read -rsp "Admin şifresi belirle: " ADM; echo
if [ -d /run/systemd/system ]; then
  $SUDO tee /etc/systemd/system/zenith.service >/dev/null <<UNIT
[Unit]
Description=ZenithCompMaker
After=network.target
[Service]
WorkingDirectory=$DIR
Environment=ZENITH_ADMIN_PASSWORD=$ADM
Environment=PORT=5000
Environment=PATH=/usr/local/bin:/usr/bin:/bin
ExecStart=$DIR/venv/bin/python app.py
Restart=always
User=$(id -un)
[Install]
WantedBy=multi-user.target
UNIT
  $SUDO systemctl daemon-reload && $SUDO systemctl enable --now zenith
else
  echo "systemd yok (Docker pod): arka planda başlatılıyor, log: zenith.log"
  ZENITH_ADMIN_PASSWORD="$ADM" PORT=5000 nohup venv/bin/python app.py > zenith.log 2>&1 &
fi
if [ -n "$DOMAIN" ]; then
  $SUDO apt-get install -y -qq debian-keyring debian-archive-keyring apt-transport-https gpg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | $SUDO gpg --dearmor --yes -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | $SUDO tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
  $SUDO apt-get update -qq && $SUDO apt-get install -y -qq caddy
  echo "$DOMAIN { reverse_proxy localhost:5000 }" | $SUDO tee /etc/caddy/Caddyfile >/dev/null; $SUDO systemctl restart caddy
  echo "Hazır: https://$DOMAIN  (domain A kaydı bu sunucunun IP'sine bakmalı)"
else
  echo "Hazır: http://$(curl -s ifconfig.me || echo SUNUCU_IP):5000  (portu aç) | HTTPS linki için: cloudflared tunnel --url http://localhost:5000"
fi
echo "Kontrol: venv/bin/python selftest.py"
