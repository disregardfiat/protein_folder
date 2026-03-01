#!/bin/bash
# Run from repo root on server (e.g. ~/protein_folder after clone).
# Sets up venv, installs deps, installs systemd user service for gunicorn.
set -e
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
echo "Repo root: $ROOT"
python3 -m venv .venv
.venv/bin/pip install -q -r requirements-server.txt
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/casp-server.service << EOF
[Unit]
Description=HQIV CASP prediction server
After=network.target

[Service]
Type=simple
WorkingDirectory=$ROOT
# 259200s = 72h (CASP submission window)
ExecStart=$ROOT/.venv/bin/gunicorn -w 1 -b 127.0.0.1:8050 --timeout 259200 casp_server:app
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
EOF
echo "Created ~/.config/systemd/user/casp-server.service"
echo "Enable and start with:"
echo "  systemctl --user daemon-reload"
echo "  systemctl --user enable --now casp-server"
echo "  loginctl enable-linger \$USER   # if needed for user service to run without login"
