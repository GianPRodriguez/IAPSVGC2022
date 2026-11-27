#!/usr/bin/env bash
# Prepara una EC2 Ubuntu 24.04 (AWS Academy Learner Lab) para la web de retos BO3 (scripts/online/bo3_web.py).
# Idempotente: se puede volver a ejecutar para actualizar (git pull + reinicio del servicio).
#
#   Antes (desde la laptop):
#     scp logs/online_ddqn/*.pt            ubuntu@IP:~/online_ddqn/     (no están en git)
#     scp <token DuckDNS>                  ubuntu@IP:~/duckdns_token
#   En la EC2:
#     bash setup_ec2.sh
#
# Resultado: https://$SUB.duckdns.org -> Caddy (HTTPS automático) -> 127.0.0.1:8000 (bo3web.service).
# DuckDNS se actualiza al arrancar y cada 5 min (la IP pública cambia en cada sesión del Lab).
set -euo pipefail

SUB="${SUB:-teamofabrexuxa}"
BRANCH="${BRANCH:-feature/Champions}"
B="$HOME"
REPO="$B/IAPSVGC2022"
log() { echo "== $*"; }

log "paquetes del sistema"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git python3-venv python3-pip curl ca-certificates gnupg \
  debian-keyring debian-archive-keyring apt-transport-https >/dev/null

if ! command -v node >/dev/null || [ "$(node -v | cut -d. -f1 | tr -d v)" -lt 20 ]; then
  log "node 22 (NodeSource)"
  curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash - >/dev/null
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq nodejs >/dev/null
fi

if ! command -v caddy >/dev/null; then
  log "caddy"
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
    | sudo gpg --batch --yes --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
    | sudo tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq caddy >/dev/null
fi

log "repos (versiones fijadas como en docs/SERVER.md)"
if [ ! -d "$REPO/.git" ]; then
  git clone -q --depth 1 --branch "$BRANCH" --single-branch https://github.com/GianPRodriguez/IAPSVGC2022.git "$REPO"
else
  git -C "$REPO" pull -q --ff-only
fi
if [ ! -d "$B/poke-env/.git" ]; then
  git clone -q https://github.com/hsahovic/poke-env.git "$B/poke-env"
fi
git -C "$B/poke-env" checkout -q 7ff6212
if [ ! -d "$B/damage-calc/.git" ]; then
  git clone -q https://github.com/smogon/damage-calc.git "$B/damage-calc"
fi
git -C "$B/damage-calc" checkout -q bacbbfc
if [ ! -f "$B/damage-calc/calc/dist/index.js" ]; then
  log "compilando damage-calc"
  (cd "$B/damage-calc/calc" && npm ci --ignore-scripts --no-audit --no-fund --loglevel=error && rm -rf dist .tsbuildinfo && npm run -s compile)
fi

log "entorno de Python (torch CPU)"
[ -x "$B/venv/bin/python" ] || python3 -m venv "$B/venv"
"$B/venv/bin/pip" install -q -U pip
"$B/venv/bin/python" -c "import torch" 2>/dev/null \
  || "$B/venv/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cpu
"$B/venv/bin/pip" install -q -r "$REPO/requirements-rl.txt" -e "$B/poke-env"

log "checkpoints de DDQN v1 (copiados por scp; no están en git)"
mkdir -p "$REPO/logs/online_ddqn"
if compgen -G "$B/online_ddqn/*.pt" >/dev/null; then
  cp -u "$B"/online_ddqn/*.pt "$REPO/logs/online_ddqn/"
fi
ls "$REPO/logs/online_ddqn/"*.pt

log "comprobación de agentes (sin conectarse)"
(cd "$REPO" && "$B/venv/bin/python" scripts/online/bo3_web.py --check)

log "DuckDNS"
sudo mkdir -p /etc/bo3web
if [ -f "$B/duckdns_token" ]; then
  sudo install -m 600 -o root -g root "$B/duckdns_token" /etc/bo3web/duckdns_token
  rm -f "$B/duckdns_token"
fi
sudo test -s /etc/bo3web/duckdns_token
sudo tee /usr/local/bin/duckdns-update >/dev/null <<'EOF'
#!/bin/sh
# ip= vacío: DuckDNS toma la IP pública de quien llama. Responde "OK" o "KO".
TOKEN=$(tr -d ' \r\n' < /etc/bo3web/duckdns_token)
curl -fsS -m 20 "https://www.duckdns.org/update?domains=$1&token=$TOKEN&ip=" | grep -q '^OK'
EOF
sudo chmod 755 /usr/local/bin/duckdns-update
sudo tee /etc/systemd/system/duckdns.service >/dev/null <<EOF
[Unit]
Description=Actualiza $SUB.duckdns.org con la IP pública
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
ExecStart=/usr/local/bin/duckdns-update $SUB
EOF
sudo tee /etc/systemd/system/duckdns.timer >/dev/null <<EOF
[Unit]
Description=DuckDNS al arrancar y cada 5 min

[Timer]
OnBootSec=15s
OnUnitActiveSec=5min

[Install]
WantedBy=timers.target
EOF

log "servicio de la web"
sudo tee /etc/systemd/system/bo3web.service >/dev/null <<EOF
[Unit]
Description=Web de retos BO3 (scripts/online/bo3_web.py)
After=network-online.target
Wants=network-online.target

[Service]
User=$USER
WorkingDirectory=$REPO
Environment=PYTHONUNBUFFERED=1
# --inbound-only: Showdown bloquea (lock) la IP de la EC2; la persona reta al bot y el bot acepta.
ExecStart=$B/venv/bin/python scripts/online/bo3_web.py --host 127.0.0.1 --port 8000 --inbound-only
Restart=on-failure
RestartSec=15

[Install]
WantedBy=multi-user.target
EOF

log "caddy (HTTPS para $SUB.duckdns.org)"
sudo tee /etc/caddy/Caddyfile >/dev/null <<EOF
$SUB.duckdns.org {
	encode gzip
	reverse_proxy 127.0.0.1:8000
}
EOF

sudo systemctl daemon-reload
sudo systemctl enable -q --now duckdns.timer
sudo systemctl start duckdns.service
sudo systemctl enable -q bo3web.service caddy
sudo systemctl restart bo3web.service caddy
log "listo: https://$SUB.duckdns.org (logs: journalctl -u bo3web -f)"
