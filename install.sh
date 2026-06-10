#!/bin/bash
#
# install.sh — llama.cpp Dashboard installer
#
# Usage: sudo bash install.sh
#
# What it does:
#   1. Installs system dependencies
#   2. Creates a Python virtualenv and installs packages
#   3. Generates a self-signed SSL certificate
#   4. Creates /etc/llama.conf from existing service (if found)
#   5. Installs /usr/local/bin/llama-server.sh wrapper
#   6. Installs and starts llama-dashboard systemd service

set -e

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
DASHBOARD_USER="${SUDO_USER:-root}"
PYTHON="python3"
VENV_DIR="$APP_DIR/venv"
SERVICE_NAME="llama-dashboard"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
CONF_FILE="/etc/llama.conf"
WRAPPER="/usr/local/bin/llama-server.sh"

log()  { printf "\033[0;32m[INFO]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[WARN]\033[0m %s\n" "$*"; }

if [ "$EUID" -ne 0 ]; then
    warn "Please run with sudo or as root."
    exit 1
fi

# ── System Dependencies ──────────────────────────────────────────
log "Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq "$PYTHON" "$PYTHON-venv" openssl curl >/dev/null

# ── Python Virtualenv ────────────────────────────────────────────
log "Creating Python virtual environment..."
"$PYTHON" -m venv "$VENV_DIR"

log "Installing Python dependencies..."
"$VENV_DIR/bin/pip" install --quiet --upgrade pip
"$VENV_DIR/bin/pip" install --quiet flask gunicorn werkzeug

# ── SSL Certificate ──────────────────────────────────────────────
CERT="$APP_DIR/cert.pem"
KEY="$APP_DIR/key.pem"

if [ ! -f "$CERT" ] || [ ! -f "$KEY" ]; then
    log "Generating self-signed SSL certificate (10-year validity)..."
    openssl req -x509 -newkey rsa:4096 -keyout "$KEY" -out "$CERT" \
        -days 3650 -nodes -subj "/CN=llama-dashboard" 2>/dev/null
    chmod 600 "$KEY" "$CERT"
    log "Certificate generated."
else
    log "Certificate already exists, skipping."
fi

# ── llama-server Wrapper Script ──────────────────────────────────
log "Installing llama-server wrapper script..."
cat > "$WRAPPER" << 'WRAPPER'
#!/bin/bash
. /etc/llama.conf
exec "$LLAMA_BIN" -m "$LLAMA_MODEL" $LLAMA_OPTS
WRAPPER
chmod 755 "$WRAPPER"

# ── Config File ──────────────────────────────────────────────────
if [ -f "$CONF_FILE" ]; then
    log "$CONF_FILE already exists, keeping it."
else
    log "Creating $CONF_FILE from existing service configuration..."
    LLAMA_BIN=""
    LLAMA_MODEL=""
    LLAMA_OPTS=""

    # Try to extract from running service unit
    if systemctl cat llama-server &>/dev/null; then
        EXEC_START=$(systemctl cat llama-server 2>/dev/null | grep ^ExecStart= | head -1 | sed 's/^ExecStart=//')
        if [ -n "$EXEC_START" ]; then
            # shellcheck disable=SC2086
            set -- $EXEC_START
            BIN=""
            MODEL=""
            OPTS=()
            while [ $# -gt 0 ]; do
                case "$1" in
                    -m|--model)
                        MODEL="$2"; shift 2 ;;
                    --model=*)
                        MODEL="${1#*=}"; shift ;;
                    -m?*)
                        MODEL="${1#??}"; shift ;;
                    *)
                        if [ -z "$BIN" ] && [ -f "$1" ] && [ -x "$1" ]; then
                            BIN="$1"
                        else
                            OPTS+=("$1")
                        fi
                        shift ;;
                esac
            done
            LLAMA_BIN="$BIN"
            LLAMA_MODEL="$MODEL"
            LLAMA_OPTS="${OPTS[*]}"
        fi
    fi

    # Fallback defaults
    [ -z "$LLAMA_BIN" ]    && LLAMA_BIN="/usr/local/llama.cpp/llama-server"
    [ -z "$LLAMA_MODEL" ]  && LLAMA_MODEL=""
    [ -z "$LLAMA_OPTS" ]   && LLAMA_OPTS="--host 0.0.0.0 --port 8080 --threads 8 --n-gpu-layers 99"

    cat > "$CONF_FILE" << CONFEOF
LLAMA_BIN=$LLAMA_BIN
LLAMA_MODEL=$LLAMA_MODEL
LLAMA_OPTS="$LLAMA_OPTS"
CONFEOF
    chmod 644 "$CONF_FILE"
    log "$CONF_FILE created."
fi

# ── Systemd Service ──────────────────────────────────────────────
log "Installing systemd service..."
cat > "$SERVICE_FILE" << UNIT
[Unit]
Description=llama.cpp Dashboard (Flask)
After=network.target

[Service]
Type=simple
User=root
Group=root
WorkingDirectory=$APP_DIR
ExecStart=$VENV_DIR/bin/gunicorn -w 2 -b 0.0.0.0:4043 \
    --certfile $APP_DIR/cert.pem \
    --keyfile $APP_DIR/key.pem \
    --pid /run/${SERVICE_NAME}.pid \
    app:app
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable "$SERVICE_NAME"
systemctl restart "$SERVICE_NAME"

log "Service $SERVICE_NAME enabled and started."
echo ""
echo "─────────────────────────────────────────────────────"
echo "  llama-switcher installed successfully"
echo "─────────────────────────────────────────────────────"
echo ""
echo "  Dashboard URL:  https://$(hostname -f 2>/dev/null || hostname):4043"
echo "  Username:       admin"
echo "  Password:       admin"
echo ""
echo "  Change the password immediately from the"
echo "  Administration page in the dashboard."
echo ""
echo "  To view logs:   journalctl -u $SERVICE_NAME -f"
echo "  To restart:     systemctl restart $SERVICE_NAME"
echo ""
