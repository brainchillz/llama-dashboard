# Llama-Dash

Flask web UI for managing [llama.cpp](https://github.com/ggml-org/llama.cpp) servers. Start/stop/restart the service, switch models, edit CLI arguments, view startup logs, and monitor server health — all from a browser.

## Prerequisites

- llama.cpp built with `llama-server` binary
- Python 3.10+
- systemd (the dashboard manages llama-server as a systemd unit)
- .gguf model files accessible on the filesystem

## Directory Layout (Expected)

```
/usr/share/models/          # scanned recursively for .gguf files
/etc/llama.conf             # config consumed by the wrapper script
/usr/local/bin/llama-server.sh  # wrapper that sources conf and execs llama-server
/etc/systemd/system/llama-server.service   # systemd unit
/opt/llama-dashboard/       # the dashboard itself
```

Models are discovered recursively under `/usr/share/models`. If your models live elsewhere, symlink:

```bash
ln -s /path/to/your/models /usr/share/models
```

## Quick Install

```bash
git clone https://github.com/brainchillz/llama-dashboard.git /opt/llama-dashboard
cd /opt/llama-dashboard
./install.sh
```

The installer:

1. Installs system dependencies (`python3`, `python3-venv`, `openssl`, `curl`)
2. Creates a Python virtualenv and installs Flask + gunicorn + werkzeug
3. Generates a self-signed SSL certificate (10-year) for `cert.pem` / `key.pem`
4. Creates `/etc/llama.conf` from current running config (or a sensible default)
5. Installs `/usr/local/bin/llama-server.sh` wrapper script
6. Registers and starts the `llama-dashboard` systemd service on port **4043** (HTTPS)

After install, open `https://<host>:4043` and log in with `admin` / `admin`. Change the password immediately from the Administration page.

## Manual Setup

### 1. Config File (`/etc/llama.conf`)

```ini
LLAMA_BIN=/usr/local/llama.cpp/llama-server
LLAMA_MODEL=/usr/share/models/your-model.gguf
LLAMA_OPTS="--host 0.0.0.0 --port 8080 --threads 8 --n-gpu-layers 99 --metrics"
```

- `LLAMA_BIN` — path to the llama-server binary
- `LLAMA_MODEL` — path to the .gguf model file
- `LLAMA_OPTS` — all other CLI flags (must be quoted)

### 2. Wrapper Script (`/usr/local/bin/llama-server.sh`)

```bash
#!/bin/bash
. /etc/llama.conf
exec "$LLAMA_BIN" -m "$LLAMA_MODEL" $LLAMA_OPTS
```

Make it executable:

```bash
chmod 755 /usr/local/bin/llama-server.sh
```

### 3. Systemd Service (`/etc/systemd/system/llama-server.service`)

```ini
[Unit]
Description=llama.cpp llama-server
After=network.target

[Service]
User=llama
Group=llama
WorkingDirectory=/usr/local/llama.cpp
Environment="LD_LIBRARY_PATH=/usr/local/llama.cpp:/opt/rocm/lib:/opt/rocm/lib64"
ExecStart=/usr/local/bin/llama-server.sh
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Adjust `User`, `Group`, `WorkingDirectory`, and `Environment` for your setup. The dashboard manages this service via `systemctl start/stop/restart llama-server`.

### 4. Register Libraries (if needed)

```bash
echo /usr/local/llama.cpp > /etc/ld.so.conf.d/llama.conf
ldconfig
```

## Dashboard Service

The dashboard runs under its own systemd unit (`llama-dashboard.service`) on port **4043** (HTTPS).

| Command | Description |
|---|---|
| `systemctl start llama-dashboard` | Start the dashboard |
| `systemctl stop llama-dashboard` | Stop the dashboard |
| `systemctl restart llama-dashboard` | Restart the dashboard |
| `journalctl -u llama-dashboard -f` | Follow dashboard logs |

## Pages

### Dashboard (`/`)

- **Server Health** — status indicator (green/red) and real-time metrics from llama-server's `/health` and `/metrics` endpoints (prompt tok/s, gen tok/s, token counts, active/deferred requests)
- **Service Status** — current systemd state with Start / Stop / Restart buttons
- **Startup Log** — last 50 lines of `journalctl -u llama-server`

### Server Arguments (`/args`)

- **Model** — dropdown listing all `.gguf` files found under `/usr/share/models`. Switching writes the new path to `/etc/llama.conf` (you must restart the service from the Dashboard to apply)
- **Server Arguments** — editable table of CLI flags. Add, edit, or remove flags. Saving updates `/etc/llama.conf` and automatically restarts the service

### Administration (`/admin`)

- Change the web UI password

## llama.cpp UI Link

The header includes a "llama.cpp UI" button that links to `http://<your-host>:8080` (the server's address is derived from the request host, so it works across machines).

## API Endpoints

All API routes require a valid session cookie (`/api/login` first).

| Method | Path | Description |
|---|---|---|
| POST | `/api/login` | Authenticate (`{password}`) |
| POST | `/api/logout` | Clear session |
| PUT | `/api/password` | Change password (`{current_password, new_password}`) |
| GET | `/api/status` | Service active state + substate |
| POST | `/api/start` | Start llama-server |
| POST | `/api/stop` | Stop llama-server |
| POST | `/api/restart` | Restart llama-server |
| GET | `/api/model` | Currently configured model |
| PUT | `/api/model` | Set model (`{model_path}`) |
| GET | `/api/args` | Current CLI arguments |
| PUT | `/api/args` | Update CLI arguments (`{args: [{flag, value}]}`) |
| GET | `/api/logs` | Last 50 lines of service journal |
| GET | `/api/llama-health` | Proxy to llama-server `/health` |
| GET | `/api/llama-metrics` | Parsed metrics from llama-server `/metrics` |

## Security Notes

- Always change the default password after first login
- The dashboard uses a self-signed certificate generated during install — replace with a trusted cert in production
- The dashboard runs as root (required for `systemctl` calls). Restrict access by firewall to trusted networks
- API keys/tokens in this README are examples only; never commit real credentials

## License

MIT
