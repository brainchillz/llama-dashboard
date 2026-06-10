import subprocess
import os
import re
import time
import json
import secrets
from functools import wraps
from flask import (
    Flask, render_template_string, jsonify, request, session, redirect
)
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)

MODELS_DIR = "/usr/share/models"
SERVICE_NAME = "llama-server"
SERVICE_FILE = "/etc/systemd/system/llama-server.service"
CONF_FILE = "/etc/llama.conf"
CONFIG_FILE = "/opt/llama-dashboard/config.json"


def _load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            return json.load(f)
    sk = secrets.token_hex(32)
    ph = generate_password_hash("admin")
    cfg = {"secret_key": sk, "password_hash": ph}
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)
    app.logger.warning("=== Default password is 'admin' — change it immediately ===")
    return cfg


_first_config = _load_config()
app.secret_key = _first_config["secret_key"]


def _read_config():
    return _load_config()


def _write_config(cfg):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)


def login_required(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        if not session.get("user"):
            if request.path.startswith("/api/"):
                return jsonify(ok=False, error="Unauthorized"), 401
            return redirect("/login")
        return fn(*a, **kw)
    return wrapper


def get_models():
    models = []
    for root, dirs, files in os.walk(MODELS_DIR):
        for f in files:
            if f.endswith(".gguf") and "mmproj-" not in f:
                full = os.path.join(root, f)
                display = full.replace(MODELS_DIR + "/", "")
                models.append({"path": full, "name": display})
    return sorted(models, key=lambda m: m["name"])


def get_service_status():
    r = subprocess.run(
        ["systemctl", "is-active", SERVICE_NAME],
        capture_output=True, text=True
    )
    active_state = r.stdout.strip()
    r2 = subprocess.run(
        ["systemctl", "show", "--property=SubState", SERVICE_NAME],
        capture_output=True, text=True
    )
    substate = r2.stdout.strip().replace("SubState=", "")
    return {
        "active": active_state == "active",
        "status": active_state,
        "substate": substate,
    }


def get_current_model():
    return _read_conf()["model"] or None


# Flags that take no value (boolean/presence-only flags)
KNOWN_BOOL_FLAGS = {
    # general
    "--version", "-v", "--license", "--help", "-h", "--usage",
    "--list-devices", "--check-tensors", "--log-disable",
    "--log-colors", "--verbose", "--log-verbose",
    "--offline",
    # inference
    "--escape", "--no-escape",
    "--ignore-eos", "--perf", "--no-perf",
    "--flash-attn", "-fa",
    # memory / compute
    "--mlock", "--no-mmap", "--mmap",
    "--no-host", "--repack", "--no-repack",
    "--kv-offload", "-kvo", "--no-kv-offload", "-nkvo",
    "--direct-io", "-dio", "--no-direct-io", "-ndio",
    "--op-offload", "--no-op-offload",
    "--cpu-moe", "-cmoe",
    # server
    "--reuse-port", "--metrics", "--props",
    "--slots", "--no-slots",
    "--embedding", "--embeddings",
    "--rerank", "--reranking",
    "--jinja", "--no-jinja",
    "--cont-batching", "-cb", "--no-cont-batching", "-nocb",
    "--cache-prompt", "--no-cache-prompt",
    "--cache-idle-slots", "--no-cache-idle-slots",
    "--context-shift", "--no-context-shift",
    "--warmup", "--no-warmup",
    "--spm-infill",
    "--mmproj-auto", "--no-mmproj", "--no-mmproj-auto",
    "--mmproj-offload", "--no-mmproj-offload",
    "--prefill-assistant", "--no-prefill-assistant",
    "--skip-chat-parsing", "--no-skip-chat-parsing",
    "--kv-unified", "-kvu", "--no-kv-unified", "-no-kvu",
    "--models-autoload", "--no-models-autoload",
    "--log-prefix", "--no-log-prefix",
    "--log-timestamps", "--no-log-timestamps",
    "--lora-init-without-apply",
    "--backend-sampling", "-bs",
    "--ui-mcp-proxy", "--no-ui-mcp-proxy",
    "--webui-mcp-proxy", "--no-webui-mcp-proxy",
    "--webui", "--no-webui",
    "--ui", "--no-ui",
    # speculative
    "--spec-draft-backend-sampling", "--no-spec-draft-backend-sampling",
    "--cpu-moe-draft", "-cmoed",
    # fit
    "--fit",
}
KNOWN_BOOL_SHORT = {f"-{k.lstrip('-')}" for k in KNOWN_BOOL_FLAGS if k.startswith("--") and not k.startswith("---")}


def _read_conf():
    """Read /etc/llama.conf and return {bin, model, opts}."""
    conf = {"bin": "/usr/local/llama.cpp/llama-server", "model": "", "opts": ""}
    if os.path.exists(CONF_FILE):
        with open(CONF_FILE) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, val = line.partition("=")
                    val = val.strip().strip('"').strip("'")
                    if key == "LLAMA_BIN":
                        conf["bin"] = val
                    elif key == "LLAMA_MODEL":
                        conf["model"] = val
                    elif key == "LLAMA_OPTS":
                        conf["opts"] = val
    # strip any -m from opts as a safety measure
    conf["opts"] = re.sub(r'-m\s+\S+', '', conf["opts"]).strip()
    return conf


def _write_conf(conf):
    """Write {bin, model, opts} to /etc/llama.conf."""
    with open(CONF_FILE, "w") as f:
        f.write(f'LLAMA_BIN={conf["bin"]}\n')
        f.write(f'LLAMA_MODEL={conf["model"]}\n')
        f.write(f'LLAMA_OPTS="{conf["opts"]}"\n')


def _parse_opts(opts_string):
    """Parse an opts string into a list of {flag, value} dicts.

    Flags in KNOWN_BOOL_FLAGS produce value=''. All other flags take the
    next non-flag token as their value.
    """
    tokens = opts_string.split()
    args = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("-"):
            flag = tok
            if flag in KNOWN_BOOL_FLAGS:
                args.append({"flag": flag, "value": ""})
                i += 1
                continue
            if "=" in flag:
                f, v = flag.split("=", 1)
                args.append({"flag": f, "value": v})
                i += 1
                continue
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                args.append({"flag": flag, "value": tokens[i + 1]})
                i += 2
            else:
                args.append({"flag": flag, "value": ""})
                i += 1
        else:
            args.append({"flag": "", "value": tok})
            i += 1
    return args


def _format_opts(args):
    """Format a list of {flag, value} dicts into an opts string."""
    parts = []
    for a in args:
        if not a["flag"]:
            continue
        if a["value"]:
            parts.append(f"{a['flag']} {a['value']}")
        else:
            parts.append(a["flag"])
    return " ".join(parts)


LOGIN_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Llama-Dash — Login</title>
<style>
  :root {
    --bg: #f5f5f5; --card: #fff; --text: #1a1a1a; --text-secondary: #666;
    --border: #ddd; --accent: #2563eb; --accent-hover: #1d4ed8;
    --shadow: rgba(0,0,0,0.08); --code-bg: #f0f0f0; --input-bg: #fff;
  }
  [data-theme="dark"] {
    --bg: #111827; --card: #1f2937; --text: #f3f4f6;
    --text-secondary: #9ca3af; --border: #374151; --accent: #3b82f6;
    --accent-hover: #60a5fa; --shadow: rgba(0,0,0,0.3);
    --code-bg: #374151; --input-bg: #374151;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg); color: var(--text);
    min-height: 100vh; display: flex; align-items: center; justify-content: center;
    transition: background .2s, color .2s;
  }
  .login-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 2rem; width: 360px;
    box-shadow: 0 4px 12px var(--shadow);
  }
  .login-card h1 { font-size: 1.25rem; margin-bottom: .25rem; }
  .login-card p { color: var(--text-secondary); font-size: .875rem; margin-bottom: 1.5rem; }
  label { display: block; font-size: .875rem; font-weight: 500; margin-bottom: .25rem; }
  input {
    width: 100%; padding: .5rem .75rem; margin-bottom: 1rem;
    border: 1px solid var(--border); border-radius: 6px;
    background: var(--input-bg); color: var(--text); font-size: .875rem;
  }
  input:focus { outline: 2px solid var(--accent); outline-offset: -1px; }
  .btn {
    width: 100%; padding: .5rem; border: none; border-radius: 6px;
    background: var(--accent); color: #fff; font-size: .875rem; font-weight: 500;
    cursor: pointer; transition: background .15s;
  }
  .btn:hover { background: var(--accent-hover); }
  .msg { margin-top: 1rem; padding: .5rem; border-radius: 6px; font-size: .8125rem; display: none; }
  .msg.error { display: block; background: #7f1d1d; color: #fecaca; }
  [data-theme="dark"] .msg.error { background: #7f1d1d; color: #fecaca; }
  .header-bar {
    display: flex; align-items: center; justify-content: flex-end;
    padding: .75rem 1.5rem; background: var(--card);
    border-bottom: 1px solid var(--border);
  }
  .theme-toggle {
    background: var(--code-bg); border: 1px solid var(--border);
    color: var(--text); padding: .3rem .7rem; border-radius: 6px;
    cursor: pointer; font-size: .8rem;
  }
  .theme-toggle:hover { opacity: .8; }
</style>
</head>
<body>
<div style="width:100%">
<div class="header-bar">
  <button class="theme-toggle" onclick="toggleTheme()" id="themeBtn">Dark Mode</button>
</div>
<div style="display:flex;justify-content:center;padding-top:2rem">
<div class="login-card">
  <h1>Llama-Dash</h1>
  <p>Authenticate to continue</p>
  <form onsubmit="return login(event)">
    <label for="user">Username</label>
    <input type="text" id="user" value="admin" readonly
           style="opacity:.7;cursor:not-allowed">
    <label for="pass">Password</label>
    <input type="password" id="pass" autofocus required>
    <button type="submit" class="btn" id="loginBtn">Sign In</button>
  </form>
  <div class="msg" id="msg"></div>
</div>
</div>
</div>
<script>
function getCookie(name) {
  const v = document.cookie.match('(^|; )' + name + '=([^;]*)');
  return v ? decodeURIComponent(v[2]) : null;
}
function setCookie(name, val) {
  document.cookie = name + '=' + encodeURIComponent(val) + ';path=/;max-age=31536000';
}
(function() {
  const theme = getCookie('theme');
  if (theme === 'dark') {
    document.documentElement.setAttribute('data-theme', 'dark');
    document.getElementById('themeBtn').textContent = 'Light Mode';
  }
})();
function toggleTheme() {
  const html = document.documentElement;
  const btn = document.getElementById('themeBtn');
  if (html.getAttribute('data-theme') === 'dark') {
    html.removeAttribute('data-theme'); setCookie('theme', 'light');
    btn.textContent = 'Dark Mode';
  } else {
    html.setAttribute('data-theme', 'dark'); setCookie('theme', 'dark');
    btn.textContent = 'Light Mode';
  }
}
async function login(e) {
  e.preventDefault();
  const btn = document.getElementById('loginBtn');
  const msg = document.getElementById('msg');
  btn.disabled = true; msg.className = 'msg';
  try {
    const r = await fetch('/api/login', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({password: document.getElementById('pass').value})
    });
    const d = await r.json();
    if (d.ok) { window.location.href = '/'; return; }
    msg.textContent = d.error || 'Invalid credentials';
    msg.className = 'msg error';
  } catch(_) {
    msg.textContent = 'Network error';
    msg.className = 'msg error';
  }
  btn.disabled = false;
}
</script>
</body>
</html>
"""

DASHBOARD_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Llama-Dash</title>
<style>
  :root {
    --bg: #f5f5f5; --card: #fff; --text: #1a1a1a; --text-secondary: #666;
    --border: #ddd; --accent: #2563eb; --accent-hover: #1d4ed8;
    --success: #16a34a; --danger: #dc2626; --danger-hover: #b91c1c;
    --warning: #d97706; --shadow: rgba(0,0,0,0.08); --code-bg: #f0f0f0;
    --select-bg: #fff; --input-bg: #fff;
  }
  [data-theme="dark"] {
    --bg: #111827; --card: #1f2937; --text: #f3f4f6;
    --text-secondary: #9ca3af; --border: #374151; --accent: #3b82f6;
    --accent-hover: #60a5fa; --success: #22c55e; --danger: #ef4444;
    --danger-hover: #f87171; --warning: #f59e0b; --shadow: rgba(0,0,0,0.3);
    --code-bg: #374151; --select-bg: #374151; --input-bg: #374151;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg); color: var(--text); min-height: 100vh;
    transition: background .2s, color .2s;
  }
  .header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 1rem 2rem; background: var(--card);
    border-bottom: 1px solid var(--border); box-shadow: 0 1px 3px var(--shadow);
  }
  .header h1 { font-size: 1.25rem; font-weight: 600; }
  .header-actions { display: flex; align-items: center; gap: .5rem; }
  .header-actions .btn { padding: .3rem .7rem; font-size: .8rem; }
  .theme-toggle {
    background: var(--code-bg); border: 1px solid var(--border);
    color: var(--text); padding: .3rem .7rem; border-radius: 6px;
    cursor: pointer; font-size: .8rem;
  }
  .btn-logout {
    background: var(--danger); border: none; color: #fff;
    padding: .3rem .7rem; border-radius: 6px; cursor: pointer; font-size: .8rem;
  }
  .btn-logout:hover { background: var(--danger-hover); }
  .btn-llama {
    background: var(--code-bg); border: 1px solid var(--border);
    color: var(--text); padding: .3rem .7rem; border-radius: 6px;
    cursor: pointer; font-size: .8rem; text-decoration: none;
  }
  .btn-llama:hover { opacity: .8; }
  .container { max-width: 800px; margin: 2rem auto; padding: 0 1rem; }
  .card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 1.5rem; margin-bottom: 1.5rem;
    box-shadow: 0 1px 3px var(--shadow);
  }
  .card h2 { font-size: 1.1rem; margin-bottom: 1rem; }
  .status-row {
    display: flex; align-items: center; gap: .75rem; margin-bottom: 1rem;
  }
  .indicator { width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }
  .indicator.active { background: var(--success); }
  .indicator.inactive { background: var(--danger); }
  .label { color: var(--text-secondary); font-size: .875rem; }
  .value { font-weight: 500; }
  .btn-group { display: flex; gap: .5rem; flex-wrap: wrap; }
  .btn {
    padding: .5rem 1.25rem; border: none; border-radius: 6px;
    font-size: .875rem; font-weight: 500; cursor: pointer; transition: background .15s;
  }
  .btn:disabled { opacity: .5; cursor: not-allowed; }
  .btn-start { background: var(--success); color: #fff; }
  .btn-start:hover:not(:disabled) { filter: brightness(1.1); }
  .btn-stop { background: var(--danger); color: #fff; }
  .btn-stop:hover:not(:disabled) { background: var(--danger-hover); }
  .btn-primary { background: var(--accent); color: #fff; }
  .btn-primary:hover:not(:disabled) { background: var(--accent-hover); }
  select {
    width: 100%; padding: .5rem .75rem; border: 1px solid var(--border);
    border-radius: 6px; background: var(--select-bg); color: var(--text);
    font-size: .875rem; margin-bottom: .75rem;
  }
  input {
    width: 100%; padding: .5rem .75rem; margin-bottom: .75rem;
    border: 1px solid var(--border); border-radius: 6px;
    background: var(--input-bg); color: var(--text); font-size: .875rem;
  }
  input:focus { outline: 2px solid var(--accent); outline-offset: -1px; }
  .current-model {
    font-family: ui-monospace, "SF Mono", monospace; font-size: .8125rem;
    background: var(--code-bg); padding: .5rem .75rem; border-radius: 6px;
    word-break: break-all; margin-bottom: .75rem;
  }
  .msg {
    padding: .75rem; border-radius: 6px; margin-top: 1rem;
    font-size: .875rem; display: none;
  }
  .msg.success { display: block; background: #166534; color: #bbf7d0; }
  .msg.error { display: block; background: #7f1d1d; color: #fecaca; }
  [data-theme="dark"] .msg.success { background: #14532d; color: #bbf7d0; }
  [data-theme="dark"] .msg.error { background: #7f1d1d; color: #fecaca; }
  .spinner { display: none; }
  .spinner.show {
    display: inline-block; width: 14px; height: 14px;
    border: 2px solid var(--border); border-top-color: var(--accent);
    border-radius: 50%; animation: spin .6s linear infinite;
    vertical-align: middle; margin-left: .5rem;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .nav { display: flex; gap: .25rem; margin-left: 2rem; }
  .nav a {
    padding: .3rem .75rem; border-radius: 6px; text-decoration: none;
    color: var(--text-secondary); font-size: .875rem; font-weight: 500;
  }
  .nav a:hover { background: var(--code-bg); }
  .nav a.active { background: var(--accent); color: #fff; }
</style>
</head>
<body>
<div class="header">
  <div style="display:flex;align-items:center">
    <h1>Llama-Dash</h1>
    <nav class="nav">
      <a href="/" class="{{ 'active' if active_page == 'dashboard' else '' }}">Dashboard</a>
      <a href="/args" class="{{ 'active' if active_page == 'args' else '' }}">Server Arguments</a>
      <a href="/admin" class="{{ 'active' if active_page == 'admin' else '' }}">Administration</a>
    </nav>
  </div>
    <div class="header-actions">
    <a href="{{ llama_ui_url }}" target="_blank" rel="noopener noreferrer" class="btn btn-llama">llama.cpp UI</a>
    <button class="theme-toggle" onclick="toggleTheme()" id="themeBtn">Dark Mode</button>
    <button class="btn-logout" onclick="logout()">Logout</button>
  </div>
</div>
<div class="container">

  <!-- Server Health -->
  <div class="card">
    <h2>Server Health</h2>
    <div class="status-row" id="healthRow">
      <span class="indicator" id="healthIndicator"></span>
      <span class="value" id="healthText">checking...</span>
    </div>
    <div id="metricsGrid" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:.75rem;margin-top:.75rem"></div>
  </div>

  <!-- Status -->
  <div class="card">
    <h2>Service Status</h2>
    <div class="status-row">
      <span class="indicator {{ 'active' if status.active else 'inactive' }}" id="indicator"></span>
      <span class="value" id="statusText">{{ status.status }}</span>
      <span class="label" id="substateText">({{ status.substate }})</span>
    </div>
    <div class="btn-group">
      <button class="btn btn-start" id="startBtn" {{ 'disabled' if status.active else '' }} onclick="action('start')">Start</button>
      <button class="btn btn-stop" id="stopBtn" {{ 'disabled' if not status.active else '' }} onclick="action('stop')">Stop</button>
      <button class="btn btn-primary" onclick="action('restart')">Restart</button>
      <span class="spinner" id="spinner"></span>
    </div>
    <div class="msg" id="msg"></div>
  </div>

  <!-- Logs -->
  <div class="card">
    <h2>Startup Log</h2>
    <pre id="logOutput" style="background:var(--code-bg);border:1px solid var(--border);border-radius:6px;padding:.75rem;font-size:.75rem;line-height:1.4;max-height:400px;overflow-y:auto;white-space:pre-wrap;word-break:break-all;color:var(--text);margin-bottom:.75rem"></pre>
    <button class="btn btn-primary" onclick="loadLogs()">Refresh Logs</button>
    <span class="spinner" id="logSpinner"></span>
  </div>

</div>

<script>
function getCookie(name) {
  const v = document.cookie.match('(^|; )' + name + '=([^;]*)');
  return v ? decodeURIComponent(v[2]) : null;
}
function setCookie(name, val) {
  document.cookie = name + '=' + encodeURIComponent(val) + ';path=/;max-age=31536000';
}
(function() {
  const theme = getCookie('theme');
  if (theme === 'dark') {
    document.documentElement.setAttribute('data-theme', 'dark');
    document.getElementById('themeBtn').textContent = 'Light Mode';
  }
})();
function toggleTheme() {
  const html = document.documentElement;
  const btn = document.getElementById('themeBtn');
  if (html.getAttribute('data-theme') === 'dark') {
    html.removeAttribute('data-theme'); setCookie('theme', 'light');
    btn.textContent = 'Dark Mode';
  } else {
    html.setAttribute('data-theme', 'dark'); setCookie('theme', 'dark');
    btn.textContent = 'Light Mode';
  }
}
function showMsg(el, text, type) {
  el.textContent = text; el.className = 'msg ' + type;
  if (type) setTimeout(() => { el.className = 'msg'; }, 5000);
}
function setBusy(active) {
  document.getElementById('spinner').className = 'spinner' + (active ? ' show' : '');
  document.getElementById('startBtn').disabled = active;
  document.getElementById('stopBtn').disabled = active;
}
async function action(cmd) {
  const msg = document.getElementById('msg');
  setBusy(true); showMsg(msg, '', '');
  try {
    const r = await fetch('/api/' + cmd, { method: 'POST' });
    const data = await r.json();
    if (data.ok) { showMsg(msg, data.message || 'OK', 'success'); await refreshStatus(); await loadLogs(); await loadHealth(); await loadMetrics(); }
    else { showMsg(msg, data.error || 'Failed', 'error'); }
  } catch (e) { showMsg(msg, 'Network error', 'error'); }
  setBusy(false);
}
async function refreshStatus() {
  try {
    const r = await fetch('/api/status');
    const data = await r.json();
    document.getElementById('indicator').className = 'indicator ' + (data.active ? 'active' : 'inactive');
    document.getElementById('statusText').textContent = data.status;
    document.getElementById('substateText').textContent = '(' + data.substate + ')';
    document.getElementById('startBtn').disabled = data.active;
    document.getElementById('stopBtn').disabled = !data.active;
  } catch (_) {}
}
async function loadLogs() {
  const el = document.getElementById('logOutput');
  const spinner = document.getElementById('logSpinner');
  spinner.className = 'spinner show';
  try {
    const r = await fetch('/api/logs');
    const data = await r.json();
    el.textContent = data.logs || 'No logs available';
  } catch (_) { el.textContent = 'Failed to fetch logs'; }
  spinner.className = 'spinner';
}
async function loadHealth() {
  const indicator = document.getElementById('healthIndicator');
  const text = document.getElementById('healthText');
  try {
    const r = await fetch('/api/llama-health');
    const d = await r.json();
    if (d.ok) {
      indicator.className = 'indicator active';
      text.textContent = d.status;
    } else {
      indicator.className = 'indicator inactive';
      text.textContent = 'unreachable';
    }
  } catch (_) {
    indicator.className = 'indicator inactive';
    text.textContent = 'error';
  }
}
async function loadMetrics() {
  const grid = document.getElementById('metricsGrid');
  try {
    const r = await fetch('/api/llama-metrics');
    const d = await r.json();
    if (!d.ok || !d.metrics) { grid.innerHTML = ''; return; }
    const m = d.metrics;
    const items = [
      { label: 'Prompt Tok/s',  val: m.prompt_tokens_seconds },
      { label: 'Gen Tok/s',     val: m.predicted_tokens_seconds },
      { label: 'Prompt Tokens', val: m.prompt_tokens_total },
      { label: 'Generated Tok', val: m.tokens_predicted_total },
      { label: 'Processing',    val: m.requests_processing },
      { label: 'Deferred',      val: m.requests_deferred },
    ].filter(x => x.val !== undefined);
    grid.innerHTML = items.map(x => `
      <div style="background:var(--code-bg);border-radius:6px;padding:.6rem;text-align:center">
        <div style="font-size:1.1rem;font-weight:600">${typeof x.val === 'number' ? x.val.toFixed(x.val < 10 ? 2 : 1) : x.val}</div>
        <div style="font-size:.7rem;color:var(--text-secondary);margin-top:.15rem">${x.label}</div>
      </div>
    `).join('');
  } catch (_) { grid.innerHTML = ''; }
}
async function changePassword(e) {
  e.preventDefault();
  const cur = document.getElementById('curPass').value;
  const nw = document.getElementById('newPass').value;
  const cf = document.getElementById('confPass').value;
  const msg = document.getElementById('pwMsg');
  const spinner = document.getElementById('pwSpinner');
  if (nw !== cf) { showMsg(msg, 'Passwords do not match', 'error'); return false; }
  if (nw.length < 4) { showMsg(msg, 'Password must be at least 4 characters', 'error'); return false; }
  spinner.className = 'spinner show'; showMsg(msg, '', '');
  try {
    const r = await fetch('/api/password', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ current_password: cur, new_password: nw })
    });
    const data = await r.json();
    if (data.ok) {
      showMsg(msg, 'Password updated', 'success');
      document.getElementById('curPass').value = '';
      document.getElementById('newPass').value = '';
      document.getElementById('confPass').value = '';
    } else { showMsg(msg, data.error || 'Failed', 'error'); }
  } catch (e) { showMsg(msg, 'Network error', 'error'); }
  spinner.className = 'spinner';
  return false;
}

async function logout() {
  await fetch('/api/logout', { method: 'POST' });
  window.location.href = '/login';
}
(function() {
  refreshStatus();
  loadLogs();
  loadHealth();
  loadMetrics();
})();
</script>
</body>
</html>
"""


ADMIN_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Llama-Dash — Administration</title>
<style>
  :root {
    --bg: #f5f5f5; --card: #fff; --text: #1a1a1a; --text-secondary: #666;
    --border: #ddd; --accent: #2563eb; --accent-hover: #1d4ed8;
    --success: #16a34a; --danger: #dc2626; --danger-hover: #b91c1c;
    --shadow: rgba(0,0,0,0.08); --code-bg: #f0f0f0; --input-bg: #fff;
  }
  [data-theme="dark"] {
    --bg: #111827; --card: #1f2937; --text: #f3f4f6;
    --text-secondary: #9ca3af; --border: #374151; --accent: #3b82f6;
    --accent-hover: #60a5fa; --success: #22c55e; --danger: #ef4444;
    --danger-hover: #f87171; --shadow: rgba(0,0,0,0.3);
    --code-bg: #374151; --input-bg: #374151;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg); color: var(--text); min-height: 100vh;
    transition: background .2s, color .2s;
  }
  .header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 1rem 2rem; background: var(--card);
    border-bottom: 1px solid var(--border); box-shadow: 0 1px 3px var(--shadow);
  }
  .header h1 { font-size: 1.25rem; font-weight: 600; }
  .header-actions { display: flex; align-items: center; gap: .5rem; }
  .nav { display: flex; gap: .25rem; margin-left: 2rem; }
  .nav a {
    padding: .3rem .75rem; border-radius: 6px; text-decoration: none;
    color: var(--text-secondary); font-size: .875rem; font-weight: 500;
  }
  .nav a:hover { background: var(--code-bg); }
  .nav a.active { background: var(--accent); color: #fff; }
  .theme-toggle {
    background: var(--code-bg); border: 1px solid var(--border);
    color: var(--text); padding: .3rem .7rem; border-radius: 6px;
    cursor: pointer; font-size: .8rem;
  }
  .btn-logout {
    background: var(--danger); border: none; color: #fff;
    padding: .3rem .7rem; border-radius: 6px; cursor: pointer; font-size: .8rem;
  }
  .btn-logout:hover { background: var(--danger-hover); }
  .btn-llama {
    background: var(--code-bg); border: 1px solid var(--border);
    color: var(--text); padding: .3rem .7rem; border-radius: 6px;
    cursor: pointer; font-size: .8rem; text-decoration: none;
  }
  .btn-llama:hover { opacity: .8; }
  .container { max-width: 800px; margin: 2rem auto; padding: 0 1rem; }
  .card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 1.5rem; margin-bottom: 1.5rem;
    box-shadow: 0 1px 3px var(--shadow);
  }
  .card h2 { font-size: 1.1rem; margin-bottom: 1rem; }
  input {
    width: 100%; padding: .5rem .75rem; margin-bottom: .75rem;
    border: 1px solid var(--border); border-radius: 6px;
    background: var(--input-bg); color: var(--text); font-size: .875rem;
  }
  input:focus { outline: 2px solid var(--accent); outline-offset: -1px; }
  .btn {
    padding: .5rem 1.25rem; border: none; border-radius: 6px;
    font-size: .875rem; font-weight: 500; cursor: pointer; transition: background .15s;
  }
  .btn-primary { background: var(--accent); color: #fff; }
  .btn-primary:hover:not(:disabled) { background: var(--accent-hover); }
  .msg {
    padding: .75rem; border-radius: 6px; margin-top: 1rem;
    font-size: .875rem; display: none;
  }
  .msg.success { display: block; background: #166534; color: #bbf7d0; }
  .msg.error { display: block; background: #7f1d1d; color: #fecaca; }
  [data-theme="dark"] .msg.success { background: #14532d; color: #bbf7d0; }
  [data-theme="dark"] .msg.error { background: #7f1d1d; color: #fecaca; }
  .spinner { display: none; }
  .spinner.show {
    display: inline-block; width: 14px; height: 14px;
    border: 2px solid var(--border); border-top-color: var(--accent);
    border-radius: 50%; animation: spin .6s linear infinite;
    vertical-align: middle; margin-left: .5rem;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<div class="header">
  <div style="display:flex;align-items:center">
    <h1>Llama-Dash</h1>
    <nav class="nav">
      <a href="/">Dashboard</a>
      <a href="/args">Server Arguments</a>
      <a href="/admin" class="active">Administration</a>
    </nav>
  </div>
    <div class="header-actions">
    <a href="{{ llama_ui_url }}" target="_blank" rel="noopener noreferrer" class="btn btn-llama">llama.cpp UI</a>
    <button class="theme-toggle" onclick="toggleTheme()" id="themeBtn">Dark Mode</button>
    <button class="btn-logout" onclick="logout()">Logout</button>
  </div>
</div>
<div class="container">

  <!-- Password -->
  <div class="card">
    <h2>Change Password</h2>
    <form onsubmit="return changePassword(event)">
      <input type="password" id="curPass" placeholder="Current password" required>
      <input type="password" id="newPass" placeholder="New password" required minlength="4">
      <input type="password" id="confPass" placeholder="Confirm new password" required minlength="4">
      <button type="submit" class="btn btn-primary">Update Password</button>
      <span class="spinner" id="pwSpinner"></span>
    </form>
    <div class="msg" id="pwMsg"></div>
  </div>

</div>

<script>
function getCookie(name) {
  const v = document.cookie.match('(^|; )' + name + '=([^;]*)');
  return v ? decodeURIComponent(v[2]) : null;
}
function setCookie(name, val) {
  document.cookie = name + '=' + encodeURIComponent(val) + ';path=/;max-age=31536000';
}
(function() {
  const theme = getCookie('theme');
  if (theme === 'dark') {
    document.documentElement.setAttribute('data-theme', 'dark');
    document.getElementById('themeBtn').textContent = 'Light Mode';
  }
})();
function toggleTheme() {
  const html = document.documentElement;
  const btn = document.getElementById('themeBtn');
  if (html.getAttribute('data-theme') === 'dark') {
    html.removeAttribute('data-theme'); setCookie('theme', 'light');
    btn.textContent = 'Dark Mode';
  } else {
    html.setAttribute('data-theme', 'dark'); setCookie('theme', 'dark');
    btn.textContent = 'Light Mode';
  }
}
function showMsg(el, text, type) {
  el.textContent = text; el.className = 'msg ' + type;
  if (type) setTimeout(() => { el.className = 'msg'; }, 5000);
}
async function changePassword(e) {
  e.preventDefault();
  const cur = document.getElementById('curPass').value;
  const nw = document.getElementById('newPass').value;
  const cf = document.getElementById('confPass').value;
  const msg = document.getElementById('pwMsg');
  const spinner = document.getElementById('pwSpinner');
  if (nw !== cf) { showMsg(msg, 'Passwords do not match', 'error'); return false; }
  if (nw.length < 4) { showMsg(msg, 'Password must be at least 4 characters', 'error'); return false; }
  spinner.className = 'spinner show'; showMsg(msg, '', '');
  try {
    const r = await fetch('/api/password', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ current_password: cur, new_password: nw })
    });
    const data = await r.json();
    if (data.ok) {
      showMsg(msg, 'Password updated', 'success');
      document.getElementById('curPass').value = '';
      document.getElementById('newPass').value = '';
      document.getElementById('confPass').value = '';
    } else { showMsg(msg, data.error || 'Failed', 'error'); }
  } catch (e) { showMsg(msg, 'Network error', 'error'); }
  spinner.className = 'spinner';
  return false;
}
async function logout() {
  await fetch('/api/logout', { method: 'POST' });
  window.location.href = '/login';
}
</script>
</body>
</html>
"""


ARGS_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Llama-Dash — Arguments</title>
<style>
  :root {
    --bg: #f5f5f5; --card: #fff; --text: #1a1a1a; --text-secondary: #666;
    --border: #ddd; --accent: #2563eb; --accent-hover: #1d4ed8;
    --success: #16a34a; --danger: #dc2626; --danger-hover: #b91c1c;
    --warning: #d97706; --shadow: rgba(0,0,0,0.08); --code-bg: #f0f0f0;
    --select-bg: #fff; --input-bg: #fff;
  }
  [data-theme="dark"] {
    --bg: #111827; --card: #1f2937; --text: #f3f4f6;
    --text-secondary: #9ca3af; --border: #374151; --accent: #3b82f6;
    --accent-hover: #60a5fa; --success: #22c55e; --danger: #ef4444;
    --danger-hover: #f87171; --warning: #f59e0b; --shadow: rgba(0,0,0,0.3);
    --code-bg: #374151; --select-bg: #374151; --input-bg: #374151;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg); color: var(--text); min-height: 100vh;
    transition: background .2s, color .2s;
  }
  .header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 1rem 2rem; background: var(--card);
    border-bottom: 1px solid var(--border); box-shadow: 0 1px 3px var(--shadow);
  }
  .header h1 { font-size: 1.25rem; font-weight: 600; }
  .header-actions { display: flex; align-items: center; gap: .5rem; }
  .header-actions .btn { padding: .3rem .7rem; font-size: .8rem; }
  .theme-toggle {
    background: var(--code-bg); border: 1px solid var(--border);
    color: var(--text); padding: .3rem .7rem; border-radius: 6px;
    cursor: pointer; font-size: .8rem;
  }
  .btn-logout {
    background: var(--danger); border: none; color: #fff;
    padding: .3rem .7rem; border-radius: 6px; cursor: pointer; font-size: .8rem;
  }
  .btn-logout:hover { background: var(--danger-hover); }
  .btn-llama {
    background: var(--code-bg); border: 1px solid var(--border);
    color: var(--text); padding: .3rem .7rem; border-radius: 6px;
    cursor: pointer; font-size: .8rem; text-decoration: none;
  }
  .btn-llama:hover { opacity: .8; }
  .container { max-width: 800px; margin: 2rem auto; padding: 0 1rem; }
  .card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 1.5rem; margin-bottom: 1.5rem;
    box-shadow: 0 1px 3px var(--shadow);
  }
  .card h2 { font-size: 1.1rem; margin-bottom: 1rem; }
  .btn-group { display: flex; gap: .5rem; flex-wrap: wrap; }
  .btn {
    padding: .5rem 1.25rem; border: none; border-radius: 6px;
    font-size: .875rem; font-weight: 500; cursor: pointer; transition: background .15s;
  }
  .btn:disabled { opacity: .5; cursor: not-allowed; }
  .btn-primary { background: var(--accent); color: #fff; }
  .btn-primary:hover:not(:disabled) { background: var(--accent-hover); }
  .btn-start { background: var(--success); color: #fff; }
  .btn-start:hover:not(:disabled) { filter: brightness(1.1); }
  input {
    width: 100%; padding: .5rem .75rem;
    border: 1px solid var(--border); border-radius: 6px;
    background: var(--input-bg); color: var(--text); font-size: .875rem;
  }
  input:focus { outline: 2px solid var(--accent); outline-offset: -1px; }
  .msg {
    padding: .75rem; border-radius: 6px; margin-top: 1rem;
    font-size: .875rem; display: none;
  }
  .msg.success { display: block; background: #166534; color: #bbf7d0; }
  .msg.error { display: block; background: #7f1d1d; color: #fecaca; }
  [data-theme="dark"] .msg.success { background: #14532d; color: #bbf7d0; }
  [data-theme="dark"] .msg.error { background: #7f1d1d; color: #fecaca; }
  .spinner { display: none; }
  .spinner.show {
    display: inline-block; width: 14px; height: 14px;
    border: 2px solid var(--border); border-top-color: var(--accent);
    border-radius: 50%; animation: spin .6s linear infinite;
    vertical-align: middle; margin-left: .5rem;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .nav { display: flex; gap: .25rem; margin-left: 2rem; }
  .nav a {
    padding: .3rem .75rem; border-radius: 6px; text-decoration: none;
    color: var(--text-secondary); font-size: .875rem; font-weight: 500;
  }
  .nav a:hover { background: var(--code-bg); }
  .nav a.active { background: var(--accent); color: #fff; }
</style>
</head>
<body>
<div class="header">
  <div style="display:flex;align-items:center">
    <h1>Llama-Dash</h1>
    <nav class="nav">
      <a href="/">Dashboard</a>
      <a href="/args" class="active">Server Arguments</a>
      <a href="/admin">Administration</a>
    </nav>
  </div>
  <div class="header-actions">
    <a href="{{ llama_ui_url }}" target="_blank" rel="noopener noreferrer" class="btn btn-llama">llama.cpp UI</a>
    <button class="theme-toggle" onclick="toggleTheme()" id="themeBtn">Dark Mode</button>
    <button class="btn-logout" onclick="logout()">Logout</button>
  </div>
</div>
<div class="container">

  <!-- Model -->
  <div class="card">
    <h2>Model</h2>
    <div class="current-model" id="currentModel" style="font-family:ui-monospace,'SF Mono',monospace;font-size:.8125rem;background:var(--code-bg);padding:.5rem .75rem;border-radius:6px;word-break:break-all;margin-bottom:.75rem">{{ current_model or 'not set' }}</div>
    <form id="modelForm" onsubmit="return changeModel(event)">
      <select name="model" id="modelSelect" style="width:100%;padding:.5rem .75rem;border:1px solid var(--border);border-radius:6px;background:var(--select-bg);color:var(--text);font-size:.875rem;margin-bottom:.75rem">
        {% for m in models %}
        <option value="{{ m.path }}" {{ 'selected' if m.path == current_model else '' }}>{{ m.name }}</option>
        {% endfor %}
      </select>
      <button type="submit" class="btn btn-primary">Switch Model</button>
      <span class="spinner" id="modelSpinner"></span>
    </form>
    <div class="msg" id="modelMsg"></div>
    <div style="margin-top:.5rem;font-size:.8125rem;color:var(--text-secondary)">Restart the service from the Dashboard for the change to take effect.</div>
  </div>

  <div class="card">
    <h2>Server Arguments</h2>
    <div style="margin-bottom:.75rem;font-size:.8125rem;color:var(--text-secondary)">
      All CLI flags passed to llama-server. The <code>-m</code> flag is set in the Model card above.
    </div>
    <table id="argsTable" style="width:100%;border-collapse:collapse">
      <thead>
        <tr style="font-size:.8125rem;color:var(--text-secondary);text-align:left">
          <th style="width:45%;padding-bottom:.5rem">Flag</th>
          <th style="width:45%;padding-bottom:.5rem">Value</th>
          <th style="width:10%;padding-bottom:.5rem"></th>
        </tr>
      </thead>
      <tbody id="argsBody"></tbody>
    </table>
    <div style="display:flex;gap:.5rem;margin-top:.75rem">
      <button class="btn btn-primary" onclick="addArgRow()">Add Argument</button>
      <button class="btn btn-start" onclick="saveArgs()">Save Changes</button>
      <span class="spinner" id="argsSpinner"></span>
    </div>
    <div class="msg" id="argsMsg"></div>
  </div>

  <datalist id="flagSuggestions">
    <option value="--threads"><option value="--threads-batch">
    <option value="--n-gpu-layers"><option value="--main-gpu">
    <option value="--split-mode"><option value="--tensor-split">
    <option value="--ctx-size"><option value="--batch-size"><option value="--ubatch-size">
    <option value="--flash-attn"><option value="--mlock"><option value="--no-mmap">
    <option value="--host"><option value="--port">
    <option value="--spec-type"><option value="--spec-draft-n-max"><option value="--spec-draft-n-min">
    <option value="--spec-draft-model"><option value="--spec-draft-ngl"><option value="--spec-draft-p-split">
    <option value="--spec-draft-threads"><option value="--model-draft">
    <option value="--temp"><option value="--top-k"><option value="--top-p"><option value="--min-p">
    <option value="--seed"><option value="--repeat-penalty">
    <option value="--parallel"><option value="--cont-batching"><option value="--timeout">
    <option value="--metrics"><option value="--api-key">
    <option value="--cache-type-k"><option value="--cache-type-v">
    <option value="--numa"><option value="--device">
    <option value="--rope-scaling"><option value="--rope-scale">
    <option value="--mlock"><option value="--no-mmap">
    <option value="--embedding"><option value="--rerank">
    <option value="--model-url"><option value="--offline">
    <option value="--jinja"><option value="--no-jinja">
    <option value="--chat-template"><option value="--grammar"><option value="--grammar-file">
    <option value="--json-schema"><option value="--reasoning">
    <option value="--reasoning-budget"><option value="--lora">
    <option value="--control-vector"><option value="--mmproj">
    <option value="--alias"><option value="--tags">
    <option value="--prio"><option value="--cpu-mask">
    <option value="--cache-reuse"><option value="--slot-prompt-similarity">
    <option value="--cache-ram"><option value="--sleep-idle-seconds">
    <option value="--prefill-assistant"><option value="--no-prefill-assistant">
    <option value="--embeddings"><option value="--context-shift">
    <option value="--model-draft"><option value="--spec-draft-n-max">
    <option value="--spec-type">
  </datalist>

</div>

<script>
function getCookie(name) {
  const v = document.cookie.match('(^|; )' + name + '=([^;]*)');
  return v ? decodeURIComponent(v[2]) : null;
}
function setCookie(name, val) {
  document.cookie = name + '=' + encodeURIComponent(val) + ';path=/;max-age=31536000';
}
(function() {
  const theme = getCookie('theme');
  if (theme === 'dark') {
    document.documentElement.setAttribute('data-theme', 'dark');
    document.getElementById('themeBtn').textContent = 'Light Mode';
  }
})();
function toggleTheme() {
  const html = document.documentElement;
  const btn = document.getElementById('themeBtn');
  if (html.getAttribute('data-theme') === 'dark') {
    html.removeAttribute('data-theme'); setCookie('theme', 'light');
    btn.textContent = 'Dark Mode';
  } else {
    html.setAttribute('data-theme', 'dark'); setCookie('theme', 'dark');
    btn.textContent = 'Light Mode';
  }
}
function showMsg(el, text, type) {
  el.textContent = text; el.className = 'msg ' + type;
  if (type) setTimeout(() => { el.className = 'msg'; }, 5000);
}

// ── Model Switcher ───────────────────────────────────────────
async function changeModel(e) {
  e.preventDefault();
  const select = document.getElementById('modelSelect');
  const path = select.value;
  const msg = document.getElementById('modelMsg');
  const spinner = document.getElementById('modelSpinner');
  spinner.className = 'spinner show'; showMsg(msg, '', '');
  try {
    const r = await fetch('/api/model', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_path: path })
    });
    const data = await r.json();
    if (data.ok) {
      showMsg(msg, data.message || 'Model switched', 'success');
      document.getElementById('currentModel').textContent = path;
    } else { showMsg(msg, data.error || 'Failed', 'error'); }
  } catch (e) { showMsg(msg, 'Network error', 'error'); }
  spinner.className = 'spinner';
  return false;
}

// ── Arguments Editor ─────────────────────────────────────────
async function loadArgs() {
  try {
    const r = await fetch('/api/args');
    const data = await r.json();
    if (!data.ok) return;
    const tbody = document.getElementById('argsBody');
    tbody.innerHTML = '';
    for (const a of data.args) {
      if (a.flag === '-m') continue;
      addArgRow(a.flag, a.value);
    }
  } catch (_) {}
}
function addArgRow(flag, value) {
  const tbody = document.getElementById('argsBody');
  const tr = document.createElement('tr');
  tr.style.borderBottom = '1px solid var(--border)';
  const flagTd = document.createElement('td');
  flagTd.style.padding = '.35rem .25rem';
  const flagInput = document.createElement('input');
  flagInput.name = 'flag';
  flagInput.placeholder = '--flag';
  flagInput.value = flag || '';
  flagInput.setAttribute('list', 'flagSuggestions');
  flagInput.style.margin = '0';
  flagTd.appendChild(flagInput);
  const valTd = document.createElement('td');
  valTd.style.padding = '.35rem .25rem';
  const valInput = document.createElement('input');
  valInput.name = 'value';
  valInput.placeholder = 'value';
  valInput.value = value || '';
  valInput.style.margin = '0';
  valTd.appendChild(valInput);
  const delTd = document.createElement('td');
  delTd.style.padding = '.35rem .25rem';
  delTd.style.textAlign = 'center';
  const delBtn = document.createElement('button');
  delBtn.textContent = 'x';
  delBtn.className = 'btn';
  delBtn.style.cssText = 'background:var(--danger);color:#fff;padding:.2rem .5rem;font-size:.8rem';
  delBtn.type = 'button';
  delBtn.onclick = function(){ tr.remove(); };
  delTd.appendChild(delBtn);
  tr.appendChild(flagTd);
  tr.appendChild(valTd);
  tr.appendChild(delTd);
  tbody.appendChild(tr);
}
async function saveArgs() {
  const msg = document.getElementById('argsMsg');
  const spinner = document.getElementById('argsSpinner');
  spinner.className = 'spinner show'; showMsg(msg, '', '');
  const rows = document.querySelectorAll('#argsBody tr');
  const args = [];
  for (const tr of rows) {
    const flag = tr.querySelector('[name=flag]').value.trim();
    const value = tr.querySelector('[name=value]').value.trim();
    if (!flag) continue;
    args.push({ flag, value });
  }
  try {
    const r = await fetch('/api/args', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ args })
    });
    const data = await r.json();
    if (data.ok) {
      showMsg(msg, data.message || 'Saved', 'success');
      await loadArgs();
    } else {
      showMsg(msg, data.error || 'Failed', 'error');
    }
  } catch (e) { showMsg(msg, 'Network error', 'error'); }
  spinner.className = 'spinner';
}
(function() {
  const raf = window.requestAnimationFrame || function(fn){ setTimeout(fn, 0); };
  raf(function() {
    if (document.getElementById('argsBody')) loadArgs();
  });
})();

async function logout() {
  await fetch('/api/logout', { method: 'POST' });
  window.location.href = '/login';
}
</script>
</body>
</html>
"""


# ── Routes ──────────────────────────────────────────────────────

@app.route("/login")
def login_page():
    if session.get("user"):
        return redirect("/")
    return render_template_string(LOGIN_TEMPLATE)


def _llama_ui_url():
    host = request.host.split(":")[0]
    return f"http://{host}:8080"


@app.route("/")
@login_required
def index():
    return render_template_string(
        DASHBOARD_TEMPLATE,
        active_page="dashboard",
        status=get_service_status(),
        models=get_models(),
        current_model=get_current_model(),
        llama_ui_url=_llama_ui_url(),
    )


@app.route("/admin")
@login_required
def admin_page():
    return render_template_string(ADMIN_TEMPLATE, llama_ui_url=_llama_ui_url())


@app.route("/args")
@login_required
def args_page():
    return render_template_string(
        ARGS_TEMPLATE,
        llama_ui_url=_llama_ui_url(),
        models=get_models(),
        current_model=get_current_model(),
    )


# ── Auth API ────────────────────────────────────────────────────

@app.route("/api/login", methods=["POST"])
def api_login():
    data = json.loads(request.data)
    pw = data.get("password", "")
    cfg = _read_config()
    if check_password_hash(cfg["password_hash"], pw):
        session["user"] = "admin"
        session.permanent = True
        return jsonify(ok=True)
    return jsonify(ok=False, error="Invalid password")


@app.route("/api/logout", methods=["POST"])
def api_logout():
    session.pop("user", None)
    return jsonify(ok=True)


@app.route("/api/password", methods=["PUT"])
@login_required
def api_password():
    data = json.loads(request.data)
    cur = data.get("current_password", "")
    new = data.get("new_password", "")
    if len(new) < 4:
        return jsonify(ok=False, error="Password must be at least 4 characters")
    cfg = _read_config()
    if not check_password_hash(cfg["password_hash"], cur):
        return jsonify(ok=False, error="Current password is incorrect")
    cfg["password_hash"] = generate_password_hash(new)
    _write_config(cfg)
    return jsonify(ok=True, message="Password updated")


# ── Service API ─────────────────────────────────────────────────

@app.route("/api/status")
@login_required
def api_status():
    return jsonify(get_service_status())


def _run_systemctl(action):
    subprocess.run(["systemctl", action, SERVICE_NAME], capture_output=True, text=True)
    time.sleep(2 if action == "stop" else 1)
    return get_service_status()


@app.route("/api/start", methods=["POST"])
@login_required
def api_start():
    if get_service_status()["active"]:
        return jsonify(ok=False, error="Service is already running")
    _run_systemctl("start")
    s = get_service_status()
    return jsonify(ok=s["active"], message="Service started" if s["active"] else "Failed to start")


@app.route("/api/stop", methods=["POST"])
@login_required
def api_stop():
    if not get_service_status()["active"]:
        return jsonify(ok=False, error="Service is not running")
    _run_systemctl("stop")
    s = get_service_status()
    return jsonify(ok=not s["active"], message="Service stopped" if not s["active"] else "Failed to stop")


@app.route("/api/restart", methods=["POST"])
@login_required
def api_restart():
    _run_systemctl("restart")
    s = get_service_status()
    return jsonify(ok=s["active"], message="Service restarted" if s["active"] else "Failed to restart")


@app.route("/api/model", methods=["PUT"])
@login_required
def api_set_model():
    data = json.loads(request.data)
    model_path = data.get("model_path")
    if not model_path:
        return jsonify(ok=False, error="No model_path provided")
    if not os.path.isfile(model_path):
        return jsonify(ok=False, error="Model file not found")

    conf = _read_conf()
    conf["model"] = model_path
    _write_conf(conf)

    return jsonify(ok=True, message=f"Model set to {model_path}. Restart the service to apply.")


# ── Args API ────────────────────────────────────────────────────

@app.route("/api/args", methods=["GET"])
@login_required
def api_get_args():
    conf = _read_conf()
    args = _parse_opts(conf["opts"])
    return jsonify(ok=True, args=args)


@app.route("/api/args", methods=["PUT"])
@login_required
def api_set_args():
    data = json.loads(request.data)
    args = data.get("args", [])
    if not isinstance(args, list):
        return jsonify(ok=False, error="args must be a list")
    for a in args:
        if not isinstance(a, dict) or "flag" not in a or "value" not in a:
            return jsonify(ok=False, error="Each arg must have 'flag' and 'value' keys")
        if a["flag"] and not a["flag"].startswith("-"):
            return jsonify(ok=False, error=f"Invalid flag: {a['flag']}")

    conf = _read_conf()
    was_active = get_service_status()["active"]
    if was_active:
        _run_systemctl("stop")
        time.sleep(1)

    # Build new opts from submitted args (exclude -m — it's managed separately)
    filtered = [a for a in args if a["flag"] and a["flag"] != "-m"]
    conf["opts"] = _format_opts(filtered)
    _write_conf(conf)

    if was_active:
        _run_systemctl("start")
        time.sleep(1)
        s = get_service_status()
        if not s["active"]:
            return jsonify(ok=False, error="Service failed to start with new args")

    return jsonify(ok=True, message="Arguments updated" + (" and service restarted" if was_active else ""))


@app.route("/api/logs")
@login_required
def api_logs():
    r = subprocess.run(
        ["journalctl", "-u", SERVICE_NAME, "--no-pager", "-n", "50", "--output=short-precise"],
        capture_output=True, text=True
    )
    logs = r.stdout or r.stderr or "No output from journalctl"
    return jsonify(ok=True, logs=logs)


@app.route("/api/llama-health")
@login_required
def api_llama_health():
    import urllib.request
    try:
        r = urllib.request.urlopen("http://localhost:8080/health", timeout=5)
        data = json.loads(r.read().decode())
        return jsonify(ok=True, status=data.get("status", "unknown"))
    except Exception as e:
        return jsonify(ok=False, error=str(e))


@app.route("/api/llama-metrics")
@login_required
def api_llama_metrics():
    import re, urllib.request
    try:
        r = urllib.request.urlopen("http://localhost:8080/metrics", timeout=5)
        text = r.read().decode()
        metrics = {}
        for m in re.finditer(r'^(\w[\w:]*) ([\d.]+)', text, re.MULTILINE):
            name = m.group(1)
            val = m.group(2)
            val = float(val) if "." in val else int(val)
            short = name.split(":", 1)[-1] if ":" in name else name
            metrics[short] = val
        return jsonify(ok=True, metrics=metrics)
    except Exception as e:
        return jsonify(ok=False, error=str(e))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4043,
            ssl_context=("/opt/llama-dashboard/cert.pem", "/opt/llama-dashboard/key.pem"))
