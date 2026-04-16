#!/usr/bin/env bash
# check_connections.sh — Hawk-I pre-flight connection validator
#
# Run this before every flight to catch silent config failures early.
# Works on Linux (Jetson), macOS, and Windows Git Bash / WSL.
#
# Usage:
#   bash scripts/check_connections.sh
#
# All values are read from .env in the repo root. Override on the command line:
#   GCS_HOST=10.0.0.5 bash scripts/check_connections.sh

set -euo pipefail

# ── Load .env ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../.env"

if [[ -f "$ENV_FILE" ]]; then
    # Export only lines that look like KEY=VALUE (skip comments and blanks)
    set -a
    # shellcheck disable=SC1090
    source <(grep -E '^[A-Z_]+=.+' "$ENV_FILE")
    set +a
else
    echo "[WARN] .env not found at $ENV_FILE — using defaults"
fi

# Defaults (if not in .env)
GCS_HOST="${GCS_HOST:-172.18.239.242}"
GCS_PORT="${GCS_PORT:-8000}"
RECEIVER_PORT="${RECEIVER_PORT:-8001}"
OLLAMA_HOST="${OLLAMA_HOST:-localhost}"
OLLAMA_PORT="${OLLAMA_PORT:-11434}"
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"

# ── Helpers ───────────────────────────────────────────────────────────────────
PASS="\033[0;32m✓\033[0m"
FAIL="\033[0;31m✗\033[0m"
WARN="\033[0;33m!\033[0m"

check_http() {
    local label="$1" url="$2"
    local code
    code=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 3 "$url" 2>/dev/null || echo "000")
    if [[ "$code" == "200" ]]; then
        echo -e "  $PASS $label  ($url) → HTTP $code"
    else
        echo -e "  $FAIL $label  ($url) → HTTP $code  (expected 200)"
    fi
}

check_tcp() {
    local label="$1" host="$2" port="$3"
    if timeout 3 bash -c "echo > /dev/tcp/$host/$port" 2>/dev/null; then
        echo -e "  $PASS $label  ($host:$port) → port open"
    else
        echo -e "  $FAIL $label  ($host:$port) → connection refused / timeout"
    fi
}

check_ping() {
    local label="$1" host="$2"
    if ping -c 1 -W 2 "$host" &>/dev/null; then
        echo -e "  $PASS $label  ($host) → reachable"
    else
        echo -e "  $FAIL $label  ($host) → unreachable"
    fi
}

# ── Print config being used ────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Hawk-I Pre-Flight Connection Check"
echo "═══════════════════════════════════════════════════════"
echo "  GCS (laptop)  : $GCS_HOST:$GCS_PORT"
echo "  Receiver port : $RECEIVER_PORT"
echo "  Ollama        : $OLLAMA_HOST:$OLLAMA_PORT"
echo "  PostgreSQL    : $DB_HOST:$DB_PORT"
echo "═══════════════════════════════════════════════════════"
echo ""

# ── 1. GCS backend ────────────────────────────────────────────────────────────
echo "[ 1 ] FastAPI backend"
check_http "GET /health"   "http://$GCS_HOST:$GCS_PORT/health"
check_http "GET /api/session" "http://$GCS_HOST:$GCS_PORT/api/session"
echo ""

# ── 2. WebSocket port (TCP reachability) ──────────────────────────────────────
echo "[ 2 ] WebSocket port"
check_tcp  "ws/drone port" "$GCS_HOST" "$GCS_PORT"
echo ""

# ── 3. Standalone receiver port ───────────────────────────────────────────────
echo "[ 3 ] Standalone receiver (receiver.py)"
# Just check whether the port is free — if something is already bound here it means
# receiver.py is running (or there's a conflict)
if timeout 3 bash -c "echo > /dev/tcp/localhost/$RECEIVER_PORT" 2>/dev/null; then
    echo -e "  $WARN Receiver port $RECEIVER_PORT is already bound — receiver.py may be running"
else
    echo -e "  $PASS Receiver port $RECEIVER_PORT is free"
fi
echo ""

# ── 4. Port conflict check ────────────────────────────────────────────────────
echo "[ 4 ] Port conflict check (FastAPI vs receiver.py)"
if [[ "$GCS_PORT" == "$RECEIVER_PORT" ]]; then
    echo -e "  $FAIL GCS_PORT ($GCS_PORT) == RECEIVER_PORT ($RECEIVER_PORT) — THIS WILL CAUSE A CONFLICT"
    echo "        Set RECEIVER_PORT to a different value in .env (e.g. 8001)"
else
    echo -e "  $PASS Ports are separate: backend=$GCS_PORT, receiver=$RECEIVER_PORT"
fi
echo ""

# ── 5. PostgreSQL ─────────────────────────────────────────────────────────────
echo "[ 5 ] PostgreSQL"
check_tcp "PostgreSQL" "$DB_HOST" "$DB_PORT"
echo ""

# ── 6. Ollama ─────────────────────────────────────────────────────────────────
echo "[ 6 ] Ollama (LLM)"
check_http "GET /api/tags" "http://$OLLAMA_HOST:$OLLAMA_PORT/api/tags"
echo ""

# ── 7. Network reachability ───────────────────────────────────────────────────
echo "[ 7 ] Network reachability"
check_ping "GCS self-ping" "$GCS_HOST"
echo ""

echo "═══════════════════════════════════════════════════════"
echo "  Done. Fix any ✗ items before arming the drone."
echo "═══════════════════════════════════════════════════════"
echo ""
