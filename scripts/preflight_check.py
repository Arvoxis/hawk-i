#!/usr/bin/env python3
"""
preflight_check.py — Hawk-I pre-flight connection validator.

Run this on the Jetson OR the GCS laptop before every flight.
Reads GCS_HOST and GCS_PORT from .env in the repo root.

Usage:
    python scripts/preflight_check.py
    python scripts/preflight_check.py --host 172.18.239.242 --port 8000
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Load .env from repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_REPO_ROOT / ".env")
except ImportError:
    print("[WARN] python-dotenv not installed — install with: pip3 install python-dotenv")

# ── Required .env keys ────────────────────────────────────────────────────────
REQUIRED_KEYS = ["GCS_HOST", "GCS_PORT", "DB_HOST", "DB_PORT", "MAVLINK_SERIAL"]

# ── ANSI colours ──────────────────────────────────────────────────────────────
_GREEN  = "\033[0;32m"
_RED    = "\033[0;31m"
_YELLOW = "\033[0;33m"
_RESET  = "\033[0m"

def _ok(msg: str)   -> str: return f"{_GREEN}[OK  ]{_RESET}  {msg}"
def _fail(msg: str) -> str: return f"{_RED}[FAIL]{_RESET}  {msg}"
def _warn(msg: str) -> str: return f"{_YELLOW}[WARN]{_RESET}  {msg}"


# ── Check 1: .env present + required keys set ────────────────────────────────

def check_env() -> bool:
    env_path = _REPO_ROOT / ".env"
    if not env_path.exists():
        print(_fail(f".env not found at {env_path}"))
        print("        Create it: cp .env.example .env  then fill in your IPs")
        return False

    missing = [k for k in REQUIRED_KEYS if not os.getenv(k)]
    if missing:
        print(_fail(f".env present but missing keys: {', '.join(missing)}"))
        return False

    print(_ok(f".env present — all required keys set ({', '.join(REQUIRED_KEYS)})"))
    return True


# ── Check 2: HTTP reachability of GCS backend ────────────────────────────────

async def check_http(host: str, port: int) -> bool:
    try:
        import httpx
        async with httpx.AsyncClient(timeout=4.0) as client:
            resp = await client.get(f"http://{host}:{port}/health")
            if resp.status_code == 200:
                print(_ok(f"GCS reachable at {host}:{port}  (HTTP {resp.status_code})"))
                return True
            else:
                print(_fail(f"GCS returned HTTP {resp.status_code} at {host}:{port}/health"))
                return False
    except ImportError:
        print(_warn("httpx not installed — skipping HTTP check (pip3 install httpx)"))
        return True
    except Exception as exc:
        print(_fail(f"Cannot reach GCS at {host}:{port} — {exc}"))
        print(f"        Check GCS_HOST in .env (current: {host})")
        print(f"        Is the FastAPI backend running? → python run.py")
        return False


# ── Check 3: WebSocket handshake ─────────────────────────────────────────────

async def check_websocket(host: str, port: int) -> bool:
    uri = f"ws://{host}:{port}/ws/drone"
    try:
        import websockets
        async with websockets.connect(uri, open_timeout=4) as ws:
            print(_ok(f"WebSocket handshake OK  ({uri})"))
            return True
    except ImportError:
        print(_warn("websockets not installed — skipping WS check (pip3 install websockets)"))
        return True
    except Exception as exc:
        print(_fail(f"WebSocket connect failed: {uri}"))
        print(f"        Error: {exc}")
        print(f"        Confirm backend has @app.websocket(\"/ws/drone\")")
        return False


# ── Check 4: DB TCP port reachable ───────────────────────────────────────────

async def check_db(host: str, port: int) -> bool:
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=3
        )
        writer.close()
        await writer.wait_closed()
        print(_ok(f"PostgreSQL port open at {host}:{port}"))
        return True
    except Exception as exc:
        print(_fail(f"Cannot reach PostgreSQL at {host}:{port} — {exc}"))
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(host: str, port: int, db_host: str, db_port: int) -> bool:
    print()
    print("═" * 54)
    print("  Hawk-I Pre-Flight Check")
    print(f"  GCS  : {host}:{port}")
    print(f"  DB   : {db_host}:{db_port}")
    print("═" * 54)
    print()

    results = []
    results.append(check_env())
    results.append(await check_http(host, port))
    results.append(await check_websocket(host, port))
    results.append(await check_db(db_host, db_port))

    print()
    print("═" * 54)
    if all(results):
        print(f"{_GREEN}  GO — all checks passed. Safe to fly.{_RESET}")
    else:
        failed = results.count(False)
        print(f"{_RED}  NO-GO — {failed} check(s) failed. Fix before flying.{_RESET}")
    print("═" * 54)
    print()

    return all(results)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Hawk-I pre-flight connection check")
    ap.add_argument("--host", default=os.getenv("GCS_HOST", "172.18.239.242"),
                    help="GCS backend IP")
    ap.add_argument("--port", type=int, default=int(os.getenv("GCS_PORT", "8000")),
                    help="GCS backend port")
    ap.add_argument("--db-host", default=os.getenv("DB_HOST", "localhost"),
                    help="PostgreSQL host")
    ap.add_argument("--db-port", type=int, default=int(os.getenv("DB_PORT", "5432")),
                    help="PostgreSQL port")
    args = ap.parse_args()

    ok = asyncio.run(main(args.host, args.port, args.db_host, args.db_port))
    sys.exit(0 if ok else 1)
