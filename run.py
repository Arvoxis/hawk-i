#!/usr/bin/env python3
"""
run.py — Entry point for the Hawk-I ground-control-station backend.

    python run.py                      # 0.0.0.0:8000, auto-reload on
    python run.py --port 9000          # different port
    python run.py --no-reload          # production-ish: no file watcher

Why the sys.path juggling: modules inside backend/ import each other by bare
name (``import config``, ``from database import ...``) rather than as a
package.  Both the repo root and backend/ therefore have to be importable —
the root so ``backend.main`` and ``edge.multi_query_yoloworld`` resolve, and
backend/ so those bare intra-package imports resolve.
"""

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.join(_ROOT, "backend")

for _path in (_BACKEND, _ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# The WatchFiles reloader spawns a child process that does not inherit sys.path
# mutations, so bake the same two entries into PYTHONPATH for it to pick up.
_existing = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = os.pathsep.join(
    [p for p in [_ROOT, _BACKEND, _existing] if p]
)

import uvicorn  # noqa: E402 — must follow the sys.path setup above

# Windows consoles default to cp1252.  The backend's log lines contain arrows
# and check marks, and the uvicorn child process inherits these streams, so
# force UTF-8 here or startup dies with UnicodeEncodeError on a stock cmd.exe.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # pragma: no cover - non-TTY streams
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Hawk-I GCS backend")
    parser.add_argument("--host", default=os.getenv("BACKEND_BIND_HOST", "0.0.0.0"),
                        help="Interface to bind")
    parser.add_argument("--port", type=int, default=int(os.getenv("GCS_PORT", "8000")),
                        help="Port to listen on")
    parser.add_argument("--no-reload", action="store_true",
                        help="Disable the auto-reloader (recommended for flights)")
    parser.add_argument("--log-level", default="info",
                        choices=["critical", "error", "warning", "info", "debug"])
    args = parser.parse_args()

    uvicorn.run(
        "backend.main:app",
        host=args.host,
        port=args.port,
        reload=not args.no_reload,
        # Watch only backend/ — watching the repo root would retrigger on every
        # frame and report the pipeline writes into data/.
        reload_dirs=[_BACKEND] if not args.no_reload else None,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
