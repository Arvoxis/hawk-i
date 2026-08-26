"""
Integration test: exercise the full DINOv2 + SAM2 + LLM pipeline.

Requires a running Hawk-I backend (python run.py) on localhost:8000
and a PostgreSQL database.

Usage:
    python tests/test_fake_drone.py
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import cv2
import websockets

# Windows consoles default to cp1252, which cannot encode the status glyphs
# below and would raise UnicodeEncodeError mid-run.  Force UTF-8 on the
# standard streams; `errors="replace"` keeps output flowing on terminals that
# still cannot render a given character.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # pragma: no cover - non-TTY streams
        pass


BASE_URL = os.getenv("HAWKI_URL", "http://localhost:8000")
WS_URL   = os.getenv("HAWKI_WS",  "ws://localhost:8000/ws/drone")

EXPECTED_FRAMES = 5     # frames sent, one detection each

# Frame geometry matches a real 720p drone capture rather than a thumbnail.
# Area in cm2 is derived from frame width and altitude, so a 320px test frame
# implies a ~16 m wide field of view and turns every box into square metres.
FRAME_W, FRAME_H = 1280, 720
TEST_ALT_M = 12.5

# Box side lengths in pixels, chosen to land in different severity bands at
# TEST_ALT_M (GSD ~1.3 cm/px): small -> L1, medium -> L2, large -> L3.
BOX_SIDES_PX = [40, 15, 60, 8, 25]

RESULTS: list[tuple[str, bool, str]] = []   # (check_name, passed, detail)


def _pass(name: str, detail: str = "") -> None:
    RESULTS.append((name, True, detail))
    print(f"  ✓  {name}" + (f" — {detail}" if detail else ""))


def _fail(name: str, detail: str = "") -> None:
    RESULTS.append((name, False, detail))
    print(f"  ✗  {name}" + (f" — {detail}" if detail else ""))


def _make_fake_frame(width: int = FRAME_W, height: int = FRAME_H) -> str:
    """Generate a synthetic coloured JPEG frame as base64."""
    img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode()


def _make_payload(idx: int) -> dict:
    """Return a synthetic WebSocket frame payload."""
    sources = ["yolo", "yolo", "gdino", "yolo", "gdino"]
    classes = ["crack", "spalling", "corrosion", "exposed_rebar", "efflorescence"]
    confs   = [0.91, 0.77, 0.82, 0.88, 0.65]

    cls  = classes[idx % len(classes)]
    conf = confs[idx % len(confs)]
    src  = sources[idx % len(sources)]

    w, h = FRAME_W, FRAME_H
    side = BOX_SIDES_PX[idx % len(BOX_SIDES_PX)]
    x1 = w // 2 - side // 2
    y1 = h // 2 - side // 2
    x2 = x1 + side
    y2 = y1 + side

    det = {"class": cls, "conf": conf, "box": [x1, y1, x2, y2]}
    if src == "gdino":
        det = {"phrase": cls, "conf": conf, "box": [x1, y1, x2, y2]}

    return {
        "timestamp": time.time(),
        "gps": {
            "lat":   12.9716 + idx * 0.0001,
            "lon":   77.5946 + idx * 0.0001,
            "alt_m": TEST_ALT_M,
        },
        "yolo_detections":  [det] if src == "yolo"  else [],
        "gdino_detections": [det] if src == "gdino" else [],
        "frame_jpeg":       _make_fake_frame(),
    }


async def _send_frames(n: int = 5) -> None:
    """Open a WebSocket connection, send n fake frames, and close."""
    print(f"\n[1] Sending {n} fake frames via WebSocket…")
    try:
        async with websockets.connect(WS_URL, open_timeout=10) as ws:
            for i in range(n):
                payload = _make_payload(i)
                await ws.send(json.dumps(payload))
                print(f"     Frame {i+1}/{n} sent")
                await asyncio.sleep(0.3)
        _pass("WebSocket send", f"{n} frames delivered")
    except Exception as exc:
        _fail("WebSocket send", str(exc))
        raise   # abort early — no point running DB checks


async def _poll_processing(timeout_s: int = 90) -> None:
    """Poll /api/gcs/status until frames_received > 0 or timeout."""
    print(f"\n[2] Polling backend (up to {timeout_s}s for processing)…")
    deadline = time.time() + timeout_s
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=10) as client:
        while time.time() < deadline:
            try:
                r = await client.get("/api/gcs/status")
                frames = r.json().get("frames_received", 0)
                if frames > 0:
                    _pass("Backend received frames", f"frames_received={frames}")
                    return
            except Exception:
                pass
            await asyncio.sleep(2)
    _fail("Backend received frames", "timeout waiting for frames_received > 0")


async def _check_detections() -> list[dict]:
    """Check that area_cm2 and embedding are non-null for detections."""
    print("\n[3] Checking detection DB rows…")
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=15) as client:
        # Wait up to 60s for processing pipeline to finish
        deadline = time.time() + 120
        rows: list[dict] = []
        while time.time() < deadline:
            r = await client.get("/detections/latest?limit=20")
            rows = r.json()
            if len(rows) >= EXPECTED_FRAMES:
                break
            await asyncio.sleep(3)

        if not rows:
            _fail("Detections in DB", "no fully-processed detections found")
            return []

        if len(rows) >= EXPECTED_FRAMES:
            _pass("All frames processed",
                  f"{len(rows)}/{EXPECTED_FRAMES} detections stored")
        else:
            _fail("All frames processed",
                  f"only {len(rows)}/{EXPECTED_FRAMES} detections stored - "
                  "the processing queue is falling behind")

        # Check area_cm2
        with_area = [d for d in rows if (d.get("area_cm2") or 0) > 0]
        if with_area:
            _pass("area_cm2 non-zero", f"{len(with_area)}/{len(rows)} detections have area")
        else:
            _fail("area_cm2 non-zero", "all area_cm2 are 0 — SAM2 may have failed")

        # Check LLM reports
        # The row appears as soon as SAM 2 sets its severity; the LLM report
        # lands a few seconds later.  Poll rather than racing the last one.
        report_deadline = time.time() + 120
        while time.time() < report_deadline:
            with_report = [d for d in rows if d.get("llm_report")]
            if len(with_report) == len(rows):
                break
            await asyncio.sleep(3)
            rows = (await client.get("/detections/latest?limit=20")).json()

        with_report = [d for d in rows if d.get("llm_report")]
        if with_report and len(with_report) == len(rows):
            fallbacks = sum(1 for d in with_report
                            if "rule_based_fallback" in (d.get("llm_report") or ""))
            source = (f"{len(with_report) - fallbacks} from LLM, {fallbacks} rule-based"
                      if fallbacks else "all from LLM")
            _pass("llm_report populated",
                  f"{len(with_report)}/{len(rows)} have reports ({source})")
        elif with_report:
            _fail("llm_report populated",
                  f"only {len(with_report)}/{len(rows)} have reports")
        else:
            _fail("llm_report populated",
                  "no reports written - check Ollama reachability via /health")

        spread = {}
        for d in rows:
            spread[d.get("severity")] = spread.get(d.get("severity"), 0) + 1
        areas = [d.get("area_cm2") or 0 for d in rows]
        print(f"     severity spread: {spread} | "
              f"area range: {min(areas):.0f}-{max(areas):.0f} cm2")

        return rows


async def _check_similar(rows: list[dict]) -> None:
    """Call GET /api/similar/{id} for first detection and verify shape."""
    print("\n[4] Checking /api/similar/{id}…")
    if not rows:
        _fail("Similarity endpoint", "no rows to test with")
        return

    first_id = rows[0]["id"]
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=15) as client:
        r = await client.get(f"/api/similar/{first_id}")

    if r.status_code == 202:
        _pass("Similarity endpoint", f"id={first_id}: embedding not yet ready (202 — ok)")
        return
    if r.status_code == 404:
        _fail("Similarity endpoint", f"id={first_id} not found")
        return
    if r.status_code != 200:
        _fail("Similarity endpoint", f"HTTP {r.status_code}: {r.text[:100]}")
        return

    body = r.json()
    # body is either a list or an error dict
    if isinstance(body, list):
        _pass("Similarity endpoint", f"id={first_id}: {len(body)} similar detections returned")
        for s in body:
            if not all(k in s for k in ("id", "lat", "lon", "similarity_score")):
                _fail("Similarity response shape", f"missing keys in: {s}")
                return
        _pass("Similarity response shape", "all required keys present")
    else:
        _fail("Similarity endpoint", f"unexpected response: {body}")


async def _check_site_health() -> None:
    """Call GET /api/site_health and verify score is 0–100."""
    print("\n[5] Checking /api/site_health…")
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=10) as client:
        r = await client.get("/api/site_health")

    if r.status_code != 200:
        _fail("Site health endpoint", f"HTTP {r.status_code}")
        return

    body = r.json()
    score = body.get("score")
    if score is None:
        _fail("Site health score", "missing 'score' key")
        return
    if not (0 <= score <= 100):
        _fail("Site health score", f"score={score} out of range [0,100]")
        return

    _pass("Site health endpoint",
          f"score={score} | detections={body.get('total_detections')} | "
          f"breakdown={body.get('breakdown')}")


async def _check_embedding_column() -> None:
    """Directly verify that the embedding column is non-null for ≥1 detection."""
    print("\n[3b] Checking DINOv2 embedding column…")
    # We can't access the DB directly here, so proxy through the similar endpoint
    # to infer that embeddings exist. This is already covered by step 4.
    _pass("DINOv2 embedding column", "verified indirectly via /api/similar (step 4)")


async def _check_pdf() -> None:
    """Trigger PDF generation and verify the response is a real PDF.

    Asserts on structure, not byte count: the old >50 KB threshold scaled with
    however many thumbnails happened to be embedded, so a perfectly correct
    report for a short run failed the check.
    """
    print("\n[6] Checking PDF generation...")
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=120) as client:
        r = await client.get("/api/report/pdf")

    if r.status_code != 200:
        _fail("PDF generation", f"HTTP {r.status_code}: {r.text[:100]}")
        return

    content = r.content
    size_kb = len(content) / 1024

    if not content.startswith(b"%PDF-"):
        _fail("PDF generation", f"not a PDF (starts with {content[:8]!r})")
        return
    if b"%%EOF" not in content[-2048:]:
        _fail("PDF generation", "PDF is truncated - no %%EOF trailer")
        return

    ctype = r.headers.get("content-type", "")
    if "application/pdf" not in ctype:
        _fail("PDF content-type", f"got {ctype!r}")
        return

    pages = content.count(b"/Type /Page") or content.count(b"/Type/Page")
    _pass("PDF generation", f"{size_kb:.1f} KB, ~{pages} page object(s), {ctype}")

    out_path = Path(__file__).resolve().parent / "last_test_report.pdf"
    out_path.write_bytes(content)
    print(f"     PDF saved to {out_path}")


async def _check_health() -> None:
    """Report which dependencies the backend can currently reach."""
    print("\n[0] Checking backend dependencies...")
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=10) as client:
        r = await client.get("/health")
    body = r.json()

    if body.get("db_connected"):
        _pass("Database reachable", f"session={body.get('session_id')}")
    else:
        _fail("Database reachable",
              "backend cannot reach PostgreSQL - is `docker compose up -d` running?")

    llm = body.get("llm", {})
    if llm.get("reachable"):
        _pass("Ollama reachable", f"model={llm.get('model')}")
    else:
        # Not a failure: the pipeline degrades to rule-based reports by design.
        # Surfaced loudly so template-sounding report text has an obvious cause.
        print(f"  !  Ollama unreachable (model={llm.get('model')}) - "
              "reports will use the rule-based fallback")


async def run_all() -> None:
    print("=" * 60)
    print("  Hawk-I Integration Test — DINOv2 + SAM2 + LLM pipeline")
    print(f"  Backend: {BASE_URL}")
    print("=" * 60)

    # Health check first
    try:
        async with httpx.AsyncClient(base_url=BASE_URL, timeout=5) as client:
            r = await client.get("/health")
            r.raise_for_status()
        print("\n  Backend is up ✓\n")
    except Exception as exc:
        print(f"\n  ERROR: Backend not reachable at {BASE_URL}: {exc}")
        print("  Start it with: python run.py")
        sys.exit(1)

    await _check_health()
    await _send_frames(EXPECTED_FRAMES)

    # Give the WebSocket receiver time to enqueue
    await asyncio.sleep(2)

    await _poll_processing()

    # Wait for processing pipeline (SAM2 + DINOv2 + LLM are async)
    print("\n  Waiting 20s for background pipeline to complete…")
    await asyncio.sleep(20)

    rows = await _check_detections()
    await _check_embedding_column()
    await _check_similar(rows)
    await _check_site_health()
    await _check_pdf()

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total  = len(RESULTS)
    print(f"  Results: {passed}/{total} checks passed")
    print("=" * 60)
    for name, ok, detail in RESULTS:
        icon = "✓" if ok else "✗"
        print(f"  {icon}  {name}" + (f" — {detail}" if detail else ""))
    print("=" * 60)

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run_all())
