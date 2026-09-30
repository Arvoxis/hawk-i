#!/usr/bin/env python3
"""
run_all.py — Run every Hawk-I test suite and print one verdict.

    python tests/run_all.py            # unit suites only (no services needed)
    python tests/run_all.py --all      # also the live integration test

Unit suites need nothing running: no GPU, no database, no model weights, no
network. They are safe in CI and safe on the Jetson. The integration suite
needs the full stack up (docker compose + run.py) and is opt-in for that
reason, so a missing service reads as "skipped", not "failed".

Exit code is non-zero if any selected suite fails.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # pragma: no cover
        pass

_TESTS = Path(__file__).resolve().parent
_ROOT = _TESTS.parent

UNIT_SUITES = [
    ("geometry      ", "test_geometry.py",    "GSD / area / severity maths"),
    ("query mapping ", "test_multi_query.py", "class normalisation, expansion, NMS"),
    ("llm resilience", "test_llm_worker.py",  "circuit breaker, parsing, fallback"),
    ("defect growth ", "test_growth.py",      "cross-inspection growth comparison"),
    ("pdf report    ", "test_pdf.py",         "report rendering on degraded rows"),
]

INTEGRATION_SUITE = ("integration   ", "test_fake_drone.py",
                     "end-to-end against a running backend")


def run(path: Path, verbose: bool) -> tuple[bool, str, float]:
    started = time.time()
    proc = subprocess.run(
        [sys.executable, str(path)],
        cwd=str(_ROOT),
        capture_output=not verbose,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.time() - started
    output = "" if verbose else (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, output, elapsed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true",
                    help="Also run the integration suite (needs the stack up)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Stream each suite's own output")
    args = ap.parse_args()

    suites = list(UNIT_SUITES)
    if args.all:
        suites.append(INTEGRATION_SUITE)

    print("=" * 66)
    print("  Hawk-I test suites")
    print("=" * 66)

    results = []
    for label, filename, description in suites:
        path = _TESTS / filename
        if not path.exists():
            print(f"  ?  {label}  {filename} not found")
            results.append((label, None, 0.0, ""))
            continue

        print(f"\n▶  {label.strip()} — {description}")
        ok, output, elapsed = run(path, args.verbose)
        results.append((label, ok, elapsed, output))
        print(f"   {'PASS' if ok else 'FAIL'}  ({elapsed:.1f}s)")
        if not ok and not args.verbose:
            tail = [ln for ln in output.strip().splitlines() if ln.strip()][-15:]
            for line in tail:
                print(f"     | {line}")

    print("\n" + "=" * 66)
    passed = sum(1 for _, ok, _, _ in results if ok)
    total = sum(1 for _, ok, _, _ in results if ok is not None)
    print(f"  {passed}/{total} suite(s) passed")
    print("=" * 66)
    for label, ok, elapsed, _ in results:
        mark = "?" if ok is None else ("✓" if ok else "✗")
        print(f"  {mark}  {label}  {elapsed:5.1f}s")
    print("=" * 66)

    if not args.all:
        print("\n  Integration suite skipped. To include it, start the stack:")
        print("    docker compose up -d  &&  python run.py")
        print("  then:  python tests/run_all.py --all")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
