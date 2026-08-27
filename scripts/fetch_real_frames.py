#!/usr/bin/env python3
"""
fetch_real_frames.py — Build a corpus of real infrastructure-defect photographs.

Synthetic noise frames prove the pipeline *runs*; they prove nothing about
whether it *detects*. This pulls genuine photographs of concrete cracking,
spalling, exposed reinforcement, corrosion and efflorescence so the detector,
SAM 2 segmentation and the DINOv2 verification stage can be exercised against
imagery they might actually see in the field.

Source is Wikimedia Commons rather than a general image search: every file
carries an explicit licence, the URLs are stable, and a large share of the
structural-survey material is US federal public domain (HAER/HABS bridge
surveys — exactly the subject matter this project targets). Licence and
author for every downloaded file are recorded in a manifest next to the
images, so the corpus can be redistributed or cited without guesswork.

    python scripts/fetch_real_frames.py
    python scripts/fetch_real_frames.py --per-class 6 --out data/real_frames

Downloaded frames are gitignored: this script is the reproducible artefact,
not the pixels.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # pragma: no cover
        pass

_REPO_ROOT = Path(__file__).resolve().parent.parent

API = "https://commons.wikimedia.org/w/api.php"
# Wikimedia rejects generic clients under its robot policy, and asks that
# bulk callers identify themselves and use the published thumbnail sizes
# (which is what iiurlwidth below requests).
UA = ("HawkI-InspectionPipeline/1.0 "
      "(+https://github.com/Arvoxis/hawk-i; research corpus builder) "
      "python-urllib")

# Search terms per canonical defect class.  Phrased the way survey
# photographers caption structural damage, not the way a textbook names it.
SEARCHES: dict[str, list[str]] = {
    "Crack": [
        "concrete bridge pier crack damage",
        "cracked concrete wall structural",
    ],
    "Spalling": [
        "spalling concrete bridge deck",
        "concrete spalling deterioration structure",
    ],
    "Exposed_reinforcement": [
        "exposed rebar deteriorated concrete bridge",
        "reinforcement bar exposed concrete damage",
    ],
    "Corrosion": [
        "corroded steel bridge girder rust",
        "rusted steel truss bridge deterioration",
    ],
    "Efflorescence": [
        "efflorescence concrete wall",
        "efflorescence masonry salt deposit",
    ],
    "Scaling": [
        "concrete surface scaling deterioration",
        "weathered concrete surface erosion",
    ],
}

# Commons categories, which are curated by subject.  Full-text search on
# these topics mostly surfaces scanned engineering PDFs; categories return
# actual photographs.
CATEGORIES: dict[str, list[str]] = {
    "Crack": [
        "Category:Cracks in concrete",
        "Category:Cracked walls",
    ],
    "Spalling": [
        "Category:Spalling",
        "Category:Concrete degradation",
    ],
    "Exposed_reinforcement": [
        "Category:Rebar",
        "Category:Concrete degradation",
    ],
    "Corrosion": [
        "Category:Corrosion of steel in bridges",
        "Category:Rust on bridges",
        "Category:Corrosion",
    ],
    "Efflorescence": [
        "Category:Efflorescence",
    ],
    "Scaling": [
        "Category:Weathered concrete",
        "Category:Concrete degradation",
    ],
}

# Subject filter.  The first corpus build pulled in close-ups of rusting
# nails and an aerial of a grounded ship -- technically "corrosion" and
# "damage", useless as infrastructure inspection frames.  A title has to look
# like built infrastructure to be kept.
SUBJECT_HINTS = (
    "bridge", "viaduct", "concrete", "wall", "pier", "abutment", "deck",
    "tunnel", "culvert", "dam", "building", "facade", "column", "beam",
    "girder", "masonry", "parapet", "haer", "habs", "structure", "overpass",
    "retaining", "foundation", "aqueduct", "pavement", "kerb", "curb",
)
REJECT_HINTS = (
    "nail", "ship", "boat", "car ", "vehicle", "coin", "tool", "knife",
    "microscope", "specimen", "diagram", "chart", "map of", "portrait",
)

GOOD_MIME = {"image/jpeg", "image/png", "image/tiff"}
MIN_WIDTH = 800
RETRY_STATUSES = {429, 503}


def _get(url: str, timeout: int = 45, attempts: int = 4) -> bytes:
    """Fetch a URL, backing off on the rate limits Commons applies to bursts."""
    delay = 1.5
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code not in RETRY_STATUSES:
                raise
            if attempt < attempts - 1:
                time.sleep(delay)
                delay *= 2
        except Exception as exc:
            last = exc
            if attempt < attempts - 1:
                time.sleep(delay)
                delay *= 2
    raise last if last else RuntimeError("request failed")


def _api(params: dict) -> dict:
    params = {**params, "format": "json", "formatversion": "2"}
    url = f"{API}?{urllib.parse.urlencode(params)}"
    return json.loads(_get(url).decode("utf-8"))


def _looks_like_infrastructure(title: str) -> bool:
    lowered = title.lower()
    if any(bad in lowered for bad in REJECT_HINTS):
        return False
    return any(good in lowered for good in SUBJECT_HINTS)


def _records_from_pages(pages: list, require_subject: bool = True) -> list[dict]:
    out = []
    for page in pages:
        info = (page.get("imageinfo") or [{}])[0]
        if info.get("mime") not in GOOD_MIME:
            continue
        if (info.get("width") or 0) < MIN_WIDTH:
            continue
        if require_subject and not _looks_like_infrastructure(page.get("title", "")):
            continue
        thumb = info.get("thumburl")
        if not thumb:
            continue
        meta = info.get("extmetadata", {})
        out.append({
            "title":   page.get("title", ""),
            "url":     thumb,
            "page":    info.get("descriptionurl", ""),
            "licence": (meta.get("LicenseShortName") or {}).get("value", "unknown"),
            "author":  _strip_html((meta.get("Artist") or {}).get("value", "unknown")),
            "width":   info.get("width"),
            "height":  info.get("height"),
        })
    return out


def category_images(category: str, limit: int, require_subject: bool = True) -> list[dict]:
    """Return image records from a Commons category."""
    try:
        data = _api({
            "action": "query",
            "generator": "categorymembers",
            "gcmtitle": category,
            "gcmtype": "file",
            "gcmlimit": max(limit * 5, 20),
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata",
            "iiurlwidth": 1280,
        })
    except Exception as exc:
        print(f"    ! category {category!r} failed: {exc}")
        return []
    return _records_from_pages((data.get("query") or {}).get("pages", []), require_subject)


def search_images(term: str, limit: int, require_subject: bool = True) -> list[dict]:
    """Return candidate image records for one search term."""
    try:
        data = _api({
            "action": "query",
            "generator": "search",
            "gsrnamespace": 6,          # File:
            "gsrsearch": term,
            "gsrlimit": limit * 4,      # over-fetch; most hits are PDFs
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata",
            "iiurlwidth": 1280,         # server-side resize to a drone-ish width
        })
    except Exception as exc:
        print(f"    ! search failed for {term!r}: {exc}")
        return []

    return _records_from_pages((data.get("query") or {}).get("pages", []), require_subject)


def _strip_html(text: str) -> str:
    """Commons returns author fields as HTML fragments."""
    import re
    text = re.sub(r"<[^>]+>", "", text or "")
    return " ".join(text.split())[:120]


def _safe_name(cls: str, idx: int, title: str) -> str:
    """ASCII-only, so every downstream consumer can open the file.

    Commons titles carry accents and umlauts; cv2.imread on Windows cannot
    open a path outside the active code page and fails silently.
    """
    stem = title.replace("File:", "").rsplit(".", 1)[0]
    stem = "".join(c if c.isascii() and c.isalnum() else "_" for c in stem)
    stem = "_".join(filter(None, stem.split("_")))[:48].strip("_")
    return f"{cls}__{idx:02d}__{stem or 'frame'}.jpg"


def download(rec: dict, dest: Path) -> bool:
    try:
        data = _get(rec["url"], timeout=60)
        if len(data) < 8_000:
            return False
        dest.write_bytes(data)
        return True
    except Exception as exc:
        print(f"    ! download failed: {exc}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(_REPO_ROOT / "data" / "real_frames"),
                    help="Directory to write frames into")
    ap.add_argument("--per-class", type=int, default=4,
                    help="Images to keep per defect class")
    ap.add_argument("--classes", default="",
                    help="Comma-separated subset of classes (default: all)")
    ap.add_argument("--delay", type=float, default=2.0,
                    help="Seconds between requests; Commons rate-limits bursts")
    ap.add_argument("--any-subject", action="store_true",
                    help="Skip the infrastructure subject filter")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    wanted = ([c.strip() for c in args.classes.split(",") if c.strip()]
              or list(SEARCHES))

    manifest: list[dict] = []
    seen_urls: set[str] = set()

    for cls in wanted:
        if cls not in SEARCHES and cls not in CATEGORIES:
            print(f"  ? unknown class {cls!r} — skipping")
            continue

        print(f"\n{cls}")
        kept = 0

        # Categories first (photographs), search second (fills any gap).
        candidates: list[dict] = []
        for category in CATEGORIES.get(cls, []):
            if len(candidates) >= args.per_class * 3:
                break
            candidates += category_images(category, args.per_class,
                                          require_subject=not args.any_subject)
            time.sleep(args.delay)
        for term in SEARCHES.get(cls, []):
            if len(candidates) >= args.per_class * 3:
                break
            candidates += search_images(term, args.per_class,
                                        require_subject=not args.any_subject)
            time.sleep(args.delay)

        for rec in candidates:
            if kept >= args.per_class:
                break
            if rec["url"] in seen_urls:
                continue
            seen_urls.add(rec["url"])

            dest = out_dir / _safe_name(cls, kept, rec["title"])
            if download(rec, dest):
                kept += 1
                manifest.append({"file": dest.name, "class": cls, **rec})
                print(f"  ✓ {dest.name}")
                print(f"      {rec['licence']} · {rec['author']}")
            time.sleep(args.delay)   # be polite to the API

        if kept == 0:
            print("  ! nothing usable found")

    manifest_path = out_dir / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps({
            "source": "Wikimedia Commons",
            "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "note": ("每 file retains its own licence, recorded below. "
                     "Check each before redistributing.").replace("每", "Each"),
            "files": manifest,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\n{len(manifest)} image(s) → {out_dir}")
    print(f"Attribution manifest → {manifest_path}")
    return 0 if manifest else 1


if __name__ == "__main__":
    sys.exit(main())
