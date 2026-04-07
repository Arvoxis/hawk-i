import asyncio, websockets, json, random, time
import base64, io
import numpy as np
import requests
from PIL import Image

DEFECTS = ["cracked concrete", "rust stain", "spalling", "exposed rebar", "water seepage"]
BASE_LAT, BASE_LON = 12.9716, 77.5946
FRAME_W, FRAME_H   = 1280, 720


def make_fake_frame_b64() -> str:
    """Generate a synthetic 1280×720 JPEG (grey gradient) as a base64 string."""
    arr = np.random.randint(80, 180, (FRAME_H, FRAME_W, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode()


def print_sam2_results(since_ts: float):
    """Poll /detections/latest and print any rows that have SAM2 area data."""
    try:
        resp = requests.get("http://localhost:8000/detections/latest", timeout=3)
        rows = resp.json()
        for r in rows:
            area = r.get("area_cm2")
            if area and area > 0:
                print(
                    f"  [SAM2] {r['class_name']} | area_cm2={area:.1f} | "
                    f"severity={r['severity']} | conf={r['confidence']:.2f}"
                )
    except Exception as e:
        print(f"  [SAM2] Could not fetch results: {e}")


async def fake_drone():
    uri = "ws://localhost:8000/ws/drone"
    async with websockets.connect(uri) as ws:
        print("✓ Fake drone connected to backend")
        while True:
            phrase   = random.choice(DEFECTS)
            conf     = round(random.uniform(0.55, 0.95), 2)
            frame_b64 = make_fake_frame_b64()

            payload = {
                "timestamp": time.time(),
                "gps": {
                    "lat":   BASE_LAT + random.uniform(-0.001, 0.001),
                    "lon":   BASE_LON + random.uniform(-0.001, 0.001),
                    "alt_m": 12.0,
                },
                "yolo_detections": [],
                "gdino_detections": [{
                    "phrase": phrase,
                    "conf":   conf,
                    "box":    [100, 200, 300, 400],
                }],
                "frame_jpeg": frame_b64,
            }

            await ws.send(json.dumps(payload))
            print(f"  → Sent: {phrase} (conf={conf})  [frame included]")

            # Give the backend ~2 s to run SAM2, then print results
            await asyncio.sleep(2)
            print_sam2_results(since_ts=payload["timestamp"])

            await asyncio.sleep(1)   # total ~3 s cadence


asyncio.run(fake_drone())
