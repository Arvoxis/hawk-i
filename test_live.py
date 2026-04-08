# test_live.py — replace entire file with this
import asyncio
import websockets
import json
import base64
import numpy as np
import cv2

async def watch():
    # Connect to dashboard broadcast endpoint instead
    uri = "ws://localhost:8000/ws/dashboard"
    print(f"Connecting to {uri}...")
    async with websockets.connect(uri) as ws:
        print("Connected — waiting for frames...")
        frame_count = 0
        while True:
            raw  = await ws.recv()
            data = json.loads(raw)
            print(f"RAW KEYS: {list(data.keys())}")  # see what's being broadcast
            b64  = data.get("frame_jpeg", "")
            if not b64:
                print(f"No frame — payload: {data}")
                continue
            img_bytes = base64.b64decode(b64)
            img_np    = np.frombuffer(img_bytes, dtype=np.uint8)
            frame     = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            if frame is None:
                print("Decode failed")
                continue
            frame_count += 1
            cv2.imshow("Hawki Live Feed", frame)
            print(f"Frame {frame_count} — shape: {frame.shape}")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

asyncio.run(watch())