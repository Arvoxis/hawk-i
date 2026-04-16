import asyncio
import json
import base64
import os
import cv2
import numpy as np
import websockets
from dotenv import load_dotenv

load_dotenv()

_RECEIVER_PORT = int(os.getenv("RECEIVER_PORT", "8000"))

async def handle_client(websocket):
    print("\n[+] Jetson pipeline connected!")
    cv2.namedWindow("Hawk-I Live Stream", cv2.WINDOW_NORMAL)
    
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                
                yolo_dets = data.get("yolo_detections", [])
                world_dets = data.get("gdino_detections", [])
                
                # Terminal logging
                print(f"Received Frame | YOLO detections: {len(yolo_dets)} | YOLO-World detections: {len(world_dets)}", end="\r")
                
                # Extract and decode JPEG
                b64_img = data.get("frame_jpeg")
                if b64_img:
                    # Ignore the base64 prefix if somehow included
                    if "," in b64_img:
                        b64_img = b64_img.split(",")[1]
                    
                    img_bytes = base64.b64decode(b64_img)
                    np_arr = np.frombuffer(img_bytes, np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        # Draw GPS info head-up display text
                        gps = data.get("gps", {})
                        if gps:
                            lat = gps.get("lat") or "N/A"
                            lon = gps.get("lon") or "N/A"
                            alt = gps.get("alt_m") or "N/A"
                            
                            hud_text = f"GPS: {lat}, {lon} | Alt: {alt}m"
                            cv2.putText(frame, hud_text, (15, 35), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                        # Display the video frame
                        cv2.imshow("Hawk-I Live Stream", frame)
                        
                        # Wait 1 ms to allow OpenCV to update the frame
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("\n[!] Stream viewing stopped by user (Press Q).")
                            break
                            
            except json.JSONDecodeError:
                print("Error decoding JSON from message.")
            except Exception as e:
                print(f"Error viewing frame: {e}")
                
    except websockets.exceptions.ConnectionClosed:
        print("\n[-] Jetson disconnected.")
    finally:
        cv2.destroyAllWindows()


async def main():
    print("=" * 60)
    print("  Hawk-I Live Stream Receiver")
    print(f"  Listening on ws://0.0.0.0:{_RECEIVER_PORT}")
    print("  Waiting for Jetson to connect... (Press Ctrl+C to exit)")
    print("=" * 60)

    async with websockets.serve(handle_client, "0.0.0.0", _RECEIVER_PORT):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nReceiver shut down safely.")
