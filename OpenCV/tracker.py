
import threading
import queue as queue_module  # Rename to avoid shadowing
import time
import cv2
import numpy as np
from ultralytics import YOLO
import os

# Configuration
#RTSP_URL = "rtsp://admin:4PEL%232025@192.168.29.128:554/h264"
RTSP_URL = "rtsp://admin:rays%232022@192.168.1.161:554/h264"
DETECT_RES = (1280, 720)
QUEUE_MAXSIZE = 5
CONF_THRESHOLD = 0.3
EMA_ALPHA = 0.2
RETRY_DELAY = 0.5
RECONNECT_ATTEMPTS = 3

# Load model
model = YOLO("yolov8n.pt")
# Queues and events
frame_queue = queue_module.Queue(maxsize=QUEUE_MAXSIZE)
stop_event = threading.Event()

def frame_grabber(rtsp_url, frame_queue, stop_event):
    attempts = 0
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "rtsp_transport;tcp;"
        "fflags;genpts+nobuffer;"
        "flags;low_delay;"
        "flags2;showall;"
        "vsync;0;"
        "reset_timestamps;1"
    )
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("[ERROR] Unable to open stream.")
        stop_event.set()
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    full_res = (width, height)

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            attempts += 1
            print(f"[WARN] Frame read failed ({attempts}/{RECONNECT_ATTEMPTS})")
            time.sleep(RETRY_DELAY)
            if attempts >= RECONNECT_ATTEMPTS:
                cap.release()
                cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                attempts = 0
            continue
        attempts = 0

        if frame.shape[1::-1] != full_res:
            frame = cv2.resize(frame, full_res)

        try:
            frame_queue.put(frame, timeout=0.5)
        except queue_module.Full:
            _ = frame_queue.get_nowait()
            try:
                frame_queue.put(frame, timeout=0.5)
            except queue_module.Full:
                pass

    cap.release()

# Start grabber
grab_thread = threading.Thread(
    target=frame_grabber,
    args=(RTSP_URL, frame_queue, stop_event),
    daemon=True
)
grab_thread.start()

fps_ema = None
try:
    while not stop_event.is_set():
        start = time.time()
        try:
            frame = frame_queue.get(timeout=1.0)
        except queue_module.Empty:
            continue

        small = cv2.resize(frame, DETECT_RES)
        results = model.track(small, classes=[0], persist=True, verbose=False)

        output = frame.copy()
        if results and results[0].boxes is not None:
            x_scale = frame.shape[1] / DETECT_RES[0]
            y_scale = frame.shape[0] / DETECT_RES[1]
            for box in results[0].boxes:
                conf = float(box.conf)
                if conf < CONF_THRESHOLD:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, x2 = int(x1 * x_scale), int(x2 * x_scale)
                y1, y2 = int(y1 * y_scale), int(y2 * y_scale)
                cv2.rectangle(output, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(output, f"person {conf:.2f}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        fps = 1.0 / max(time.time() - start, 1e-6)
        fps_ema = fps if fps_ema is None else EMA_ALPHA*fps + (1-EMA_ALPHA)*fps_ema
        cv2.putText(output, f"FPS: {fps_ema:.2f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Person Tracker", output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

except KeyboardInterrupt:
    stop_event.set()

grab_thread.join()
cv2.destroyAllWindows()
print("[INFO] Shutdown complete.")
