import os
import cv2
import time
import threading
import queue as queue_module
from ultralytics import YOLO
import uuid
from utils import send_telegram_image_alert,send_telegram_text_alert,send_alert_async,send_ip_alert, _play_siren_blocking

# Configurations
#RTSP_URL = "rtsp://admin:4PEL%232025@192.168.29.128:554/h264"
RTSP_URL = "rtsp://admin:rays%402022@192.168.1.161:554/h264"
DETECT_RES = (1280, 720)
QUEUE_MAXSIZE = 5
CONF_THRESHOLD = 0.3
EMA_ALPHA = 0.2
RETRY_DELAY = 0.5
RECONNECT_ATTEMPTS = 3


# Logger Setup
import logging
import os
from datetime import datetime

# Simplified logging setup
LOGS_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

LOG_FILE = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
LOG_FILE_PATH = os.path.join(LOGS_DIR, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger("PersonTracker")
logger.info("Logger initialized.")

model = YOLO("yolov8n.pt")
logger.info("YOLOv8 model loaded.")

# Frame Grabber Thread
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
        logger.error("Unable to open RTSP stream.")
        stop_event.set()
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    full_res = (width, height)

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            attempts += 1
            logger.warning(f"Frame read failed ({attempts}/{RECONNECT_ATTEMPTS})")
            time.sleep(RETRY_DELAY)
            if attempts >= RECONNECT_ATTEMPTS:
                logger.warning("Attempting to reconnect to RTSP stream.")
                cap.release()
                cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                attempts = 0
            continue
        attempts = 0

        try:
            frame_queue.put(frame, timeout=0.5)
        except queue_module.Full:
            _ = frame_queue.get_nowait()
            try:
                frame_queue.put(frame, timeout=0.5)
            except queue_module.Full:
                pass

    cap.release()
    logger.info("Frame grabber thread stopped.")

# Schedule Alert
from datetime import datetime, time as dt_time

def is_alert_allowed():
    now = datetime.now().time()
    start = dt_time(7, 0)  # 21:00
    end = dt_time(6, 0)     # 06:00

    return now >= start or now <= end


# main loop
person_last_alert = {}         # Track alert time per ID
person_cooldown = 10           # Cooldown in seconds
CAMERA_NAME = "Office"     # Customize per camera if needed


def main():
    grab_thread = threading.Thread(
        target=frame_grabber,
        args=(RTSP_URL, frame_queue, stop_event),
        daemon=True
    )
    grab_thread.start()

    logger.info("Frame grabber thread started.")
    fps_ema = None
    last_log_time = time.time()

    try:
        while not stop_event.is_set():
            try:
                frame = frame_queue.get(timeout=1.0)
                start = time.time()
            except queue_module.Empty:
                continue

            small = cv2.resize(frame, DETECT_RES)
            results = model.track(small, classes=[0], persist=True, verbose=False)

            output = frame.copy()
            if results and results[0].boxes is not None:
                x_scale = frame.shape[1] / DETECT_RES[0]
                y_scale = frame.shape[0] / DETECT_RES[1]
                now_ts = time.time()
                current_time = time.time()
                for box in results[0].boxes:
                    conf = float(box.conf)
                    if conf < CONF_THRESHOLD:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, x2 = int(x1 * x_scale), int(x2 * x_scale)
                    y1, y2 = int(y1 * y_scale), int(y2 * y_scale)

                    person_id = getattr(box, "id", None)
                    if person_id is None:
                        continue
                    person_id = int(person_id.item())  # Tensor → int

                    # Draw rectangle and label
                    cv2.rectangle(output, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    label = f"ID:{person_id} | {conf:.2f}"
                    cv2.putText(output, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    last_alert = person_last_alert.get(person_id, 0)

                    if is_alert_allowed() and current_time - last_alert > person_cooldown:
                        alert_msg = f"Warning Intruder Detected by Camera {CAMERA_NAME}"
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        #send_alert_async(send_ip_alert, alert_msg)
                        send_alert_async(_play_siren_blocking)
                        person_last_alert[person_id] = now_ts

                        message = (
                            f"🚨 Person Detected!\n"
                            f"🧍 ID: {person_id}\n"
                            f"🎯 Confidence: {conf:.2f}\n"
                            f"📍 Camera: {CAMERA_NAME}\n"
                            f"🕒 {timestamp}"
                        )

                        if conf >= 0.6:
                            filename = f"alert_{uuid.uuid4().hex[:8]}.jpg"
                            filepath = os.path.join("logs", filename)
                            cv2.imwrite(filepath, output)

                            send_alert_async(send_telegram_image_alert, filepath, caption=message)
                            logger.info(f"[ALERT] Queued image alert for ID {person_id}")
                            os.remove(filepath)  # optional

                        else:
                            send_alert_async(send_telegram_text_alert, message)
                            logger.info(f"[ALERT] Queued text alert for ID {person_id}")

                        person_last_alert[person_id] = current_time

            # FPS
            fps = 1.0 / max(time.time() - start, 1e-6)
            fps_ema = fps if fps_ema is None else EMA_ALPHA * fps + (1 - EMA_ALPHA) * fps_ema
            cv2.putText(output, f"FPS: {fps_ema:.2f}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Log FPS every 10s
            if time.time() - last_log_time > 10:
                logger.info(f"Current FPS (EMA): {fps_ema:.2f}")
                last_log_time = time.time()

            cv2.imshow("Person Tracker", output)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Exit key pressed.")
                stop_event.set()
                break

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
        stop_event.set()

    grab_thread.join()
    cv2.destroyAllWindows()
    logger.info("Application shutdown complete.")

if __name__ == "__main__":
    main()
