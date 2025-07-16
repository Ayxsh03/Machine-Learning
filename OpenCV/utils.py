import requests

TELEGRAM_TOKEN = "8105706453:AAGy8KT8smq7mhpFCvsD82nbgxBJlfOFlAc"
CHAT_ID = "5420629863"

def send_telegram_text_alert(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {
        "chat_id": CHAT_ID,
        "text": message,
        "disable_notification": False
    }
    try:
        requests.post(url, data=data)
    except Exception as e:
        print(f"[ERROR] Failed to send Telegram text alert: {e}")

def send_telegram_image_alert(image_path, caption="Person detected!"):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    with open(image_path, "rb") as img:
        files = {"photo": img}
        data = {
            "chat_id": CHAT_ID,
            "caption": caption,
            "disable_notification": False
        }
        try:
            requests.post(url, files=files, data=data)
        except Exception as e:
            print(f"[ERROR] Failed to send Telegram image alert: {e}")

import threading

#UDP alert
import pathlib
from playsound import playsound
import socket

ALERT_IP   = "192.168.29.51" #asad sir's
ALERT_PORT = 5051
SOCKET_FAMILY = socket.AF_INET
SOCKET_TYPE   = socket.SOCK_DGRAM

SIREN_PATH = pathlib.Path(__file__).with_name("siren.wav")


def _play_siren_blocking():
    try:
        playsound(str(SIREN_PATH))
    except Exception as exc:
        print(f"[ERROR] Could not play siren: {exc}")


def send_ip_alert(message: str):
    try:
        with socket.socket(SOCKET_FAMILY, SOCKET_TYPE) as sock:
            sock.sendto(message.encode("utf-8"), (ALERT_IP, ALERT_PORT))
    except Exception as exc:
        print(f"[ERROR] Failed to send alert to {ALERT_IP}:{ALERT_PORT} – {exc}")

def send_alert_async(func, *args, **kwargs):
    threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True).start()