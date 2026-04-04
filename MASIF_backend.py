# ============================================================
# MASIF — Colab/Kaggle Backend
# Integrated Backend for MASIF-interpolation-bot
# ============================================================

# ── CELL 1: Imports & System Setup ──────────────────────────
import os
import io
import shutil
import subprocess
import threading
import logging
import socket
import requests
from flask import Flask, request, send_file
from pyngrok import ngrok

# ── CELL 2: Kaggle/Local Authentication ─────────────────────
try:
    from kaggle_secrets import UserSecretsClient
    NGROK_TOKEN = UserSecretsClient().get_secret("NGROK_AUTH_TOKEN")
except Exception:
    # Fallback for local testing
    NGROK_TOKEN = "YOUR_TOKEN_HERE"

# Disable noisy logs
logging.getLogger('werkzeug').disabled = True
ngrok.set_auth_token(NGROK_TOKEN)

# ── CELL 3: Flask Application Logic ────────────────────────
app = Flask(__name__)

@app.route('/')
def home():
    return "MASIF Backend is running. ✅"

@app.route('/process', methods=['POST'])
def process_video():
    input_path  = "input_from_bot.mp4"
    output_path = "masif_output.mp4"

    try:
        # Parse incoming request from Telegram Bot
        data = request.get_json(force=True, silent=True) or {}
        video_url    = data.get('video_url')
        speed_option = float(data.get('speed', 0.5))

        if not video_url:
            return {"error": "No video_url provided"}, 400

        # 1. Download video
        print(f"\n📥 Downloading video from Bot...")
        r = requests.get(video_url, timeout=60)
        with open(input_path, 'wb') as f:
            f.write(r.content)

        # 2. Run MASIF pipeline
        # Ensure run_pipeline is defined in your main environment
        print(f"⚙️ Running MASIF pipeline at {speed_option}x speed...")
        out_file, original_fps = run_pipeline(input_path, speed=speed_option, output_path=output_path)

        # 3. Return output mp4 directly
        print(f"📤 Sending result back...")
        buf = io.BytesIO()
        with open(out_file, 'rb') as f:
            buf.write(f.read())
        buf.seek(0)

        return send_file(buf, mimetype='video/mp4')

    except Exception as e:
        print(f"❌ Error: {e}")
        return {"error": str(e)}, 500

    finally:
        # Cleanup temporary files and frame folders
        for p in [input_path, output_path]:
            if os.path.exists(p):
                os.remove(p)
        for folder in ["frames", "output_frames"]:
            if os.path.exists(folder):
                shutil.rmtree(folder, ignore_errors=True)

# ── CELL 4: Network & Tunneling Setup ───────────────────────
def get_free_port(start_port=5000):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port = start_port
    while True:
        try:
            sock.bind(('127.0.0.1', port))
            sock.close()
            return port
        except OSError:
            port += 1

def start_backend():
    ngrok.kill()
    free_port = get_free_port()

    try:
        public_url = ngrok.connect(free_port).public_url

        print("\n" + "=" * 60)
        print("🚀 MASIF BACKEND IS LIVE!")
        print(f"Endpoint: {public_url}/process")
        print("Note: Copy this URL to your Telegram Bot config.")
        print("=" * 60 + "\n")

        # Start Flask
        app.run(port=free_port, debug=False, use_reloader=False)

    except Exception as e:
        print(f"\n❌ Ngrok Error: {e}")

# ── CELL 5: Execution ───────────────────────────────────────
if __name__ == "__main__":
    # If running in a notebook, you might use a thread.
    # For a standalone script, call start_backend() directly.
    start_backend()
