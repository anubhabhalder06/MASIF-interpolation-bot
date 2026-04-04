# ============================================================
# MASIF — Colab/Kaggle Backend
# Run this in Google Colab or Kaggle with GPU enabled.
# Runtime → Change runtime type → T4 GPU
# ============================================================

# ── CELL 1: Install dependencies ────────────────────────────
# !pip install tensorflow tensorflow-hub opencv-python-headless ffmpeg-python numpy Pillow flask pyngrok absl-py mediapy scikit-image scipy

# ── CELL 2: Clone FILM repo ─────────────────────────────────
# !git clone https://github.com/google-research/frame-interpolation

# ── CELL 3: Imports & Model Loading ─────────────────────────
import os
import shutil
import subprocess
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image

print("TensorFlow Version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

print("Loading FILM model... (takes 1-2 minutes, please wait)")
film_model = hub.load("https://tfhub.dev/google/film/1")
print("FILM model loaded successfully! ✅")


# ── CELL 4: Core MASIF Functions ─────────────────────────────

def extract_frames(video_path, output_folder="frames", max_size=1280):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    subprocess.run([
        "ffmpeg", "-y",
        "-i", video_path,
        "-vsync", "0",
        "-vf", f"scale='if(gt(iw,ih),{max_size},-2)':'if(gt(iw,ih),-2,{max_size})'",
        "-q:v", "1",
        f"{output_folder}/frame_%04d.png"
    ])

    result = subprocess.run([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ], capture_output=True, text=True)

    fps_str = result.stdout.strip()
    num, den = fps_str.split("/")
    original_fps = float(num) / float(den)

    frames = sorted(os.listdir(output_folder))
    test = Image.open(f"{output_folder}/{frames[0]}")
    print(f"✅ Extracted {len(frames)} frames at {original_fps:.2f} FPS — size: {test.size}")
    return frames, original_fps


def load_frame(path, max_size=1280):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    img_array = np.array(img) / 255.0
    return img_array.astype(np.float32)


def film_interpolate(frame1_path, frame2_path, num_passes):
    f1 = load_frame(frame1_path)
    f2 = load_frame(frame2_path)
    all_frames = [f1]

    def recursive_interpolate(left, right, depth):
        if depth == 0:
            return
        inputs = {
            'x0': tf.expand_dims(left, 0),
            'x1': tf.expand_dims(right, 0),
            'time': tf.constant([[0.5]], dtype=tf.float32)
        }
        mid = film_model(inputs)['image'][0].numpy()
        recursive_interpolate(left, mid, depth - 1)
        all_frames.append(mid)
        recursive_interpolate(mid, right, depth - 1)

    recursive_interpolate(f1, f2, num_passes)
    all_frames.append(f2)
    return all_frames


# ── CELL 5: Main Pipeline ────────────────────────────────────

def run_pipeline(video_path, speed=0.5, output_path="output_slowmo.mp4"):
    speed_to_passes = {
        0.5:   1,
        0.25:  2,
        0.125: 3
    }

    num_passes = speed_to_passes.get(speed, 1)
    print(f"Speed: {speed}x → applying {num_passes} FILM pass(es) per frame pair")

    if os.path.exists("output_frames"):
        shutil.rmtree("output_frames")
    os.makedirs("output_frames")

    print("Stage 1: Extracting frames... ⏳")
    frames, fps = extract_frames(video_path)

    print("Stage 2: Running FILM interpolation... ⏳")
    output_idx = 0

    for i in range(len(frames) - 1):
        f1_path = f"frames/{frames[i]}"
        f2_path = f"frames/{frames[i+1]}"

        interpolated = film_interpolate(f1_path, f2_path, num_passes)

        for frame_array in interpolated[:-1]:
            out_frame = Image.fromarray((frame_array * 255).astype(np.uint8))
            out_frame.save(f"output_frames/frame_{output_idx:06d}.png")
            output_idx += 1

        if i % 10 == 0:
            print(f"  Processed {i}/{len(frames)} frame pairs...")

    last = Image.open(f"frames/{frames[-1]}")
    last.save(f"output_frames/frame_{output_idx:06d}.png")

    print("Stage 3: Encoding final video... ⏳")
    subprocess.run([
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", "output_frames/frame_%06d.png",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        output_path
    ])

    print(f"✅ Done! Saved to: {output_path}")
    return output_path, fps


print("Pipeline ready! ✅")


# ── CELL 6: Flask + ngrok Server ────────────────────────────

import io
import zipfile
import threading
import logging
import socket
import requests
from flask import Flask, request, send_file
from pyngrok import ngrok

# ── For Kaggle: use this line ──
from kaggle_secrets import UserSecretsClient
NGROK_TOKEN = UserSecretsClient().get_secret("NGROK_AUTH_TOKEN")

# ── For Colab: use these two lines instead (comment out the Kaggle lines above) ──
# from google.colab import userdata
# NGROK_TOKEN = userdata.get('NGROK_AUTH_TOKEN')

logging.getLogger('werkzeug').disabled = True

ngrok.set_auth_token(NGROK_TOKEN)

# Replace with your static ngrok domain
NGROK_STATIC_DOMAIN = "georgeann-calcaneal-colorimetrically.ngrok-free.app"

app = Flask(__name__)


@app.route('/')
def home():
    return "MASIF Backend is running. ✅"


@app.route('/process', methods=['POST'])
def process_video():
    input_path      = "input_from_bot.mp4"
    output_path     = "masif_output.mp4"
    raw_path        = "raw_slowmo.mp4"
    comparison_path = "comparison.mp4"

    try:
        data = request.get_json(force=True, silent=True) or {}
        video_url    = data.get('video_url')
        speed_option = float(data.get('speed', 0.5))

        if not video_url:
            return {"error": "No video_url provided"}, 400

        # ── 1. Download video ─────────────────────────────────
        print(f"\n📥 Downloading video...")
        r = requests.get(video_url, timeout=60)
        with open(input_path, 'wb') as f:
            f.write(r.content)

        # ── 2. Run MASIF pipeline ─────────────────────────────
        print(f"⚙️ Running MASIF pipeline at {speed_option}x speed...")
        output_path, original_fps = run_pipeline(input_path, speed=speed_option, output_path=output_path)

        # ── 3. Generate raw slow-mo (WITHOUT MASIF) ───────────
        # This just duplicates/drops frames using ffmpeg — no AI
        print(f"🎞️ Generating raw slow-mo for comparison...")
        subprocess.run([
            "ffmpeg", "-y",
            "-i", input_path,
            "-vf", f"setpts={1/speed_option}*PTS",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            raw_path
        ])

        # ── 4. Stitch side-by-side comparison video ───────────
        # Left = WITHOUT MASIF, Right = WITH MASIF, labels burned in
        print(f"🎬 Creating side-by-side comparison video...")
        subprocess.run([
            "ffmpeg", "-y",
            "-i", raw_path,
            "-i", output_path,
            "-filter_complex",
            (
                "[0:v]scale=640:-2,"
                "drawtext=text='WITHOUT MASIF':fontcolor=white:fontsize=22"
                ":x=10:y=10:box=1:boxcolor=black@0.5:boxborderw=5[left];"

                "[1:v]scale=640:-2,"
                "drawtext=text='WITH MASIF':fontcolor=white:fontsize=22"
                ":x=10:y=10:box=1:boxcolor=black@0.5:boxborderw=5[right];"

                "[left][right]hstack=inputs=2[out]"
            ),
            "-map", "[out]",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "20",
            comparison_path
        ])

        # ── 5. Zip both files and return ──────────────────────
        print(f"📤 Zipping and sending back to bot...")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(comparison_path, "comparison.mp4")
            zf.write(output_path,     "masif_output.mp4")
        buf.seek(0)

        return send_file(buf, mimetype='application/zip')

    except Exception as e:
        print(f"❌ Error: {e}")
        return {"error": str(e)}, 500

    finally:
        # Clean up ALL temp files
        for path in [
            input_path, output_path, raw_path, comparison_path,
            "frames", "output_frames"
        ]:
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            elif os.path.isfile(path):
                try:
                    os.remove(path)
                except:
                    pass


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


free_port = get_free_port()
ngrok.kill()

try:
    public_url = ngrok.connect(free_port, domain=NGROK_STATIC_DOMAIN).public_url
    print("\n" + "=" * 60)
    print("🚀 MASIF BACKEND IS LIVE!")
    print(f"   {public_url}/process")
    print("=" * 60 + "\n")

    threading.Thread(
        target=lambda: app.run(port=free_port, debug=False, use_reloader=False)
    ).start()

except Exception as e:
    print(f"\n❌ Ngrok Error: {e}")
