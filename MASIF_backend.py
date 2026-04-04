# ============================================================
# MASIF — Kaggle Backend  (FULL PIPELINE)
# Pipeline:
#   FFmpeg extract frames → OpenCV optical flow classify
#   → FILM interpolate → Frame interleave → FFmpeg stitch
# ============================================================


# ── CELL 1: Install dependencies ─────────────────────────────
# !pip install pyngrok tensorflow-hub opencv-python-headless ffmpeg-python Pillow flask -q


# ── CELL 2: Download FILM model ──────────────────────────────
import os

MODEL_PATH = "/kaggle/working/film_net/pretrained_models/film_net/Style/saved_model"

if not os.path.exists(MODEL_PATH):
    print("Downloading FILM model weights...")
    os.system("pip install -q gdown")
    os.system('gdown "https://drive.google.com/uc?id=1rEABCoyQFkmHGieKDhHXW2ZYJi12lofI" -O pretrained_models.zip')

    print("Extracting model archive...")
    os.system("unzip -q pretrained_models.zip -d /kaggle/working/film_net/")
    os.system("rm pretrained_models.zip")
    print("Model ready.")
else:
    print("Model already present. Skipping download.")


# ── CELL 3: Imports ──────────────────────────────────────────
import io
import cv2
import glob
import shutil
import subprocess
import logging
import socket
import requests
import numpy as np
import tensorflow as tf
from pathlib import Path
from flask import Flask, request, send_file
from pyngrok import ngrok


# ── CELL 4: ngrok auth ───────────────────────────────────────
try:
    from kaggle_secrets import UserSecretsClient
    NGROK_TOKEN = UserSecretsClient().get_secret("NGROK_AUTH_TOKEN")
except Exception:
    NGROK_TOKEN = "YOUR_TOKEN_HERE"

logging.getLogger('werkzeug').disabled = True
ngrok.set_auth_token(NGROK_TOKEN)


# ── CELL 5: Load FILM model ───────────────────────────────────
print("Loading FILM model...")
_film_model = tf.saved_model.load(MODEL_PATH)
_film_infer = _film_model.signatures["serving_default"]
print("FILM model loaded successfully.")


# ══════════════════════════════════════════════════════════════
# STEP 1 — FFmpeg: Extract frames as PNG files
# ══════════════════════════════════════════════════════════════

def extract_frames(video_path: str, frames_dir: str) -> tuple[list[str], float]:
    """
    Use FFmpeg to extract every frame from the video as a PNG file.
    Returns (sorted list of PNG paths, original FPS).
    """
    os.makedirs(frames_dir, exist_ok=True)

    probe = subprocess.run([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ], capture_output=True, text=True)

    try:
        num, den = probe.stdout.strip().split("/")
        fps = float(num) / float(den)
    except Exception:
        fps = 30.0

    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        os.path.join(frames_dir, "frame_%06d.png")
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    print(f"[Step 1] Extracted {len(frame_paths)} frames at {fps:.2f} FPS -> {frames_dir}/")
    return frame_paths, fps


# ══════════════════════════════════════════════════════════════
# STEP 2 — OpenCV Optical Flow: Classify each segment
# ══════════════════════════════════════════════════════════════

SLOW_THRESHOLD   = 2.0
MEDIUM_THRESHOLD = 8.0

def classify_motion(frame_path_a: str, frame_path_b: str) -> str:
    """
    Compute dense optical flow between two frames.
    Returns 'slow', 'medium', or 'fast'.
    """
    img_a = cv2.imread(frame_path_a, cv2.IMREAD_GRAYSCALE)
    img_b = cv2.imread(frame_path_b, cv2.IMREAD_GRAYSCALE)

    if img_a is None or img_b is None:
        return "medium"

    flow      = cv2.calcOpticalFlowFarneback(
                    img_a, img_b, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    mean_mag  = float(np.mean(magnitude))

    if mean_mag < SLOW_THRESHOLD:
        return "slow"
    elif mean_mag < MEDIUM_THRESHOLD:
        return "medium"
    else:
        return "fast"

def classify_all_segments(frame_paths: list[str]) -> list[str]:
    """
    Classify motion for every consecutive frame pair.
    Returns a list of labels, length = len(frame_paths) - 1.
    """
    labels = []
    total  = len(frame_paths) - 1
    for i in range(total):
        label = classify_motion(frame_paths[i], frame_paths[i + 1])
        labels.append(label)
        if (i + 1) % 30 == 0 or (i + 1) == total:
            slow   = labels.count("slow")
            medium = labels.count("medium")
            fast   = labels.count("fast")
            print(f"   Classified {i+1}/{total} segments  |  slow: {slow}  medium: {medium}  fast: {fast}")
    return labels


# ══════════════════════════════════════════════════════════════
# STEP 3 — FILM: Generate new frames between each pair
# ══════════════════════════════════════════════════════════════

MOTION_PASSES = {
    "slow":   0,
    "medium": 0,
    "fast":   1,
}

def _film_midpoint(img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
    """Run FILM once to get the midpoint frame. Inputs: float32 [H,W,3] in [0,1]."""
    x0 = tf.cast(tf.expand_dims(img_a, 0), tf.float32)
    x1 = tf.cast(tf.expand_dims(img_b, 0), tf.float32)
    dt = tf.constant([[0.5]], dtype=tf.float32)

    result = _film_infer(x0=x0, x1=x1, time=dt)

    if "interpolated_image" in result:
        out_key = "interpolated_image"
    elif "image" in result:
        out_key = "image"
    else:
        out_key = [k for k in result.keys() if result[k].shape[-1] == 3][0]

    return tf.squeeze(result[out_key], axis=0).numpy()

def _load_frame(path: str) -> np.ndarray:
    """Load PNG as float32 RGB in [0, 1]."""
    bgr = cv2.imread(path).astype(np.float32) / 255.0
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def _save_frame(img: np.ndarray, path: str):
    """Save float32 RGB [0,1] as PNG."""
    bgr = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)

def generate_between_pair(img_a: np.ndarray, img_b: np.ndarray,
                           passes: int) -> list[np.ndarray]:
    """
    Recursively generate intermediate frames using FILM.
    passes=1 -> [mid]           (1 new frame)
    passes=2 -> [q1, mid, q3]  (3 new frames)
    passes=3 -> 7 new frames
    Returns frames in chronological order, NOT including a or b.
    """
    if passes == 0:
        return []
    mid   = _film_midpoint(img_a, img_b)
    left  = generate_between_pair(img_a, mid, passes - 1)
    right = generate_between_pair(mid, img_b, passes - 1)
    return left + [mid] + right

def interpolate_all_segments(frame_paths: list[str],
                              motion_labels: list[str],
                              interp_dir: str,
                              global_passes: int) -> dict[int, list[str]]:
    """
    For each consecutive pair, generate new frames and save to interp_dir.
    Returns dict: {pair_index -> [ordered list of new frame file paths]}
    """
    os.makedirs(interp_dir, exist_ok=True)
    new_frames_map = {}
    total = len(frame_paths) - 1

    for i in range(total):
        label  = motion_labels[i]
        passes = global_passes + MOTION_PASSES.get(label, 0)

        img_a   = _load_frame(frame_paths[i])
        img_b   = _load_frame(frame_paths[i + 1])
        between = generate_between_pair(img_a, img_b, passes)

        saved = []
        for j, frame in enumerate(between):
            out_path = os.path.join(interp_dir, f"new_{i:06d}_{j:04d}.png")
            _save_frame(frame, out_path)
            saved.append(out_path)

        new_frames_map[i] = saved

        if (i + 1) % 20 == 0 or (i + 1) == total:
            print(f"   Pair {i+1}/{total}  [{label}, {passes} pass(es) -> {len(between)} new frames]")

    return new_frames_map


# ══════════════════════════════════════════════════════════════
# STEP 4 — Frame Interleaver: Sort original + new frames
# ══════════════════════════════════════════════════════════════

def interleave_frames(frame_paths: list[str],
                      new_frames_map: dict[int, list[str]],
                      output_dir: str) -> list[str]:
    """
    Weave original and generated frames into final chronological order.
    Copies everything into output_dir with sequential names for FFmpeg.
    """
    os.makedirs(output_dir, exist_ok=True)
    counter = 0

    for i, orig_path in enumerate(frame_paths):
        dst = os.path.join(output_dir, f"final_{counter:07d}.png")
        shutil.copy2(orig_path, dst)
        counter += 1

        if i in new_frames_map:
            for new_path in new_frames_map[i]:
                dst = os.path.join(output_dir, f"final_{counter:07d}.png")
                shutil.copy2(new_path, dst)
                counter += 1

    final_paths = sorted(glob.glob(os.path.join(output_dir, "final_*.png")))
    print(f"[Step 4] Interleaved -> {len(final_paths)} total frames")
    return final_paths


# ══════════════════════════════════════════════════════════════
# STEP 5 — FFmpeg: Stitch frames back into .mp4
# ══════════════════════════════════════════════════════════════

def stitch_frames(final_frames_dir: str, fps: float,
                  output_path: str, original_video_path: str) -> str:
    """
    Encode all final_*.png frames into MP4 at original FPS,
    then mux back the original audio (skipped silently if no audio track).
    """
    temp_video    = output_path.replace(".mp4", "_noaudio.mp4")
    frame_pattern = os.path.join(final_frames_dir, "final_%07d.png")

    subprocess.run([
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i",         frame_pattern,
        "-vcodec",    "libx264",
        "-preset",    "fast",
        "-crf",       "18",
        "-pix_fmt",   "yuv420p",
        temp_video
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    result = subprocess.run([
        "ffmpeg", "-y",
        "-i", temp_video,
        "-i", original_video_path,
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        output_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if result.returncode != 0:
        shutil.move(temp_video, output_path)
    elif os.path.exists(temp_video):
        os.remove(temp_video)

    print(f"[Step 5] Output written -> {output_path}")
    return output_path


# ══════════════════════════════════════════════════════════════
# MASTER PIPELINE
# ══════════════════════════════════════════════════════════════

SPEED_TO_PASSES = {0.5: 1, 0.25: 2, 0.125: 3}

def run_pipeline(input_path: str, speed: float = 0.5,
                 output_path: str = "masif_output.mp4") -> tuple[str, float]:
    """
    Full MASIF pipeline:
      1. FFmpeg       -> extract frames as PNG
      2. OpenCV       -> classify each segment (slow / medium / fast)
      3. FILM         -> generate new frames per pair
      4. Interleaver  -> merge original + new frames in order
      5. FFmpeg       -> stitch back into .mp4 with audio
    """
    work_dir   = f"masif_work_{os.getpid()}"
    frames_dir = os.path.join(work_dir, "frames")
    interp_dir = os.path.join(work_dir, "interpolated")
    final_dir  = os.path.join(work_dir, "final")

    try:
        print(f"\n{'='*55}")
        print(f"MASIF  |  speed={speed}x  |  {os.path.basename(input_path)}")
        print(f"{'='*55}")

        global_passes = SPEED_TO_PASSES.get(round(speed, 4), 1)

        print("\n[Step 1] Extracting frames with FFmpeg...")
        frame_paths, fps = extract_frames(input_path, frames_dir)
        if len(frame_paths) < 2:
            raise RuntimeError("Video too short — need at least 2 frames.")

        print("\n[Step 2] Classifying motion with OpenCV optical flow...")
        motion_labels = classify_all_segments(frame_paths)
        print(f"   slow: {motion_labels.count('slow')}  "
              f"medium: {motion_labels.count('medium')}  "
              f"fast: {motion_labels.count('fast')}")

        print(f"\n[Step 3] FILM interpolation  (base {global_passes} pass(es) + motion bonus)...")
        new_frames_map = interpolate_all_segments(
            frame_paths, motion_labels, interp_dir, global_passes
        )

        print("\n[Step 4] Interleaving frames...")
        interleave_frames(frame_paths, new_frames_map, final_dir)

        print("\n[Step 5] Encoding output with FFmpeg...")
        stitch_frames(final_dir, fps, output_path, input_path)

        print(f"\nPipeline complete -> {output_path}\n")
        return output_path, fps

    finally:
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)


# ══════════════════════════════════════════════════════════════
# Flask API
# ══════════════════════════════════════════════════════════════

app = Flask(__name__)

@app.route('/')
def home():
    return "MASIF backend is running."

@app.route('/process', methods=['POST'])
def process_video():
    input_path  = f"input_{os.getpid()}.mp4"
    output_path = f"masif_output_{os.getpid()}.mp4"

    try:
        data         = request.get_json(force=True, silent=True) or {}
        video_url    = data.get('video_url')
        speed_option = float(data.get('speed', 0.5))

        if not video_url:
            return {"error": "No video_url provided"}, 400

        print(f"\nDownloading input video...")
        r = requests.get(video_url, timeout=60)
        r.raise_for_status()
        with open(input_path, 'wb') as f:
            f.write(r.content)
        print(f"   {os.path.getsize(input_path)/1024/1024:.1f} MB received")

        out_file, _ = run_pipeline(input_path, speed=speed_option,
                                   output_path=output_path)

        print("Sending result to client...")
        buf = io.BytesIO()
        with open(out_file, 'rb') as f:
            buf.write(f.read())
        buf.seek(0)
        return send_file(buf, mimetype='video/mp4')

    except Exception as e:
        print(f"Error: {e}")
        import traceback; traceback.print_exc()
        return {"error": str(e)}, 500

    finally:
        for p in [input_path, output_path]:
            if os.path.exists(p):
                os.remove(p)


# ══════════════════════════════════════════════════════════════
# Startup
# ══════════════════════════════════════════════════════════════

def get_free_port(start=5000):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    p = start
    while True:
        try:
            s.bind(('127.0.0.1', p)); s.close(); return p
        except OSError:
            p += 1

def start_backend():
    ngrok.kill()
    port = get_free_port()
    try:
        url = ngrok.connect(port).public_url
        print("\n" + "=" * 60)
        print("MASIF BACKEND IS LIVE")
        print(f"Endpoint : {url}/process")
        print("Paste this URL into your bot .env as COLAB_BACKEND_URL")
        print("=" * 60 + "\n")
        app.run(port=port, debug=False, use_reloader=False)
    except Exception as e:
        print(f"ngrok error: {e}")

if __name__ == "__main__":
    start_backend()
