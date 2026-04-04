# M.A.S.I.F — Motion-Aware Scheduling & Interpolation Framework

![MASIF Banner](https://img.shields.io/badge/MASIF-AI%20Slow%20Motion-blueviolet?style=for-the-badge&logo=tensorflow)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![Telegram](https://img.shields.io/badge/Telegram-Bot-26A5E4?style=flat-square&logo=telegram)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Cost](https://img.shields.io/badge/Cost-₹0-brightgreen?style=flat-square)

**AI-powered video frame interpolation with adaptive motion scheduling — delivered through Telegram. No installs. No GPU. No cost.**

---

## Why MASIF is Different

<p align="center">
  <img src="demo.gif" alt="MASIF AI Slow Motion Demo" width="100%" />
</p>

> **AI-powered smooth interpolation:** MASIF generates genuine new frames to transform standard video into cinematic slow-motion.

---

## What Makes MASIF Unique

Most tools that use Google's FILM model just wrap it and run it uniformly on every frame — no intelligence, no scheduling, no delivery system. MASIF is different in three ways:

**I-01 · Adaptive Motion-Aware Scheduling** — Before interpolating, MASIF analyses every frame pair using optical flow and classifies motion as low, medium, or high. It then applies 1×, 2×, or 3× FILM passes per segment accordingly. Static scenes don't waste compute. Fast scenes get the passes they actually need. No existing open-source FILM wrapper does this.

**I-02 · Visual Speed-Curve Editor** *(in progress)* — A Gradio interface where you draw a curve to control which part of the video slows down and by how much — like After Effects' time-remapping, but completely free and open-source. First implementation of its kind on top of FILM.

**I-03 · Telegram Delivery** — The entire GPU pipeline runs on the cloud. Users just send a video in Telegram and receive slow-mo back. No install, no account, no local GPU needed. First ML-heavy video pipeline deployed this way.

---

## What is MASIF?

MASIF is a full-stack AI video processing framework that takes any standard-FPS video and transforms it into smooth, cinematic slow-motion by intelligently generating the missing frames using **Google's FILM (Frame Interpolation for Large Motion)** model.

Unlike naive slow-motion tools that simply duplicate frames (causing choppy, stuttery output), MASIF uses a deep learning model that **pixel-predicts entirely new frames** — not blends, not duplicates. Each generated frame is a genuine AI reconstruction of what would have existed between two real frames.

The entire pipeline runs on **free Kaggle/Colab GPU**, and users interact with it through a **Telegram bot** — no installation, no account, no local GPU required on their end.

---

## The Problem It Solves

| Problem | How MASIF Fixes It |
|---|---|
| **Choppy slow-motion** — 24–30 FPS footage doesn't have enough frames for fluid slow-mo. Simply reducing playback speed reveals jarring jumps. | MASIF generates genuine new frames using FILM AI, so slowed footage stays silky smooth. |
| **Frame warping artifacts** — Most tools use basic optical flow blending, causing ghosting and pixel bending in fast-motion scenes. | FILM's large-motion-aware architecture handles fast movement without ghosting. |
| **High cost & complexity** — Tools like After Effects or Twixtor require expensive licences and powerful local hardware. | MASIF runs 100% on free cloud GPU. Zero licence. Zero hardware cost for the user. |

---



## System Architecture

```
┌─────────────────────────────────────────────────────┐
│                  USER INTERFACE LAYER                │
│                                                     │
│   Telegram Bot          │   Gradio Web UI           │
│   (Video in → out)      │   (Speed-curve editor)    │
└───────────────┬─────────────────────────────────────┘
                │ HTTP POST (video_url + speed)
                ▼
┌─────────────────────────────────────────────────────┐
│                  PROCESSING ENGINE                   │
│                                                     │
│  1. Frame Extraction    →  ffmpeg → PNG frames       │
│  2. Motion Classifier   →  OpenCV optical flow       │
│  3. FILM Interpolator   →  Google FILM TF Hub        │
│  4. Frame Interleaver   →  Sort + sequence all       │
└───────────────┬─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────┐
│                    OUTPUT LAYER                      │
│                                                     │
│  ffmpeg Encoder  →  Frames → .mp4 at original FPS   │
│  Speed Options   →  0.5× · 0.25× · 0.125×           │
│  Delivery        →  Telegram message / web download  │
└─────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
masif/
│
├── reference.mp4           # Demo video
├── bot/
│   ├── bot.py              # Telegram bot — queue system, user sessions, Colab bridge
│   └── .env.example        # Template for your environment variables
│
├── colab/
│   └── MASIF_backend.py    # Full Colab/Kaggle backend — FILM model + Flask + ngrok
│
├── requirements.txt        # Python dependencies for the Telegram bot
├── Procfile                # Render deployment entry point
├── railway.toml            # Railway config
└── README.md               # This file
```

---

## How to Run MASIF

MASIF has **two parts** that run independently:

### Part 1 — The AI Backend (Google Colab / Kaggle)

This is where the actual AI processing happens. It needs a GPU.

1. Open [Google Colab](https://colab.research.google.com/) or [Kaggle Notebooks](https://www.kaggle.com/code)
2. Enable GPU runtime: **Runtime → Change runtime type → T4 GPU**
3. Upload `colab/MASIF_backend.py` or paste its contents into a notebook
4. Add your `NGROK_AUTH_TOKEN` to Colab Secrets
5. Run all cells — the backend will print a live URL like:

```
🚀 MASIF BACKEND IS LIVE!
   https://your-domain.ngrok-free.app/process
```

6. Copy this URL into your `.env` as `COLAB_BACKEND_URL`

> **Note:** The Colab backend needs to be running for video processing to work. The Telegram bot can run 24/7 on Render independently — it will queue requests and attempt processing whenever the backend is live.

### Part 2 — The Telegram Bot (Render — runs 24/7)

**Option A: Run locally (for testing)**

```bash
git clone https://github.com/YOUR_USERNAME/masif.git
cd masif
pip install -r requirements.txt
cp bot/.env.example bot/.env
# Fill in your TELEGRAM_BOT_TOKEN and COLAB_BACKEND_URL
python bot/bot.py
```

**Option B: Deploy to Render (recommended — stays alive always)**

1. Go to [render.com](https://render.com) and sign up free
2. Click **New → Web Service**
3. Connect your GitHub account and select this repo
4. Set the following:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python bot/bot.py`
5. Go to **Environment** and add:

```
TELEGRAM_BOT_TOKEN=your_token_here
COLAB_BACKEND_URL=https://your-ngrok-url.ngrok-free.app/process
```

6. Click **Deploy**. Your bot is now online 24/7 — even when your laptop is off.

> **Updating the backend URL:** Every time you restart Colab (unless you have a static ngrok domain), the URL changes. Just update `COLAB_BACKEND_URL` in your Render service's Environment settings and redeploy.

---

## Environment Variables

Copy `bot/.env.example` to `bot/.env` and fill in your values:

| Variable | Description |
|---|---|
| `TELEGRAM_BOT_TOKEN` | From [@BotFather](https://t.me/botfather) on Telegram |
| `COLAB_BACKEND_URL` | The `/process` endpoint URL from your running Colab/Kaggle backend |

---

## Using the Bot

1. Find your bot on Telegram and send `/start`
2. Choose **🎬 Slow-Mo** or **🔄 Fix Jitter**
3. Send your video (MP4/MOV/AVI, max 20MB, max 60s)
4. Choose your speed: **0.5× · 0.25× · 0.125×**
5. Your video is queued and processed automatically
6. Receive the smooth slow-motion video back in chat

---

## Speed Options Explained

| Option | Speed | FILM Passes | Frames Generated | Best For |
|---|---|---|---|---|
| ⚡ Fast | 0.5× | 1 pass | 2× frames | General slow-mo |
| 🔥 Popular | 0.25× | 2 passes | 4× frames | Sports, action |
| 💎 Extreme | 0.125× | 3 passes | 8× frames | Cinematic, water, fire |

---

## Tech Stack

| Component | Technology |
|---|---|
| AI Model | [Google FILM](https://github.com/google-research/frame-interpolation) via TensorFlow Hub |
| Frame Extraction | ffmpeg |
| Motion Analysis | OpenCV optical flow (Farneback) |
| Video Encoding | ffmpeg libx264 |
| Bot Framework | python-telegram-bot v20 |
| Backend Server | Flask + ngrok |
| GPU | Google Colab / Kaggle (free) |
| Bot Hosting | Render |

---

## Queue System

MASIF includes a full async queue so multiple users can use the bot simultaneously:

- Each user gets a position number when they submit
- Estimated wait times are shown in real-time
- Users can cancel at any time — files are deleted immediately
- Temp files (input + output) are cleaned up automatically after each job

---

## Project Stats

| Metric | Value |
|---|---|
| Training Required | 0 (uses pre-trained FILM) |
| Cost to Run | ₹0 |
| Paid Dependencies | 0 |
| Original Contributions | 3 |

---

## Limitations

- Colab/Kaggle sessions expire — the backend must be manually restarted
- Max video size: 20MB / 60 seconds (Telegram bot limit)
- Processing time: ~3 minutes per video on T4 GPU
- Not suitable for 4K — MASIF downscales to 1280px on the longest edge

---

## Roadmap

- [ ] Kaggle auto-restart script
- [ ] Visual speed-curve editor (Gradio UI)
- [ ] Per-segment adaptive scheduling (full I-01 implementation)
- [ ] Support for audio preservation in output
- [ ] Web frontend (upload → download)

---

## License

MIT License — free to use, modify, and distribute.

---

## Acknowledgements

- [Google FILM](https://github.com/google-research/frame-interpolation) — the AI model at the heart of MASIF
- [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot)
- [ngrok](https://ngrok.com/) for tunneling the Colab backend

---

*Built with love by Team 140 · Project Exhibition II*
