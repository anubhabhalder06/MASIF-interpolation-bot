import os
import asyncio
import aiohttp
import time
from datetime import datetime
from collections import deque
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv

load_dotenv()


class VideoProcessingQueue:
    def __init__(self):
        self.waiting_queue = deque()
        self.is_processing = False
        self.current_user = None
        self.processing_start_time = None
        self.estimated_time_per_video = 180
        self.max_queue_size = 50
        self.cancelled_during_processing = set()

    def add_to_queue(self, user_id, video_info, processing_params, bot):
        if len(self.waiting_queue) >= self.max_queue_size:
            return False, "Queue is full. Try again later."

        position = len(self.waiting_queue) + 1
        self.waiting_queue.append({
            "user_id": user_id,
            "video_info": video_info,
            "processing_params": processing_params,
            "joined_at": datetime.now(),
            "bot": bot,
            "input_file": None,
            "output_file": None,
        })
        return True, position

    def get_queue_position(self, user_id):
        for i, item in enumerate(self.waiting_queue):
            if item["user_id"] == user_id:
                return i + 1
        return None

    def is_user_processing(self, user_id):
        return self.is_processing and self.current_user and self.current_user["user_id"] == user_id

    def get_wait_time(self, position):
        if self.is_processing:
            elapsed = (datetime.now() - self.processing_start_time).seconds
            remaining = max(0, self.estimated_time_per_video - elapsed)
            return remaining + (position - 1) * self.estimated_time_per_video
        return (position - 1) * self.estimated_time_per_video

    def cancel_queued(self, user_id):
        for i, item in enumerate(self.waiting_queue):
            if item["user_id"] == user_id:
                self._cleanup_files(item)
                del self.waiting_queue[i]
                return True
        return False

    def cancel_processing(self, user_id):
        if self.is_user_processing(user_id):
            self.cancelled_during_processing.add(user_id)
            return True
        return False

    def is_cancelled(self, user_id):
        return user_id in self.cancelled_during_processing

    def clear_cancelled(self, user_id):
        self.cancelled_during_processing.discard(user_id)

    def _cleanup_files(self, queue_item):
        for key in ("input_file", "output_file"):
            path = queue_item.get(key)
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                    print(f"🗑️ Deleted: {path}")
                except Exception as e:
                    print(f"⚠️ Could not delete {path}: {e}")

    def get_queue_info(self):
        if not self.waiting_queue and not self.is_processing:
            return "🟢 Queue is empty — instant processing available!"
        info = []
        if self.is_processing:
            elapsed = (datetime.now() - self.processing_start_time).seconds
            remaining = max(0, self.estimated_time_per_video - elapsed)
            info.append(f"🟡 Processing 1 video — ~{remaining // 60}m {remaining % 60}s left")
        if self.waiting_queue:
            info.append(f"📋 {len(self.waiting_queue)} video(s) waiting")
        return "\n".join(info)

    async def process_next(self, bot_instance):
        if not self.is_processing and self.waiting_queue:
            self.is_processing = True
            self.current_user = self.waiting_queue[0]
            self.processing_start_time = datetime.now()
            bot = self.current_user["bot"]

            keyboard = [[InlineKeyboardButton("❌ Cancel & Delete", callback_data="cancel_active")]]
            await bot.send_message(
                self.current_user["user_id"],
                f"⚙️ *Processing started!*\n\n"
                f"Speed: {self.current_user['processing_params']['slowmo_factor']}x\n"
                f"Est. time: ~{self.estimated_time_per_video // 60} min\n\n"
                f"_You'll receive your video automatically._",
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode='Markdown'
            )
            asyncio.create_task(self._process(bot_instance))

    async def _process(self, bot_instance):
        bot = self.current_user["bot"]
        user_id = self.current_user["user_id"]
        try:
            success = await bot_instance.send_to_colab(self.current_user)

            if self.is_cancelled(user_id):
                self.clear_cancelled(user_id)
                await bot.send_message(user_id,
                    "🗑️ *Cancelled.* Your video and all files have been deleted.",
                    parse_mode='Markdown'
                )
            elif success:
                await bot.send_message(user_id,
                    "✅ *Done!* Your video is above.\n\n_All temp files deleted from server._",
                    parse_mode='Markdown'
                )
            else:
                await bot.send_message(user_id,
                    "❌ *Processing failed.* Colab may be offline.\n\nUse /start to try again.",
                    parse_mode='Markdown'
                )
        except Exception as e:
            print(f"Queue process error: {e}")
            await bot.send_message(user_id,
                f"❌ *Error:* {str(e)[:100]}\n\nUse /start to try again.",
                parse_mode='Markdown'
            )
        finally:
            self._cleanup_files(self.current_user)
            self.waiting_queue.popleft()
            self.is_processing = False
            self.current_user = None
            await self.process_next(bot_instance)


class SlowMoBot:
    def __init__(self):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.colab_url = os.getenv("COLAB_BACKEND_URL")

        if not self.token:
            raise ValueError("TELEGRAM_BOT_TOKEN missing from .env")
        if not self.colab_url:
            raise ValueError("COLAB_BACKEND_URL missing from .env")

        self.queue = VideoProcessingQueue()
        self.sessions = {}

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        name = update.effective_user.first_name

        text = (
            f"👋 Hi {name}!\n\n"
            f"*MASIF* — AI slow-motion & video smoother\n\n"
            f"• Convert any FPS video to buttery slow-mo\n"
            f"• Fix jittery or choppy footage\n"
            f"• Free, no watermarks\n\n"
            f"{self.queue.get_queue_info()}"
        )
        keyboard = [
            [InlineKeyboardButton("🎬 Slow-Mo", callback_data="create_slowmo"),
             InlineKeyboardButton("🔄 Fix Jitter", callback_data="fix_jitter")],
            [InlineKeyboardButton("📊 Queue", callback_data="queue_status"),
             InlineKeyboardButton("❌ Cancel", callback_data="cancel_request")],
            [InlineKeyboardButton("❓ How it works", callback_data="how_it_works")]
        ]
        if update.message:
            await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='Markdown')
        else:
            await update.callback_query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='Markdown')

    async def how_it_works(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        text = (
            "*How MASIF works*\n\n"
            "1️⃣ Send your video\n"
            "2️⃣ Choose speed / fix level\n"
            "3️⃣ AI fills in missing frames\n"
            "4️⃣ Receive your smooth video\n\n"
            "⚠️ Requires Colab backend to be running."
        )
        await query.edit_message_text(text,
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("◀️ Back", callback_data="back_to_menu")]]),
            parse_mode='Markdown'
        )

    async def create_slowmo_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        user_id = update.effective_user.id

        if self.queue.get_queue_position(user_id) or self.queue.is_user_processing(user_id):
            await query.edit_message_text(
                "⚠️ You already have a video in progress.\nUse /cancel to remove it first.",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("◀️ Back", callback_data="back_to_menu")]]),
                parse_mode='Markdown'
            )
            return

        self.sessions[user_id] = {"mode": "slowmo"}
        await query.edit_message_text(
            "*🎬 Send your video*\n\n"
            "• MP4 / MOV / AVI\n"
            "• Max 20MB, max 60s\n"
            "• Any FPS works",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("◀️ Back", callback_data="back_to_menu")]]),
            parse_mode='Markdown'
        )

    async def fix_jitter_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        user_id = update.effective_user.id

        if self.queue.get_queue_position(user_id) or self.queue.is_user_processing(user_id):
            await query.edit_message_text(
                "⚠️ You already have a video in progress.\nUse /cancel to remove it first.",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("◀️ Back", callback_data="back_to_menu")]]),
                parse_mode='Markdown'
            )
            return

        self.sessions[user_id] = {"mode": "jitter_fix"}
        await query.edit_message_text(
            "*🔄 Send your jittery video*\n\n"
            "• MP4 / MOV / AVI\n"
            "• Max 20MB, max 60s",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("◀️ Back", callback_data="back_to_menu")]]),
            parse_mode='Markdown'
        )

    async def handle_video(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id

        if user_id not in self.sessions:
            await update.message.reply_text("Use /start first to set up your request.")
            return

        if self.queue.get_queue_position(user_id) or self.queue.is_user_processing(user_id):
            await update.message.reply_text("⚠️ You already have a video in progress. Use /cancel to remove it.")
            return

        video = update.message.video
        if not video:
            await update.message.reply_text("Please send a valid video file.")
            return

        if video.file_size > 20 * 1024 * 1024:
            await update.message.reply_text("❌ File too large. Max 20MB.")
            return

        video_info = {
            "file_id": video.file_id,
            "duration": video.duration,
            "size_mb": round(video.file_size / (1024 * 1024), 2),
        }
        self.sessions[user_id]["video_info"] = video_info

        mode = self.sessions[user_id]["mode"]
        label = "Slow-Mo Speed" if mode == "slowmo" else "Fix Level"
        prefix = "slowmo" if mode == "slowmo" else "jitter"

        await update.message.reply_text(
            f"✅ *Video received* ({video_info['duration']}s, {video_info['size_mb']}MB)\n\n*Choose {label}:*",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("⚡ 0.5x — Fast", callback_data=f"{prefix}_60_0.5")],
                [InlineKeyboardButton("🔥 0.25x — Popular", callback_data=f"{prefix}_120_0.25")],
                [InlineKeyboardButton("💎 0.125x — Extreme", callback_data=f"{prefix}_240_0.125")],
                [InlineKeyboardButton("◀️ Cancel", callback_data="back_to_menu")]
            ]),
            parse_mode='Markdown'
        )

    async def add_to_queue(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        user_id = update.effective_user.id

        try:
            parts = query.data.split("_", maxsplit=2)
            mode, target_fps, slowmo_factor = parts[0], int(parts[1]), float(parts[2])
        except (IndexError, ValueError):
            await query.edit_message_text("❌ Invalid selection. Please try again with /start")
            return

        if user_id not in self.sessions or "video_info" not in self.sessions.get(user_id, {}):
            await query.edit_message_text(
                "⚠️ Session expired. Please /start again and re-upload your video.",
                parse_mode='Markdown'
            )
            return

        video_info = self.sessions[user_id]["video_info"]
        params = {"mode": mode, "target_fps": target_fps, "slowmo_factor": slowmo_factor}

        success, result = self.queue.add_to_queue(user_id, video_info, params, context.bot)
        if not success:
            await query.edit_message_text(f"❌ {result}")
            return

        position = result
        wait_time = self.queue.get_wait_time(position)
        wait_display = f"~{wait_time // 60}m {wait_time % 60}s" if wait_time > 0 else "Starting now"

        await query.edit_message_text(
            f"✅ *Queued!*\n\n"
            f"Position: #{position}\n"
            f"Est. wait: {wait_display}\n\n"
            f"_You'll be notified when processing starts._",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("◀️ Menu", callback_data="back_to_menu")]]),
            parse_mode='Markdown'
        )

        if not self.queue.is_processing:
            await self.queue.process_next(self)

    async def cancel_request(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        if query:
            await query.answer()
        user_id = update.effective_user.id

        if self.queue.is_user_processing(user_id):
            self.queue.cancel_processing(user_id)
            text = (
                "⏳ *Cancellation requested.*\n\n"
                "Will stop after current step and delete all files.\n"
                "_May take up to 30 seconds._"
            )
        elif self.queue.cancel_queued(user_id):
            text = "🗑️ *Cancelled.* Removed from queue and files deleted."
        else:
            text = "ℹ️ No active request found."

        keyboard = [[InlineKeyboardButton("◀️ Menu", callback_data="back_to_menu")]]
        if query:
            await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='Markdown')
        else:
            await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='Markdown')

    async def cancel_active(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        user_id = update.effective_user.id

        if self.queue.is_user_processing(user_id):
            self.queue.cancel_processing(user_id)
            await query.edit_message_text(
                "⏳ *Cancellation requested.*\n\n"
                "Stopping after current step. Files will be deleted.\n"
                "_May take up to 30 seconds._",
                parse_mode='Markdown'
            )
        else:
            await query.edit_message_text("ℹ️ Nothing is currently processing for you.")

    async def queue_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        if query:
            await query.answer()
        user_id = update.effective_user.id

        processing = self.queue.is_user_processing(user_id)
        position = self.queue.get_queue_position(user_id)

        if processing:
            elapsed = (datetime.now() - self.queue.processing_start_time).seconds
            remaining = max(0, self.queue.estimated_time_per_video - elapsed)
            text = (
                f"*Your video is processing now*\n\n"
                f"⏱️ ~{remaining // 60}m {remaining % 60}s remaining"
            )
            keyboard = [
                [InlineKeyboardButton("❌ Cancel & Delete", callback_data="cancel_active")],
                [InlineKeyboardButton("◀️ Menu", callback_data="back_to_menu")]
            ]
        elif position:
            wait = self.queue.get_wait_time(position)
            text = f"*Queue position: #{position}*\n\nEst. wait: ~{wait // 60}m {wait % 60}s"
            keyboard = [
                [InlineKeyboardButton("❌ Cancel & Delete", callback_data="cancel_request")],
                [InlineKeyboardButton("◀️ Menu", callback_data="back_to_menu")]
            ]
        else:
            text = f"*Queue Status*\n\n{self.queue.get_queue_info()}"
            keyboard = [[InlineKeyboardButton("◀️ Menu", callback_data="back_to_menu")]]

        if query:
            await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='Markdown')
        else:
            await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='Markdown')

    async def send_to_colab(self, queue_item):
        bot = queue_item["bot"]
        user_id = queue_item["user_id"]
        input_file = None
        output_file = None
        video_sent = False

        try:
            print(f"📥 Downloading video for user {user_id}...")
            tg_file = await bot.get_file(queue_item["video_info"]["file_id"])
            file_url = tg_file.file_path

            input_file = f"input_{user_id}_{int(time.time())}.mp4"
            queue_item["input_file"] = input_file

            async with aiohttp.ClientSession() as dl:
                async with dl.get(file_url) as r:
                    with open(input_file, 'wb') as f:
                        f.write(await r.read())

            if self.queue.is_cancelled(user_id):
                return False

            print(f"🔗 Sending to Colab...")
            payload = {
                "video_url": file_url,
                "speed": str(queue_item["processing_params"]["slowmo_factor"])
            }

            timeout = aiohttp.ClientTimeout(total=600)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self.colab_url,
                    json=payload,
                    headers={
                        "User-Agent": "Mozilla/5.0",
                        "ngrok-skip-browser-warning": "true",
                        "Content-Type": "application/json"
                    },
                    ssl=False
                ) as response:

                    if self.queue.is_cancelled(user_id):
                        return False

                    if response.status == 200:
                        output_file = f"output_{user_id}_{int(time.time())}.mp4"
                        queue_item["output_file"] = output_file

                        with open(output_file, 'wb') as f:
                            f.write(await response.read())

                        if self.queue.is_cancelled(user_id):
                            return False

                        print(f"📤 Sending result to user {user_id}...")
                        with open(output_file, 'rb') as v:
                            await bot.send_video(chat_id=user_id, video=v)

                        video_sent = True

                        if os.path.exists(output_file):
                            os.remove(output_file)
                            print(f"🗑️ Output deleted: {output_file}")
                        queue_item["output_file"] = None

                        return True
                    else:
                        err = await response.text()
                        print(f"❌ Colab error {response.status}: {err[:200]}")
                        return False

        except asyncio.TimeoutError:
            if video_sent:
                print("⚠️ Timeout after send — video was already delivered successfully")
                return True
            print("❌ Colab timed out")
            await bot.send_message(user_id,
                "⏰ *Timed out.* Your video took too long. Try a shorter clip.",
                parse_mode='Markdown'
            )
            return False

        except aiohttp.ClientConnectorError:
            print("❌ Cannot connect to Colab")
            await bot.send_message(user_id,
                "🔌 *Backend offline.* Colab is not running. Notify the admin.",
                parse_mode='Markdown'
            )
            return False

        except Exception as e:
            print(f"❌ send_to_colab error: {e}")
            return False

        finally:
            if input_file and os.path.exists(input_file):
                os.remove(input_file)
                print(f"🗑️ Input deleted: {input_file}")
            queue_item["input_file"] = None

    async def back_to_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        await self.start(update, context)

    async def cancel_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self.cancel_request(update, context)

    async def queue_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self.queue_status(update, context)

    async def setup_commands(self, application: Application):
        await application.bot.set_my_commands([
            BotCommand("start", "Open main menu"),
            BotCommand("queue", "Check your queue position"),
            BotCommand("cancel", "Cancel & delete your request"),
        ])

    def run(self):
        app = Application.builder().token(self.token).post_init(self.setup_commands).build()

        app.add_handler(CommandHandler("start", self.start))
        app.add_handler(CommandHandler("queue", self.queue_command))
        app.add_handler(CommandHandler("cancel", self.cancel_command))

        app.add_handler(CallbackQueryHandler(self.how_it_works,        pattern="^how_it_works$"))
        app.add_handler(CallbackQueryHandler(self.create_slowmo_start, pattern="^create_slowmo$"))
        app.add_handler(CallbackQueryHandler(self.fix_jitter_start,    pattern="^fix_jitter$"))
        app.add_handler(CallbackQueryHandler(self.queue_status,        pattern="^queue_status$"))
        app.add_handler(CallbackQueryHandler(self.cancel_request,      pattern="^cancel_request$"))
        app.add_handler(CallbackQueryHandler(self.cancel_active,       pattern="^cancel_active$"))
        app.add_handler(CallbackQueryHandler(self.back_to_menu,        pattern="^back_to_menu$"))
        app.add_handler(CallbackQueryHandler(self.add_to_queue,        pattern="^slowmo_"))
        app.add_handler(CallbackQueryHandler(self.add_to_queue,        pattern="^jitter_"))

        app.add_handler(MessageHandler(filters.VIDEO, self.handle_video))

        print("🤖 Bot running. Colab backend must also be active for processing.")
        app.run_polling()


if __name__ == "__main__":
    bot = SlowMoBot()
    bot.run()
