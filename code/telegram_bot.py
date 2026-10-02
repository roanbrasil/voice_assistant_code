"""The assistant on your phone without writing an app: a Telegram bot on the same Pi/Jetson.
Accepts text and voice notes, replies in text. Only talks to YOUR chat_id."""
import os
import tempfile

from faster_whisper import WhisperModel
from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from brain import make_brain

ALLOWED = {int(x) for x in os.environ["TG_ALLOWED_CHAT_IDS"].split(",")}
brain = make_brain()
stt = WhisperModel(os.getenv("WHISPER", "small"), device="cpu", compute_type="int8")
histories: dict[int, list] = {}


async def handle(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat.id
    if chat not in ALLOWED:
        return                                   # strangers get silence
    msg = update.message
    if msg.voice:
        f = await msg.voice.get_file()
        with tempfile.NamedTemporaryFile(suffix=".ogg") as tmp:
            await f.download_to_drive(tmp.name)
            segs, _ = stt.transcribe(tmp.name, language="en", vad_filter=True)
            text = " ".join(s.text for s in segs).strip()
        await msg.reply_text(f"🎙️ {text}")
    else:
        text = msg.text
    h = histories.setdefault(chat, [])
    h.append({"role": "user", "content": text})
    answer = brain.ask(h[-9:])
    h.append({"role": "assistant", "content": answer})
    await msg.reply_text(answer)


def main():
    app = Application.builder().token(os.environ["TG_TOKEN"]).build()
    app.add_handler(MessageHandler(filters.TEXT | filters.VOICE, handle))
    app.run_polling()                            # polling: no inbound port opened on your network


if __name__ == "__main__":
    main()
