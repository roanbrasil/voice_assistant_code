# Voice Assistant

A household voice assistant that runs on a Raspberry Pi 5 or a Jetson. It listens for a
wake word, transcribes locally, answers through a tool-calling loop, and speaks the answer
back. The brain is swappable: a hosted model or a local one through Ollama. Nothing but the
brain has to leave the machine, and even that is optional.

Three ways to talk to it: the local voice loop, a full-duplex live-audio variant you can
interrupt, and a Telegram bot for when you are not home.

## How it works

```
wake word          speech to text        brain                text to speech
openWakeWord  ->   faster-whisper   ->   tool-calling loop ->  Piper
```

`voice_local.py` holds that loop. It reads 80 ms blocks of 16 kHz mono audio, waits for the
wake word to cross 0.5, records until 0.8 s of silence (capped at 12 s), transcribes, asks
the brain, and pipes the answer through Piper into `aplay`. The last nine turns go to the
brain as context.

## Files

| File | What it is |
| --- | --- |
| `code/voice_local.py` | the local loop: wake word, recording, STT, brain, TTS |
| `code/voice_live.py` | full-duplex variant — live audio model as ears and mouth, the brain reached by client delegation, so you can interrupt it |
| `code/brain.py` | the tool-calling loop, with two interchangeable backends behind one `ask()` |
| `code/tools.py` | the four tools and their JSON schemas |
| `code/telegram_bot.py` | a Telegram bot that accepts text and voice notes, restricted to known chat ids |
| `code/assistant.service` | systemd unit for the local loop |

## The brain

One interface, two implementations, chosen by the `BRAIN` variable:

| `BRAIN` | Backend | Needs |
| --- | --- | --- |
| `claude` (default) | Messages API | `ANTHROPIC_API_KEY` |
| `ollama` | a local model over Ollama's OpenAI-compatible endpoint | Ollama running |

Both run the same loop: ask, run any requested tools, feed the results back, repeat. The
loop is capped at six iterations so a confused agent stops instead of spinning. The system
prompt keeps answers to two sentences, because they are read aloud, and forbids inventing a
temperature, a time or a device state.

## The tools

| Tool | What it does |
| --- | --- |
| `weather` | current conditions and today's forecast from Open-Meteo — no API key |
| `take_note` | appends a timestamped line to the notes file |
| `read_notes` | reads back the most recent notes |
| `home` | calls a Home Assistant service |

`home` only accepts the domains `light`, `switch`, `cover`, `media_player` and `scene`. The
model does not get to name any domain it likes. Tool errors are returned to the model as
JSON rather than raised, so it can explain the failure instead of crashing the loop.

## Running it

Install the Python dependencies:

```bash
pip install numpy sounddevice faster-whisper openwakeword httpx anthropic openai python-telegram-bot
```

Piper and ALSA come from the system:

```bash
sudo apt install alsa-utils    # provides aplay
# install piper and download a voice: https://github.com/rhasspy/piper
```

Then:

```bash
python code/voice_local.py     # the local loop
python code/voice_live.py      # the full-duplex variant
python code/telegram_bot.py    # the bot
```

## Configuration

Everything comes from the environment; nothing is hardcoded.

| Variable | Default | Used by |
| --- | --- | --- |
| `BRAIN` | `claude` | which backend to use |
| `ANTHROPIC_API_KEY` | — | the hosted brain |
| `CLAUDE_MODEL` | `claude-haiku-4-5` | the hosted brain |
| `OLLAMA_URL` | `http://localhost:11434/v1` | the local brain |
| `OLLAMA_MODEL` | `qwen3:4b` | the local brain |
| `WAKE_MODEL` | `hey_jarvis` | wake word — swap in an `.onnx` you trained |
| `WHISPER` | `small` | STT model size |
| `WHISPER_DEVICE` | `cpu` | `cpu` on a Pi, `cuda` on a Jetson |
| `WHISPER_COMPUTE` | `int8` | quantisation |
| `PIPER_VOICE` | `en_US-lessac-medium.onnx` | the voice |
| `ASSIST_NOTES` | `~/assistant/notes.md` | where notes are kept |
| `HA_URL` | `http://homeassistant.local:8123` | Home Assistant |
| `HA_TOKEN` | — | Home Assistant long-lived token |
| `TG_TOKEN` | — | the Telegram bot |
| `TG_ALLOWED_CHAT_IDS` | — | comma-separated chat ids allowed to talk to the bot |
| `OPENAI_API_KEY` | — | the live-audio variant |

On a Pi 5, `small` at `int8` on the CPU is the usable setting. On a Jetson,
`large-v3-turbo` on `cuda` fits comfortably.

## Running as a service

`assistant.service` expects the project at `/home/pi/assistant`, a virtualenv at
`.venv`, and the variables above in `/home/pi/assistant/.env`. Adjust the paths and user
if yours differ.

```bash
sudo cp code/assistant.service /etc/systemd/system/
sudo systemctl enable --now assistant
journalctl -u assistant -f
```

The `.env` file is gitignored. Keep it that way — it holds your Telegram and Home Assistant
tokens.

## Notes on the other two front ends

**The live variant** uses a live-audio model for the ears and mouth and delegates the actual
thinking back to the same brain, so the tools and the system prompt stay in one place.
Delegation events do not carry the request text, so the recent transcript is kept locally
and passed along. It is full duplex, which means a microphone with acoustic echo
cancellation is not optional — otherwise it hears itself.

**The Telegram bot** polls rather than listening, so no inbound port is opened on your
network. Messages from a chat id outside the allowlist get silence, not an error. Voice
notes are transcribed with the same Whisper model and echoed back so you can see what it
heard.

## Limits

- The wake-word threshold is a flat 0.5 with no adaptation to room noise.
- History is a flat window of the last nine turns, with no summarisation; a long
  conversation simply forgets its beginning.
- Notes are a flat Markdown file. There is no search beyond reading the last few lines.
- `home` reports the HTTP status of the call but does not read device state back, so the
  assistant cannot confirm that a light actually turned on.
