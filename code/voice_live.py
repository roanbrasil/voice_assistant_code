"""GPT-Live-1 as the ears and mouth, Claude (or Ollama) as the brain, via client delegation.
Full-duplex: you can interrupt it. Needs OPENAI_API_KEY and a mic with AEC (e.g. XVF3800)."""
import asyncio
import base64

import numpy as np
import sounddevice as sd
from openai import AsyncOpenAI

from brain import make_brain

RATE, BLOCK = 24000, 480                      # 20 ms of mono PCM16 at 24 kHz
brain = make_brain()
transcript: list[str] = []                    # delegation events do NOT carry the request text: we keep it ourselves

LIVE_PROMPT = ("You are a home assistant. Speak English in short sentences. "
               "For weather, notes, reminders or controlling the house, delegate the task "
               "and say you are checking. Never make up the result.")


async def mic_to_live(conn):
    q: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    stream = sd.InputStream(samplerate=RATE, channels=1, dtype="int16", blocksize=BLOCK,
                            callback=lambda d, *_: loop.call_soon_threadsafe(q.put_nowait, d.tobytes()))
    with stream:
        while True:
            pcm = await q.get()
            await conn.send({"type": "session.input_audio.append",
                             "audio": base64.b64encode(pcm).decode()})


async def delegate(conn, delegation_id: str):
    request = " ".join(transcript[-40:])      # recent user speech as context
    answer = await asyncio.to_thread(brain.ask, [{"role": "user", "content": request}])
    await conn.send({"type": "session.commentary.append", "delegation_id": delegation_id,
                     "content": answer[:1500]})


async def main():
    client = AsyncOpenAI()
    speaker = sd.OutputStream(samplerate=RATE, channels=1, dtype="int16")
    speaker.start()
    async with client.live.connect() as conn:
        await conn.send({"type": "session.start", "session": {
            "model": "gpt-live-1", "instructions": LIVE_PROMPT,
            "audio": {"format": {"type": "audio/pcm", "rate": RATE}, "output": {"voice": "marin"}},
            "delegation": {"type": "client"}}})
        mic_task = None
        async for ev in conn:
            if ev.type == "session.started":
                mic_task = asyncio.create_task(mic_to_live(conn))
            elif ev.type == "session.output_audio.delta":
                speaker.write(np.frombuffer(base64.b64decode(ev.audio), dtype=np.int16))
            elif ev.type == "session.input_transcript.delta":
                transcript.append(ev.delta)
            elif ev.type == "session.delegation.created":
                asyncio.create_task(delegate(conn, ev.delegation.id))
            elif ev.type == "error":
                print("error:", ev)
        if mic_task:
            mic_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
