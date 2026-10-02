"""Fully local voice assistant (except the brain, if BRAIN=claude).
wake word (openWakeWord) -> record until silence -> STT (faster-whisper) -> brain -> TTS (Piper)"""
import os
import subprocess
import time

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from openwakeword.model import Model as WakeModel

from brain import make_brain

RATE, FRAME = 16000, 1280                       # openWakeWord expects 80 ms blocks at 16 kHz
WAKE = os.getenv("WAKE_MODEL", "hey_jarvis")      # swap in the .onnx you trained
PIPER_VOICE = os.getenv("PIPER_VOICE", "en_US-lessac-medium.onnx")
SILENCE_RMS, SILENCE_S, MAX_S = 500, 0.8, 12

wake = WakeModel(wakeword_models=[WAKE], inference_framework="onnx")
stt = WhisperModel(os.getenv("WHISPER", "small"),          # Pi 5: small int8 | Jetson: large-v3-turbo
                   device=os.getenv("WHISPER_DEVICE", "cpu"),
                   compute_type=os.getenv("WHISPER_COMPUTE", "int8"))
brain = make_brain()
history: list = []


def speak(text: str):
    piper = subprocess.Popen(["piper", "--model", PIPER_VOICE, "--output-raw"],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    aplay = subprocess.Popen(["aplay", "-q", "-r", "22050", "-f", "S16_LE", "-t", "raw", "-"],
                             stdin=piper.stdout)
    piper.stdin.write(text.encode()); piper.stdin.close()
    piper.wait(); aplay.wait()


def record_until_silence(stream) -> np.ndarray:
    chunks, quiet, start = [], 0.0, time.time()
    while time.time() - start < MAX_S:
        block, _ = stream.read(FRAME)
        chunks.append(block[:, 0].copy())
        rms = np.sqrt(np.mean(block.astype(np.float32) ** 2))
        quiet = quiet + FRAME / RATE if rms < SILENCE_RMS else 0.0
        if quiet > SILENCE_S and len(chunks) * FRAME / RATE > 1.0:
            break
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def main():
    with sd.InputStream(samplerate=RATE, channels=1, dtype="int16", blocksize=FRAME) as stream:
        print("listening…")
        while True:
            block, _ = stream.read(FRAME)
            if max(wake.predict(block[:, 0]).values()) < 0.5:
                continue
            wake.reset()
            t0 = time.time()
            audio = record_until_silence(stream)
            segs, _ = stt.transcribe(audio, language="en", vad_filter=True, beam_size=1)
            text = " ".join(s.text for s in segs).strip()
            t_stt = time.time()
            if not text:
                continue
            history.append({"role": "user", "content": text})
            answer = brain.ask(history[-9:])   # odd-sized window always starts on a user turn
            history.append({"role": "assistant", "content": answer})
            print(f"[{t_stt - t0:.1f}s stt | {time.time() - t_stt:.1f}s brain] {text!r} -> {answer!r}")
            speak(answer)


if __name__ == "__main__":
    main()
