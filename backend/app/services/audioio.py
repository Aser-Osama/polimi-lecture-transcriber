"""In-memory audio loading for the MLX provider.

Reads the 16 kHz mono PCM WAV produced by our own FFmpeg extraction in chunks
so a 90-minute lecture does not require the PCM bytes and the float array to
exist twice in memory.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def load_wav_float32(path: Path, chunk_frames: int = 4_000_000) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        framerate = wav.getframerate()
        total_frames = wav.getnframes()
        if channels != 1 or sample_width != 2 or framerate != 16000:
            raise ValueError(
                "Prepared audio must be 16 kHz mono 16-bit PCM; "
                f"got {framerate} Hz, {channels} ch, {sample_width * 8} bit"
            )
        output = np.empty(total_frames, dtype=np.float32)
        offset = 0
        while offset < total_frames:
            frames = wav.readframes(min(chunk_frames, total_frames - offset))
            if not frames:
                break
            chunk = np.frombuffer(frames, dtype=np.int16)
            output[offset : offset + chunk.size] = chunk.astype(np.float32) / 32768.0
            offset += chunk.size
        if offset != total_frames:
            output = output[:offset]
    return output
