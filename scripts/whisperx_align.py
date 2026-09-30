#!/usr/bin/env python
"""WhisperX forced-alignment helper (runs inside the isolated alignment venv).

Protocol:
    whisperx_align.py request.json output.json

request.json:
    {
      "audio_path": "/abs/path/audio.wav",
      "language": "en",
      "device": "auto" | "mps" | "cpu",
      "segments": [{"start": 0.0, "end": 4.2, "text": "..."}]
    }

stdout markers (parsed by the backend):
    WHISPERX_STAGE: <message>
    WHISPERX_PROGRESS: <0..1>
    WHISPERX_DEVICE: <mps|cpu>

output.json:
    {"words": [{"text": str, "start": float, "end": float, "score": float|null}],
     "device": "mps", "language": "en"}

The transcript text is never re-generated here: WhisperX only aligns the given
segments against the audio with a CTC model.
"""

from __future__ import annotations

import json
import sys


def emit(line: str) -> None:
    print(line, flush=True)


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: whisperx_align.py request.json output.json", file=sys.stderr)
        return 2

    request_path, output_path = sys.argv[1], sys.argv[2]
    with open(request_path, encoding="utf-8") as handle:
        request = json.load(handle)

    language = request.get("language") or "en"
    requested_device = request.get("device") or "auto"
    segments = request.get("segments") or []
    if not segments:
        print("whisperx_align: no segments to align", file=sys.stderr)
        return 3

    import torch
    from whisperx.alignment import align, load_align_model

    def pick_device(preference: str) -> str:
        if preference in ("mps", "cpu"):
            return preference
        return "mps" if torch.backends.mps.is_available() else "cpu"

    def run_alignment(device: str):
        emit("WHISPERX_STAGE: Loading alignment model")
        model_a, metadata = load_align_model(language_code=language, device=device)
        emit("WHISPERX_STAGE: Aligning words")
        progress_state = {"last": 0.0}

        def report(fraction: float) -> None:
            value = max(0.0, min(1.0, float(fraction)))
            if value - progress_state["last"] >= 0.01 or value >= 1.0:
                progress_state["last"] = value
                emit(f"WHISPERX_PROGRESS: {value:.4f}")

        return align(
            segments,
            model_a,
            metadata,
            request["audio_path"],
            device=device,
            progress_callback=report,
        )

    device = pick_device(requested_device)
    try:
        result = run_alignment(device)
    except Exception as first_error:  # MPS can fail on some ops; fall back to CPU
        if device == "mps":
            emit(f"WHISPERX_STAGE: MPS failed ({type(first_error).__name__}), retrying on CPU")
            device = "cpu"
            result = run_alignment(device)
        else:
            raise

    words = []
    for word in result.get("word_segments", []):
        text = str(word.get("word", "")).strip()
        if not text:
            continue
        try:
            start = float(word["start"])
            end = float(word["end"])
        except (KeyError, TypeError, ValueError):
            continue
        score = word.get("score")
        words.append(
            {
                "text": text,
                "start": round(start, 3),
                "end": round(end, 3),
                "score": round(float(score), 4) if isinstance(score, (int, float)) else None,
            }
        )

    emit(f"WHISPERX_DEVICE: {device}")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump({"words": words, "device": device, "language": language}, handle)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        print(f"WHISPERX_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
