"""WhisperX forced alignment provider.

Runs scripts/whisperx_align.py inside the isolated `.venv-whisperx` and maps
the resulting words to the internal models. The transcript text is passed
through unchanged; WhisperX only re-times it against the audio.
"""

from __future__ import annotations

import json
import logging
import select
import subprocess
import tempfile
import time
from pathlib import Path

from app.alignment.base import AlignmentProvider, AlignmentResult
from app.core.errors import CancelledError
from app.models.result import AlignedWord
from app.providers.base import CancellationToken, ProgressCallback, RawTranscription
from app.services import whisperx as whisperx_service

log = logging.getLogger(__name__)


class WhisperXAligner(AlignmentProvider):
    name = "whisperx"

    def __init__(
        self,
        interpreter: Path | None = None,
        script_path: Path | None = None,
        device: str = "auto",
        timeout_seconds: float = 3600.0,
    ):
        self._interpreter = interpreter or whisperx_service.venv_python()
        self._script = script_path or whisperx_service.ALIGN_SCRIPT
        self._device = device
        self._timeout = timeout_seconds

    def is_available(self) -> bool:
        return self._interpreter.exists() and self._script.is_file()

    def align(
        self,
        transcription: RawTranscription,
        media_duration: float | None = None,
        progress: ProgressCallback | None = None,
        cancel: CancellationToken | None = None,
    ) -> AlignmentResult:
        notify = progress or (lambda stage, message=None, fraction=None: None)
        warnings: list[str] = []

        segments = [
            {"start": s.start, "end": s.end, "text": s.text}
            for s in transcription.segments
            if s.text.strip() and s.end > s.start
        ]
        if not segments:
            return AlignmentResult(
                words=[],
                warnings=["WhisperX alignment needs segment timing, but none was returned."],
            )
        if not self.is_available():
            return AlignmentResult(
                words=[],
                warnings=["WhisperX alignment is not installed; keeping native timestamps."],
            )

        notify("aligning", "Aligning words with WhisperX")
        with tempfile.TemporaryDirectory(prefix="whisperx-") as temp_dir:
            request_path = Path(temp_dir) / "request.json"
            output_path = Path(temp_dir) / "output.json"
            request_path.write_text(
                json.dumps(
                    {
                        "audio_path": str(transcription.meta.get("audio_path", "")),
                        "language": transcription.language or "en",
                        "device": self._device,
                        "segments": segments,
                    }
                ),
                encoding="utf-8",
            )
            returncode, device, error = self._run(request_path, output_path, notify, cancel)
            if cancel is not None and cancel.cancelled:
                raise CancelledError("Alignment cancelled")
            if returncode != 0 or not output_path.exists():
                detail = error or f"exit code {returncode}"
                log.warning("WhisperX alignment failed: %s", detail)
                warnings.append(f"WhisperX alignment failed ({detail}); keeping native timestamps.")
                return AlignmentResult(words=[], warnings=warnings)

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            words = [
                AlignedWord(
                    text=entry["text"],
                    start=float(entry["start"]),
                    end=float(entry["end"]),
                    probability=entry.get("score"),
                )
                for entry in payload.get("words", [])
            ]
            if not words:
                warnings.append("WhisperX returned no aligned words; keeping native timestamps.")
            if device:
                notify("aligning", f"Aligned {len(words)} words with WhisperX ({device.upper()})")
            return AlignmentResult(words=words, warnings=warnings)

    def _run(
        self,
        request_path: Path,
        output_path: Path,
        notify: ProgressCallback,
        cancel: CancellationToken | None,
    ) -> tuple[int, str | None, str | None]:
        process = subprocess.Popen(
            [str(self._interpreter), str(self._script), str(request_path), str(output_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        device: str | None = None
        stderr_lines: list[str] = []
        deadline = time.monotonic() + self._timeout

        try:
            assert process.stdout is not None
            while True:
                if cancel is not None and cancel.cancelled:
                    process.terminate()
                    process.wait(timeout=10)
                    return 1, device, "cancelled"
                if time.monotonic() > deadline:
                    process.kill()
                    return 1, device, "timeout"
                ready, _, _ = select.select([process.stdout], [], [], 0.2)
                if not ready:
                    if process.poll() is not None:
                        break
                    continue
                line = process.stdout.readline()
                if not line:
                    break
                line = line.strip()
                if line.startswith("WHISPERX_STAGE:"):
                    notify("aligning", line.split(":", 1)[1].strip())
                elif line.startswith("WHISPERX_PROGRESS:"):
                    try:
                        fraction = float(line.split(":", 1)[1])
                        notify("aligning", None, max(0.0, min(1.0, fraction)))
                    except ValueError:
                        continue
                elif line.startswith("WHISPERX_DEVICE:"):
                    device = line.split(":", 1)[1].strip()
            returncode = process.wait(timeout=30)
        finally:
            if process.stderr is not None:
                stderr_lines = process.stderr.read().splitlines()[-6:]
                process.stderr.close()
            if process.stdout:
                process.stdout.close()

        error = None
        if returncode != 0:
            joined = " | ".join(line for line in stderr_lines if line)
            error = joined[-300:] if joined else None
        return returncode, device, error
