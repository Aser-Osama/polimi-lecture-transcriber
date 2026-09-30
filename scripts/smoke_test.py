#!/usr/bin/env python
"""Real end-to-end transcription smoke test using MLX Whisper.

Generates (or accepts) a short speech sample, runs the real pipeline with the
real LocalMLXProvider, and validates the produced outputs.

This downloads the selected model on first use (several GB for Large V3).

Usage:
    .venv/bin/python scripts/smoke_test.py --model-key fast
    .venv/bin/python scripts/smoke_test.py --audio /path/to/lecture-clip.wav
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.config import AppPaths
from app.models.domain import AppSettings, Job, JobConfig, LanguageChoice
from app.providers.base import CancellationToken
from app.services.pipeline import PipelineContext, run_pipeline

SPEECH = (
    "Good morning. Today we continue the operating systems lecture. "
    "We will discuss virtual memory, the page table, and the TLB. "
    "Remember that cache coherence in multicore processors uses the MESI protocol. "
    "A spinlock busy waits, while a mutex can put the current thread to sleep. "
    "That concludes today's lecture at Politecnico di Milano."
)


def generate_speech(destination: Path) -> Path:
    aiff = destination.with_suffix(".aiff")
    say_bin = shutil.which("say")
    if not say_bin:
        raise SystemExit(
            "No --audio provided and the macOS 'say' command is unavailable; "
            "pass --audio explicitly."
        )
    subprocess.run([say_bin, "-o", str(aiff), SPEECH], check=True)
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(aiff),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            "-y",
            str(destination),
        ],
        check=True,
    )
    aiff.unlink(missing_ok=True)
    return destination


def validate_outputs(outputs: dict, result_json: dict) -> list[str]:
    problems: list[str] = []
    srt_path = Path(outputs["srt"])
    vtt_path = Path(outputs["vtt"])
    txt_path = Path(outputs["txt"])
    json_path = Path(outputs["json"])
    for path in (srt_path, vtt_path, txt_path, json_path):
        if not path.is_file() or path.stat().st_size == 0:
            problems.append(f"missing or empty: {path}")

    srt_text = srt_path.read_text(encoding="utf-8")
    stamps = re.findall(r"(\d\d):(\d\d):(\d\d),(\d\d\d) --> (\d\d):(\d\d):(\d\d),(\d\d\d)", srt_text)

    def ms(groups: tuple[str, ...]) -> int:
        h, m, s, millis = (int(value) for value in groups)
        return ((h * 60 + m) * 60 + s) * 1000 + millis

    previous_end: int | None = None
    for match in stamps:
        start = ms(match[:4])
        end = ms(match[4:])
        if end <= start:
            problems.append(f"SRT cue end <= start: {match}")
        if previous_end is not None and start < previous_end:
            problems.append("SRT cues overlap")
        previous_end = end

    vtt_text = vtt_path.read_text(encoding="utf-8")
    if not vtt_text.startswith("WEBVTT"):
        problems.append("VTT does not start with WEBVTT")

    if not result_json.get("words"):
        problems.append("no word-level timestamps in JSON result")
    if not result_json.get("cues"):
        problems.append("no cues in JSON result")
    if result_json.get("schema_version") != 1:
        problems.append("unexpected schema_version")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", choices=["quality", "fast", "balanced"], default="fast")
    parser.add_argument("--audio", type=Path, default=None, help="optional existing audio file")
    parser.add_argument("--language", choices=["en", "it", "auto"], default="en")
    parser.add_argument("--keep", action="store_true", help="keep temporary work directory")
    args = parser.parse_args()

    # The pipeline requires the ffmpeg binaries regardless of input format.
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        print("FFmpeg/ffprobe are required. Install with: brew install ffmpeg")
        return 2

    work_root = Path(tempfile.mkdtemp(prefix="polimi-smoke-"))
    paths = AppPaths(
        data_dir=work_root / "data",
        logs_dir=work_root / "logs",
        temp_dir=work_root / "temp",
        default_output_dir=work_root / "out",
        db_path=work_root / "data" / "smoke.db",
    )
    paths.ensure_directories()

    if args.audio:
        audio = args.audio.expanduser()
        if not audio.is_file():
            print(f"Audio file not found: {audio}")
            return 2
    else:
        print("Generating speech sample with macOS 'say'...")
        audio = generate_speech(work_root / "sample.wav")

    from app.services.registry import get_model_spec

    spec = get_model_spec(args.model_key)
    print(f"Model: {spec.display_name} ({spec.repo_id})")
    print(f"Audio: {audio} ({audio.stat().st_size} bytes)")

    language = {"en": LanguageChoice.ENGLISH, "it": LanguageChoice.ITALIAN, "auto": LanguageChoice.AUTO}[
        args.language
    ]
    job = Job(
        source_path=str(audio),
        source_filename=audio.name,
        config=JobConfig(
            model_key=args.model_key,
            language=language,
            glossary="Politecnico di Milano, virtual memory, page table, TLB, MESI, spinlock, mutex",
        ),
    )

    seen_stages: list[str] = []

    def progress(stage: str, message: str | None = None, fraction: float | None = None) -> None:
        label = message or stage
        if fraction is not None:
            label = f"{label} ({fraction * 100:.0f}%)"
        if not seen_stages or seen_stages[-1] != f"{stage}:{label}":
            seen_stages.append(f"{stage}:{label}")
            print(f"  [{stage}] {label}")

    context = PipelineContext(
        job=job,
        settings=AppSettings(),
        paths=paths,
        output_dir=paths.default_output_dir,
        work_dir=paths.temp_dir / "jobs" / job.id,
        progress=progress,
        cancel=CancellationToken(),
    )

    started = time.monotonic()
    try:
        outcome = run_pipeline(context)
    except Exception as exc:  # noqa: BLE001 - the CLI reports any failure
        print(f"\nFAILED: {exc}")
        return 1
    wall = time.monotonic() - started

    result = json.loads(Path(outcome.outputs["json"]).read_text(encoding="utf-8"))
    problems = validate_outputs(outcome.outputs, result)

    timings = outcome.timings
    print("\n--- results -------------------------------------------------")
    print(f"Media duration:      {outcome.media_duration:.2f}s")
    print(f"Wall clock:          {wall:.2f}s")
    print(f"Model load:          {timings.model_load:.2f}s")
    print(f"Inference:           {timings.inference:.2f}s")
    print(f"Audio preparation:   {timings.audio_prepare:.2f}s")
    print(f"Alignment:           {timings.alignment:.3f}s")
    print(f"Formatting/saving:   {timings.formatting:.3f}s / {timings.saving:.3f}s")
    if outcome.media_duration:
        print(f"Speed:               {outcome.media_duration / wall:.2f}x realtime")
    print(f"Words:               {len(result['words'])}")
    print(f"Cues:                {len(result['cues'])}")
    print(f"Detected language:   {result.get('language_detected')}")
    print(f"Warnings:            {len(result['warnings'])}")
    for warning in result["warnings"][:8]:
        print(f"  - {warning}")
    print("\nFirst cues:")
    for cue in result["cues"][:5]:
        text = cue["text"].replace("\n", " ")
        print(f"  [{cue['start']:7.2f} -> {cue['end']:7.2f}] {text[:90]}")
    print("\nTranscript head:")
    print("  " + Path(outcome.outputs["txt"]).read_text(encoding="utf-8")[:400].replace("\n", "\n  "))
    print("\nOutput files:")
    for kind in ("txt", "srt", "vtt", "json"):
        print(f"  {outcome.outputs[kind]}")

    if problems:
        print("\nVALIDATION PROBLEMS:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    if args.keep:
        print(f"\nWork directory kept at: {work_root}")
    else:
        shutil.rmtree(work_root, ignore_errors=True)
    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
