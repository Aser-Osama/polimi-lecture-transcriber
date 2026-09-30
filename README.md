# Polimi Lecture Transcriber

A polished local macOS application that transcribes university lecture videos and audio files
with Apple Silicon acceleration (MLX Whisper), producing accurate transcripts and
word-timestamp-based SRT/WebVTT subtitles.

Everything runs on your Mac. Nothing is uploaded anywhere.

> Screenshot placeholder: `docs/screenshot.png` (main window: drop area, options, job queue).

## Features

- Local transcription with **MLX Whisper** on Apple Silicon (no CUDA, no cloud) **or** an
  optional **OpenRouter** cloud backend (5 STT models — mai-transcribe-2, Whisper large-v3 /
  turbo, Qwen3-ASR) selectable per job in the UI
- **Word-level timestamps** (Whisper cross-attention + DTW, or the remote model's own word
  timings) used to build subtitle cues — never fabricated timings
- Optional **WhisperX forced alignment** (local, Apple-silicon accelerated): re-times the
  transcript of *any* backend against the original audio **without re-transcribing**; for
  text-only remote models a quick local Whisper tiny anchor pass supplies real speech windows
- Three quality tiers with real, verified Hugging Face models (downloaded on demand)
- TXT, SRT, WebVTT and versioned JSON output per job; JSON retains raw Whisper segments,
  word timestamps and final cues so subtitles can be regenerated later
- **Batch queue** processed sequentially (one transcription at a time), with per-job cancel,
  retry and history
- **Cancellation that actually stops the work**: the model runs in a dedicated worker process;
  a stuck inference is force-terminated via its process group and the queue continues
- Live stage progress over SSE (honest stage-based progress; no fake percentages)
- Subtitle preview synchronized with the source media, plus the generated VTT attached as a
  real `<track>`
- Course vocabulary glossary fed to Whisper as an initial prompt (cleaned and size-limited)
- English / Italian / Auto Detect language selection; English and Italian are passed explicitly
- Privacy: binds to `127.0.0.1` only, no telemetry, no analytics, no remote services

## Requirements

- macOS on Apple Silicon (built and verified on an M5 Pro / 48 GB, macOS 26)
- Python 3.11 or 3.12
- Node.js 20+ and npm
- FFmpeg + ffprobe (`brew install ffmpeg`)
- ~4 GB free disk for the default Quality model (cached under the standard Hugging Face cache)

## Installation

```bash
git clone <your-repo-url> && cd Polimi-Lecture-Transcriber
./setup.sh                  # verifies platform/tools, creates .venv, installs deps, builds UI
```

`setup.sh` never installs Homebrew silently. If FFmpeg is missing it prints
`brew install ffmpeg` and exits; you can also run `./setup.sh --install-ffmpeg` to have it run
that command for you.

### First launch

```bash
./run.sh
```

This starts the backend on `http://127.0.0.1:8765`, serves the built interface and opens your
browser. The first transcription downloads the selected model into the standard Hugging Face
cache (`~/.cache/huggingface/hub`) — the UI shows "Downloading model (first use)" while it
happens. Nothing else is downloaded at install time.

## Models

| UI option | Hugging Face repository | Approx. download |
| --- | --- | --- |
| Quality — Whisper Large V3 (default) | `mlx-community/whisper-large-v3-mlx` | 3.08 GB |
| Fast — Whisper Large V3 Turbo | `mlx-community/whisper-large-v3-turbo` | 1.61 GB |
| Balanced — Whisper Medium | `mlx-community/whisper-medium-mlx` | 1.52 GB |

The mapping lives in exactly one place: `backend/app/services/registry.py`. Models can be
downloaded ahead of time from **Settings → Models** (real byte-level progress), and the cache
can be revealed in Finder. Batch jobs reuse the loaded model in the same worker process via
mlx-whisper's built-in `ModelHolder` cache; switching models replaces it (one model in memory).

### Memory behavior

The worker process keeps the model loaded while you are batching, so consecutive lectures do
not pay the load time again. When no work has been queued for 60 seconds
(`PT_WORKER_IDLE_SECONDS`), the worker exits and **all** of its unified memory is returned to
macOS — including MLX's Metal buffer cache and the model weights (~3 GB for Large V3). The
next job simply starts a fresh worker and reloads the model from the cache in a few seconds.
Transient MLX buffers are also released after every job.

## OpenRouter backend (optional)

Choose **OpenRouter** in the backend selector, pick a model and add your API key in
Settings. The key is stored in the **macOS Keychain** (service `PolimiLectureTranscriber`) —
never in the database, logs or the browser tab.

| Model | Notes |
| --- | --- |
| `microsoft/mai-transcribe-2` | default; multilingual, word timestamps, $0.10/hour |
| `openai/whisper-large-v3` | OpenAI Whisper large-v3 through OpenRouter providers |
| `openai/whisper-large-v3-turbo` | faster Whisper variant |
| `qwen/qwen3-asr-1.7b` | Qwen multilingual ASR, word timestamps |
| `qwen/qwen3-asr-flash-2026-02-10` | fastest Qwen; may return text only (use alignment) |

How it works: the prepared 16 kHz audio is split into ~5-minute chunks **at detected
silences** (OpenRouter providers time out after ~60 s per request), each chunk is encoded to
mono MP3 and sent as base64 JSON with `response_format=verbose_json` and word/segment
granularities. Chunk progress, detected language, per-job cost (`usage.cost`) and the model id
are recorded in the job and result JSON. Models that cannot return structured output are
retried as plain text and flagged in the job warnings — enable WhisperX alignment to still
get subtitles. Course vocabulary is local-only (OpenRouter ignores prompts).

## WhisperX alignment (optional)

Enable **“Improve subtitle alignment (WhisperX)”** per batch in the transcribe view. It runs
[WhisperX](https://github.com/m-bain/whisperX) *forced alignment* locally (wav2vec2 CTC on MPS
or CPU, ~50–100× realtime) and re-times the transcript against the audio — the text itself is
never regenerated. Works with Local and OpenRouter backends, with native word timestamps or
segment-only results. If a remote model returns no timing at all, a small local Whisper tiny
pass creates speech-anchor windows first.

WhisperX lives in an isolated environment (`.venv-whisperx`, ~1.5 GB, torch/transformers) so
it never touches the app's dependencies. Install it either way:

- **Settings → WhisperX alignment → Install** (recommended; live status in the UI), or
- `./setup.sh --with-whisperx`.

The first alignment for a language downloads its small wav2vec2 model automatically. Without
alignment, the local and remote workflows behave exactly as before.

## Supported media

MP4, MOV, MKV, WEBM, M4V, MP3, M4A, AAC, WAV, FLAC, OGG/Opus, AIFF, MPEG/TS, WMA.
Files are probed with ffprobe; a file without an audio track is rejected with a clear message.

### File handling: no unnecessary copies

- **"choose files without copying"** opens the native macOS picker and processes the original
  file path directly — recommended for multi-gigabyte lectures.
- **Drag & drop** streams the file through localhost into an application-managed temporary
  folder (browser sandboxing makes this unavoidable), never loading it into RAM. Temporary
  copies are deleted after processing; enable *Settings → Keep temporary copies* if you want
  preview to keep working for dropped files.

## Output

Default output folder: `~/Documents/Polimi Transcripts/` (changeable in Settings).
Each successful job writes four files sharing one basename:

```
Lecture Name.txt    plain transcript (conservative whitespace cleanup only)
Lecture Name.srt    SubRip subtitles from word-level timing
Lecture Name.vtt    WebVTT subtitles (also used by the in-app preview)
Lecture Name.json   versioned result: segments + words + cues + timings + metadata
```

Existing files are never overwritten: duplicates become `Lecture Name (2).txt` etc.
The JSON result (`schema_version: 1`) lets you **regenerate subtitles after improving the
segmentation algorithm** without retranscribing: Preview → Regenerate, or
`POST /api/jobs/regenerate {"job_id": "..."}`.

## Subtitle timing design

1. Whisper runs with `word_timestamps=True`, producing per-word `{word, start, end, probability}`.
2. The aligner (`backend/app/alignment/`) normalizes those timestamps and never invents any.
3. The segmenter (`backend/app/subtitles/segmenter.py`) streams words into cues using, in
   priority order: meaningful silences (≥ 0.6 s), sentence punctuation, weak punctuation with a
   pause, maximum cue duration (7 s), and the character budget (42 chars × 2 lines by default).
   Tiny cues are merged or padded (bounded by the next cue), and over-long cues are split at
   real word boundaries. No timestamp is ever evenly distributed.
4. Validation (`validate.py`) repairs/clamps and reports warnings; quantized millisecond values
   are re-validated so serialized SRT/VTT can never overlap by rounding.
5. Cues are displayed as soon as they are spoken; end padding is small (≤ 0.3 s) and never
   crosses the next cue or the media duration.

## Batch processing

Add multiple files (picker or drop) and start once. Jobs run strictly sequentially in FIFO
order. Each row shows status, live stage, elapsed time, and actions (Cancel / Retry / Remove /
Preview / Finder). Finished jobs can be cleared from the queue without deleting outputs.
History shows every job with its model, language, duration, processing time, status and outputs.

## Cancellation

The heavy work runs in a dedicated spawned worker process (`spawn`, never `fork`, for MLX
safety). Cancel first asks the worker to stop cooperatively (FFmpeg extraction is terminated
immediately; the token is checked between stages). If a blocking MLX call does not return within
5 seconds, the worker's process group is force-terminated — including any child FFmpeg — the job
is marked cancelled, temporary audio is removed, and the next queued job starts in a fresh
worker (the model is simply reloaded after a forced cancel).

If the app is closed while jobs were active, those jobs are marked **interrupted** at next
startup. Nothing expensive restarts automatically; use Retry.

## Architecture

```
Browser UI (React + TypeScript, served by FastAPI)
        │  SSE for live status
FastAPI on 127.0.0.1 (backend/app/api)
        │
JobManager: SQLite queue + dispatcher (backend/app/queue/manager.py)
        │  spawn
Worker process ── FFmpeg extract ── LocalMLXProvider ── MLXWordTimestampAligner
        │                                                  │
   SubtitleSegmenter ── validation ── TXT/SRT/VTT/JSON exporters
```

The provider abstraction (`backend/app/providers/base.py`) is the extension point: a future
`VastProvider` implements the same `transcribe_sync` contract and the queue/export layers stay
untouched. See `docs/vast-provider-plan.md`.

### Data locations (macOS conventions)

| What | Where |
| --- | --- |
| Application data + SQLite DB | `~/Library/Application Support/Polimi Lecture Transcriber/` |
| Logs (rotating, 5 × 5 MB) | `~/Library/Logs/Polimi Lecture Transcriber/app.log` |
| Temporary media | `~/Library/Application Support/Polimi Lecture Transcriber/tmp/` |
| Transcripts | `~/Documents/Polimi Transcripts/` |
| Whisper model cache | `~/.cache/huggingface/hub` |
| OpenRouter API key | macOS Keychain (`PolimiLectureTranscriber`) |
| WhisperX environment | `.venv-whisperx/` in the project folder (optional) |

## Privacy

**Local mode: processing is performed locally on this Mac; your lecture is not uploaded
anywhere.** No telemetry, no analytics, no silent remote services. The only network access is
downloading open-source models from Hugging Face and WhisperX alignment models when requested.

If you explicitly select the **OpenRouter backend**, the prepared audio *is* uploaded to
OpenRouter for transcription (that is the point of the feature) and billed per second. Nothing
is uploaded unless you choose that backend, and no other data ever leaves the machine.

## Testing

```bash
.venv/bin/python -m pytest backend/tests        # backend unit + integration tests
cd frontend && npm run typecheck && npm run build
```

The suite covers timestamp formatting, SRT/VTT serialization, segmentation edge cases
(long silences, rapid speech, 30 s+ raw segments, one-word fragments, malformed/overlapping
word times), cue validation and quantization, output naming (duplicates, Unicode), settings
persistence/recovery, SQLite recovery of interrupted jobs, the sequential queue, cancellation
(including a real spawned worker that is force-killed), the API with a real subprocess worker,
and SSE against a real uvicorn server.

### Real transcription smoke test

```bash
.venv/bin/python scripts/smoke_test.py --model-key fast
```

Generates a speech sample with macOS `say`, runs the real MLX pipeline, validates the outputs
and prints timings/RTF. It downloads the model on first use unless already cached.

## Development

```bash
./dev.sh      # backend with reload on 127.0.0.1:8765 + Vite dev server on 127.0.0.1:5173
```

The Vite dev server proxies `/api` to the backend, so no CORS configuration is needed.

Useful environment variables: `PT_HOST` (default `127.0.0.1`), `PT_PORT` (default `8765`),
`PT_DATA_DIR`, `PT_LOGS_DIR`, `PT_TEMP_DIR`, `PT_OUTPUT_DIR`, `PT_DB_PATH`, `PT_DEBUG=1`
(keeps temporary audio for inspection), `PT_PROVIDER` / `PT_FAKE_DELAY` (development with the
fake provider), `HF_HOME` (model cache override), `PT_OPENROUTER_BASE_URL` (tests only),
`PT_WHISPERX_VENV` (alternate alignment environment), `PT_KEYCHAIN_SERVICE` (tests only).

## Troubleshooting

- **"FFmpeg is missing"** — `brew install ffmpeg`, then restart the app.
- **First transcription takes minutes** — it is downloading the model. Watch the stage in the
  queue row; models are cached afterwards.
- **Out of disk space** — the app checks free space before audio extraction and stops with a
  clear message; transcripts and temp audio need roughly 30 MB per hour of 16 kHz PCM16 audio
  plus whatever the model download requires.
- **A job shows "Interrupted"** — the app closed while it was running. Click Retry.
- **Memory stays high right after a job** — expected while the app is batching: the model stays
  warm for the next job. It is released automatically after 60 s without queued work; the next
  job reloads it from the local cache.
- **OpenRouter job fails with 401** — the key was rejected; re-enter it in Settings.
- **OpenRouter job has no subtitles** — the chosen model returned text only. Enable WhisperX
  alignment (or pick a model with word timestamps) and re-run.
- **WhisperX install fails** — check the log; usually a network hiccup during the pip install.
  Press Install again (the installer is idempotent).
- **Where are the details?** — `~/Library/Logs/Polimi Lecture Transcriber/app.log`
  (Settings → Reveal logs).

## Future: Vast.ai provider

V1 is fully local by design. `docs/vast-provider-plan.md` describes the planned optional
`VastProvider` (upload only extracted audio, GPU worker lifecycle, same result schema) and why
the current architecture already supports it without rewriting queue, history or exports.
