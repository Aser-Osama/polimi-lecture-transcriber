# Polimi Lecture Transcriber

A polished local macOS application that transcribes university lecture videos and audio files
with Apple Silicon acceleration (MLX Whisper), producing accurate transcripts and
word-timestamp-based SRT/WebVTT subtitles. On **Windows and Linux** the same app runs the
**OpenRouter cloud path** (transcription + Cloud alignment); the local MLX/WhisperX options are
macOS-only and are hidden automatically.

On macOS everything runs on your Mac by default. Nothing is uploaded anywhere unless you pick
the OpenRouter backend.

> Screenshot placeholder: `docs/screenshot.png` (main window: drop area, options, job queue).

## Features

- Local transcription with **MLX Whisper** on Apple Silicon (no CUDA, no cloud) **or** an
  optional **OpenRouter** cloud backend (5 STT models — mai-transcribe-2, Whisper large-v3 /
  turbo, Qwen3-ASR) selectable per job in the UI
- **Word-level timestamps** (Whisper cross-attention + DTW, or the remote model's own word
  timings) used to build subtitle cues — never fabricated timings
- **Subtitle alignment with three modes**: none (native timestamps), **Cloud** (word timestamps
  from the chosen OpenRouter model, or a MAI-Transcribe 2 anchor pass when it has none — nothing
  runs on your Mac), or local **WhisperX** forced alignment (Apple-silicon accelerated)
- **Parallel cloud jobs**: jobs that run fully on OpenRouter execute concurrently (1–32,
  default 6) since they do not use this Mac; local jobs always run one at a time
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

macOS (local + cloud):

- macOS on Apple Silicon (built and verified on an M5 Pro / 48 GB, macOS 26)
- Python 3.11 or 3.12
- Node.js 20+ and npm
- FFmpeg + ffprobe (`brew install ffmpeg`)
- ~4 GB free disk for the default Quality model (cached under the standard Hugging Face cache)

Windows / Linux (cloud-only):

- Windows 10/11 or a modern Linux distribution
- Python 3.11 or 3.12
- Node.js 20+ and npm (only needed to build the interface)
- FFmpeg + ffprobe (`winget install Gyan.FFmpeg` / `sudo apt install ffmpeg`)
- An OpenRouter API key (Settings → OpenRouter)

## Installation

```bash
git clone <your-repo-url> && cd Polimi-Lecture-Transcriber
./setup.sh                  # macOS: verifies platform/tools, creates .venv, installs deps, builds UI
```

On **Windows and Linux** use the cross-platform launcher instead of `setup.sh`/`run.sh`:

```bash
python run.py --check       # verify Python/Node/FFmpeg and print platform capabilities
python run.py               # first run sets up .venv, installs deps, builds the UI, starts the app
```

`run.py` is also the recommended entry point on macOS; it mirrors `./run.sh`. Flags:
`--check` (prerequisites only), `--no-browser`, `--skip-build`. `PT_HOST`/`PT_PORT` are honored
on every platform.

`setup.sh` never installs Homebrew silently. If FFmpeg is missing it prints
`brew install ffmpeg` and exits; you can also run `./setup.sh --install-ffmpeg` to have it run
that command for you.

### First launch

```bash
./run.sh          # macOS
python run.py     # Windows / Linux / macOS
```

This starts the backend on `http://127.0.0.1:8765`, serves the built interface and opens your
browser. The first transcription downloads the selected model into the standard Hugging Face
cache (`~/.cache/huggingface/hub`) — the UI shows "Downloading model (first use)" while it
happens. Nothing else is downloaded at install time.

## Windows and Linux (cloud-only)

The app detects the platform and adapts the whole UI:

- **Backend**: OpenRouter only (Local MLX Whisper is not offered)
- **Alignment**: `None` and `Cloud` (Cloud uses the chosen model's native word timestamps or a
  MAI-Transcribe 2 anchor pass — see below); Local WhisperX is not offered
- **API key storage**: no Keychain on these platforms, so the key is kept in a user-only file
  (`openrouter.key`, permissions 0600) inside the app data folder
  (`%APPDATA%\Polimi Lecture Transcriber` on Windows,
  `~/.local/share/Polimi Lecture Transcriber` on Linux; override with `PT_DATA_DIR`)
- **File selection**: drag & drop or the browser file picker (the macOS native picker is not
  available); dropped files are copied to a temporary folder
- **Model downloads, WhisperX install and local model settings** are hidden

Everything else (queue, parallel cloud jobs, history, preview, subtitles, courses/projects,
cost tracking) works exactly like on macOS. `PT_SIMULATE_PLATFORM=win32|linux` forces the
cloud-only mode on any machine (development/tests only).

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

## Hierarchical transcription context (optional)

Context helps every backend recognize names, acronyms and domain terminology. Three levels are
merged at transcription time as **Global + Course/Project + Current file**, where more specific
terms override broader ones (case-insensitive de-duplication, most specific spelling wins):

| Level | Where | For |
| --- | --- | --- |
| Global | Settings → Global context | persistent names, terminology, preferences — applies to every job |
| Course/Project | Settings → Courses / Projects (or "New..." in the transcribe view) | recurring vocabulary, professor names, acronyms per course/project |
| Current file | Transcribe view → context field | one-off topics, guest speakers, unusual terms |

Context is never mandatory: with all three empty the app behaves exactly as before. PDF, TXT
and Markdown files can be uploaded per course/project to extract candidate names/acronyms/terms
with explainable frequency heuristics (no LLM, no embeddings, no vector database) — the
extracted terms are appended to the project's context and stay fully editable.

How context reaches the models (adapted per backend, gracefully degrading):

| Backend / model | Context mechanism |
| --- | --- |
| Local MLX Whisper (all sizes) | Whisper `initial_prompt` (cleaned, de-duplicated, 224-token-safe) |
| OpenRouter `microsoft/mai-transcribe-2` | Keyword biasing via `provider.options.azure.phraseList.phrases` (max 100 terms) |
| Other OpenRouter models | No biasing support — the job runs normally and a warning notes the terms were not sent |

Every result JSON stores the full context breakdown (`context.global_context`,
`project_name`, `per_file_context`, `effective`, `terms`) and `provider_meta.context_applied`,
so you can see exactly what was used.

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

## Subtitle alignment

Choose per batch in the transcribe view:

| Mode | Where it runs | How it works |
| --- | --- | --- |
| **None** | — | Uses the model's own timestamps (Whisper DTW locally, native word timestamps on OpenRouter models that provide them). |
| **Cloud** (OpenRouter only) | OpenRouter | If the chosen model returns word timestamps (MAI-Transcribe 2, Whisper, Qwen3-ASR 1.7B) they are used **directly — no extra pass, no extra cost**. If it returns none (e.g. Qwen3 ASR Flash), a second OpenRouter call to **MAI-Transcribe 2** ($0.10/hour) supplies timing anchors that the transcript is matched onto. The transcript text stays 100% from the model you selected, and both costs are shown per job. |
| **Local WhisperX** | This Mac | wav2vec2 forced alignment (MPS/CPU) re-times any transcript without re-transcribing it; text-only results get a local Whisper tiny anchor pass first. |

There is no dedicated forced-alignment model on OpenRouter (verified against their model list),
so Cloud mode uses the strongest word-timestamp models available there. Note for later:
Qwen's `Qwen3-ForcedAligner-0.6B` is roughly 3× more accurate than WhisperX but is not served
by OpenRouter; it could be added via a GPU provider (see `docs/vast-provider-plan.md`).

### Parallel cloud jobs

Settings → Defaults → **Parallel cloud jobs** (1–32, default 6). A job counts as cloud-only when
its transcription provider is OpenRouter *and* its alignment is not Local WhisperX; those jobs
each get their own worker and run concurrently. Jobs that touch this Mac (MLX transcription or
WhisperX) share a single worker and stay strictly sequential, and they can overlap with cloud
jobs.

The Mac is not the bottleneck for cloud jobs: audio preparation takes ~13 s per 100-minute
video (FFmpeg extraction + MP3 chunking) and each job uploads only ~29 MB per hour of audio.
The real limit is the OpenRouter/provider rate limit. Measured on an M5 Pro against real
lecture recordings:

| Model | Cost | Speed (per request) | 16 parallel | 32 parallel |
| --- | --- | --- | --- | --- |
| MAI-Transcribe 2 | ~$0.10/h | 16–40x realtime | clean | rare 429s, auto-retried |
| Whisper Large V3 | ~$0.043/h | 16–43x realtime | clean | clean |
| Whisper Large V3 Turbo | ~$0.012/h | 13–24x realtime | clean | clean |
| Qwen3 ASR 1.7B | ~$0.027/h | 30–53x realtime | clean | clean |
| Qwen3 ASR Flash | ~$0.126/h | 9–13x realtime | clean | clean |

16–32 concurrent jobs are realistic for a batch (e.g. 60 h of lectures ≈ 15–25 min total at
16-way). The client honors provider `Retry-After` hints and retries up to 5 times, so brief
rate limits do not fail jobs. Budget examples for 60 h: Turbo ≈ $0.75, Qwen 1.7B ≈ $1.60,
Whisper V3 ≈ $2.60, MAI-2 ≈ $6.

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
is uploaded unless you choose that backend, and no other data ever leaves the machine. On
Windows/Linux, where the cloud path is the only backend, the UI says so explicitly.

The OpenRouter API key is stored in the macOS Keychain on macOS and in a user-only file
(0600) inside the app data folder on Windows/Linux. It is never written to the database, logs
or the browser, and is never sent anywhere except OpenRouter as the `Authorization` header.

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
- **OpenRouter job has no subtitles** — the chosen model returned text only and alignment was
  "None". Switch alignment to **Cloud** (adds a cheap MAI-Transcribe 2 anchor pass) or pick a
  model with word timestamps, then re-run.
- **Cloud alignment rejected at start** — Cloud mode requires the OpenRouter backend; switch the
  backend or use Local WhisperX.
- **WhisperX install fails** — check the log; usually a network hiccup during the pip install.
  Press Install again (the installer is idempotent).
- **Local options missing (Windows/Linux)** — expected: local MLX Whisper and WhisperX are
  macOS-only; the app shows the OpenRouter cloud path only. Cloud alignment is fully available.
- **`run.py` cannot create the virtual environment (Linux)** — install the venv package first
  (`sudo apt install python3-venv`) and re-run.
- **Where are the details?** — `~/Library/Logs/Polimi Lecture Transcriber/app.log` on macOS,
  `%LOCALAPPDATA%\Polimi Lecture Transcriber\Logs` on Windows,
  `~/.local/state/Polimi Lecture Transcriber/logs` on Linux (Settings → Reveal logs).

## Future: Vast.ai provider

V1 is fully local by design. `docs/vast-provider-plan.md` describes the planned optional
`VastProvider` (upload only extracted audio, GPU worker lifecycle, same result schema) and why
the current architecture already supports it without rewriting queue, history or exports.

## Future: per-job chunk parallelism (cloud)

A long lecture is currently bounded by its sequential chunk chain (a 106-minute video is
22 chunks × 3–5 s at MAI-2 speed). `docs/chunk-parallelism-plan.md` describes the planned
bounded parallelism (a few chunks in flight per job, ~3–4x faster per lecture) and the
request-budget rule that keeps provider rate limits safe.
