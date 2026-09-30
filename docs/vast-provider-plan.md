# Future plan: optional Vast.ai processing backend

Status: **not implemented in V1**. This document describes a realistic design so the option can
be added later without rewriting the queue, history, exports or UI.

## Why the current architecture already supports it

- `TranscriptionProvider` (`backend/app/providers/base.py`) is the only thing the pipeline knows
  about transcription engines. A `VastProvider` implements the same `transcribe_sync`
  contract and returns the same `RawTranscription` (segments + word timestamps).
- Provider selection already exists in the data model: `JobConfig.provider`
  (`local_mlx` today, `vast` later) and is persisted per job.
- The worker process boundary does not change: the worker simply instantiates a different
  provider based on the job config.
- Results, subtitles, exports and the JSON schema are provider-independent.
- Secrets stay server-side: the browser only talks to `127.0.0.1`, so a Vast API key would live
  in the backend keychain/config and never reach the frontend.

## Design

### 1. Credentials

- Store the Vast API key in the macOS Keychain (via `security` CLI or `keyring`) under
  `Polimi Lecture Transcriber / vast-api-key`.
- Settings exposes *Enable Vast Cloud* and a key entry field. The key is sent once to the
  local backend over loopback; it is never echoed back to the browser (write-only field).
- Every API route that touches Vast requires the key to be present; errors are actionable.

### 2. Job intake

- The UI adds a processing-engine selector:

  ```text
  Processing engine:
  ● Local Mac
  ○ Vast Cloud
  ```

  This maps to `JobConfig.provider`; nothing else in the queue UI changes.
- For `vast` jobs the pipeline still runs **locally**: ffprobe validation, 16 kHz mono PCM16
  WAV extraction and word alignment happen on the Mac. Only the compressed/plain audio leaves
  the machine.

### 3. Uploading only what is necessary

- Ask the GPU worker to run Whisper on the extracted audio:
  - best: upload the extracted WAV (172 MB/hour) — deterministic and lossless;
  - better: encode to FLAC (~60 % of WAV) or 16 kHz Opus (~10 MB/hour) when the remote
    runner can decode it; chosen by a setting, default FLAC.
- Upload with resumable, chunked HTTP (e.g. `httpx` streaming in 8 MB chunks).
- Never upload the original video. Never store the upload in the app database.

### 4. Worker lifecycle

1. `GET /bundles/` — find an image running `faster-whisper`/`WhisperX` with word timestamps
   and a small HTTP control API (FastAPI on the instance, authenticated with a per-job token).
2. `PUT /asks/{id}/` — create an instance from the selected offer (filter: verified,
   CUDA ≥ 12, ≥ 24 GB VRAM, region EU preferred for latency).
3. Poll instance state until `running`; fetch the mapped SSH/HTTP endpoint.
4. Submit `{"audio_url": ..., "model": "large-v3", "language": "en", "initial_prompt": ...}`.
5. Poll progress (`processed_seconds / total_seconds`, genuinely chunk-based on the runner).
6. Download the JSON result (segments + words + probabilities) and map it into
   `RawTranscription`.
7. Convert/validate exactly like the local path.
8. `DELETE /instances/{id}/` — destroy the worker. Delete the uploaded audio from the instance
   filesystem before destruction and, as defense in depth, expire it with a short TTL.
9. Cleanup on failure: a `finally` block always attempts destroy + remote delete; a startup
   recovery pass destroys instances tagged with this app if they were left running.

### 5. Progress and cancellation

- Remote progress is real (chunks processed), so the UI can show a determinate bar for the
  cloud provider — still honest, because the number comes from the runner.
- Cancellation: set a cancel flag → worker process sends `DELETE /instances/{id}` (destroy is
  the strongest form of cancel) → mark job cancelled. Local extraction cancellation works as
  today.
- The worker process owns all remote calls, so killing the local worker (force cancel path)
  cannot leave a half-cancelled HTTP call in the API process.

### 6. Reliability

- Uploads and downloads retried with exponential backoff; job holds an `attempt` counter in
  `JobConfig.options` (already supported).
- If the remote runner dies mid-job, the job fails with a concise message and can be retried on
  either engine (Retry copies the config; the user can switch engine by re-adding with Vast
  selected).
- Cost guardrails: user sets a maximum price/hour and hard timeout (e.g. 2× media duration);
  the job aborts if exceeded.
- Privacy: the model download happens on the GPU instance, not from your Mac. The UI must show
  a clear notice that audio *will* be uploaded for Vast jobs before the first cloud job runs.

### 7. What does not change

- `SubtitleSegmenter`, validation, SRT/VTT/TXT/JSON exporters, duplicate-safe naming.
- Queue semantics: still sequential by default; the queue can later allow N concurrent Vast
  jobs by dropping the single-worker assumption in `JobManager` (it is already centralized in
  `_dispatcher`).
- History and preview: results are identical, so both work unchanged.

## Estimated implementation size

| Piece | Where |
| --- | --- |
| `VastProvider` + HTTP runner contract | `backend/app/providers/vast.py` (new) |
| Vast client (instances, auth, retries) | `backend/app/services/vast_client.py` (new) |
| Keychain storage | `backend/app/services/secrets.py` (new) |
| Engine selector + settings section | `frontend/src/components/*` (small) |
| Tests with a stub Vast API | `backend/tests/test_vast_provider.py` (new) |

The provider contract and result schema stay frozen, which is the reason this was designed
first: adding Vast is additive, not a rewrite.
