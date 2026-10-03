# Future plan: per-job chunk parallelism (cloud)

Status: **not implemented**. Today each OpenRouter job sends its audio chunks one at a time
(`OpenRouterProvider.transcribe_sync`), so a single long lecture is bounded by its sequential
chunk chain. This document describes the planned bounded-parallelism change so it can be added
without touching the queue, exports or UI contracts.

## Why it matters (measured, M5 Pro + MAI-Transcribe 2)

For a 106-minute lecture: local prep ~13 s (2.3 s extraction + 10.8 s MP3 chunking), upload
~51 MB (~7 s at 56 Mbps), then **22 chunks × 3–5 s round trip, sequential** = 4–7 min. Batch
wall time is therefore ≈ the slowest single video, not the number of jobs. Processing 4 chunks
concurrently should cut per-lecture time ~3–4x (to ~1.5–2.5 min), with no changes to the
provider or output formats.

## Design

1. **Bounded pool inside the provider.** `transcribe_sync` runs chunk requests in a
   `ThreadPoolExecutor` with `K` workers; `K` comes from a new setting
   `chunk_parallelism` (1–8, default 4) or `request.options["chunk_parallelism"]` for tests.
   `K = 1` reproduces today's behavior exactly.
2. **Order-safe merge is already there.** The retry-pass refactor stores payloads in a
   `{position: payload}` dict and merges in chunk order; parallel completion needs no extra
   ordering logic.
3. **Format probing before fan-out.** `_verbose_supported` is shared mutable state. Probe the
   first chunk once (verbose or plain) before starting the pool, then fan out with the decided
   format so threads never race on the flag.
4. **Retries stay per chunk.** Each chunk keeps its 5 attempts + `Retry-After` handling; the
   second pass over failed chunks runs after the first wave drains, with the same
   chronological merge.
5. **Progress + cancellation.** Completed-chunk counter under a lock feeds the existing
   `progress("transcribing", ...)` messages; `CancellationToken` is checked before starting
   each new chunk. In-flight HTTP requests finish within the provider timeout; the worker
   force-kill path is unchanged.
6. **Global request budget.** In-flight requests = active jobs × K. Workers are separate
   processes, so a cross-process semaphore is awkward; instead the scheduler should treat the
   pair as one budget, e.g. `max_parallel_cloud_jobs × chunk_parallelism ≤ 64`, or expose a
   single "cloud request budget" setting that the manager divides into jobs. Defaults
   (16 jobs × 4 chunks) stay inside observed provider limits; MAI-2 throttles earlier than
   Whisper/Qwen, so the UI hint should recommend K=4 for MAI-2 and 6–8 for the others.
7. **Memory/CPU.** Each in-flight chunk holds its base64 body (~1–2 MB) plus one response;
   K=8 per job is negligible on any supported machine. Chunk files are already encoded before
   upload, so local CPU does not change.

## Tests

- Mock server with a peak counter: one job with K=4 must reach 4 concurrent requests and still
  produce chunks in chronological order.
- One chunk returning 429 must not cancel the others; the retry pass still recovers it.
- Cancellation during a wave stops new chunks and raises `CancelledError`.
- Settings validation for `chunk_parallelism` (1–8) and the budget rule.
- Real E2E: compare wall time of one long lecture at K=1 vs K=4 (expect ~3x).

## Non-goals

- Parallelizing across different models/providers, resumable uploads, or streaming partials.
- Changing `RawTranscription`, result schema, exporters or the queue UI.
