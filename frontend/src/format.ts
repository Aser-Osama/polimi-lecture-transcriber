import type { JobStatus } from "./types";

export function formatBytes(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined) return "--";
  if (bytes < 1000) return `${bytes} B`;
  // Decimal units to match Hugging Face download sizes shown in the catalog.
  const units = ["KB", "MB", "GB", "TB"];
  let value = bytes / 1000;
  let unit = 0;
  while (value >= 1000 && unit < units.length - 1) {
    value /= 1000;
    unit += 1;
  }
  const digits = value >= 100 ? 0 : value >= 10 ? 1 : 2;
  return `${value.toFixed(digits)} ${units[unit]}`;
}

export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || seconds < 0) return "--";
  const total = Math.round(seconds);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  if (hours > 0) return `${hours}h ${String(minutes).padStart(2, "0")}m ${String(secs).padStart(2, "0")}s`;
  if (minutes > 0) return `${minutes}m ${String(secs).padStart(2, "0")}s`;
  return `${secs}s`;
}

export function formatClock(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(total / 60);
  const secs = total % 60;
  return `${minutes}:${String(secs).padStart(2, "0")}`;
}

export function formatDateTime(iso: string | null | undefined): string {
  if (!iso) return "--";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "--";
  return date.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function formatRealtimeFactor(job: {
  realtime_factor: number | null;
  processing_duration: number | null;
}): string {
  if (job.realtime_factor) return `${job.realtime_factor.toFixed(1)}x realtime`;
  if (job.processing_duration) return `${formatDuration(job.processing_duration)} elapsed`;
  return "--";
}

export const STATUS_LABELS: Record<JobStatus, string> = {
  queued: "Queued",
  preparing: "Reading media",
  extracting_audio: "Preparing audio",
  loading_model: "Loading model",
  transcribing: "Transcribing",
  aligning: "Aligning words",
  formatting: "Formatting subtitles",
  saving: "Saving results",
  completed: "Completed",
  failed: "Failed",
  cancelled: "Cancelled",
  interrupted: "Interrupted",
};

export const STAGE_ORDER: JobStatus[] = [
  "preparing",
  "extracting_audio",
  "loading_model",
  "transcribing",
  "aligning",
  "formatting",
  "saving",
];

export function stageIndex(status: JobStatus): number {
  return STAGE_ORDER.indexOf(status);
}

export const LANGUAGE_LABELS: Record<string, string> = {
  en: "English",
  it: "Italian",
  auto: "Auto Detect",
};

export function elapsedSince(iso: string | null | undefined): string {
  if (!iso) return "--";
  const start = new Date(iso).getTime();
  if (Number.isNaN(start)) return "--";
  return formatDuration((Date.now() - start) / 1000);
}
