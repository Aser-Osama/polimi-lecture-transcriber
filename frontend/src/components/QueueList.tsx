import { useState } from "react";
import {
  formatBytes,
  formatDateTime,
  formatDuration,
  formatRealtimeFactor,
  LANGUAGE_LABELS,
  STATUS_LABELS,
} from "../format";
import type { StageFraction } from "../hooks/useJobs";
import { useNow } from "../hooks/useNow";
import { ACTIVE_STATUSES, type Job, type ModelInfo } from "../types";
import { StatusChip } from "./StatusChip";

export interface JobActionHandlers {
  onCancel: (id: string) => void;
  onRetry: (id: string) => void;
  onRemove: (job: Job) => void;
  onArchive: (job: Job) => void;
  onPreview: (job: Job) => void;
  onReveal: (id: string) => void;
  onRegenerate: (id: string) => void;
  onDeleteHistory: (job: Job) => void;
  onClearFinished: () => void;
}

interface Props extends JobActionHandlers {
  jobs: Job[];
  fractions: Record<string, StageFraction>;
  models: ModelInfo[];
  connection: string;
}

export function QueueList({
  jobs,
  fractions,
  models,
  connection,
  ...handlers
}: Props) {
  const visible = jobs.filter((job) => !job.archived);
  const hasFinished = visible.some(
    (job) => !ACTIVE_STATUSES.includes(job.status) && job.status !== "queued",
  );
  return (
    <section className="panel queue-panel" aria-label="Job queue">
      <div className="panel-header">
        <h2 className="panel-title">Queue</h2>
        {hasFinished && (
          <button type="button" className="btn btn-ghost" onClick={handlers.onClearFinished}>
            Clear finished
          </button>
        )}
      </div>
      {visible.length === 0 ? (
        <p className="empty-state">
          {connection === "live"
            ? "Nothing queued. Add lecture files above to start."
            : "No connection to the local server."}
        </p>
      ) : (
        <ul className="queue-list">
          {visible.map((job) => (
            <JobRow
              key={job.id}
              job={job}
              fraction={fractions[job.id]}
              models={models}
              {...handlers}
            />
          ))}
        </ul>
      )}
    </section>
  );
}

function JobRow({
  job,
  fraction,
  models,
  onCancel,
  onRetry,
  onRemove,
  onArchive,
  onPreview,
  onReveal,
  onRegenerate,
  onDeleteHistory,
}: { job: Job; fraction?: StageFraction; models: ModelInfo[] } & JobActionHandlers) {
  const [menuOpen, setMenuOpen] = useState(false);
  const isActive = ACTIVE_STATUSES.includes(job.status);
  const now = useNow(isActive);
  const localModelName =
    models.find((model) => model.key === job.config.model_key)?.display_name ??
    job.config.model_key;
  const modelName =
    job.config.provider === "openrouter"
      ? `OpenRouter · ${job.config.openrouter_model ?? ""}`
      : localModelName;
  const costUsd =
    typeof job.provider_meta?.cost_usd === "number" ? job.provider_meta.cost_usd : null;
  const alignmentProvider =
    typeof job.provider_meta?.alignment_provider === "string"
      ? job.provider_meta.alignment_provider
      : null;

  const startedMs = job.started_at ? new Date(job.started_at).getTime() : null;
  const elapsedSeconds = isActive && startedMs ? (now - startedMs) / 1000 : null;
  const elapsed = elapsedSeconds !== null ? formatDuration(elapsedSeconds) : null;

  // A fraction is only meaningful for the status it was reported for.
  const determinate =
    fraction && fraction.status === job.status && fraction.fraction !== null
      ? fraction.fraction
      : null;
  const percent = determinate !== null ? Math.round(determinate * 100) : null;
  const eta =
    determinate !== null &&
    determinate >= 0.05 &&
    determinate < 0.95 &&
    elapsedSeconds !== null &&
    elapsedSeconds > 5
      ? formatDuration(
          Math.max(1, Math.round((elapsedSeconds * (1 - determinate)) / determinate)),
        )
      : null;

  return (
    <li className="job-row">
      <div className="job-main">
        <div className="job-title-line">
          <span className="job-name" title={job.source_path}>
            {job.source_filename}
          </span>
          <StatusChip status={job.status} />
        </div>
        <div className="job-meta" aria-live="polite">
          <span>{formatDuration(job.media?.duration_seconds ?? null)}</span>
          <span>{formatBytes(job.size_bytes)}</span>
          <span>{modelName}</span>
          <span>{LANGUAGE_LABELS[job.config.language] ?? job.config.language}</span>
          {job.detected_language && !["en", "it"].includes(job.detected_language) && (
            <span>detected: {job.detected_language}</span>
          )}
          {job.completed_at && job.status === "completed" && (
            <span>{formatRealtimeFactor(job)}</span>
          )}
          {job.status === "completed" && job.processing_duration && (
            <span>processing {formatDuration(job.processing_duration)}</span>
          )}
          {costUsd !== null && costUsd > 0 && <span>${costUsd.toFixed(4)}</span>}
          {alignmentProvider?.includes("whisperx") && <span>WhisperX aligned</span>}
          {job.config.project_name && <span>project: {job.config.project_name}</span>}
          {elapsed && <span>{elapsed} elapsed</span>}
          {job.created_at && !job.started_at && (
            <span>added {formatDateTime(job.created_at)}</span>
          )}
        </div>
        {(isActive || job.status === "queued") && (
          <div
            className="job-progress"
            role="progressbar"
            aria-label={STATUS_LABELS[job.status]}
            aria-valuemin={determinate !== null ? 0 : undefined}
            aria-valuemax={determinate !== null ? 100 : undefined}
            aria-valuenow={percent ?? undefined}
          >
            {determinate !== null ? (
              <div className="progress-track">
                <div className="progress-fill" style={{ width: `${percent}%` }} />
              </div>
            ) : (
              isActive && <div className="progress-track indeterminate"><div className="progress-fill" /></div>
            )}
            <span className="progress-label">
              {job.status_message ?? STATUS_LABELS[job.status]}
              {percent !== null ? ` ${percent}%` : ""}
              {eta ? ` · ~${eta} left` : ""}
            </span>
          </div>
        )}
        {job.error && <p className="job-error">{job.error}</p>}
      </div>
      <div className="job-actions">
        {isActive && (
          <button type="button" className="btn btn-small" onClick={() => onCancel(job.id)}>
            Cancel
          </button>
        )}
        {job.status === "queued" && (
          <button type="button" className="btn btn-small" onClick={() => onRemove(job)}>
            Remove
          </button>
        )}
        {job.status === "completed" && (
          <button
            type="button"
            className="btn btn-small btn-primary"
            onClick={() => onPreview(job)}
          >
            Preview
          </button>
        )}
        {(job.status === "failed" ||
          job.status === "interrupted" ||
          job.status === "cancelled") && (
          <button type="button" className="btn btn-small" onClick={() => onRetry(job.id)}>
            Retry
          </button>
        )}
        <div className="row-menu">
          <button
            type="button"
            className="icon-btn"
            aria-haspopup="menu"
            aria-expanded={menuOpen}
            aria-label={`More actions for ${job.source_filename}`}
            onClick={() => setMenuOpen((open) => !open)}
          >
            ⋯
          </button>
          {menuOpen && (
            <>
              <div className="menu-backdrop" onClick={() => setMenuOpen(false)} />
              <div className="menu" role="menu">
                {job.status === "completed" && (
                  <>
                    <a
                      className="menu-item"
                      role="menuitem"
                      href={`/api/jobs/${job.id}/media`}
                      target="_blank"
                      rel="noreferrer"
                      onClick={() => setMenuOpen(false)}
                    >
                      Preview in player
                    </a>
                    <a
                      className="menu-item"
                      role="menuitem"
                      href={`/api/jobs/${job.id}/outputs/txt`}
                      target="_blank"
                      rel="noreferrer"
                      onClick={() => setMenuOpen(false)}
                    >
                      Open transcript (.txt)
                    </a>
                    {job.outputs["srt"] && (
                      <a
                        className="menu-item"
                        role="menuitem"
                        href={`/api/jobs/${job.id}/outputs/srt`}
                        target="_blank"
                        rel="noreferrer"
                        onClick={() => setMenuOpen(false)}
                      >
                        Open subtitles (.srt)
                      </a>
                    )}
                    {job.outputs["vtt"] && (
                      <a
                        className="menu-item"
                        role="menuitem"
                        href={`/api/jobs/${job.id}/outputs/vtt`}
                        target="_blank"
                        rel="noreferrer"
                        onClick={() => setMenuOpen(false)}
                      >
                        Open subtitles (.vtt)
                      </a>
                    )}
                    <a
                      className="menu-item"
                      role="menuitem"
                      href={`/api/jobs/${job.id}/outputs/json`}
                      target="_blank"
                      rel="noreferrer"
                      onClick={() => setMenuOpen(false)}
                    >
                      Open result (.json)
                    </a>
                    <button
                      type="button"
                      className="menu-item"
                      role="menuitem"
                      onClick={() => {
                        setMenuOpen(false);
                        onReveal(job.id);
                      }}
                    >
                      Reveal in Finder
                    </button>
                    <button
                      type="button"
                      className="menu-item"
                      role="menuitem"
                      onClick={() => {
                        setMenuOpen(false);
                        onRegenerate(job.id);
                      }}
                    >
                      Regenerate subtitles from JSON
                    </button>
                  </>
                )}
                {job.status !== "completed" && job.status !== "queued" && !isActive && (
                  <button
                    type="button"
                    className="menu-item"
                    role="menuitem"
                    onClick={() => {
                      setMenuOpen(false);
                      onRetry(job.id);
                    }}
                  >
                    Retry
                  </button>
                )}
                {!isActive && (
                  <button
                    type="button"
                    className="menu-item"
                    role="menuitem"
                    onClick={() => {
                      setMenuOpen(false);
                      if (job.status === "completed") onArchive(job);
                      else onDeleteHistory(job);
                    }}
                  >
                    {job.status === "completed" ? "Clear from queue" : "Delete entry"}
                  </button>
                )}
              </div>
            </>
          )}
        </div>
      </div>
    </li>
  );
}
