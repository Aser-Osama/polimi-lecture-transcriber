import { useState } from "react";
import {
  formatBytes,
  formatDateTime,
  formatDuration,
  formatRealtimeFactor,
  LANGUAGE_LABELS,
  STATUS_LABELS,
} from "../format";
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
  fractions: Record<string, number>;
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
}: { job: Job; fraction?: number; models: ModelInfo[] } & JobActionHandlers) {
  const [menuOpen, setMenuOpen] = useState(false);
  const isActive = ACTIVE_STATUSES.includes(job.status);
  const now = useNow(isActive);
  const modelName =
    models.find((model) => model.key === job.config.model_key)?.display_name ??
    job.config.model_key;

  const elapsed =
    isActive && job.started_at
      ? formatDuration((now - new Date(job.started_at).getTime()) / 1000)
      : null;

  const determinate =
    job.status === "extracting_audio" && fraction !== undefined && fraction > 0
      ? fraction
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
          {elapsed && <span>{elapsed} elapsed</span>}
          {job.created_at && !job.started_at && (
            <span>added {formatDateTime(job.created_at)}</span>
          )}
        </div>
        {(isActive || job.status === "queued") && (
          <div className="job-progress" role="progressbar" aria-label={STATUS_LABELS[job.status]}>
            {determinate !== null ? (
              <div className="progress-track">
                <div className="progress-fill" style={{ width: `${Math.round(determinate * 100)}%` }} />
              </div>
            ) : (
              isActive && <div className="progress-track indeterminate"><div className="progress-fill" /></div>
            )}
            <span className="progress-label">
              {job.status_message ?? STATUS_LABELS[job.status]}
              {determinate !== null ? ` ${Math.round(determinate * 100)}%` : ""}
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
