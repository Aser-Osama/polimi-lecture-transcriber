import {
  formatBytes,
  formatDateTime,
  formatDuration,
  LANGUAGE_LABELS,
} from "../format";
import type { Job, ModelInfo } from "../types";
import { StatusChip } from "./StatusChip";

interface Props {
  jobs: Job[];
  models: ModelInfo[];
  onPreview: (job: Job) => void;
  onRetry: (id: string) => void;
  onReveal: (id: string) => void;
  onDelete: (job: Job) => void;
}

export function HistoryView({ jobs, models, onPreview, onRetry, onReveal, onDelete }: Props) {
  return (
    <section className="panel history-panel" aria-label="Job history">
      <h2 className="panel-title">History</h2>
      {jobs.length === 0 ? (
        <p className="empty-state">No jobs yet. Transcribed lectures will appear here.</p>
      ) : (
        <div className="history-scroll">
          <table className="history-table">
            <thead>
              <tr>
                <th>Date</th>
                <th>File</th>
                <th>Model</th>
                <th>Language</th>
                <th>Duration</th>
                <th>Processing</th>
                <th>Status</th>
                <th aria-label="Actions" />
              </tr>
            </thead>
            <tbody>
              {jobs.map((job) => (
                <tr key={job.id}>
                  <td>{formatDateTime(job.created_at)}</td>
                  <td className="history-name" title={job.source_path}>
                    {job.source_filename}
                    <span className="history-size">{formatBytes(job.size_bytes)}</span>
                  </td>
                  <td>
                    {job.config.provider === "openrouter"
                      ? `OpenRouter · ${job.config.openrouter_model ?? ""}`
                      : (models.find((model) => model.key === job.config.model_key)
                          ?.display_name ?? job.config.model_key)}
                  </td>
                  <td>{LANGUAGE_LABELS[job.config.language] ?? job.config.language}</td>
                  <td>{formatDuration(job.media?.duration_seconds ?? null)}</td>
                  <td>
                    {job.processing_duration ? formatDuration(job.processing_duration) : "--"}
                    {job.realtime_factor ? ` (${job.realtime_factor.toFixed(1)}x)` : ""}
                  </td>
                  <td>
                    <StatusChip status={job.status} />
                    {job.error && <span className="history-error">{job.error}</span>}
                  </td>
                  <td className="history-actions">
                    {job.status === "completed" && (
                      <>
                        <button type="button" className="btn btn-small" onClick={() => onPreview(job)}>
                          Preview
                        </button>
                        <a
                          className="btn btn-small"
                          href={`/api/jobs/${job.id}/outputs/txt`}
                          target="_blank"
                          rel="noreferrer"
                        >
                          Transcript
                        </a>
                        {job.outputs["srt"] && (
                          <a
                            className="btn btn-small"
                            href={`/api/jobs/${job.id}/outputs/srt`}
                            target="_blank"
                            rel="noreferrer"
                          >
                            SRT
                          </a>
                        )}
                        <button type="button" className="btn btn-small" onClick={() => onReveal(job.id)}>
                          Finder
                        </button>
                      </>
                    )}
                    {(job.status === "failed" ||
                      job.status === "interrupted" ||
                      job.status === "cancelled") && (
                      <button type="button" className="btn btn-small" onClick={() => onRetry(job.id)}>
                        Retry
                      </button>
                    )}
                    <button
                      type="button"
                      className="btn btn-small btn-danger-ghost"
                      onClick={() => onDelete(job)}
                    >
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
