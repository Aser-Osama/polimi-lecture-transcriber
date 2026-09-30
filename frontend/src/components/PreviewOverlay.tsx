import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import { formatClock, formatDuration } from "../format";
import type { Job, PreviewData, PreviewCue } from "../types";

interface Props {
  job: Job;
  onClose: () => void;
  onError: (message: string) => void;
}

export function PreviewOverlay({ job, onClose, onError }: Props) {
  const [data, setData] = useState<PreviewData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeCue, setActiveCue] = useState<number | null>(null);
  const [showWarnings, setShowWarnings] = useState(false);
  const mediaRef = useRef<HTMLVideoElement>(null);
  const listRef = useRef<HTMLUListElement>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .preview(job.id)
      .then((result) => {
        if (!cancelled) setData(result);
      })
      .catch((err: Error) => {
        if (!cancelled) setError(err.message);
      });
    return () => {
      cancelled = true;
    };
  }, [job.id]);

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose]);

  const handleTimeUpdate = useCallback(() => {
    const media = mediaRef.current;
    if (!media || !data) return;
    const time = media.currentTime;
    const index = data.cues.findIndex((cue) => time >= cue.start && time < cue.end);
    setActiveCue(index === -1 ? null : index);
  }, [data]);

  useEffect(() => {
    if (activeCue === null || !listRef.current) return;
    const element = listRef.current.querySelector<HTMLElement>(
      `[data-cue-index="${activeCue}"]`,
    );
    element?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [activeCue]);

  const seekTo = (cue: PreviewCue) => {
    const media = mediaRef.current;
    if (!media) return;
    media.currentTime = cue.start;
    void media.play().catch(() => undefined);
  };

  const hasVideo = job.media?.has_video ?? false;

  return (
    <div className="overlay preview-overlay" role="dialog" aria-modal="true" aria-label={`Preview ${job.source_filename}`}>
      <div className="preview-panel">
        <header className="preview-header">
          <div>
            <h2>{job.source_filename}</h2>
            <p className="preview-subtitle">
              {formatDuration(data?.media_duration ?? job.media?.duration_seconds ?? null)} ·{" "}
              {job.outputs["basename"] ? `${job.outputs["basename"]}.srt` : "subtitles"} ·{" "}
              {data?.realtime_factor ? `${data.realtime_factor.toFixed(1)}x realtime` : ""}
            </p>
          </div>
          <button type="button" className="icon-btn" aria-label="Close preview" onClick={onClose}>
            ✕
          </button>
        </header>
        {error && <p className="job-error">{error}</p>}
        <div className="preview-body">
          <div className="preview-media">
            {data?.media_available && data.media_url ? (
              hasVideo ? (
                <video
                  ref={mediaRef}
                  controls
                  autoPlay={false}
                  onTimeUpdate={handleTimeUpdate}
                  src={data.media_url}
                  className="media-element"
                >
                  <track
                    kind="subtitles"
                    src={data.subtitles_url}
                    srcLang={job.detected_language ?? job.config.language}
                    label="Generated subtitles"
                    default
                  />
                </video>
              ) : (
                <div className="audio-wrap">
                  <video
                    ref={mediaRef}
                    controls
                    onTimeUpdate={handleTimeUpdate}
                    src={data.media_url}
                    className="visually-hidden-media"
                  />
                  <div className="audio-illustration" aria-hidden="true">♪</div>
                  <p>Audio-only source: cue list on the right stays synchronized.</p>
                </div>
              )
            ) : (
              <p className="empty-state">
                The source media is no longer available at its original path
                {job.source_is_temporary ? " (temporary upload was removed)" : ""}. Subtitles are
                shown from the result file.
              </p>
            )}
            {data && data.cues.length === 0 && (
              <p className="empty-state">No subtitle cues were produced for this media.</p>
            )}
            <div className="preview-links">
              <a className="btn btn-small" href={`/api/jobs/${job.id}/outputs/txt`} target="_blank" rel="noreferrer">
                Open transcript
              </a>
              <a className="btn btn-small" href={`/api/jobs/${job.id}/outputs/srt`} target="_blank" rel="noreferrer">
                Open SRT
              </a>
              <a className="btn btn-small" href={`/api/jobs/${job.id}/outputs/vtt`} target="_blank" rel="noreferrer">
                Open VTT
              </a>
              <button
                type="button"
                className="btn btn-small"
                onClick={() => {
                  api.reveal("output_dir").catch((err: Error) => onError(err.message));
                }}
              >
                Reveal output folder
              </button>
            </div>
            {data && data.warnings.length > 0 && (
              <div className="preview-warnings">
                <button
                  type="button"
                  className="link-btn"
                  aria-expanded={showWarnings}
                  onClick={() => setShowWarnings((open) => !open)}
                >
                  {data.warnings.length} warning{data.warnings.length === 1 ? "" : "s"} from
                  processing
                </button>
                {showWarnings && (
                  <ul>
                    {data.warnings.map((warning, index) => (
                      <li key={index}>{warning}</li>
                    ))}
                  </ul>
                )}
              </div>
            )}
          </div>
          <div className="preview-cues" aria-label="Subtitle cues">
            {data ? (
              <ul className="cue-list" ref={listRef}>
                {data.cues.map((cue, index) => (
                  <li key={cue.index}>
                    <button
                      type="button"
                      data-cue-index={index}
                      className={activeCue === index ? "cue-row active" : "cue-row"}
                      onClick={() => seekTo(cue)}
                      title="Jump to this cue"
                    >
                      <span className="cue-time">
                        {formatClock(cue.start)} → {formatClock(cue.end)}
                      </span>
                      <span className="cue-text">{cue.text.replace(/\n/g, " ")}</span>
                    </button>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="empty-state">Loading cues...</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
