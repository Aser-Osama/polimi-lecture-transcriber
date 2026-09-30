import { formatBytes, formatDuration } from "../format";
import type { SelectedItem } from "../types";

interface Props {
  items: SelectedItem[];
  onRemove: (key: string) => void;
}

export function SelectedFiles({ items, onRemove }: Props) {
  if (!items.length) return null;
  return (
    <section className="panel" aria-label="Selected files">
      <h2 className="panel-title">Selected files ({items.length})</h2>
      <ul className="file-list">
        {items.map((item) => (
          <li key={item.key} className="file-row">
            <div className="file-primary">
              <span className="file-name">{item.filename}</span>
              <span className="file-meta">
                {formatBytes(item.size_bytes)} · {formatDuration(item.media.duration_seconds)}
                {item.media.has_video ? ` · video ${item.media.video_codec ?? ""}` : ""}
                {item.media.has_audio ? ` · audio ${item.media.audio_codec ?? ""}` : ""}
                {item.media.sample_rate ? ` · ${item.media.sample_rate} Hz` : ""}
              </span>
            </div>
            <div className="file-badges">
              {item.media.has_video && <span className="badge">Video</span>}
              {item.media.has_audio && <span className="badge">Audio</span>}
              {!item.media.has_audio && <span className="badge badge-danger">No audio track</span>}
            </div>
            <button
              type="button"
              className="icon-btn"
              aria-label={`Remove ${item.filename}`}
              onClick={() => onRemove(item.key)}
            >
              ✕
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
