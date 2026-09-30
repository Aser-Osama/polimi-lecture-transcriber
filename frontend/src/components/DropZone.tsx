import { useRef, useState } from "react";

interface Props {
  onFiles: (files: File[]) => void;
  onPickFiles: () => void;
  busy: boolean;
}

export function DropZone({ onFiles, onPickFiles, busy }: Props) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const stop = (event: React.DragEvent) => {
    event.preventDefault();
    event.stopPropagation();
  };

  return (
    <section
      className={dragging ? "dropzone dragging" : "dropzone"}
      onDragEnter={(event) => {
        stop(event);
        setDragging(true);
      }}
      onDragOver={stop}
      onDragLeave={(event) => {
        stop(event);
        setDragging(false);
      }}
      onDrop={(event) => {
        stop(event);
        setDragging(false);
        const files = Array.from(event.dataTransfer.files);
        if (files.length) onFiles(files);
      }}
      aria-label="Drop lecture media files here"
    >
      <input
        ref={inputRef}
        type="file"
        multiple
        className="visually-hidden"
        accept=".mp4,.mov,.mkv,.webm,.m4v,.mp3,.m4a,.aac,.wav,.flac,.ogg,.opus,.aiff,.aif"
        onChange={(event) => {
          const files = Array.from(event.target.files ?? []);
          if (files.length) onFiles(files);
          event.target.value = "";
        }}
      />
      <div className="dropzone-icon" aria-hidden="true">
        <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
          <path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5" strokeLinecap="round" strokeLinejoin="round" />
          <path d="M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3" strokeLinecap="round" />
        </svg>
      </div>
      <p className="dropzone-title">
        Drop lecture video or audio here, or{" "}
        <button
          type="button"
          className="link-btn"
          onClick={onPickFiles}
          disabled={busy}
        >
          choose files without copying
        </button>
      </p>
      <p className="dropzone-hint">
        MP4, MOV, MKV, WEBM, MP3, M4A, WAV, FLAC - batch supported. The native picker processes
        original files in place; dropped files are copied to a temporary folder and removed after
        processing.
      </p>
    </section>
  );
}
