export type JobStatus =
  | "queued"
  | "preparing"
  | "extracting_audio"
  | "loading_model"
  | "transcribing"
  | "aligning"
  | "formatting"
  | "saving"
  | "completed"
  | "failed"
  | "cancelled"
  | "interrupted";

export const ACTIVE_STATUSES: JobStatus[] = [
  "preparing",
  "extracting_audio",
  "loading_model",
  "transcribing",
  "aligning",
  "formatting",
  "saving",
];

export interface MediaStreamInfo {
  index: number;
  codec_type: string;
  codec_name: string | null;
  sample_rate: number | null;
  channels: number | null;
  language: string | null;
  width: number | null;
  height: number | null;
  is_attached_picture: boolean;
}

export interface MediaInfo {
  path: string;
  filename: string;
  size_bytes: number;
  duration_seconds: number | null;
  format_name: string | null;
  has_audio: boolean;
  has_video: boolean;
  audio_codec: string | null;
  video_codec: string | null;
  sample_rate: number | null;
  channels: number | null;
  bit_rate: number | null;
  streams: MediaStreamInfo[];
}

export type LanguageChoice = "en" | "it" | "auto";

export type ProviderName = "local_mlx" | "openrouter" | "fake";

export type AlignmentMode = "none" | "local_whisperx" | "cloud";

export interface JobConfig {
  model_key: string;
  language: LanguageChoice;
  provider: ProviderName;
  openrouter_model: string | null;
  align_with_whisperx: boolean;
  alignment_mode: AlignmentMode;
  glossary: string;
  global_context: string;
  per_file_context: string;
  project_id: string | null;
  project_name: string | null;
  options: Record<string, unknown>;
}

export interface Project {
  id: string;
  name: string;
  context: string;
  created_at: string;
  updated_at: string;
}

export interface OpenRouterModel {
  id: string;
  display_name: string;
  description: string;
  timestamps: "words" | "segments" | "none" | "unknown";
  default: boolean;
}

export interface AlignmentInstallState {
  state: "idle" | "installing" | "completed" | "failed";
  message: string | null;
  fraction?: number | null;
}

export interface AlignmentStatus {
  installed: boolean;
  supported?: boolean;
  venv_path: string;
  venv_exists: boolean;
  script_exists: boolean;
  device: string | null;
  version: string | null;
  message: string;
  install: AlignmentInstallState;
}

export interface Job {
  id: string;
  source_path: string;
  source_filename: string;
  source_is_temporary: boolean;
  size_bytes: number | null;
  media: MediaInfo | null;
  config: JobConfig;
  status: JobStatus;
  status_message: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  processing_duration: number | null;
  realtime_factor: number | null;
  detected_language: string | null;
  error: string | null;
  outputs: Record<string, string>;
  timings: Record<string, number>;
  provider_meta: Record<string, unknown>;
  archived: boolean;
}

export interface SubtitlePreferences {
  max_line_chars: number;
  max_lines: number;
  max_cue_duration: number;
}

export interface AppSettings {
  output_dir: string | null;
  default_model_key: string;
  default_language: LanguageChoice;
  default_provider: ProviderName;
  default_openrouter_model: string;
  default_align_with_whisperx: boolean;
  default_alignment_mode: AlignmentMode;
  max_parallel_cloud_jobs: number;
  glossary: string;
  subtitles: SubtitlePreferences;
  keep_temp_uploads: boolean;
  reveal_outputs_on_finish: boolean;
}

export interface ModelDownloadState {
  state: "idle" | "downloading" | "completed" | "failed";
  fraction: number | null;
  message: string | null;
  error?: string | null;
}

export interface ModelInfo {
  key: string;
  display_name: string;
  repo_id: string;
  approx_size_bytes: number;
  description: string;
  default: boolean;
  cached: boolean;
  download: ModelDownloadState | null;
}

export interface PreviewCue {
  index: number;
  start: number;
  end: number;
  text: string;
}

export interface PreviewData {
  job_id: string;
  cues: PreviewCue[];
  media_available: boolean;
  media_url: string | null;
  subtitles_url: string;
  media_duration: number | null;
  warnings: string[];
  realtime_factor: number | null;
  outputs: Record<string, string>;
}

export interface Capabilities {
  platform: string;
  platform_label: string;
  local_transcription: boolean;
  local_alignment: boolean;
  native_file_picker: boolean;
  key_storage: "keychain" | "file";
}

export interface Health {
  status: string;
  app_version: string;
  result_schema_version: number;
  ffmpeg: { ok: boolean; info: string | null };
  mlx_provider: { ok: boolean; error: string | null; supported?: boolean };
  capabilities: Capabilities;
  active_job_id: string | null;
}

export interface UploadResponse {
  upload_id: string;
  path: string;
  filename: string;
  size_bytes: number;
  media: MediaInfo;
}

export interface PickFile {
  path: string;
  filename: string;
  size_bytes: number;
  media: MediaInfo;
}

export interface SelectedItem {
  key: string;
  path?: string;
  upload_id?: string;
  filename: string;
  size_bytes: number;
  media: MediaInfo;
}

export type ConnectionState = "connecting" | "live" | "offline";

export type ViewName = "transcribe" | "history" | "settings";
