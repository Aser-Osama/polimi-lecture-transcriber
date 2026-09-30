import { useEffect, useState } from "react";
import { api } from "../api/client";
import { formatBytes } from "../format";
import type { AlignmentInstallEvent } from "../hooks/useJobs";
import type {
  AlignmentStatus,
  AppSettings,
  LanguageChoice,
  ModelDownloadState,
  ModelInfo,
  OpenRouterModel,
} from "../types";
import { PrivacyNote } from "./PrivacyNote";

interface Props {
  settings: AppSettings;
  defaultOutputDir: string;
  resolvedOutputDir: string;
  models: ModelInfo[];
  modelDownloads: Record<string, ModelDownloadState>;
  openrouterModels: OpenRouterModel[];
  hasOpenRouterKey: boolean;
  alignmentStatus: AlignmentStatus | null;
  alignmentInstall: AlignmentInstallEvent | null;
  onSaved: (settings: AppSettings) => void;
  onModelsChanged: () => void;
  onKeyChanged: () => void;
  onAlignmentChanged: () => void;
  onToast: (message: string, kind?: "info" | "error") => void;
}

export function SettingsView({
  settings,
  defaultOutputDir,
  resolvedOutputDir,
  models,
  modelDownloads,
  openrouterModels,
  hasOpenRouterKey,
  alignmentStatus,
  alignmentInstall,
  onSaved,
  onModelsChanged,
  onKeyChanged,
  onAlignmentChanged,
  onToast,
}: Props) {
  const [draft, setDraft] = useState<AppSettings>(settings);
  const [saving, setSaving] = useState(false);
  const [keyInput, setKeyInput] = useState("");
  const [keyBusy, setKeyBusy] = useState(false);

  useEffect(() => {
    setDraft(settings);
  }, [settings]);

  const dirty = JSON.stringify(draft) !== JSON.stringify(settings);

  const save = async () => {
    setSaving(true);
    try {
      const result = await api.updateSettings(draft);
      onSaved(result.settings);
      onToast("Settings saved.");
    } catch (error) {
      onToast((error as Error).message, "error");
    } finally {
      setSaving(false);
    }
  };

  const reveal = async (target: string) => {
    try {
      await api.reveal(target);
    } catch (error) {
      onToast((error as Error).message, "error");
    }
  };

  return (
    <div className="settings-layout">
      <section className="panel" aria-label="Output settings">
        <h2 className="panel-title">Output</h2>
        <div className="field">
          <label htmlFor="output-dir">Output directory</label>
          <div className="inline-field">
            <input
              id="output-dir"
              type="text"
              value={draft.output_dir ?? ""}
              placeholder={defaultOutputDir}
              onChange={(event) =>
                setDraft({ ...draft, output_dir: event.target.value || null })
              }
            />
            <button
              type="button"
              className="btn"
              onClick={() => setDraft({ ...draft, output_dir: null })}
            >
              Use default
            </button>
          </div>
          <p className="field-hint">
            Files are written to <code>{resolvedOutputDir}</code>. Existing transcripts are never
            overwritten; duplicates get " (2)" suffixes.
          </p>
        </div>
        <div className="button-row">
          <button type="button" className="btn" onClick={() => reveal("output_dir")}>
            Reveal output folder
          </button>
          <button type="button" className="btn" onClick={() => reveal("logs")}>
            Reveal logs
          </button>
        </div>
      </section>

      <section className="panel" aria-label="Default transcription preferences">
        <h2 className="panel-title">Defaults</h2>
        <div className="options-grid three">
          <div className="field">
            <label htmlFor="default-backend">Backend</label>
            <select
              id="default-backend"
              value={draft.default_provider === "openrouter" ? "openrouter" : "local_mlx"}
              onChange={(event) =>
                setDraft({
                  ...draft,
                  default_provider: event.target.value as AppSettings["default_provider"],
                })
              }
            >
              <option value="local_mlx">Local</option>
              <option value="openrouter">OpenRouter</option>
            </select>
          </div>
          <div className="field">
            <label htmlFor="default-model">Local model</label>
            <select
              id="default-model"
              value={draft.default_model_key}
              onChange={(event) => setDraft({ ...draft, default_model_key: event.target.value })}
            >
              {models.map((model) => (
                <option key={model.key} value={model.key}>
                  {model.display_name}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label htmlFor="default-openrouter-model">OpenRouter model</label>
            <select
              id="default-openrouter-model"
              value={draft.default_openrouter_model}
              onChange={(event) =>
                setDraft({ ...draft, default_openrouter_model: event.target.value })
              }
            >
              {openrouterModels.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.display_name}
                </option>
              ))}
            </select>
          </div>
        </div>
        <div className="options-grid">
          <div className="field">
            <label htmlFor="default-language">Default language</label>
            <select
              id="default-language"
              value={draft.default_language}
              onChange={(event) =>
                setDraft({ ...draft, default_language: event.target.value as LanguageChoice })
              }
            >
              <option value="en">English</option>
              <option value="it">Italian</option>
              <option value="auto">Auto Detect</option>
            </select>
          </div>
          <div className="field checkbox-field">
            <label>
              <input
                type="checkbox"
                checked={draft.default_align_with_whisperx}
                onChange={(event) =>
                  setDraft({ ...draft, default_align_with_whisperx: event.target.checked })
                }
              />
              Enable WhisperX alignment by default
            </label>
          </div>
        </div>
      </section>

      <section className="panel" aria-label="OpenRouter">
        <h2 className="panel-title">OpenRouter</h2>
        <p className="field-hint">
          Cloud transcription via OpenRouter. The key is stored in the macOS Keychain — it never
          touches the app database, logs or this browser. Long lectures are split into
          silence-aligned chunks because providers time out after ~60 s per request.
        </p>
        <div className="field">
          <label htmlFor="openrouter-key">API key</label>
          <div className="inline-field">
            <input
              id="openrouter-key"
              type="password"
              value={keyInput}
              placeholder={hasOpenRouterKey ? "Key stored ✓ (enter a new one to replace)" : "sk-or-..."}
              autoComplete="off"
              onChange={(event) => setKeyInput(event.target.value)}
            />
            <button
              type="button"
              className="btn"
              disabled={keyBusy || keyInput.trim().length === 0}
              onClick={() => {
                setKeyBusy(true);
                api
                  .storeOpenRouterKey(keyInput.trim())
                  .then(() => {
                    setKeyInput("");
                    onKeyChanged();
                    onToast("OpenRouter key stored in macOS Keychain.");
                  })
                  .catch((error: Error) => onToast(error.message, "error"))
                  .finally(() => setKeyBusy(false));
              }}
            >
              {keyBusy ? "Saving..." : "Save key"}
            </button>
            {hasOpenRouterKey && (
              <button
                type="button"
                className="btn btn-danger-ghost"
                onClick={() => {
                  api
                    .deleteOpenRouterKey()
                    .then(() => {
                      onKeyChanged();
                      onToast("OpenRouter key removed.");
                    })
                    .catch((error: Error) => onToast(error.message, "error"));
                }}
              >
                Remove
              </button>
            )}
          </div>
          <p className="field-hint">
            {hasOpenRouterKey
              ? "A key is stored in the login Keychain (service PolimiLectureTranscriber)."
              : "No key stored yet."}
          </p>
        </div>
      </section>

      <section className="panel" aria-label="WhisperX alignment">
        <div className="panel-header">
          <h2 className="panel-title">WhisperX alignment</h2>
          <span className={alignmentStatus?.installed ? "chip chip-ok" : "chip chip-muted"}>
            {alignmentStatus === null
              ? "Checking..."
              : alignmentStatus.installed
                ? `Installed (${alignmentStatus.device?.toUpperCase() ?? "CPU"})`
                : "Not installed"}
          </span>
        </div>
        <p className="field-hint">
          Optional local forced alignment that re-times the transcript of any backend (Local or
          OpenRouter) against the original audio without re-transcribing it. Installs into an
          isolated environment (<code>{alignmentStatus?.venv_path ?? ".venv-whisperx"}</code>,
          ~1.5 GB) and does not touch the app's dependencies. Models without any timestamps also
          get a quick local Whisper tiny anchor pass so alignment can work.
        </p>
        {alignmentInstall?.state === "installing" && (
          <p className="field-hint">
            {alignmentInstall.message ?? "Installing..."}
            {alignmentInstall.fraction ? ` ${Math.round(alignmentInstall.fraction * 100)}%` : ""}
          </p>
        )}
        {alignmentInstall?.state === "failed" && (
          <p className="model-error">{alignmentInstall.message}</p>
        )}
        <div className="button-row">
          {!alignmentStatus?.installed && (
            <button
              type="button"
              className="btn"
              disabled={alignmentInstall?.state === "installing"}
              onClick={() => {
                api
                  .installAlignment()
                  .then(() => onToast("WhisperX installation started."))
                  .catch((error: Error) => onToast(error.message, "error"));
              }}
            >
              {alignmentInstall?.state === "installing" ? "Installing..." : "Install WhisperX alignment"}
            </button>
          )}
          {alignmentStatus?.installed && (
            <button type="button" className="btn" onClick={() => onAlignmentChanged()}>
              Re-check status
            </button>
          )}
        </div>
      </section>

      <section className="panel" aria-label="Course vocabulary">
        <h2 className="panel-title">Course vocabulary</h2>
        <div className="field">
          <textarea
            rows={6}
            value={draft.glossary}
            onChange={(event) => setDraft({ ...draft, glossary: event.target.value })}
            placeholder={"One term per line or comma-separated:\nNUMA\nTLB\nMESI\nCUDA"}
          />
          <p className="field-hint">
            Used as Whisper context at the start of transcription. Saved locally in the app
            database (future versions will support per-course glossaries).
          </p>
        </div>
      </section>

      <section className="panel" aria-label="Subtitle preferences">
        <h2 className="panel-title">Subtitles</h2>
        <div className="options-grid three">
          <div className="field">
            <label htmlFor="max-line-chars">Max characters per line</label>
            <input
              id="max-line-chars"
              type="number"
              min={20}
              max={80}
              value={draft.subtitles.max_line_chars}
              onChange={(event) =>
                setDraft({
                  ...draft,
                  subtitles: {
                    ...draft.subtitles,
                    max_line_chars: Number(event.target.value) || 42,
                  },
                })
              }
            />
          </div>
          <div className="field">
            <label htmlFor="max-lines">Max lines</label>
            <input
              id="max-lines"
              type="number"
              min={1}
              max={4}
              value={draft.subtitles.max_lines}
              onChange={(event) =>
                setDraft({
                  ...draft,
                  subtitles: {
                    ...draft.subtitles,
                    max_lines: Number(event.target.value) || 2,
                  },
                })
              }
            />
          </div>
          <div className="field">
            <label htmlFor="max-cue">Preferred max cue duration (s)</label>
            <input
              id="max-cue"
              type="number"
              min={2}
              max={20}
              step={0.5}
              value={draft.subtitles.max_cue_duration}
              onChange={(event) =>
                setDraft({
                  ...draft,
                  subtitles: {
                    ...draft.subtitles,
                    max_cue_duration: Number(event.target.value) || 7,
                  },
                })
              }
            />
          </div>
        </div>
        <div className="field checkbox-field">
          <label>
            <input
              type="checkbox"
              checked={draft.keep_temp_uploads}
              onChange={(event) =>
                setDraft({ ...draft, keep_temp_uploads: event.target.checked })
              }
            />
            Keep temporary copies of dropped files (needed for preview after processing)
          </label>
          <label>
            <input
              type="checkbox"
              checked={draft.reveal_outputs_on_finish}
              onChange={(event) =>
                setDraft({ ...draft, reveal_outputs_on_finish: event.target.checked })
              }
            />
            Reveal the output folder in Finder when a job finishes
          </label>
        </div>
      </section>

      <section className="panel" aria-label="Whisper models">
        <div className="panel-header">
          <h2 className="panel-title">Models</h2>
          <button type="button" className="btn btn-ghost" onClick={() => reveal("model_cache")}>
            Reveal cache folder
          </button>
        </div>
        <ul className="model-list">
          {models.map((model) => {
            const download = modelDownloads[model.key] ?? model.download ?? null;
            const downloading = download?.state === "downloading";
            return (
              <li key={model.key} className="model-card">
                <div className="model-info">
                  <span className="model-name">
                    {model.display_name}
                    {model.cached && <span className="badge badge-ok">Cached</span>}
                  </span>
                  <span className="model-meta">
                    {model.repo_id} · ~{formatBytes(model.approx_size_bytes)}
                  </span>
                  <span className="model-desc">{model.description}</span>
                  {downloading && (
                    <div className="progress-track">
                      <div
                        className="progress-fill"
                        style={{
                          width: download?.fraction ? `${Math.round(download.fraction * 100)}%` : "100%",
                        }}
                      />
                    </div>
                  )}
                  {downloading && (
                    <span className="model-meta">
                      {download?.message ?? "Downloading model..."}
                      {download?.fraction ? ` ${Math.round(download.fraction * 100)}%` : ""}
                    </span>
                  )}
                  {download?.state === "failed" && (
                    <span className="model-error">{download.error ?? download.message}</span>
                  )}
                </div>
                <div className="model-actions">
                  <label className="radio-label">
                    <input
                      type="radio"
                      name="default-model-radio"
                      checked={draft.default_model_key === model.key}
                      onChange={() => setDraft({ ...draft, default_model_key: model.key })}
                    />
                    Use
                  </label>
                  {!model.cached && !downloading && (
                    <button
                      type="button"
                      className="btn btn-small"
                      onClick={() => {
                        api
                          .downloadModel(model.key)
                          .then(() => onModelsChanged())
                          .catch((error: Error) => onToast(error.message, "error"));
                      }}
                    >
                      Download
                    </button>
                  )}
                </div>
              </li>
            );
          })}
        </ul>
        <p className="field-hint">
          Models download on demand into the standard Hugging Face cache and are reused between
          batch jobs.
        </p>
      </section>

      <div className="settings-footer">
        <button
          type="button"
          className="btn btn-primary"
          disabled={!dirty || saving}
          onClick={() => void save()}
        >
          {saving ? "Saving..." : "Save changes"}
        </button>
        {dirty && <span className="field-hint">Unsaved changes</span>}
      </div>
      <PrivacyNote />
    </div>
  );
}
