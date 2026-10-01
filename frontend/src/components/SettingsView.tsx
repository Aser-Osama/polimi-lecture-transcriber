import { useEffect, useState } from "react";
import { api } from "../api/client";
import { formatBytes, termCount } from "../format";
import type { AlignmentInstallEvent } from "../hooks/useJobs";
import type {
  AlignmentStatus,
  AppSettings,
  LanguageChoice,
  ModelDownloadState,
  ModelInfo,
  OpenRouterModel,
  Project,
} from "../types";
import { ConfirmDialog } from "./ConfirmDialog";
import { PrivacyNote } from "./PrivacyNote";

interface Props {
  settings: AppSettings;
  localSupported: boolean;
  platformLabel: string;
  keyStorage: "keychain" | "file";
  defaultOutputDir: string;
  resolvedOutputDir: string;
  models: ModelInfo[];
  modelDownloads: Record<string, ModelDownloadState>;
  openrouterModels: OpenRouterModel[];
  hasOpenRouterKey: boolean;
  alignmentStatus: AlignmentStatus | null;
  alignmentInstall: AlignmentInstallEvent | null;
  projects: Project[];
  onSaved: (settings: AppSettings) => void;
  onModelsChanged: () => void;
  onKeyChanged: () => void;
  onAlignmentChanged: () => void;
  onProjectsChanged: () => void;
  onToast: (message: string, kind?: "info" | "error") => void;
}

export function SettingsView({
  settings,
  localSupported,
  platformLabel,
  keyStorage,
  defaultOutputDir,
  resolvedOutputDir,
  models,
  modelDownloads,
  openrouterModels,
  hasOpenRouterKey,
  alignmentStatus,
  alignmentInstall,
  projects,
  onSaved,
  onModelsChanged,
  onKeyChanged,
  onAlignmentChanged,
  onProjectsChanged,
  onToast,
}: Props) {
  const [draft, setDraft] = useState<AppSettings>(settings);
  const [saving, setSaving] = useState(false);
  const [keyInput, setKeyInput] = useState("");
  const [keyBusy, setKeyBusy] = useState(false);
  const [newProjectName, setNewProjectName] = useState("");
  const [newProjectContext, setNewProjectContext] = useState("");
  const [editingProjectId, setEditingProjectId] = useState<string | null>(null);
  const [editName, setEditName] = useState("");
  const [editContext, setEditContext] = useState("");
  const [deleteTarget, setDeleteTarget] = useState<Project | null>(null);
  const [importingId, setImportingId] = useState<string | null>(null);

  const countTerms = (text: string) =>
    text
      .split(/[\n,;]/)
      .map((term) => term.trim())
      .filter(Boolean).length;

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
              {localSupported && <option value="local_mlx">Local</option>}
              <option value="openrouter">OpenRouter</option>
            </select>
            {!localSupported && (
              <p className="field-hint">
                Local MLX Whisper is available on macOS only; this platform uses OpenRouter.
              </p>
            )}
          </div>
          {localSupported && (
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
          )}
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
          <div className="field">
            <label htmlFor="default-alignment">Default alignment</label>
            <select
              id="default-alignment"
              value={
                draft.default_alignment_mode !== "none"
                  ? draft.default_alignment_mode
                  : draft.default_align_with_whisperx
                    ? "local_whisperx"
                    : "none"
              }
              onChange={(event) => {
                const value = event.target.value as AppSettings["default_alignment_mode"];
                setDraft({
                  ...draft,
                  default_alignment_mode: value,
                  default_align_with_whisperx: value === "local_whisperx",
                });
              }}
            >
              <option value="none">None</option>
              <option value="cloud">Cloud (OpenRouter, MAI anchors when needed)</option>
              {localSupported && <option value="local_whisperx">Local WhisperX</option>}
            </select>
          </div>
          <div className="field">
            <label htmlFor="parallel-cloud">Parallel cloud jobs</label>
            <select
              id="parallel-cloud"
              value={draft.max_parallel_cloud_jobs}
              onChange={(event) =>
                setDraft({
                  ...draft,
                  max_parallel_cloud_jobs: Number(event.target.value) || 3,
                })
              }
            >
              {[1, 2, 3, 4, 5, 6].map((count) => (
                <option key={count} value={count}>
                  {count}
                </option>
              ))}
            </select>
            <p className="field-hint">
              Jobs that run fully on OpenRouter (no local WhisperX) can run in parallel since
              they do not use this Mac. Local jobs always run one at a time.
            </p>
          </div>
        </div>
      </section>

      <section className="panel" aria-label="OpenRouter">
        <h2 className="panel-title">OpenRouter</h2>
        <p className="field-hint">
          Cloud transcription via OpenRouter.{" "}
          {keyStorage === "keychain"
            ? "The key is stored in the macOS Keychain — it never touches the app database, logs or this browser."
            : "The key is stored in a user-only file inside the app data folder — it never touches the app database, logs or this browser."}{" "}
          Long lectures are split into silence-aligned chunks because providers time out after
          ~60 s per request.
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
                    onToast(
                      keyStorage === "keychain"
                        ? "OpenRouter key stored in macOS Keychain."
                        : "OpenRouter key stored in the app data folder.",
                    );
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
              ? keyStorage === "keychain"
                ? "A key is stored in the login Keychain (service PolimiLectureTranscriber)."
                : "A key is stored in the app data folder (user-only permissions)."
              : "No key stored yet."}
          </p>
        </div>
      </section>

      {localSupported && (
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
      )}

      {!localSupported && (
        <section className="panel" aria-label="Platform">
          <h2 className="panel-title">Platform: {platformLabel}</h2>
          <p className="field-hint">
            This platform runs the OpenRouter cloud path only. Local MLX Whisper transcription,
            WhisperX alignment and local model downloads are macOS-only. Cloud transcription and
            Cloud alignment are fully supported here.
          </p>
        </section>
      )}

      <section className="panel" aria-label="Global context">
        <h2 className="panel-title">Global context</h2>
        <div className="field">
          <textarea
            rows={5}
            value={draft.glossary}
            onChange={(event) => setDraft({ ...draft, glossary: event.target.value })}
            placeholder={"Persistent names, terminology, language preferences\none term per line or comma-separated"}
          />
          <p className="field-hint">
            Applies to <strong>every</strong> transcription, on top of any course/project and
            per-file context. Saved with your settings and merged at transcription time as:
            global + course/project + current file.
          </p>
        </div>
      </section>

      <section className="panel" aria-label="Courses and projects">
        <h2 className="panel-title">Courses / Projects</h2>
        <p className="field-hint">
          Recurring technical vocabulary, professor names and acronyms for a course, project,
          meeting series or podcast. Select one in the transcribe view to apply its context
          automatically. PDF, TXT and Markdown documents can be imported to extract names,
          acronyms and terminology (no AI, just frequency heuristics — review the result).
        </p>
        <ul className="project-list">
          {projects.map((project) => (
            <li key={project.id} className="project-card">
              {editingProjectId === project.id ? (
                <div className="project-edit">
                  <input
                    type="text"
                    value={editName}
                    onChange={(event) => setEditName(event.target.value)}
                  />
                  <textarea
                    rows={4}
                    value={editContext}
                    onChange={(event) => setEditContext(event.target.value)}
                    placeholder="One term per line or comma-separated"
                  />
                  <div className="button-row">
                    <button
                      type="button"
                      className="btn btn-small btn-primary"
                      onClick={() => {
                        api
                          .updateProject(project.id, {
                            name: editName.trim(),
                            context: editContext,
                          })
                          .then(() => {
                            setEditingProjectId(null);
                            onProjectsChanged();
                            onToast("Course/project updated.");
                          })
                          .catch((error: Error) => onToast(error.message, "error"));
                      }}
                    >
                      Save
                    </button>
                    <button
                      type="button"
                      className="btn btn-small"
                      onClick={() => setEditingProjectId(null)}
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              ) : (
                <>
                  <div className="project-info">
                    <span className="project-name">{project.name}</span>
                    <span className="project-meta">
                      {termCount(countTerms(project.context))} of context
                    </span>
                    {project.context && (
                      <span className="project-preview">
                        {project.context.split(/[\n,]/).map((t) => t.trim()).filter(Boolean).slice(0, 8).join(", ")}
                        {countTerms(project.context) > 8 ? " …" : ""}
                      </span>
                    )}
                  </div>
                  <div className="project-actions">
                    <button
                      type="button"
                      className="btn btn-small"
                      onClick={() => {
                        setEditingProjectId(project.id);
                        setEditName(project.name);
                        setEditContext(project.context);
                      }}
                    >
                      Edit
                    </button>
                    <label className="btn btn-small">
                      {importingId === project.id ? "Importing..." : "Import file"}
                      <input
                        type="file"
                        accept=".pdf,.txt,.md,.markdown"
                        style={{ display: "none" }}
                        disabled={importingId === project.id}
                        onChange={(event) => {
                          const file = event.target.files?.[0];
                          event.target.value = "";
                          if (!file) return;
                          setImportingId(project.id);
                          api
                            .importProjectDocument(project.id, file)
                            .then((result) => {
                              onProjectsChanged();
                              onToast(
                                result.added.length
                                  ? `Imported ${result.added.length} new term(s) from ${file.name}.`
                                  : `No new terms found in ${file.name}.`,
                              );
                            })
                            .catch((error: Error) => onToast(error.message, "error"))
                            .finally(() => setImportingId(null));
                        }}
                      />
                    </label>
                    <button
                      type="button"
                      className="btn btn-small btn-danger-ghost"
                      onClick={() => setDeleteTarget(project)}
                    >
                      Delete
                    </button>
                  </div>
                </>
              )}
            </li>
          ))}
        </ul>
        <div className="new-project-form">
          <input
            type="text"
            placeholder="New course/project name"
            value={newProjectName}
            onChange={(event) => setNewProjectName(event.target.value)}
          />
          <textarea
            rows={3}
            placeholder={"Recurring terms (optional)\nNUMA\nTLB\nDaniele Cattaneo"}
            value={newProjectContext}
            onChange={(event) => setNewProjectContext(event.target.value)}
          />
          <button
            type="button"
            className="btn"
            disabled={!newProjectName.trim()}
            onClick={() => {
              api
                .createProject(newProjectName.trim(), newProjectContext)
                .then(() => {
                  setNewProjectName("");
                  setNewProjectContext("");
                  onProjectsChanged();
                  onToast("Course/project created.");
                })
                .catch((error: Error) => onToast(error.message, "error"));
            }}
          >
            Create course/project
          </button>
        </div>
      </section>

      {deleteTarget && (
        <ConfirmDialog
          title="Delete course/project"
          body={`Delete "${deleteTarget.name}"? Jobs that already used it keep their merged context; only the reusable project entry is removed.`}
          confirmLabel="Delete"
          danger
          onCancel={() => setDeleteTarget(null)}
          onConfirm={() => {
            const target = deleteTarget;
            setDeleteTarget(null);
            api
              .deleteProject(target.id)
              .then(() => {
                onProjectsChanged();
                onToast("Course/project deleted.");
              })
              .catch((error: Error) => onToast(error.message, "error"));
          }}
        />
      )}

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
            Reveal the output folder in the file manager when a job finishes
          </label>
        </div>
      </section>

      {localSupported && (
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
      )}

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
      <PrivacyNote cloudOnly={!localSupported} />
    </div>
  );
}
