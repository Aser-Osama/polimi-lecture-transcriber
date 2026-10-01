import { useState } from "react";
import { formatBytes, termCount } from "../format";
import type {
  AlignmentMode,
  LanguageChoice,
  ModelInfo,
  OpenRouterModel,
  Project,
  ProviderName,
} from "../types";

export type BackendChoice = Extract<ProviderName, "local_mlx" | "openrouter">;

function countTerms(text: string): number {
  return text
    .split(/[\n,;]/)
    .map((term) => term.trim())
    .filter(Boolean).length;
}

interface Props {
  backend: BackendChoice;
  localSupported: boolean;
  language: LanguageChoice;
  modelKey: string;
  openrouterModel: string;
  glossary: string;
  globalContext: string;
  projects: Project[];
  selectedProjectId: string | null;
  models: ModelInfo[];
  openrouterModels: OpenRouterModel[];
  hasOpenRouterKey: boolean;
  alignmentInstalled: boolean | null;
  alignmentDevice: string | null;
  alignmentMessage: string | null;
  alignmentMode: AlignmentMode;
  disabled: boolean;
  onBackendChange: (value: BackendChoice) => void;
  onLanguageChange: (value: LanguageChoice) => void;
  onModelChange: (value: string) => void;
  onOpenRouterModelChange: (value: string) => void;
  onGlossaryChange: (value: string) => void;
  onAlignmentModeChange: (value: AlignmentMode) => void;
  onGoToSettings: () => void;
  onProjectChange: (projectId: string | null) => void;
  onCreateProject: (name: string, context: string) => Promise<void>;
}

export function OptionsBar({
  backend,
  localSupported,
  language,
  modelKey,
  openrouterModel,
  glossary,
  globalContext,
  projects,
  selectedProjectId,
  models,
  openrouterModels,
  hasOpenRouterKey,
  alignmentInstalled,
  alignmentDevice,
  alignmentMessage,
  alignmentMode,
  disabled,
  onBackendChange,
  onLanguageChange,
  onModelChange,
  onOpenRouterModelChange,
  onGlossaryChange,
  onAlignmentModeChange,
  onGoToSettings,
  onProjectChange,
  onCreateProject,
}: Props) {
  const [glossaryOpen, setGlossaryOpen] = useState(glossary.trim().length > 0);
  const [creatingProject, setCreatingProject] = useState(false);
  const [newProjectName, setNewProjectName] = useState("");
  const [newProjectContext, setNewProjectContext] = useState("");
  const [creatingBusy, setCreatingBusy] = useState(false);
  const selectedModel = models.find((model) => model.key === modelKey);
  const selectedProject = projects.find((project) => project.id === selectedProjectId) ?? null;
  const globalTerms = countTerms(globalContext);
  const projectTerms = selectedProject ? countTerms(selectedProject.context) : 0;

  return (
    <section className="panel options-panel" aria-label="Transcription options">
      <div className="backend-row">
        <span className="field-label" id="backend-label">
          Transcription backend
        </span>
        <div className="segmented" role="radiogroup" aria-labelledby="backend-label">
          {localSupported && (
            <button
              type="button"
              role="radio"
              aria-checked={backend === "local_mlx"}
              className={backend === "local_mlx" ? "segment active" : "segment"}
              disabled={disabled}
              onClick={() => onBackendChange("local_mlx")}
            >
              Local
            </button>
          )}
          <button
            type="button"
            role="radio"
            aria-checked={backend === "openrouter"}
            className={backend === "openrouter" ? "segment active" : "segment"}
            disabled={disabled}
            onClick={() => onBackendChange("openrouter")}
          >
            OpenRouter
          </button>
        </div>
        <span className="field-hint backend-note">
          {!localSupported
            ? "This platform runs the cloud path only: audio is sent to OpenRouter for transcription (billed per second). Local MLX Whisper is available on macOS."
            : backend === "local_mlx"
              ? "Runs MLX Whisper on this Mac. Nothing is uploaded."
              : "Audio is sent to OpenRouter for transcription (billed per second)."}
        </span>
      </div>

      {backend === "openrouter" && !hasOpenRouterKey && (
        <div className="banner banner-inline" role="alert">
          <strong>No OpenRouter API key.</strong> Add one in Settings to use the cloud models.{" "}
          <button type="button" className="link-btn" onClick={onGoToSettings}>
            Open Settings
          </button>
        </div>
      )}

      <div className="options-grid">
        <div className="field">
          <label htmlFor="language-select">Language</label>
          <select
            id="language-select"
            value={language}
            disabled={disabled}
            onChange={(event) => onLanguageChange(event.target.value as LanguageChoice)}
          >
            <option value="en">English</option>
            <option value="it">Italian</option>
            <option value="auto">Auto Detect</option>
          </select>
          <p className="field-hint">
            Select the spoken language explicitly to prevent wrong auto-detection on accented
            English.
          </p>
        </div>
        {backend === "local_mlx" ? (
          <div className="field">
            <label htmlFor="model-select">Local model</label>
            <select
              id="model-select"
              value={modelKey}
              disabled={disabled}
              onChange={(event) => onModelChange(event.target.value)}
            >
              {models.map((model) => (
                <option key={model.key} value={model.key}>
                  {model.display_name}
                  {model.cached ? " (cached)" : ` (~${formatBytes(model.approx_size_bytes)})`}
                </option>
              ))}
            </select>
            <p className="field-hint">{selectedModel?.description ?? ""}</p>
          </div>
        ) : (
          <div className="field">
            <label htmlFor="openrouter-model-select">OpenRouter model</label>
            <select
              id="openrouter-model-select"
              value={openrouterModel}
              disabled={disabled}
              onChange={(event) => onOpenRouterModelChange(event.target.value)}
            >
              {openrouterModels.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.display_name}
                </option>
              ))}
            </select>
            <p className="field-hint">
              {openrouterModels.find((model) => model.id === openrouterModel)?.description ?? ""}
            </p>
          </div>
        )}
      </div>

      <div className="field alignment-field">
        <label htmlFor="alignment-mode">Subtitle alignment</label>
        <select
          id="alignment-mode"
          value={alignmentMode}
          disabled={disabled}
          onChange={(event) => onAlignmentModeChange(event.target.value as AlignmentMode)}
        >
          <option value="none">
            {backend === "openrouter"
              ? "None - use the model's own timestamps"
              : "None - native Whisper timestamps"}
          </option>
          {backend === "openrouter" && (
            <option value="cloud">Cloud - native timestamps + MAI anchors when needed</option>
          )}
          {localSupported && (
            <option value="local_whisperx" disabled={alignmentInstalled === false}>
              Local WhisperX{alignmentInstalled === false ? " (not installed)" : " (runs on this Mac)"}
            </option>
          )}
        </select>
        <p className="field-hint">
          {alignmentMode === "cloud" &&
            "Word timestamps from the chosen model are used directly when it provides them; " +
              "models without timestamps (e.g. Qwen3 ASR Flash) get timing anchors from " +
              "MAI-Transcribe 2 in the cloud (~$0.10/hour of audio). Nothing runs on this Mac."}
          {alignmentMode === "local_whisperx" && (
            <>
              {alignmentInstalled === null && "Checking WhisperX availability..."}
              {alignmentInstalled === true &&
                `Forced alignment re-times this model's transcript against the audio on this Mac${
                  alignmentDevice ? ` (${alignmentDevice.toUpperCase()})` : ""
                }.`}
              {alignmentInstalled === false && (
                <>
                  {alignmentMessage ?? "WhisperX is not installed."}{" "}
                  <button type="button" className="link-btn" onClick={onGoToSettings}>
                    Install in Settings
                  </button>
                </>
              )}
            </>
          )}
          {alignmentMode === "none" &&
            (backend === "openrouter"
              ? localSupported
                ? "Models without their own word timestamps produce no subtitles unless Cloud or Local WhisperX alignment is selected."
                : "Models without their own word timestamps produce no subtitles unless Cloud alignment is selected."
              : "Whisper's own word-level timestamps are used. WhisperX can refine them if needed.")}
        </p>
      </div>

      <div className="field project-field">
        <label htmlFor="project-select">Course / Project (optional)</label>
        <div className="inline-field">
          <select
            id="project-select"
            value={selectedProjectId ?? ""}
            disabled={disabled}
            onChange={(event) => onProjectChange(event.target.value || null)}
          >
            <option value="">No course/project</option>
            {projects.map((project) => (
              <option key={project.id} value={project.id}>
                {project.name}
              </option>
            ))}
          </select>
          <button
            type="button"
            className="btn"
            disabled={disabled}
            onClick={() => setCreatingProject((open) => !open)}
          >
            New...
          </button>
        </div>
        {creatingProject && (
          <div className="new-project-form">
            <input
              type="text"
              placeholder="Course/project name"
              value={newProjectName}
              onChange={(event) => setNewProjectName(event.target.value)}
            />
            <textarea
              rows={3}
              placeholder={"Recurring terms, acronyms, professor names (optional)\nNUMA\nTLB\nDaniele Cattaneo"}
              value={newProjectContext}
              onChange={(event) => setNewProjectContext(event.target.value)}
            />
            <div className="button-row">
              <button
                type="button"
                className="btn btn-primary"
                disabled={creatingBusy || !newProjectName.trim()}
                onClick={() => {
                  setCreatingBusy(true);
                  onCreateProject(newProjectName.trim(), newProjectContext)
                    .then(() => {
                      setCreatingProject(false);
                      setNewProjectName("");
                      setNewProjectContext("");
                    })
                    .finally(() => setCreatingBusy(false));
                }}
              >
                {creatingBusy ? "Creating..." : "Create"}
              </button>
              <button type="button" className="btn" onClick={() => setCreatingProject(false)}>
                Cancel
              </button>
            </div>
          </div>
        )}
        <p className="field-hint">
          {selectedProject
            ? `"${selectedProject.name}" adds ${termCount(projectTerms)} of context to every transcription you start with it selected.`
            : "Select a course/project to apply its recurring context automatically. Manage them in Settings."}
        </p>
      </div>

      <div className="field glossary-field">
        <button
          type="button"
          className="glossary-toggle"
          aria-expanded={glossaryOpen}
          onClick={() => setGlossaryOpen((open) => !open)}
        >
          <span>Context for this transcription (optional)</span>
          <span className="glossary-summary">
            {glossary.trim() ? termCount(countTerms(glossary)) : "optional"}
            {glossaryOpen ? " · hide" : " · show"}
          </span>
        </button>
        {glossaryOpen && (
          <>
            <textarea
              id="glossary-input"
              rows={4}
              value={glossary}
              disabled={disabled}
              placeholder={"One-off topics, guest speakers, unusual terminology\none term per line or comma-separated"}
              onChange={(event) => onGlossaryChange(event.target.value)}
            />
            <p className="field-hint">
              Merged at transcription time as global ({termCount(globalTerms)} from Settings) + course
              {selectedProject ? ` (${projectTerms})` : ""} + this file. More specific terms
              override broader ones.
              {backend === "openrouter"
                ? " OpenRouter models that support keyword biasing receive the merged terms; others transcribe without them."
                : ""}
            </p>
          </>
        )}
      </div>
    </section>
  );
}
