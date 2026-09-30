import { useState } from "react";
import { formatBytes } from "../format";
import type { LanguageChoice, ModelInfo, OpenRouterModel, ProviderName } from "../types";

export type BackendChoice = Extract<ProviderName, "local_mlx" | "openrouter">;

interface Props {
  backend: BackendChoice;
  language: LanguageChoice;
  modelKey: string;
  openrouterModel: string;
  glossary: string;
  models: ModelInfo[];
  openrouterModels: OpenRouterModel[];
  hasOpenRouterKey: boolean;
  alignmentInstalled: boolean | null;
  alignmentDevice: string | null;
  alignmentMessage: string | null;
  alignWithWhisperx: boolean;
  disabled: boolean;
  onBackendChange: (value: BackendChoice) => void;
  onLanguageChange: (value: LanguageChoice) => void;
  onModelChange: (value: string) => void;
  onOpenRouterModelChange: (value: string) => void;
  onGlossaryChange: (value: string) => void;
  onAlignChange: (value: boolean) => void;
  onGoToSettings: () => void;
}

export function OptionsBar({
  backend,
  language,
  modelKey,
  openrouterModel,
  glossary,
  models,
  openrouterModels,
  hasOpenRouterKey,
  alignmentInstalled,
  alignmentDevice,
  alignmentMessage,
  alignWithWhisperx,
  disabled,
  onBackendChange,
  onLanguageChange,
  onModelChange,
  onOpenRouterModelChange,
  onGlossaryChange,
  onAlignChange,
  onGoToSettings,
}: Props) {
  const [glossaryOpen, setGlossaryOpen] = useState(glossary.trim().length > 0);
  const selectedModel = models.find((model) => model.key === modelKey);

  return (
    <section className="panel options-panel" aria-label="Transcription options">
      <div className="backend-row">
        <span className="field-label" id="backend-label">
          Transcription backend
        </span>
        <div className="segmented" role="radiogroup" aria-labelledby="backend-label">
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
          {backend === "local_mlx"
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

      <div className="field checkbox-field alignment-field">
        <label>
          <input
            type="checkbox"
            checked={alignWithWhisperx}
            disabled={disabled || alignmentInstalled === false}
            onChange={(event) => onAlignChange(event.target.checked)}
          />
          Improve subtitle alignment (WhisperX, local, no re-transcription)
        </label>
        <p className="field-hint">
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
        </p>
      </div>

      <div className="field glossary-field">
        <button
          type="button"
          className="glossary-toggle"
          aria-expanded={glossaryOpen}
          onClick={() => setGlossaryOpen((open) => !open)}
        >
          <span>Course vocabulary</span>
          <span className="glossary-summary">
            {glossary.trim()
              ? `${glossary.split(/[\n,]/).filter((term) => term.trim()).length} terms`
              : "optional"}
            {glossaryOpen ? " · hide" : " · show"}
          </span>
        </button>
        {glossaryOpen && (
          <>
            <textarea
              id="glossary-input"
              rows={5}
              value={glossary}
              disabled={disabled}
              placeholder={"One term per line or comma-separated:\nNUMA\nTLB\ncache coherence\nCUDA"}
              onChange={(event) => onGlossaryChange(event.target.value)}
            />
            <p className="field-hint">
              {backend === "openrouter"
                ? "OpenRouter models do not accept prompts, so the vocabulary is ignored by the cloud backend."
                : "Terms are cleaned, de-duplicated and limited before being sent to Whisper as context."}
            </p>
          </>
        )}
      </div>
    </section>
  );
}
