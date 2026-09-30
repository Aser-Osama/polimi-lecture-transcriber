import { useState } from "react";
import { formatBytes } from "../format";
import type { LanguageChoice, ModelInfo } from "../types";

interface Props {
  language: LanguageChoice;
  modelKey: string;
  glossary: string;
  models: ModelInfo[];
  disabled: boolean;
  onLanguageChange: (value: LanguageChoice) => void;
  onModelChange: (value: string) => void;
  onGlossaryChange: (value: string) => void;
}

export function OptionsBar({
  language,
  modelKey,
  glossary,
  models,
  disabled,
  onLanguageChange,
  onModelChange,
  onGlossaryChange,
}: Props) {
  const [glossaryOpen, setGlossaryOpen] = useState(glossary.trim().length > 0);
  const selectedModel = models.find((model) => model.key === modelKey);

  return (
    <section className="panel options-panel" aria-label="Transcription options">
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
        <div className="field">
          <label htmlFor="model-select">Model</label>
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
              Terms are cleaned, de-duplicated and limited before being sent to Whisper as
              context.
            </p>
          </>
        )}
      </div>
    </section>
  );
}
