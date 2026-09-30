import { useCallback, useEffect, useRef, useState } from "react";
import { api, uploadMedia } from "./api/client";
import { ConfirmDialog } from "./components/ConfirmDialog";
import { DropZone } from "./components/DropZone";
import { HistoryView } from "./components/HistoryView";
import { OptionsBar } from "./components/OptionsBar";
import { PreviewOverlay } from "./components/PreviewOverlay";
import { QueueList } from "./components/QueueList";
import { SelectedFiles } from "./components/SelectedFiles";
import { SettingsView } from "./components/SettingsView";
import { TopBar } from "./components/TopBar";
import { useJobs } from "./hooks/useJobs";
import type {
  AppSettings,
  Health,
  Job,
  LanguageChoice,
  ModelInfo,
  SelectedItem,
  ViewName,
} from "./types";

interface SettingsBundle {
  settings: AppSettings;
  default_output_dir: string;
  resolved_output_dir: string;
}

interface Toast {
  id: number;
  message: string;
  kind: "info" | "error";
}

interface UploadItem {
  key: string;
  name: string;
  progress: number;
  error?: string;
}

let toastCounter = 0;

export default function App() {
  const [view, setView] = useState<ViewName>("transcribe");
  const [health, setHealth] = useState<Health | null>(null);
  const [settingsBundle, setSettingsBundle] = useState<SettingsBundle | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selected, setSelected] = useState<SelectedItem[]>([]);
  const [uploads, setUploads] = useState<UploadItem[]>([]);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [previewJob, setPreviewJob] = useState<Job | null>(null);
  const [confirmJob, setConfirmJob] = useState<Job | null>(null);
  const [language, setLanguage] = useState<LanguageChoice>("en");
  const [modelKey, setModelKey] = useState("quality");
  const [glossary, setGlossary] = useState("");
  const [picking, setPicking] = useState(false);
  const initialized = useRef(false);

  const { jobs, connection, fractions, modelDownloads } = useJobs();

  const toast = useCallback((message: string, kind: "info" | "error" = "info") => {
    const id = ++toastCounter;
    setToasts((current) => [...current, { id, message, kind }]);
    window.setTimeout(
      () => setToasts((current) => current.filter((entry) => entry.id !== id)),
      kind === "error" ? 10000 : 5000,
    );
  }, []);

  const loadSettings = useCallback(async () => {
    const bundle = await api.getSettings();
    setSettingsBundle(bundle);
    if (!initialized.current) {
      initialized.current = true;
      setLanguage(bundle.settings.default_language);
      setModelKey(bundle.settings.default_model_key);
      setGlossary(bundle.settings.glossary);
    }
  }, []);

  const loadModels = useCallback(async () => {
    const result = await api.listModels();
    setModels(result.models);
  }, []);

  useEffect(() => {
    api.health().then(setHealth).catch(() => undefined);
    loadSettings().catch((error: Error) => toast(error.message, "error"));
    loadModels().catch((error: Error) => toast(error.message, "error"));
  }, [loadSettings, loadModels, toast]);

  useEffect(() => {
    const finished = modelDownloads;
    if (!Object.values(finished).some((state) => state.state === "completed")) return;
    const timer = window.setTimeout(() => {
      void loadModels();
    }, 500);
    return () => window.clearTimeout(timer);
  }, [modelDownloads, loadModels]);

  const addSelected = useCallback((item: SelectedItem) => {
    setSelected((current) => {
      const duplicate = current.some(
        (entry) =>
          (item.path && entry.path === item.path) ||
          (item.upload_id && entry.upload_id === item.upload_id),
      );
      return duplicate ? current : [...current, item];
    });
  }, []);

  const handleDroppedFiles = useCallback(
    async (files: File[]) => {
      for (const file of files) {
        const key = `${file.name}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
        setUploads((current) => [...current, { key, name: file.name, progress: 0 }]);
        try {
          const response = await uploadMedia(file, (fraction) => {
            setUploads((current) =>
              current.map((entry) => (entry.key === key ? { ...entry, progress: fraction } : entry)),
            );
          });
          addSelected({
            key: response.upload_id,
            upload_id: response.upload_id,
            filename: response.filename,
            size_bytes: response.size_bytes,
            media: response.media,
          });
        } catch (error) {
          toast(`${file.name}: ${(error as Error).message}`, "error");
        } finally {
          setUploads((current) => current.filter((entry) => entry.key !== key));
        }
      }
    },
    [addSelected, toast],
  );

  const handlePickFiles = useCallback(async () => {
    setPicking(true);
    try {
      const result = await api.pickFiles();
      if (result.cancelled) return;
      for (const file of result.files) {
        addSelected({
          key: file.path,
          path: file.path,
          filename: file.filename,
          size_bytes: file.size_bytes,
          media: file.media,
        });
      }
      for (const error of result.errors) {
        toast(error, "error");
      }
    } catch (error) {
      toast((error as Error).message, "error");
    } finally {
      setPicking(false);
    }
  }, [addSelected, toast]);

  const startTranscription = useCallback(async () => {
    if (!selected.length) return;
    try {
      const result = await api.createJobs({
        sources: selected.map((item) =>
          item.upload_id ? { upload_id: item.upload_id } : { path: item.path },
        ),
        modelKey,
        language,
        glossary,
      });
      setSelected([]);
      toast(`Queued ${result.jobs.length} job${result.jobs.length === 1 ? "" : "s"}.`);
    } catch (error) {
      toast((error as Error).message, "error");
    }
  }, [selected, modelKey, language, glossary, toast]);

  const wrap = useCallback(
    <T,>(promise: Promise<T>, onSuccess?: (value: T) => void) => {
      promise
        .then((value) => onSuccess?.(value))
        .catch((error: Error) => toast(error.message, "error"));
    },
    [toast],
  );

  if (view === "settings" && settingsBundle) {
    return (
      <AppShell health={health} view={view} setView={setView} connection={connection}>
        <SettingsView
          settings={settingsBundle.settings}
          defaultOutputDir={settingsBundle.default_output_dir}
          resolvedOutputDir={settingsBundle.resolved_output_dir}
          models={models}
          modelDownloads={modelDownloads}
          onSaved={() => {
            void loadSettings();
          }}
          onModelsChanged={() => void loadModels()}
          onToast={toast}
        />
        <Toasts toasts={toasts} />
      </AppShell>
    );
  }

  return (
    <AppShell health={health} view={view} setView={setView} connection={connection}>
      {view === "transcribe" && (
        <>
          <DropZone onFiles={(files) => void handleDroppedFiles(files)} onPickFiles={() => void handlePickFiles()} busy={picking} />
          {uploads.length > 0 && (
            <section className="panel uploads-panel" aria-label="Uploading files">
              <h2 className="panel-title">Copying files to temporary storage</h2>
              {uploads.map((upload) => (
                <div key={upload.key} className="upload-row">
                  <span className="upload-name">{upload.name}</span>
                  <div className="progress-track">
                    <div className="progress-fill" style={{ width: `${Math.round(upload.progress * 100)}%` }} />
                  </div>
                  <span className="upload-pct">{Math.round(upload.progress * 100)}%</span>
                </div>
              ))}
            </section>
          )}
          <SelectedFiles items={selected} onRemove={(key) => setSelected((current) => current.filter((item) => item.key !== key))} />
          <OptionsBar
            language={language}
            modelKey={modelKey}
            glossary={glossary}
            models={models}
            disabled={false}
            onLanguageChange={setLanguage}
            onModelChange={setModelKey}
            onGlossaryChange={setGlossary}
          />
          <div className="start-row">
            <button
              type="button"
              className="btn btn-primary btn-large"
              disabled={selected.length === 0}
              onClick={() => void startTranscription()}
            >
              Start Transcription
            </button>
            <span className="field-hint">
              {selected.length
                ? `${selected.length} file${selected.length === 1 ? "" : "s"} ready`
                : "Add files to begin"}
            </span>
          </div>
          <QueueList
            jobs={jobs}
            fractions={fractions}
            models={models}
            connection={connection}
            onCancel={(id) => wrap(api.cancelJob(id))}
            onRetry={(id) => {
              wrap(api.retryJob(id), () => toast("Retry queued."));
            }}
            onRemove={(job) => wrap(api.removeJob(job.id), () => toast("Removed from queue."))}
            onArchive={(job) => wrap(api.archiveJob(job.id), () => undefined)}
            onPreview={setPreviewJob}
            onReveal={(id) => wrap(api.revealJob(id))}
            onRegenerate={(id) =>
              wrap(api.regenerate(id), (result) =>
                toast(`Regenerated ${result.cues} cues into ${result.srt.split("/").pop()}.`),
              )
            }
            onDeleteHistory={setConfirmJob}
            onClearFinished={() => wrap(api.clearFinished(), () => undefined)}
          />
        </>
      )}
      {view === "history" && (
        <HistoryView
          jobs={jobs}
          models={models}
          onPreview={setPreviewJob}
          onRetry={(id) => wrap(api.retryJob(id), () => toast("Retry queued."))}
          onReveal={(id) => wrap(api.revealJob(id))}
          onDelete={setConfirmJob}
        />
      )}
      {previewJob && (
        <PreviewOverlay job={previewJob} onClose={() => setPreviewJob(null)} onError={(message) => toast(message, "error")} />
      )}
      {confirmJob && (
        <ConfirmDialog
          title="Delete history entry"
          body={`Remove "${confirmJob.source_filename}" from history? Generated files are kept unless you check the box below.`}
          confirmLabel="Delete entry"
          danger
          checkboxLabel="Also delete generated txt/srt/vtt/json files"
          onCancel={() => setConfirmJob(null)}
          onConfirm={(deleteOutputs) => {
            const job = confirmJob;
            setConfirmJob(null);
            wrap(api.removeJob(job.id, deleteOutputs), () =>
              toast(deleteOutputs ? "Entry and files deleted." : "History entry deleted."),
            );
          }}
        />
      )}
      <Toasts toasts={toasts} />
    </AppShell>
  );
}

function AppShell({
  children,
  health,
  view,
  setView,
  connection,
}: {
  children: React.ReactNode;
  health: Health | null;
  view: ViewName;
  setView: (view: ViewName) => void;
  connection: import("./types").ConnectionState;
}) {
  return (
    <div className="app">
      <TopBar view={view} onNavigate={setView} connection={connection} />
      {health && !health.ffmpeg.ok && (
        <div className="banner banner-error" role="alert">
          <strong>FFmpeg is missing.</strong> Install it with <code>brew install ffmpeg</code> and
          restart the app. ({health.ffmpeg.info})
        </div>
      )}
      {health && !health.mlx_provider.ok && (
        <div className="banner banner-error" role="alert">
          <strong>MLX Whisper is unavailable.</strong> Run <code>./setup.sh</code> to repair the
          Python environment. ({health.mlx_provider.error})
        </div>
      )}
      <main className="app-main">{children}</main>
    </div>
  );
}

function Toasts({ toasts }: { toasts: Toast[] }) {
  return (
    <div className="toasts" role="status" aria-live="polite">
      {toasts.map((toast) => (
        <div key={toast.id} className={toast.kind === "error" ? "toast toast-error" : "toast"}>
          {toast.message}
        </div>
      ))}
    </div>
  );
}
