import type {
  AlignmentStatus,
  AppSettings,
  Health,
  Job,
  LanguageChoice,
  ModelInfo,
  OpenRouterModel,
  PickFile,
  PreviewData,
  Project,
  UploadResponse,
} from "../types";

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const isFormData =
    typeof FormData !== "undefined" && init?.body instanceof FormData;
  let response: Response;
  try {
    response = await fetch(path, {
      headers: init?.body && !isFormData ? { "Content-Type": "application/json" } : undefined,
      ...init,
    });
  } catch (error) {
    throw new ApiError(0, "The local server is not reachable. Is the app running?");
  }
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const payload = await response.json();
      if (typeof payload.detail === "string") {
        message = payload.detail;
      } else if (Array.isArray(payload.detail)) {
        message = payload.detail
          .map((entry: { msg?: string }) => entry.msg ?? "invalid value")
          .join("; ");
      }
    } catch {
      // keep the generic message
    }
    throw new ApiError(response.status, message);
  }
  return (await response.json()) as T;
}

export interface CreateJobOptions {
  sources: Array<{ path?: string; upload_id?: string }>;
  modelKey: string;
  language: LanguageChoice;
  glossary?: string;
  provider?: string;
  openrouterModel?: string | null;
  alignWithWhisperx?: boolean;
  alignmentMode?: string;
  projectId?: string | null;
}

export const api = {
  health: () => request<Health>("/api/health"),

  getSettings: () =>
    request<{
      settings: AppSettings;
      default_output_dir: string;
      resolved_output_dir: string;
      openrouter_key_present: boolean;
    }>("/api/settings"),

  storeOpenRouterKey: (key: string) =>
    request<{ stored: boolean }>("/api/settings/openrouter_key", {
      method: "PUT",
      body: JSON.stringify({ key }),
    }),

  deleteOpenRouterKey: () =>
    request<{ deleted: boolean }>("/api/settings/openrouter_key", { method: "DELETE" }),

  listOpenRouterModels: () => request<{ models: OpenRouterModel[] }>("/api/models/openrouter"),

  alignmentStatus: () => request<AlignmentStatus>("/api/alignment/status"),

  installAlignment: () => request<{ state: string }>("/api/alignment/install", { method: "POST" }),

  updateSettings: (patch: Partial<AppSettings>) =>
    request<{ settings: AppSettings; resolved_output_dir: string }>("/api/settings", {
      method: "PUT",
      body: JSON.stringify(patch),
    }),

  listJobs: () => request<{ jobs: Job[] }>("/api/jobs"),

  createJobs: (options: CreateJobOptions) =>
    request<{ jobs: Job[] }>("/api/jobs", {
      method: "POST",
      body: JSON.stringify({
        sources: options.sources,
        model_key: options.modelKey,
        language: options.language,
        per_file_context: options.glossary ?? null,
        provider: options.provider ?? "local_mlx",
        openrouter_model: options.openrouterModel ?? null,
        align_with_whisperx: options.alignWithWhisperx ?? false,
        alignment_mode: options.alignmentMode ?? "none",
        project_id: options.projectId ?? null,
      } satisfies Record<string, unknown>),
    }),

  cancelJob: (id: string) => request<Job>(`/api/jobs/${id}/cancel`, { method: "POST" }),
  retryJob: (id: string) => request<Job>(`/api/jobs/${id}/retry`, { method: "POST" }),
  archiveJob: (id: string) =>
    request<{ archived: boolean }>(`/api/jobs/${id}/archive`, { method: "POST" }),
  revealJob: (id: string) =>
    request<{ opened: string }>(`/api/jobs/${id}/reveal`, { method: "POST" }),
  removeJob: (id: string, deleteOutputs = false) =>
    request<{ deleted: boolean }>(
      `/api/jobs/${id}?delete_outputs=${deleteOutputs ? "true" : "false"}`,
      { method: "DELETE" },
    ),
  clearFinished: () =>
    request<{ archived: number }>("/api/jobs/clear-finished", { method: "POST" }),

  regenerate: (jobId: string, overwrite = false) =>
    request<{ srt: string; vtt: string; cues: number; warnings: string[] }>(
      "/api/jobs/regenerate",
      { method: "POST", body: JSON.stringify({ job_id: jobId, overwrite }) },
    ),

  preview: (jobId: string) => request<PreviewData>(`/api/jobs/${jobId}/preview`),

  listModels: () => request<{ models: ModelInfo[] }>("/api/models"),

  listProjects: () => request<{ projects: Project[] }>("/api/projects"),

  createProject: (name: string, context = "") =>
    request<Project>("/api/projects", {
      method: "POST",
      body: JSON.stringify({ name, context }),
    }),

  updateProject: (id: string, patch: { name?: string; context?: string }) =>
    request<Project>(`/api/projects/${id}`, {
      method: "PUT",
      body: JSON.stringify(patch),
    }),

  deleteProject: (id: string) =>
    request<{ deleted: boolean }>(`/api/projects/${id}`, { method: "DELETE" }),

  importProjectDocument: (id: string, file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<{
      project: Project;
      extracted: number;
      added: string[];
      text_chars: number;
    }>(`/api/projects/${id}/import`, { method: "POST", body: form });
  },

  downloadModel: (key: string) =>
    request<{ state: string }>(`/api/models/${key}/download`, { method: "POST" }),

  reveal: (target: string) =>
    request<{ opened: string }>("/api/actions/reveal", {
      method: "POST",
      body: JSON.stringify({ target }),
    }),

  pickFiles: () =>
    request<{ files: PickFile[]; errors: string[]; cancelled: boolean }>("/api/media/pick", {
      method: "POST",
    }),
};

export function uploadMedia(
  file: File,
  onProgress: (fraction: number) => void,
): Promise<UploadResponse> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/media/upload");
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        onProgress(event.loaded / event.total);
      }
    };
    xhr.onerror = () => reject(new ApiError(0, `Upload of ${file.name} failed.`));
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText) as UploadResponse);
        } catch {
          reject(new ApiError(0, "Unexpected server response during upload."));
        }
        return;
      }
      let message = `Upload failed (${xhr.status})`;
      try {
        const payload = JSON.parse(xhr.responseText);
        if (typeof payload.detail === "string") message = payload.detail;
      } catch {
        // keep generic
      }
      reject(new ApiError(xhr.status, message));
    };
    const form = new FormData();
    form.append("file", file);
    xhr.send(form);
  });
}
