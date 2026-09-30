import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import type { ConnectionState, Job, JobStatus, ModelDownloadState } from "../types";

export interface StageFraction {
  status: JobStatus;
  fraction: number | null;
}

interface StreamEvent {
  type: string;
  jobs?: Job[];
  job?: Job;
  job_id?: string;
  key?: string;
  fraction?: number;
  state?: string;
  message?: string;
  error?: string;
}

export interface AlignmentInstallEvent {
  state: string;
  message: string | null;
  fraction?: number | null;
}

export function useJobs() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [connection, setConnection] = useState<ConnectionState>("connecting");
  const [fractions, setFractions] = useState<Record<string, StageFraction>>({});
  const [modelDownloads, setModelDownloads] = useState<Record<string, ModelDownloadState>>({});
  const [alignmentInstall, setAlignmentInstall] = useState<AlignmentInstallEvent | null>(null);
  const [lastEvent, setLastEvent] = useState<StreamEvent | null>(null);
  const reconnectCount = useRef(0);

  const refresh = useCallback(async () => {
    const data = await api.listJobs();
    setJobs(data.jobs);
  }, []);

  useEffect(() => {
    let source: EventSource | null = new EventSource("/api/events");

    source.onopen = () => {
      setConnection("live");
      reconnectCount.current = 0;
    };
    source.onerror = () => {
      setConnection("offline");
    };
    source.onmessage = (message) => {
      let event: StreamEvent;
      try {
        event = JSON.parse(message.data) as StreamEvent;
      } catch {
        return;
      }
      setLastEvent(event);
      switch (event.type) {
        case "snapshot":
          if (event.jobs) setJobs(event.jobs);
          break;
        case "job":
          if (event.job) {
            const incoming = event.job;
            setJobs((current) => {
              const index = current.findIndex((job) => job.id === incoming.id);
              if (index === -1) return [incoming, ...current];
              const next = [...current];
              next[index] = incoming;
              return next;
            });
            // Fractions are only valid for the status they were reported for;
            // a stage change without a fraction resets the bar.
            setFractions((current) => ({
              ...current,
              [incoming.id]: {
                status: incoming.status,
                fraction: event.fraction ?? null,
              },
            }));
          }
          break;
        case "job_removed":
          if (event.job_id) {
            setJobs((current) => current.filter((job) => job.id !== event.job_id));
          }
          break;
        case "model_download":
          if (event.key) {
            setModelDownloads((current) => ({
              ...current,
              [event.key!]: {
                state: (event.state ?? "idle") as ModelDownloadState["state"],
                fraction: event.fraction ?? null,
                message: event.message ?? null,
                error: event.error ?? null,
              },
            }));
          }
          break;
        case "alignment_install":
          setAlignmentInstall({
            state: event.state ?? "idle",
            message: event.message ?? null,
            fraction: event.fraction ?? null,
          });
          break;
        case "refresh":
          void refresh();
          break;
        default:
          break;
      }
    };

    return () => {
      source?.close();
      source = null;
    };
  }, [refresh]);

  return {
    jobs,
    connection,
    fractions,
    modelDownloads,
    alignmentInstall,
    lastEvent,
    refresh,
  };
}
