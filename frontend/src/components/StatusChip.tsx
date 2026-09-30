import { STATUS_LABELS } from "../format";
import type { JobStatus } from "../types";

function group(status: JobStatus): string {
  switch (status) {
    case "completed":
      return "ok";
    case "failed":
      return "error";
    case "cancelled":
    case "interrupted":
      return "warn";
    case "queued":
      return "muted";
    default:
      return "active";
  }
}

export function StatusChip({ status }: { status: JobStatus }) {
  return <span className={`chip chip-${group(status)}`}>{STATUS_LABELS[status]}</span>;
}
