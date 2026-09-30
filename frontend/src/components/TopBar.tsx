import type { ConnectionState, ViewName } from "../types";

const TABS: Array<{ key: ViewName; label: string }> = [
  { key: "transcribe", label: "Transcribe" },
  { key: "history", label: "History" },
  { key: "settings", label: "Settings" },
];

interface Props {
  view: ViewName;
  onNavigate: (view: ViewName) => void;
  connection: ConnectionState;
}

export function TopBar({ view, onNavigate, connection }: Props) {
  return (
    <header className="topbar">
      <div className="topbar-brand">
        <span className="brand-dot" aria-hidden="true" />
        <h1>Polimi Lecture Transcriber</h1>
      </div>
      <nav className="topbar-nav" aria-label="Main navigation">
        {TABS.map((tab) => (
          <button
            key={tab.key}
            type="button"
            className={view === tab.key ? "nav-tab active" : "nav-tab"}
            aria-current={view === tab.key ? "page" : undefined}
            onClick={() => onNavigate(tab.key)}
          >
            {tab.label}
          </button>
        ))}
      </nav>
      <div className="topbar-status" role="status">
        <span className={`connection-dot ${connection}`} aria-hidden="true" />
        <span className="connection-label">
          {connection === "live"
            ? "Local Mac - live"
            : connection === "connecting"
              ? "Connecting..."
              : "Reconnecting..."}
        </span>
      </div>
    </header>
  );
}
