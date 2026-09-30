import { useState } from "react";

interface Props {
  title: string;
  body: string;
  confirmLabel: string;
  danger?: boolean;
  checkboxLabel?: string;
  onConfirm: (checked: boolean) => void;
  onCancel: () => void;
}

export function ConfirmDialog({
  title,
  body,
  confirmLabel,
  danger,
  checkboxLabel,
  onConfirm,
  onCancel,
}: Props) {
  const [checked, setChecked] = useState(false);
  return (
    <div className="overlay" role="dialog" aria-modal="true" aria-label={title}>
      <div className="dialog">
        <h2>{title}</h2>
        <p>{body}</p>
        {checkboxLabel && (
          <label className="dialog-checkbox">
            <input
              type="checkbox"
              checked={checked}
              onChange={(event) => setChecked(event.target.checked)}
            />
            {checkboxLabel}
          </label>
        )}
        <div className="dialog-actions">
          <button type="button" className="btn" onClick={onCancel}>
            Cancel
          </button>
          <button
            type="button"
            className={danger ? "btn btn-danger" : "btn btn-primary"}
            onClick={() => onConfirm(checked)}
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
