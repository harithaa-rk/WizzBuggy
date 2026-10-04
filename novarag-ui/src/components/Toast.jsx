import { useState, useCallback, useRef } from "react";

// ── Hook ──────────────────────────────────────────────
export function useToast() {
  const [toasts, setToasts] = useState([]);
  const timers = useRef({});

  const dismiss = useCallback((id) => {
    setToasts((t) => t.map((x) => x.id === id ? { ...x, hiding: true } : x));
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 280);
  }, []);

  const toast = useCallback((msg, type = "info", duration = 3000) => {
    const id = Date.now() + Math.random();
    setToasts((t) => [...t, { id, msg, type, hiding: false }]);
    timers.current[id] = setTimeout(() => dismiss(id), duration);
    return id;
  }, [dismiss]);

  return { toasts, toast, dismiss };
}

// ── Renderer ─────────────────────────────────────────
const ICONS = { success: "✓", error: "✗", info: "◈" };

export function ToastContainer({ toasts, dismiss }) {
  return (
    <div className="toast-container">
      {toasts.map((t) => (
        <div key={t.id} className={`toast ${t.type} ${t.hiding ? "hiding" : ""}`}>
          <span className="toast-icon">{ICONS[t.type]}</span>
          <span className="toast-msg">{t.msg}</span>
          <button className="toast-close" onClick={() => dismiss(t.id)}>✕</button>
        </div>
      ))}
    </div>
  );
}