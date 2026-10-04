import { useEffect, useState } from "react";

/**
 * Global keyboard shortcuts for NovaRAG.
 *
 * Ctrl+K   → focus chat input
 * Ctrl+/   → toggle sidebar
 * Ctrl+M   → toggle mic
 * Ctrl+L   → clear / new session
 * ?        → show shortcut cheatsheet (when not typing)
 * Escape   → close cheatsheet / blur input
 */
export function useKeyboardShortcuts({ onFocusInput, onToggleSidebar, onToggleMic, onNewSession }) {
  const [showHint, setShowHint] = useState(false);

  useEffect(() => {
    const handler = (e) => {
      const tag = document.activeElement?.tagName?.toLowerCase();
      const isTyping = tag === "input" || tag === "textarea";

      // ? key — show cheatsheet (only when not typing)
      if (e.key === "?" && !isTyping) {
        setShowHint((v) => !v);
        return;
      }

      // Escape — hide cheatsheet or blur
      if (e.key === "Escape") {
        setShowHint(false);
        document.activeElement?.blur();
        return;
      }

      if (!e.ctrlKey && !e.metaKey) return;

      switch (e.key.toLowerCase()) {
        case "k":
          e.preventDefault();
          setShowHint(false);
          onFocusInput?.();
          break;
        case "/":
          e.preventDefault();
          onToggleSidebar?.();
          break;
        case "m":
          e.preventDefault();
          onToggleMic?.();
          break;
        case "l":
          e.preventDefault();
          onNewSession?.();
          break;
        default:
          break;
      }
    };

    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onFocusInput, onToggleSidebar, onToggleMic, onNewSession]);

  return { showHint, setShowHint };
}