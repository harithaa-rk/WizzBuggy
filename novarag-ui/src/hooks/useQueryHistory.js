import { useState, useCallback } from "react";

const MAX_HISTORY = 50;
const STORAGE_KEY = "novarag_query_history";

function load() {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]"); }
  catch { return []; }
}

function save(items) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(items));
}

export function useQueryHistory() {
  const [history, setHistory] = useState(() => load());
  const [cursor, setCursor]   = useState(-1);  // -1 = not browsing

  const push = useCallback((query) => {
    if (!query.trim()) return;
    setHistory((prev) => {
      // Remove duplicate if already exists, then prepend
      const deduped = prev.filter((q) => q !== query);
      const next    = [query, ...deduped].slice(0, MAX_HISTORY);
      save(next);
      return next;
    });
    setCursor(-1);
  }, []);

  // Returns the query to navigate to, or null if at bounds
  const navigate = useCallback((direction, currentInput) => {
    setHistory((prev) => {
      if (!prev.length) return prev;

      let next;
      if (direction === "up") {
        next = Math.min(cursor + 1, prev.length - 1);
      } else {
        next = Math.max(cursor - 1, -1);
      }
      setCursor(next);
      return prev;
    });
    return null; // actual value read via getCurrent
  }, [cursor]);

  const navigateUp = useCallback((setInput) => {
    setHistory((prev) => {
      if (!prev.length) return prev;
      const next = Math.min(cursor + 1, prev.length - 1);
      setCursor(next);
      if (prev[next] !== undefined) setInput(prev[next]);
      return prev;
    });
  }, [cursor]);

  const navigateDown = useCallback((setInput, originalInput) => {
    const next = cursor - 1;
    setCursor(next);
    setHistory((prev) => {
      if (next < 0) {
        setInput(originalInput || "");
      } else if (prev[next] !== undefined) {
        setInput(prev[next]);
      }
      return prev;
    });
  }, [cursor]);

  const reset = useCallback(() => setCursor(-1), []);

  const clear = useCallback(() => {
    setHistory([]);
    save([]);
    setCursor(-1);
  }, []);

  return { history, cursor, push, navigateUp, navigateDown, reset, clear };
}