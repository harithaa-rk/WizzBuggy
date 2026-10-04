import { useState, useCallback } from "react";

const STORAGE_KEY = "novarag_sessions";

function load() {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]"); }
  catch { return []; }
}

function save(sessions) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
}

function makeSession(name) {
  return {
    id: `session_${Date.now()}_${Math.random().toString(36).slice(2,7)}`,
    name: name || "New conversation",
    createdAt: Date.now(),
    messages: [],
  };
}

export function useSessions() {
  const [sessions, setSessions] = useState(() => {
    const stored = load();
    return stored.length ? stored : [makeSession("Session 1")];
  });
  const [activeId, setActiveId] = useState(() => {
    const stored = load();
    return stored.length ? stored[0].id : makeSession("Session 1").id;
  });

  const persist = useCallback((next) => { setSessions(next); save(next); }, []);

  const activeSession = sessions.find((s) => s.id === activeId) || sessions[0];

  // Accepts either a messages array OR a functional updater (prev => next)
  // This is critical for streaming — updater must always see latest state
  const saveMessages = useCallback((updaterOrMessages) => {
    setSessions((prev) => {
      const next = prev.map((s) => {
        if (s.id !== activeId) return s;

        // Resolve: functional updater or plain array
        const messages = typeof updaterOrMessages === "function"
          ? updaterOrMessages(s.messages || [])
          : updaterOrMessages;

        const firstName = messages.find((m) => m.role === "user")?.message;
        return {
          ...s,
          messages,
          name: firstName
            ? firstName.slice(0, 42) + (firstName.length > 42 ? "…" : "")
            : s.name,
        };
      });
      save(next);
      return next;
    });
  }, [activeId]);

  const newSession = useCallback(() => {
    setSessions((prev) => {
      const s = makeSession(`Session ${prev.length + 1}`);
      const next = [s, ...prev];
      save(next);
      setActiveId(s.id);
      return next;
    });
  }, []);

  const switchSession = useCallback((id) => setActiveId(id), []);

  const deleteSession = useCallback((id) => {
    setSessions((prev) => {
      const next = prev.filter((s) => s.id !== id);
      if (!next.length) {
        const s = makeSession("Session 1");
        save([s]); setActiveId(s.id); return [s];
      }
      save(next);
      setActiveId((cur) => (cur === id ? next[0].id : cur));
      return next;
    });
  }, []);

  return { sessions, activeId, activeSession, saveMessages, newSession, switchSession, deleteSession };
}