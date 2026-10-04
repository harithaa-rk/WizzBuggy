import { useState, useEffect, useRef, useCallback } from "react";
import Sidebar from "./components/Sidebar";
import IngestPanel from "./components/IngestPanel";
import ChatPanel from "./components/ChatPanel";
import SearchPanel from "./components/SearchPanel";
import ComparePanel from "./components/ComparePanel";
import KeyboardShortcuts from "./components/KeyboardShortcuts";
import { ToastContainer, useToast } from "./components/Toast";
import { useSessions } from "./hooks/useSessions";
import { useKeyboardShortcuts } from "./hooks/useKeyboardShortcuts";
import "./index.css";

const API_URL = "http://127.0.0.1:8000";

export default function App() {
  const [indexedFiles, setIndexedFiles] = useState([]);
  const [sidebarOpen, setSidebarOpen]   = useState(true);
  const [dark, setDark]                 = useState(true);
  const [showKbd, setShowKbd]           = useState(false);
  const [fileFilter, setFileFilter]     = useState("");
  const [mode, setMode]                 = useState("chat"); // chat | search | compare

  const chatInputRef = useRef(null);
  const micToggleRef = useRef(null);

  const { toasts, toast, dismiss } = useToast();
  const {
    sessions, activeId, activeSession,
    saveMessages, newSession, switchSession, deleteSession,
  } = useSessions();

  const history    = activeSession?.messages || [];
  const setHistory = useCallback((updater) => saveMessages(updater), [saveMessages]);

  useEffect(() => {
    document.documentElement.classList.toggle("light", !dark);
  }, [dark]);

  const fetchFiles = async () => {
    try {
      const r    = await fetch(`${API_URL}/files`);
      const data = await r.json();
      setIndexedFiles(data);
    } catch { setIndexedFiles([]); }
  };
  useEffect(() => { fetchFiles(); }, []);

  const clearConversation = async () => {
    try {
      await fetch(`${API_URL}/reset`, { method: "POST" });
      setHistory([]);
      toast("Conversation cleared", "info");
    } catch { toast("Could not reset memory", "error"); }
  };

  const handleNewSession = () => {
    newSession();
    fetch(`${API_URL}/reset`, { method: "POST" }).catch(() => {});
    toast("New session started", "info");
  };

  const handleDeleteFile = async (filename) => {
    if (!window.confirm('Delete "' + filename + '" from the index?')) return;
    try {
      const res = await fetch(`${API_URL}/file/${encodeURIComponent(filename)}`, { method: "DELETE" });
      if (!res.ok) throw new Error("HTTP " + res.status);
      const data = await res.json();
      toast("🗑 " + filename + " removed (" + (data.chunks_removed || 0) + " chunks)", "info");
      fetchFiles();
    } catch (e) {
      toast("Delete failed: " + e.message, "error");
    }
  };

  useKeyboardShortcuts([
    { key: "k",      ctrl: true, action: () => chatInputRef.current?.focus(), allowInInput: false },
    { key: "/",      ctrl: true, action: () => setSidebarOpen((v) => !v) },
    { key: "m",      ctrl: true, action: () => micToggleRef.current?.click() },
    { key: "n",      ctrl: true, action: handleNewSession },
    { key: "l",      ctrl: true, action: clearConversation },
    { key: "f",      ctrl: true, action: () => setMode((m) => m === "search" ? "chat" : "search") },
    { key: "Escape", action: () => { if (showKbd) setShowKbd(false); } },
    { key: "?",      action: () => setShowKbd((v) => !v) },
  ]);

  const TABS = [
    { id: "chat",    label: "💬 Chat"    },
    { id: "search",  label: "⌕ Search"  },
    { id: "compare", label: "⚡ Compare" },
  ];

  return (
    <div className="app-shell">
      <header className="top-bar">
        <button className="sidebar-toggle" onClick={() => setSidebarOpen((v) => !v)}>
          <span /><span /><span />
        </button>
        <div className="logo">
          <span className="logo-nova">Nova</span>
          <span className="logo-rag">RAG</span>
          <span className="logo-tag">Offline Multimodal AI</span>
        </div>
        <div className="top-bar-right">
          <button className="kbd-hint-btn" onClick={() => setShowKbd(true)} title="Shortcuts">
            <span className="kbd" style={{ fontSize: 9, padding: "1px 5px" }}>?</span>
            Shortcuts
          </button>
          <div className="status-dot" />
          <span className="status-text">Local</span>
          <div className="theme-toggle" title={dark ? "Light mode" : "Dark mode"}>
            <span className="toggle-icon">☀️</span>
            <div className={"toggle-track" + (dark ? " dark-on" : "")} onClick={() => setDark((v) => !v)}>
              <div className="toggle-thumb" />
            </div>
            <span className="toggle-icon">🌙</span>
          </div>
        </div>
      </header>

      <div className="main-layout">
        <Sidebar
          open={sidebarOpen}
          files={indexedFiles}
          onClear={clearConversation}
          onDeleteFile={handleDeleteFile}
          sessions={sessions}
          activeId={activeId}
          onNewSession={handleNewSession}
          onSwitch={switchSession}
          onDelete={deleteSession}
        />
        <div className="content-area">
          <IngestPanel
            apiUrl={API_URL}
            onIngested={fetchFiles}
            toast={toast}
            activeFiles={indexedFiles}
            onFilterChange={setFileFilter}
          />

          {/* Mode tabs */}
          <div className="mode-tabs">
            {TABS.map((t) => (
              <button
                key={t.id}
                className={"mode-tab" + (mode === t.id ? " active" : "")}
                onClick={() => setMode(t.id)}
              >
                {t.label}
              </button>
            ))}
          </div>

          {mode === "chat" && (
            <ChatPanel
              apiUrl={API_URL}
              history={history}
              setHistory={setHistory}
              toast={toast}
              sessionId={activeId}
              sessionName={activeSession?.name}
              chatInputRef={chatInputRef}
              micToggleRef={micToggleRef}
            />
          )}
          {mode === "search" && (
            <SearchPanel apiUrl={API_URL} files={indexedFiles} />
          )}
          {mode === "compare" && (
            <ComparePanel
              apiUrl={API_URL}
              sessionId={activeId}
              toast={toast}
            />
          )}
        </div>
      </div>

      {showKbd && <KeyboardShortcuts onClose={() => setShowKbd(false)} />}
      <ToastContainer toasts={toasts} dismiss={dismiss} />
    </div>
  );
}