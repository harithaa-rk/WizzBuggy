import { useState, useEffect, useCallback, useRef } from "react";
import Sidebar from "./components/Sidebar";
import IngestPanel from "./components/IngestPanel";
import ChatPanel from "./components/ChatPanel";
import SearchPanel from "./components/SearchPanel";
import ComparePanel from "./components/ComparePanel";
import KeyboardShortcuts from "./components/KeyboardShortcuts";
import { ToastContainer, useToast } from "./components/Toast";
import { useSessions } from "./hooks/useSessions";
import { useKeyboardShortcuts } from "./hooks/useKeyboardShortcuts";
import AuthPage from "./components/AuthPage";
import AdminDashboard from "./components/AdminDashboard";
import "./index.css";

const API_URL = "http://127.0.0.1:8000";
const STORAGE_KEY = "novarag_auth";

const CLEARANCE_COLORS = {
  PUBLIC:       "#30d0b0",
  INTERNAL:     "#5b6af0",
  CONFIDENTIAL: "#f0a030",
  RESTRICTED:   "#e05080",
};

const ROLE_ICONS = {
  admin:    "🛡️",
  manager:  "🏢",
  employee: "👤",
  auditor:  "🔍",
};


export default function App() {
  // ── Auth state ──
  const [authUser, setAuthUser]       = useState(null);   // { username, role, clearance_level, token, ... }
  const [authChecked, setAuthChecked] = useState(false);
  const [showAdmin, setShowAdmin]     = useState(false);

  // ── App state ──
  const [indexedFiles, setIndexedFiles] = useState([]);
  const [sidebarOpen, setSidebarOpen]   = useState(true);
  const [dark, setDark]                 = useState(true);
  const [showKbd, setShowKbd]           = useState(false);
  const [fileFilter, setFileFilter]     = useState("");
  const [mode, setMode]                 = useState("chat");

  const chatInputRef  = useRef(null);
  const micToggleRef  = useRef(null);

  const { toasts, toast, dismiss } = useToast();
  const {
    sessions, activeId, activeSession,
    saveMessages, newSession, switchSession, deleteSession,
  } = useSessions();

  const history    = activeSession?.messages || [];
  const setHistory = useCallback((updater) => saveMessages(updater), [saveMessages]);

  // ── Restore session from localStorage ──
  useEffect(() => {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) {
      try {
        const u = JSON.parse(saved);
        // Validate token is still alive
        fetch(`${API_URL}/auth/me`, { headers: { Authorization: u.token } })
          .then(r => r.ok ? r.json() : null)
          .then(data => {
            if (data?.user) setAuthUser({ ...data.user, token: u.token });
            else localStorage.removeItem(STORAGE_KEY);
          })
          .catch(() => localStorage.removeItem(STORAGE_KEY))
          .finally(() => setAuthChecked(true));
      } catch {
        localStorage.removeItem(STORAGE_KEY);
        setAuthChecked(true);
      }
    } else {
      setAuthChecked(true);
    }
  }, []);

  const handleAuth = (userData) => {
    setAuthUser(userData);
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ token: userData.token }));
  };

  const handleLogout = async () => {
    if (authUser?.token) {
      await fetch(`${API_URL}/auth/logout`, {
        method: "POST",
        headers: { Authorization: authUser.token },
      }).catch(() => {});
    }
    localStorage.removeItem(STORAGE_KEY);
    setAuthUser(null);
    setIndexedFiles([]);
    setShowAdmin(false);
    toast("Signed out successfully", "info");
  };

  useEffect(() => {
    document.documentElement.classList.toggle("light", !dark);
  }, [dark]);

  const authHeader = authUser?.token ? { Authorization: authUser.token } : {};

  const fetchFiles = async () => {
    try {
      const r    = await fetch(`${API_URL}/files`, { headers: authHeader });
      const data = await r.json();
      setIndexedFiles(Array.isArray(data) ? data : []);
    } catch { setIndexedFiles([]); }
  };
  useEffect(() => { if (authUser) fetchFiles(); }, [authUser]);

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
      const res = await fetch(`${API_URL}/file/${encodeURIComponent(filename)}`, {
        method: "DELETE",
        headers: authHeader,
      });
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

  // ── Loading state while checking stored session ──
  if (!authChecked) {
    return (
      <div className="auth-page" style={{ display: "flex", alignItems: "center", justifyContent: "center" }}>
        <div className="auth-logo">
          <span className="auth-logo-nova">Nova</span>
          <span className="auth-logo-rag">RAG</span>
        </div>
      </div>
    );
  }

  // ── Unauthenticated: show login/register ──
  if (!authUser) {
    return (
      <>
        <AuthPage onAuth={handleAuth} toast={toast} />
        <ToastContainer toasts={toasts} dismiss={dismiss} />
      </>
    );
  }

  return (
    <div className="app-shell">
      <header className="top-bar">
        <button className="sidebar-toggle" onClick={() => setSidebarOpen((v) => !v)}>
          <span /><span /><span />
        </button>
        <div className="logo">
          <span className="logo-nova">Nova</span>
          <span className="logo-rag">RAG</span>
          <span className="logo-tag">Enterprise Vault</span>
        </div>
        <div className="top-bar-right">
          {/* User info pill */}
          <div className="user-pill" title={`${authUser.role} · ${authUser.clearance_level} clearance`}>
            <span className="user-pill-icon">{ROLE_ICONS[authUser.role] || "👤"}</span>
            <div className="user-pill-info">
              <span className="user-pill-name">{authUser.full_name || authUser.username}</span>
              <span
                className="user-pill-clearance"
                style={{ color: CLEARANCE_COLORS[authUser.clearance_level] || "#5b6af0" }}
              >
                {authUser.clearance_level}
              </span>
            </div>
            {authUser.role === "admin" && (
              <button className="user-pill-admin-btn" onClick={() => setShowAdmin(true)} title="Admin Dashboard">
                ⚙️
              </button>
            )}
            <button className="user-pill-logout-btn" onClick={handleLogout} title="Sign out">
              ⏏
            </button>
          </div>

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
          userRole={authUser.role}
          userClearance={authUser.clearance_level}
        />
        <div className="content-area">
          <IngestPanel
            apiUrl={API_URL}
            onIngested={fetchFiles}
            toast={toast}
            activeFiles={indexedFiles}
            onFilterChange={setFileFilter}
            token={authUser.token}
            userRole={authUser.role}
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
              token={authUser.token}
            />
          )}
          {mode === "search" && (
            <SearchPanel apiUrl={API_URL} files={indexedFiles} token={authUser.token} />
          )}
          {mode === "compare" && (
            <ComparePanel
              apiUrl={API_URL}
              sessionId={activeId}
              toast={toast}
              token={authUser.token}
            />
          )}
        </div>
      </div>

      {showKbd && <KeyboardShortcuts onClose={() => setShowKbd(false)} />}
      <ToastContainer toasts={toasts} dismiss={dismiss} />

      {showAdmin && authUser.role === "admin" && (
        <AdminDashboard
          user={authUser}
          token={authUser.token}
          onClose={() => setShowAdmin(false)}
          toast={toast}
        />
      )}
    </div>
  );
}