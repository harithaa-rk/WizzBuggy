import { useState, useRef, useEffect, useCallback } from "react";
import { getModel, getK, getFileFilter } from "./IngestPanel";
import CitationItem from "./CitationItem";
import ConfidenceGauge from "./ConfidenceGauge";
import { useVoice } from "../hooks/useVoice";
import { useQueryHistory } from "../hooks/useQueryHistory";
import { exportAsMarkdown, exportAsText, exportAsJSON } from "../utils/exportChat";

export default function ChatPanel({
  apiUrl, history, setHistory, toast,
  sessionId, sessionName, chatInputRef, micToggleRef,
}) {
  const [input, setInput]         = useState("");
  const [loading, setLoading]     = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [showExport, setShowExport]   = useState(false);

  const scrollRef        = useRef();
  const internalInputRef = useRef();
  const abortRef         = useRef(null);
  const originalInputRef = useRef("");  // saves input before ↑ navigation

  const {
    history: qHistory, cursor,
    push, navigateUp, navigateDown, reset, clear: clearQHistory,
  } = useQueryHistory();

  useEffect(() => {
    if (chatInputRef) chatInputRef.current = internalInputRef.current;
  });

  useEffect(() => {
    if (scrollRef.current)
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [history, loading, streaming]);

  // Close dropdowns on outside click
  useEffect(() => {
    const handler = (e) => {
      if (!e.target.closest(".input-wrapper"))   setShowHistory(false);
      if (!e.target.closest(".export-menu-wrap")) setShowExport(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const autoResize = (el) => {
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 140) + "px";
  };

  const sendMessage = async (overrideText) => {
    const q = (overrideText !== undefined ? overrideText : input).trim();
    if (!q || loading) return;

    push(q);
    reset();
    setShowHistory(false);
    setHistory((h) => [...h, { role: "user", message: q }]);
    setInput("");
    originalInputRef.current = "";
    if (internalInputRef.current) internalInputRef.current.style.height = "auto";
    setLoading(true);

    const model      = getModel();
    const k          = getK();
    const fileFilter = getFileFilter();
    const form       = new URLSearchParams();
    form.append("q", q);
    form.append("model", model);
    form.append("k", String(k));
    form.append("session", sessionId || "default");
    form.append("stream", "true");
    if (fileFilter) form.append("file_filter", fileFilter);

    const placeholderIdx = Date.now();
    setHistory((h) => [...h, {
      role: "assistant", message: "", _streaming: true,
      _id: placeholderIdx, citations: [], confidence: null,
      chunks_used: null, trace: [], linked_sources: [], user_question: q,
    }]);
    setLoading(false);
    setStreaming(true);

    try {
      const controller = new AbortController();
      abortRef.current = controller;

      const res = await fetch(`${apiUrl}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: form.toString(),
        signal: controller.signal,
      });
      if (!res.ok) throw new Error("HTTP " + res.status);

      const reader  = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "", fullText = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const evt = JSON.parse(line);
            if (evt.type === "meta") {
              setHistory((h) => h.map((m) => m._id === placeholderIdx
                ? { ...m, confidence: evt.confidence, chunks_used: evt.chunks_used,
                    citations: evt.citations || [], trace: evt.trace || [],
                    linked_sources: evt.linked_sources || [] }
                : m));
            } else if (evt.type === "token") {
              fullText += evt.text;
              const snap = fullText;
              setHistory((h) => h.map((m) => m._id === placeholderIdx
                ? { ...m, message: snap } : m));
            } else if (evt.type === "done") {
              setHistory((h) => h.map((m) => m._id === placeholderIdx
                ? { ...m, message: fullText.trim(), _streaming: false } : m));
            }
          } catch (_) {}
        }
      }
    } catch (e) {
      if (e.name !== "AbortError") {
        toast("Query failed: " + e.message, "error");
        setHistory((h) => h.map((m) => m._id === placeholderIdx
          ? { ...m, message: "Error: " + e.message, _streaming: false } : m));
      }
    } finally {
      setStreaming(false);
      abortRef.current = null;
    }
  };

  // Voice
  const handleVoiceResult = useCallback((transcript) => {
    setInput(transcript);
    setTimeout(() => {
      if (internalInputRef.current) {
        internalInputRef.current.style.height = "auto";
        internalInputRef.current.style.height =
          Math.min(internalInputRef.current.scrollHeight, 140) + "px";
        internalInputRef.current.focus();
      }
    }, 0);
    toast('Heard: "' + transcript + '"', "info", 2500);
  }, [toast]);

  const { recording, supported: voiceSupported, toggle: toggleVoice, statusMsg } =
    useVoice(handleVoiceResult, apiUrl);

  const onKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
      return;
    }
    // ↑ navigate history
    if (e.key === "ArrowUp" && !e.shiftKey) {
      e.preventDefault();
      if (cursor === -1) originalInputRef.current = input;
      navigateUp(setInput);
      setShowHistory(true);
      return;
    }
    // ↓ navigate back
    if (e.key === "ArrowDown" && !e.shiftKey) {
      e.preventDefault();
      navigateDown(setInput, originalInputRef.current);
      if (cursor <= 0) setShowHistory(false);
      return;
    }
    // Escape closes dropdown
    if (e.key === "Escape") {
      setShowHistory(false);
      setShowExport(false);
    }
  };

  const onFocus = () => {
    if (qHistory.length > 0) setShowHistory(true);
  };

  return (
    <div className="chat-panel">
      <div className="messages-scroll" ref={scrollRef}>
        {history.length === 0 && !loading && !streaming && (
          <div className="empty-state">
            <div className="empty-glyph">N</div>
            <div className="empty-text">Ingest documents · Ask anything</div>
          </div>
        )}

        {history.map((item, i) => (
          <div key={item._id || i} className={"msg-row " + item.role}>
            <div className="msg-avatar">{item.role === "user" ? "U" : "AI"}</div>
            <div className="msg-body">
              <div className="msg-bubble">
                {item._streaming && !item.message
                  ? <div className="thinking"><span /><span /><span /></div>
                  : <MarkdownText text={item.message} />}
                {item._streaming && item.message && <span className="stream-cursor" />}
              </div>

              {item.role === "assistant" && !item._streaming && (
                <>
                  <div className="msg-actions">
                    <CopyButton text={item.message} toast={toast} />
                  </div>
                  <div className="msg-meta" style={{ alignItems: "center" }}>
                    {item.confidence != null && <ConfidenceGauge value={item.confidence} />}
                    {item.chunks_used != null && (
                      <span className="meta-chip chunks">{item.chunks_used} chunks</span>
                    )}
                  </div>
                  {item.citations && item.citations.length > 0 && (
                    <Citations citations={item.citations} apiUrl={apiUrl} userQ={item.user_question} />
                  )}
                  {item.linked_sources && item.linked_sources.length > 0 && (
                    <XModalLinks links={item.linked_sources} />
                  )}
                  {item.trace && item.trace.length > 0 && (
                    <TraceBlock trace={item.trace} />
                  )}
                </>
              )}
            </div>
          </div>
        ))}
      </div>

      {statusMsg && (
        <div style={{ padding: "4px 20px", fontSize: 11, color: "var(--teal)",
          background: "var(--bg-surface)", borderTop: "1px solid var(--border)", letterSpacing: "0.5px" }}>
          {"🎤 " + statusMsg}
        </div>
      )}

      <div className="chat-input-area">
        {voiceSupported && (
          <button ref={micToggleRef}
            className={"mic-btn" + (recording ? " recording" : "")}
            onClick={toggleVoice}
            title={recording ? "Stop (Ctrl+M)" : "Voice input (Ctrl+M)"}>
            {recording ? "⏹" : "🎤"}
          </button>
        )}

        <div className="input-wrapper">
          {/* Query history dropdown */}
          {showHistory && qHistory.length > 0 && (
            <div className="history-dropdown">
              <div className="history-header">
                <span className="history-header-label">Recent queries</span>
                <button className="history-clear-btn" onClick={() => { clearQHistory(); setShowHistory(false); }}>
                  clear
                </button>
              </div>
              {qHistory.slice(0, 12).map((q, i) => (
                <div
                  key={i}
                  className={"history-item" + (i === cursor ? " active" : "")}
                  onMouseDown={(e) => { e.preventDefault(); setInput(q); setShowHistory(false); sendMessage(q); }}
                >
                  <span style={{ color: "var(--text-muted)", fontSize: 10 }}>⏱</span>
                  {q.length > 60 ? q.slice(0, 58) + "…" : q}
                </div>
              ))}
              <div className="history-arrow-hint">↑ ↓ to navigate · Enter to send</div>
            </div>
          )}

          <textarea
            ref={internalInputRef}
            className="chat-textarea"
            rows={1}
            placeholder="Ask anything… ↑ for history · ? for shortcuts"
            value={input}
            onChange={(e) => { setInput(e.target.value); autoResize(e.target); }}
            onKeyDown={onKey}
            onFocus={onFocus}
          />
        </div>

        {/* Export menu */}
        {history.length > 0 && (
          <div className="export-menu-wrap">
            <button className="export-btn" onClick={() => setShowExport((v) => !v)} title="Export chat">
              ⬇ Export
            </button>
            {showExport && (
              <div className="export-dropdown">
                <button className="export-option" onMouseDown={() => { exportAsMarkdown(history, sessionName); setShowExport(false); toast("Exported as Markdown", "success", 2000); }}>
                  📄 Markdown (.md)
                </button>
                <div className="export-divider" />
                <button className="export-option" onMouseDown={() => { exportAsText(history, sessionName); setShowExport(false); toast("Exported as text", "success", 2000); }}>
                  📝 Plain text (.txt)
                </button>
                <div className="export-divider" />
                <button className="export-option" onMouseDown={() => { exportAsJSON(history, sessionName); setShowExport(false); toast("Exported as JSON", "success", 2000); }}>
                  {"{ }"} JSON (.json)
                </button>
              </div>
            )}
          </div>
        )}

        {streaming ? (
          <button className="send-btn" style={{ background: "var(--rose)" }}
            onClick={() => abortRef.current?.abort()} title="Stop generating">■</button>
        ) : (
          <button className="send-btn" onClick={() => sendMessage()}
            disabled={!input.trim() || loading}>↑</button>
        )}
      </div>
    </div>
  );
}

/* ── Copy Button ── */
function CopyButton({ text, toast }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      toast("Copied to clipboard", "success", 2000);
      setTimeout(() => setCopied(false), 2000);
    } catch { toast("Copy failed", "error"); }
  };
  return (
    <button className={"action-btn" + (copied ? " copied" : "")} onClick={copy}>
      {copied ? "✓ Copied" : "⎘ Copy"}
    </button>
  );
}

/* ── Markdown ── */
function MarkdownText({ text }) {
  if (!text) return null;
  return (
    <div>
      {text.split("\n").map((line, i) => {
        const isBullet = /^[•\-*] /.test(line.trim());
        const clean    = line.replace(/^[•\-*] /, "");
        const parts    = clean.split(/(\*\*.*?\*\*)/g).map((p, j) =>
          p.startsWith("**") ? <strong key={j}>{p.slice(2, -2)}</strong> : p
        );
        return isBullet ? (
          <div key={i} style={{ display: "flex", gap: 8, marginBottom: 2 }}>
            <span style={{ color: "var(--accent)", marginTop: 1 }}>▸</span>
            <span>{parts}</span>
          </div>
        ) : <div key={i}>{parts}</div>;
      })}
    </div>
  );
}

/* ── Citations ── */
function Citations({ citations, apiUrl, userQ }) {
  const seen = new Set();
  const unique = citations.filter((c) => {
    const key = c.path + "|" + c.page;
    if (seen.has(key)) return false;
    seen.add(key); return true;
  });
  return (
    <div className="citations-block">
      <div className="citations-title">Sources</div>
      {unique.map((c, i) => <CitationItem key={i} citation={c} apiUrl={apiUrl} userQ={userQ} />)}
    </div>
  );
}

/* ── XModal ── */
function XModalLinks({ links }) {
  const bn = (p) => (p ? p.split(/[/\\]/).pop() : "?");
  return (
    <div className="xmodal-block">
      <div className="xmodal-title">Cross-modal Links</div>
      {links.map((lnk, i) => {
        const t = lnk.type;
        if (t === "text\u2194audio") {
          const ts = lnk.audio_timestamp != null ? " @ " + lnk.audio_timestamp + "s" : "";
          return <div key={i} className="xmodal-item">{"🔗 "}<strong>{"Text↔Audio"}</strong>{" — " + bn(lnk.text_source) + " (p." + (lnk.text_page || "?") + ") → " + bn(lnk.audio_source) + ts}</div>;
        }
        if (t === "text\u2194image")
          return <div key={i} className="xmodal-item">{"🔗 "}<strong>{"Text↔Image"}</strong>{" — " + bn(lnk.text_source) + " (p." + (lnk.text_page || "?") + ") → " + bn(lnk.image_source)}</div>;
        if (t === "audio\u2194image") {
          const ts = lnk.audio_timestamp != null ? " @ " + lnk.audio_timestamp + "s" : "";
          return <div key={i} className="xmodal-item">{"🔗 "}<strong>{"Audio↔Image"}</strong>{" — " + bn(lnk.audio_source) + ts + " → " + bn(lnk.image_source)}</div>;
        }
        return null;
      })}
    </div>
  );
}

/* ── Trace ── */
function TraceBlock({ trace }) {
  const [open, setOpen] = useState(false);
  const bn = (p) => (p ? p.split(/[/\\]/).pop() : "?");
  return (
    <div className="trace-block">
      <div className="trace-title" onClick={() => setOpen((v) => !v)}>
        <span>▸</span> Traceability Map {open ? "▲" : "▼"}
      </div>
      {open && (
        <table className="trace-table">
          <thead><tr><th>File</th><th>Page</th><th>Chunk</th><th>Score</th></tr></thead>
          <tbody>
            {trace.map((t, i) => (
              <tr key={i}>
                <td>{bn(t.file)}</td>
                <td>{t.page != null ? t.page : "—"}</td>
                <td>{t.chunk_id != null ? t.chunk_id : "—"}</td>
                <td>{t.score}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}