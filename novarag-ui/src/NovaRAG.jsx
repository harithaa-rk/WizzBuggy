import { useState, useRef, useEffect, useCallback } from "react";

const API_URL = "http://127.0.0.1:8000";

const MODELS = [
  { id: "fast", label: "Fast", sub: "Flan-T5", icon: "⚡" },
  { id: "mistral", label: "Mistral", sub: "7B Local", icon: "🧠" },
  { id: "llama3", label: "Llama 3", sub: "Local LLM", icon: "🦙" },
];

const FILE_TYPES = ["pdf", "docx", "png", "jpg", "jpeg", "wav", "mp3", "m4a"];

function TypingIndicator() {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6, padding: "12px 0" }}>
      {[0, 1, 2].map((i) => (
        <div
          key={i}
          style={{
            width: 7,
            height: 7,
            borderRadius: "50%",
            background: "#a78bfa",
            animation: `bounce 1.2s ease-in-out ${i * 0.2}s infinite`,
          }}
        />
      ))}
    </div>
  );
}

function CitationCard({ citation, index }) {
  const [open, setOpen] = useState(false);
  const fileName = citation.path?.split(/[\\/]/).pop() || "Unknown";
  const page = citation.page || null;

  return (
    <div
      style={{
        marginTop: 6,
        borderRadius: 8,
        border: "1px solid rgba(167,139,250,0.2)",
        background: "rgba(167,139,250,0.04)",
        overflow: "hidden",
        fontSize: 14,
      }}
    >
      <button
        onClick={() => setOpen(!open)}
        style={{
          width: "100%",
          padding: "8px 12px",
          background: "none",
          border: "none",
          display: "flex",
          alignItems: "center",
          gap: 8,
          cursor: "pointer",
          color: "#a78bfa",
          textAlign: "left",
          fontFamily: "'JetBrains Mono', monospace",
        }}
      >
        <span style={{ opacity: 0.6 }}>#{index + 1}</span>
        <span style={{ flex: 1, color: "#c4b5fd" }}>{fileName}</span>
        {page && (
          <span style={{ opacity: 0.5, fontSize: 13 }}>p.{page}</span>
        )}
        <span style={{ opacity: 0.5, fontSize: 12 }}>{open ? "▲" : "▼"}</span>
      </button>
      {open && (
        <div
          style={{
            padding: "8px 12px 12px",
            borderTop: "1px solid rgba(167,139,250,0.15)",
            color: "#94a3b8",
            lineHeight: 1.7,
            fontSize: 14,
          }}
        >
          {citation.snippet}
        </div>
      )}
    </div>
  );
}

function Message({ msg }) {
  const isUser = msg.role === "user";

  if (isUser) {
    return (
      <div style={{
        display: "flex",
        justifyContent: "flex-end",
        marginBottom: 28,
        animation: "fadeSlideUp 0.3s ease forwards",
      }}>
        <div style={{
          maxWidth: "68%",
          padding: "14px 20px",
          borderRadius: "20px 20px 4px 20px",
          background: "rgba(124,58,237,0.18)",
          border: "1px solid rgba(124,58,237,0.35)",
          color: "#e2e8f0",
          lineHeight: 1.75,
          fontSize: 18,
          whiteSpace: "pre-wrap",
        }}>
          {msg.content}
        </div>
      </div>
    );
  }

  return (
    <div style={{
      display: "flex",
      gap: 14,
      marginBottom: 32,
      animation: "fadeSlideUp 0.3s ease forwards",
    }}>
      <div style={{
        width: 36,
        height: 36,
        borderRadius: "50%",
        background: "linear-gradient(135deg, #7c3aed, #4f46e5)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontSize: 18,
        flexShrink: 0,
        marginTop: 2,
      }}>
        ✦
      </div>

      <div style={{ flex: 1, minWidth: 0 }}>
        {msg.rewritten && msg.rewritten !== msg.originalQ && (
          <div style={{
            fontSize: 13,
            color: "#6b7280",
            marginBottom: 8,
            fontFamily: "'JetBrains Mono', monospace",
            display: "flex",
            alignItems: "center",
            gap: 6,
          }}>
            <span style={{ color: "#7c3aed" }}>↻</span>
            searched as: <em style={{ color: "#a78bfa" }}>{msg.rewritten}</em>
          </div>
        )}

        <div style={{
          color: "#e2e8f0",
          lineHeight: 1.8,
          fontSize: 18,
          whiteSpace: "pre-wrap",
          paddingBottom: 4,
        }}>
          {msg.content}
        </div>

        {msg.citations && msg.citations.length > 0 && (
          <div style={{ marginTop: 14 }}>
            <div style={{
              fontSize: 12,
              color: "#6b7280",
              marginBottom: 8,
              fontFamily: "'JetBrains Mono', monospace",
              letterSpacing: "0.08em",
              textTransform: "uppercase",
            }}>
              Sources
            </div>
            {msg.citations.map((c, i) => (
              <CitationCard key={i} citation={c} index={i} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function FileUploadPanel({ onIngestSuccess }) {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState(null); // null | "loading" | "success" | "error"
  const [message, setMessage] = useState("");
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef();

  const handleFile = (f) => {
    if (!f) return;
    const ext = f.name.split(".").pop().toLowerCase();
    if (!FILE_TYPES.includes(ext)) {
      setStatus("error");
      setMessage("Unsupported file type.");
      return;
    }
    setFile(f);
    setStatus(null);
    setMessage("");
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    handleFile(e.dataTransfer.files[0]);
  };

  const ingest = async () => {
    if (!file) return;
    setStatus("loading");
    setMessage("");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API_URL}/ingest`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setStatus("success");
      setMessage(`Indexed ${data.chunks || 0} chunks`);
      onIngestSuccess && onIngestSuccess(data);
    } catch (e) {
      setStatus("error");
      setMessage("Backend unreachable. Is FastAPI running?");
    }
  };

  const formatSize = (bytes) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div style={{ padding: "20px 0" }}>
      <div
        onDrop={handleDrop}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onClick={() => inputRef.current?.click()}
        style={{
          border: `2px dashed ${dragging ? "#7c3aed" : "rgba(255,255,255,0.1)"}`,
          borderRadius: 12,
          padding: "28px 20px",
          textAlign: "center",
          cursor: "pointer",
          background: dragging ? "rgba(124,58,237,0.08)" : "rgba(255,255,255,0.02)",
          transition: "all 0.2s ease",
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept={FILE_TYPES.map((t) => `.${t}`).join(",")}
          style={{ display: "none" }}
          onChange={(e) => handleFile(e.target.files[0])}
        />
        <div style={{ fontSize: 30, marginBottom: 8 }}>
          {file ? "📄" : "☁"}
        </div>
        {file ? (
          <>
            <div style={{ color: "#e2e8f0", fontSize: 15, fontWeight: 600 }}>
              {file.name}
            </div>
            <div style={{ color: "#6b7280", fontSize: 13, marginTop: 4 }}>
              {formatSize(file.size)}
            </div>
          </>
        ) : (
          <>
            <div style={{ color: "#94a3b8", fontSize: 14 }}>
              Drop file or click to browse
            </div>
            <div style={{ color: "#4b5563", fontSize: 13, marginTop: 4 }}>
              PDF · DOCX · PNG · JPG · WAV · MP3 · M4A
            </div>
          </>
        )}
      </div>

      {file && (
        <button
          onClick={ingest}
          disabled={status === "loading"}
          style={{
            marginTop: 12,
            width: "100%",
            padding: "10px 0",
            borderRadius: 8,
            border: "none",
            background: status === "loading"
              ? "rgba(124,58,237,0.3)"
              : "linear-gradient(135deg, #7c3aed, #4f46e5)",
            color: "#fff",
            fontSize: 15,
            fontWeight: 600,
            cursor: status === "loading" ? "not-allowed" : "pointer",
            letterSpacing: "0.04em",
            transition: "all 0.2s ease",
          }}
        >
          {status === "loading" ? "Ingesting..." : "Ingest Document"}
        </button>
      )}

      {status === "success" && (
        <div style={{
          marginTop: 10,
          padding: "8px 12px",
          borderRadius: 8,
          background: "rgba(16,185,129,0.1)",
          border: "1px solid rgba(16,185,129,0.2)",
          color: "#34d399",
          fontSize: 14,
          display: "flex",
          alignItems: "center",
          gap: 6,
        }}>
          ✓ {message}
        </div>
      )}

      {status === "error" && (
        <div style={{
          marginTop: 10,
          padding: "8px 12px",
          borderRadius: 8,
          background: "rgba(239,68,68,0.1)",
          border: "1px solid rgba(239,68,68,0.2)",
          color: "#f87171",
          fontSize: 14,
          display: "flex",
          alignItems: "center",
          gap: 6,
        }}>
          ✗ {message}
        </div>
      )}
    </div>
  );
}

function StatsBar({ stats }) {
  if (!stats) return null;
  return (
    <div style={{
      display: "flex",
      gap: 12,
      marginTop: 16,
      padding: "12px 0",
      borderTop: "1px solid rgba(255,255,255,0.06)",
    }}>
      {[
        { label: "Chunks", value: stats.total ?? "—" },
        { label: "Docs", value: stats.files ?? "—" },
      ].map((s) => (
        <div key={s.label} style={{
          flex: 1,
          background: "rgba(255,255,255,0.03)",
          borderRadius: 8,
          padding: "10px 12px",
          border: "1px solid rgba(255,255,255,0.06)",
        }}>
          <div style={{ fontSize: 22, fontWeight: 700, color: "#a78bfa" }}>
            {s.value}
          </div>
          <div style={{ fontSize: 12, color: "#6b7280", textTransform: "uppercase", letterSpacing: "0.1em", marginTop: 2 }}>
            {s.label}
          </div>
        </div>
      ))}
    </div>
  );
}

export default function NovaRAG() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [model, setModel] = useState("mistral");
  const [stats, setStats] = useState(null);
  const [sidebarTab, setSidebarTab] = useState("upload"); // upload | settings
  const bottomRef = useRef();
  const inputRef = useRef();

  const fetchStats = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/stats`);
      const data = await res.json();
      setStats(data);
    } catch {
      setStats(null);
    }
  }, []);

  useEffect(() => {
    fetchStats();
  }, [fetchStats]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const sendMessage = async () => {
    const q = input.trim();
    if (!q || loading) return;

    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: q }]);
    setLoading(true);

    try {
      const res = await fetch(`${API_URL}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: new URLSearchParams({ q, model }),
      });

      const data = await res.json();

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.answer || "No answer returned.",
          citations: data.citations || [],
          rewritten: data.rewritten || null,
          originalQ: q,
        },
      ]);
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "❌ Could not reach backend. Make sure FastAPI is running on port 8000.",
          citations: [],
        },
      ]);
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearChat = () => setMessages([]);

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&family=DM+Sans:wght@400;500&display=swap');

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
          background: #080b14;
          color: #e2e8f0;
          font-family: 'DM Sans', sans-serif;
          font-size: 18px;
          height: 100vh;
          overflow: hidden;
        }

        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(124,58,237,0.3); border-radius: 2px; }

        @keyframes fadeSlideUp {
          from { opacity: 0; transform: translateY(12px); }
          to { opacity: 1; transform: translateY(0); }
        }

        @keyframes bounce {
          0%, 100% { transform: translateY(0); opacity: 0.4; }
          50% { transform: translateY(-5px); opacity: 1; }
        }

        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }

        @keyframes shimmer {
          0% { background-position: -200% 0; }
          100% { background-position: 200% 0; }
        }

        .model-btn:hover {
          border-color: rgba(124,58,237,0.5) !important;
          background: rgba(124,58,237,0.1) !important;
        }

        .send-btn:hover:not(:disabled) {
          background: linear-gradient(135deg, #6d28d9, #4338ca) !important;
          transform: scale(1.05);
        }

        .send-btn:disabled {
          opacity: 0.4;
          cursor: not-allowed;
        }

        .clear-btn:hover {
          background: rgba(255,255,255,0.06) !important;
        }
      `}</style>

      {/* Background grid */}
      <div style={{
        position: "fixed",
        inset: 0,
        backgroundImage: `
          linear-gradient(rgba(124,58,237,0.03) 1px, transparent 1px),
          linear-gradient(90deg, rgba(124,58,237,0.03) 1px, transparent 1px)
        `,
        backgroundSize: "48px 48px",
        pointerEvents: "none",
        zIndex: 0,
      }} />

      {/* Ambient glow */}
      <div style={{
        position: "fixed",
        top: -200,
        left: "30%",
        width: 600,
        height: 600,
        borderRadius: "50%",
        background: "radial-gradient(circle, rgba(124,58,237,0.08) 0%, transparent 70%)",
        pointerEvents: "none",
        zIndex: 0,
      }} />

      <div style={{
        position: "relative",
        zIndex: 1,
        display: "flex",
        height: "100vh",
      }}>

        {/* ─── SIDEBAR ─── */}
        <div style={{
          width: 280,
          flexShrink: 0,
          borderRight: "1px solid rgba(255,255,255,0.06)",
          display: "flex",
          flexDirection: "column",
          background: "rgba(255,255,255,0.015)",
          backdropFilter: "blur(12px)",
        }}>
          {/* Logo */}
          <div style={{ padding: "24px 20px 16px" }}>
            <div style={{
              fontFamily: "'Syne', sans-serif",
              fontWeight: 800,
              fontSize: 26,
              letterSpacing: "-0.02em",
              display: "flex",
              alignItems: "center",
              gap: 8,
            }}>
              <span style={{
                background: "linear-gradient(135deg, #a78bfa, #818cf8)",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
              }}>Nova</span>
              <span style={{ color: "#475569" }}>RAG</span>
              <div style={{
                marginLeft: "auto",
                width: 8,
                height: 8,
                borderRadius: "50%",
                background: "#10b981",
                animation: "pulse 2s ease infinite",
                boxShadow: "0 0 8px rgba(16,185,129,0.6)",
              }} />
            </div>
            <div style={{ fontSize: 13, color: "#4b5563", marginTop: 4, fontFamily: "'JetBrains Mono', monospace" }}>
              offline · multimodal · local
            </div>
          </div>

          {/* Sidebar tabs */}
          <div style={{
            display: "flex",
            margin: "0 16px",
            background: "rgba(255,255,255,0.03)",
            borderRadius: 8,
            padding: 3,
            gap: 3,
          }}>
            {[["upload", "Upload"], ["settings", "Models"]].map(([id, label]) => (
              <button
                key={id}
                onClick={() => setSidebarTab(id)}
                style={{
                  flex: 1,
                  padding: "6px 0",
                  borderRadius: 6,
                  border: "none",
                  background: sidebarTab === id ? "rgba(124,58,237,0.3)" : "transparent",
                  color: sidebarTab === id ? "#a78bfa" : "#6b7280",
                  fontSize: 14,
                  fontWeight: 600,
                  cursor: "pointer",
                  transition: "all 0.15s ease",
                  fontFamily: "'DM Sans', sans-serif",
                }}
              >
                {label}
              </button>
            ))}
          </div>

          {/* Tab content */}
          <div style={{ flex: 1, overflow: "auto", padding: "0 16px" }}>
            {sidebarTab === "upload" ? (
              <>
                <FileUploadPanel onIngestSuccess={fetchStats} />
                <StatsBar stats={stats} />
              </>
            ) : (
              <div style={{ padding: "20px 0" }}>
                <div style={{
                  fontSize: 13,
                  color: "#6b7280",
                  textTransform: "uppercase",
                  letterSpacing: "0.1em",
                  marginBottom: 12,
                  fontFamily: "'JetBrains Mono', monospace",
                }}>
                  Reasoning Engine
                </div>
                {MODELS.map((m) => (
                  <button
                    key={m.id}
                    onClick={() => setModel(m.id)}
                    className="model-btn"
                    style={{
                      width: "100%",
                      padding: "12px 14px",
                      borderRadius: 10,
                      border: `1px solid ${model === m.id ? "rgba(124,58,237,0.5)" : "rgba(255,255,255,0.06)"}`,
                      background: model === m.id ? "rgba(124,58,237,0.12)" : "rgba(255,255,255,0.02)",
                      display: "flex",
                      alignItems: "center",
                      gap: 12,
                      cursor: "pointer",
                      marginBottom: 8,
                      transition: "all 0.2s ease",
                      textAlign: "left",
                    }}
                  >
                    <span style={{ fontSize: 18 }}>{m.icon}</span>
                    <div>
                      <div style={{
                        fontSize: 15,
                        fontWeight: 600,
                        color: model === m.id ? "#a78bfa" : "#94a3b8",
                        fontFamily: "'Syne', sans-serif",
                      }}>
                        {m.label}
                      </div>
                      <div style={{ fontSize: 13, color: "#4b5563", marginTop: 1 }}>
                        {m.sub}
                      </div>
                    </div>
                    {model === m.id && (
                      <div style={{
                        marginLeft: "auto",
                        width: 6,
                        height: 6,
                        borderRadius: "50%",
                        background: "#a78bfa",
                        boxShadow: "0 0 8px rgba(167,139,250,0.6)",
                      }} />
                    )}
                  </button>
                ))}

                <div style={{
                  marginTop: 20,
                  padding: 12,
                  borderRadius: 10,
                  background: "rgba(255,255,255,0.02)",
                  border: "1px solid rgba(255,255,255,0.05)",
                }}>
                  <div style={{ fontSize: 13, color: "#6b7280", marginBottom: 6, fontFamily: "'JetBrains Mono', monospace", textTransform: "uppercase", letterSpacing: "0.08em" }}>
                    Active
                  </div>
                  <div style={{ fontSize: 15, color: "#a78bfa", fontWeight: 600 }}>
                    {MODELS.find(m2 => m2.id === model)?.icon} {MODELS.find(m2 => m2.id === model)?.label}
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Footer */}
          <div style={{
            padding: "12px 20px",
            borderTop: "1px solid rgba(255,255,255,0.05)",
            fontSize: 13,
            color: "#374151",
            fontFamily: "'JetBrains Mono', monospace",
          }}>
            100% local · no data leaves device
          </div>
        </div>

        {/* ─── MAIN CHAT ─── */}
        <div style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          minWidth: 0,
        }}>

          {/* Topbar */}
          <div style={{
            height: 56,
            borderBottom: "1px solid rgba(255,255,255,0.06)",
            display: "flex",
            alignItems: "center",
            padding: "0 24px",
            gap: 12,
            backdropFilter: "blur(8px)",
            background: "rgba(8,11,20,0.8)",
          }}>
            <div style={{
              fontFamily: "'Syne', sans-serif",
              fontWeight: 700,
              fontSize: 19,
              color: "#64748b",
            }}>
              Chat
            </div>
            <div style={{
              marginLeft: 0,
              padding: "3px 10px",
              borderRadius: 20,
              background: "rgba(124,58,237,0.12)",
              border: "1px solid rgba(124,58,237,0.2)",
              fontSize: 12,
              color: "#7c3aed",
              fontFamily: "'JetBrains Mono', monospace",
            }}>
              {MODELS.find(m2 => m2.id === model)?.icon} {model}
            </div>

            {messages.length > 0 && (
              <button
                onClick={clearChat}
                className="clear-btn"
                style={{
                  marginLeft: "auto",
                  padding: "5px 12px",
                  borderRadius: 6,
                  border: "1px solid rgba(255,255,255,0.06)",
                  background: "rgba(255,255,255,0.02)",
                  color: "#6b7280",
                  fontSize: 13,
                  cursor: "pointer",
                  fontFamily: "'DM Sans', sans-serif",
                  transition: "all 0.15s ease",
                }}
              >
                Clear
              </button>
            )}
          </div>

          {/* Messages */}
          <div style={{
            flex: 1,
            overflow: "auto",
            padding: "32px 48px",
          }}>
            {messages.length === 0 ? (
              <div style={{
                height: "100%",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                gap: 16,
                opacity: 0.4,
              }}>
                <div style={{
                  width: 64,
                  height: 64,
                  borderRadius: "50%",
                  border: "1px solid rgba(124,58,237,0.3)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: 28,
                }}>
                  ✦
                </div>
                <div style={{
                  fontFamily: "'Syne', sans-serif",
                  fontSize: 22,
                  fontWeight: 600,
                  color: "#475569",
                }}>
                  Upload a document, then ask anything
                </div>
                <div style={{ fontSize: 16, color: "#374151", textAlign: "center", maxWidth: 320 }}>
                  PDFs, DOCX, images, and audio files are all supported.
                  Everything runs locally.
                </div>
              </div>
            ) : (
              messages.map((msg, i) => <Message key={i} msg={msg} />)
            )}

            {loading && (
              <div style={{
                display: "flex",
                gap: 14,
                animation: "fadeSlideUp 0.3s ease forwards",
                marginBottom: 32,
              }}>
                <div style={{
                  width: 36,
                  height: 36,
                  borderRadius: "50%",
                  background: "linear-gradient(135deg, #7c3aed, #4f46e5)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: 18,
                  flexShrink: 0,
                  marginTop: 2,
                }}>
                  ✦
                </div>
                <div style={{
                  display: "flex",
                  alignItems: "center",
                  paddingTop: 6,
                }}>
                  <TypingIndicator />
                </div>
              </div>
            )}

            <div ref={bottomRef} />
          </div>

          {/* Input bar */}
          <div style={{
            padding: "16px 24px 20px",
            borderTop: "1px solid rgba(255,255,255,0.06)",
            background: "rgba(8,11,20,0.9)",
            backdropFilter: "blur(12px)",
          }}>
            <div style={{
              display: "flex",
              gap: 12,
              alignItems: "flex-end",
              background: "rgba(255,255,255,0.05)",
              border: "1px solid rgba(255,255,255,0.1)",
              borderRadius: 16,
              padding: "12px 14px 12px 20px",
              transition: "border-color 0.2s ease",
            }}>
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKey}
                placeholder="Ask anything about your documents..."
                rows={1}
                style={{
                  flex: 1,
                  background: "none",
                  border: "none",
                  outline: "none",
                  color: "#e2e8f0",
                  fontSize: 18,
                  fontFamily: "'DM Sans', sans-serif",
                  resize: "none",
                  lineHeight: 1.65,
                  maxHeight: 140,
                  overflowY: "auto",
                }}
                onInput={(e) => {
                  e.target.style.height = "auto";
                  e.target.style.height = Math.min(e.target.scrollHeight, 140) + "px";
                }}
              />
              <button
                onClick={sendMessage}
                disabled={loading || !input.trim()}
                className="send-btn"
                style={{
                  width: 40,
                  height: 40,
                  borderRadius: 12,
                  border: "none",
                  background: "linear-gradient(135deg, #7c3aed, #4f46e5)",
                  color: "#fff",
                  fontSize: 20,
                  cursor: "pointer",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  flexShrink: 0,
                  transition: "all 0.15s ease",
                }}
              >
                ↑
              </button>
            </div>
            <div style={{
              textAlign: "center",
              marginTop: 8,
              fontSize: 13,
              color: "#374151",
              fontFamily: "'JetBrains Mono', monospace",
            }}>
              Enter to send · Shift+Enter for newline
            </div>
          </div>
        </div>
      </div>
    </>
  );
}