import { useState, useRef } from "react";
import { getK, getFileFilter } from "./IngestPanel";
import CitationItem from "./CitationItem";
import ConfidenceGauge from "./ConfidenceGauge";

const MODELS = [
  { value: "fast",    label: "Fast",     desc: "Mistral quick" },
  { value: "mistral", label: "Smart",    desc: "Mistral deep"  },
  { value: "llama3",  label: "Creative", desc: "Llama3"        },
];

function baseName(p) {
  return p ? p.split(/[/\\]/).pop() : "?";
}

function MarkdownText({ text }) {
  if (!text) return null;
  return (
    <div>
      {text.split("\n").map((line, i) => {
        const isBullet = /^[•\-*] /.test(line.trim());
        const clean    = line.replace(/^[•\-*] /, "");
        const parts    = clean.split(/(\*\*.*?\*\*)/g).map((p, j) =>
          p.startsWith("**") ? <strong key={j}>{p.slice(2,-2)}</strong> : p
        );
        return isBullet ? (
          <div key={i} style={{ display:"flex", gap:8, marginBottom:2 }}>
            <span style={{ color:"var(--accent)" }}>▸</span>
            <span>{parts}</span>
          </div>
        ) : <div key={i}>{parts}</div>;
      })}
    </div>
  );
}

function ModelPane({ apiUrl, sessionId, model, result, loading, streaming }) {
  const info = MODELS.find((m) => m.value === model) || MODELS[0];
  const streamColor = model === "fast" ? "var(--accent)" : model === "mistral" ? "var(--teal)" : "var(--amber)";

  return (
    <div className="compare-pane">
      <div className="compare-pane-header" style={{ borderTopColor: streamColor }}>
        <span className="compare-model-label" style={{ color: streamColor }}>
          {info.label}
        </span>
        <span className="compare-model-desc">{info.desc}</span>
        {loading && <div className="thinking" style={{ marginLeft: "auto" }}><span /><span /><span /></div>}
        {streaming && !loading && (
          <span className="stream-cursor" style={{ marginLeft: "auto", background: streamColor }} />
        )}
      </div>

      <div className="compare-pane-body">
        {!result && !loading && !streaming && (
          <div className="compare-empty">Ask a question to compare</div>
        )}

        {result && (
          <>
            <div className="compare-answer">
              <MarkdownText text={result.message} />
              {streaming && <span className="stream-cursor" style={{ background: streamColor }} />}
            </div>

            {result.confidence != null && (
              <div style={{ marginTop: 10 }}>
                <ConfidenceGauge value={result.confidence} />
              </div>
            )}

            {result.chunks_used != null && (
              <div style={{ marginTop: 6 }}>
                <span className="meta-chip chunks">{result.chunks_used} chunks</span>
              </div>
            )}

            {result.citations && result.citations.length > 0 && (
              <div className="citations-block" style={{ marginTop: 10 }}>
                <div className="citations-title">Sources</div>
                {(() => {
                  const seen = new Set();
                  return result.citations
                    .filter((c) => {
                      const key = c.path + "|" + c.page;
                      if (seen.has(key)) return false;
                      seen.add(key); return true;
                    })
                    .map((c, i) => (
                      <CitationItem key={i} citation={c} apiUrl={apiUrl} userQ={result.question} />
                    ));
                })()}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

export default function ComparePanel({ apiUrl, sessionId, toast }) {
  const [query, setQuery]       = useState("");
  const [modelA, setModelA]     = useState("fast");
  const [modelB, setModelB]     = useState("mistral");
  const [resultA, setResultA]   = useState(null);
  const [resultB, setResultB]   = useState(null);
  const [loadingA, setLoadingA] = useState(false);
  const [loadingB, setLoadingB] = useState(false);
  const [streamingA, setStreamingA] = useState(false);
  const [streamingB, setStreamingB] = useState(false);
  const abortA = useRef(null);
  const abortB = useRef(null);

  const runQuery = async (model, setResult, setLoading, setStreaming, abortRef) => {
    setLoading(true);
    setResult(null);

    const k          = getK();
    const fileFilter = getFileFilter();
    const form       = new URLSearchParams();
    form.append("q", query);
    form.append("model", model);
    form.append("k", String(k));
    form.append("session", sessionId || "default");
    form.append("stream", "true");
    if (fileFilter) form.append("file_filter", fileFilter);

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
      let meta   = null;

      setLoading(false);
      setStreaming(true);
      setResult({ message: "", question: query, citations: [], confidence: null, chunks_used: null });

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
              meta = evt;
              setResult((prev) => ({
                ...prev,
                confidence:    evt.confidence,
                chunks_used:   evt.chunks_used,
                citations:     evt.citations || [],
              }));
            } else if (evt.type === "token") {
              fullText += evt.text;
              const snap = fullText;
              setResult((prev) => ({ ...prev, message: snap }));
            } else if (evt.type === "done") {
              setResult((prev) => ({ ...prev, message: fullText.trim() }));
            }
          } catch (_) {}
        }
      }
    } catch (e) {
      if (e.name !== "AbortError") {
        toast("Query failed (" + model + "): " + e.message, "error");
        setResult({ message: "Error: " + e.message, citations: [], question: query });
      }
    } finally {
      setLoading(false);
      setStreaming(false);
      abortRef.current = null;
    }
  };

  const compare = () => {
    if (!query.trim()) return;
    if (modelA === modelB) {
      toast("Choose two different models to compare", "error", 2500);
      return;
    }
    // Fire both in parallel
    runQuery(modelA, setResultA, setLoadingA, setStreamingA, abortA);
    runQuery(modelB, setResultB, setLoadingB, setStreamingB, abortB);
  };

  const stop = () => {
    abortA.current?.abort();
    abortB.current?.abort();
    setLoadingA(false); setLoadingB(false);
    setStreamingA(false); setStreamingB(false);
  };

  const isRunning = loadingA || loadingB || streamingA || streamingB;

  const onKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); compare(); }
  };

  return (
    <div className="compare-panel">
      {/* Header */}
      <div className="compare-header">
        <div className="compare-title-row">
          <span className="compare-icon">⚡</span>
          <div>
            <div className="compare-title-text">Compare Mode</div>
            <div className="compare-subtitle">Ask once, compare two models side by side</div>
          </div>
        </div>

        {/* Model selectors */}
        <div className="compare-selectors">
          <div className="compare-selector-group">
            <span className="compare-selector-label">Model A</span>
            <div className="compare-model-pills">
              {MODELS.map((m) => (
                <button
                  key={m.value}
                  className={"compare-pill" + (modelA === m.value ? " active-a" : "")}
                  onClick={() => setModelA(m.value)}
                  disabled={m.value === modelB}
                  title={m.desc}
                >
                  {m.label}
                </button>
              ))}
            </div>
          </div>

          <div className="compare-vs">VS</div>

          <div className="compare-selector-group">
            <span className="compare-selector-label">Model B</span>
            <div className="compare-model-pills">
              {MODELS.map((m) => (
                <button
                  key={m.value}
                  className={"compare-pill" + (modelB === m.value ? " active-b" : "")}
                  onClick={() => setModelB(m.value)}
                  disabled={m.value === modelA}
                  title={m.desc}
                >
                  {m.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Input */}
        <div className="compare-input-row">
          <textarea
            className="chat-textarea"
            rows={1}
            placeholder="Ask a question to compare across both models…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={onKey}
            style={{ flex: 1 }}
          />
          {isRunning ? (
            <button className="send-btn" style={{ background: "var(--rose)" }} onClick={stop}>■</button>
          ) : (
            <button className="send-btn" onClick={compare} disabled={!query.trim()}>↑</button>
          )}
        </div>
      </div>

      {/* Split panes */}
      <div className="compare-panes">
        <ModelPane
          apiUrl={apiUrl}
          model={modelA}
          result={resultA}
          loading={loadingA}
          streaming={streamingA}
        />
        <div className="compare-divider" />
        <ModelPane
          apiUrl={apiUrl}
          model={modelB}
          result={resultB}
          loading={loadingB}
          streaming={streamingB}
        />
      </div>
    </div>
  );
}