import { useState, useRef, useCallback } from "react";

function baseName(p) {
  return p ? p.split(/[/\\]/).pop() : "Unknown";
}

const EXT_ICONS = {
  pdf: "📄", docx: "📝", png: "🖼", jpg: "🖼", jpeg: "🖼",
  wav: "🎵", mp3: "🎵", m4a: "🎵",
};

function getIcon(path, type) {
  if (type === "image") return "🖼";
  if (type === "audio") return "🎵";
  const ext = baseName(path).split(".").pop()?.toLowerCase();
  return EXT_ICONS[ext] || "📄";
}

// Highlight matching query terms in text
function HighlightText({ text, query }) {
  if (!text) return null;
  if (!query) return <span>{text}</span>;

  const safe  = query.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const regex = new RegExp("(" + safe + ")", "gi");
  const parts = text.split(regex);

  return (
    <span>
      {parts.map((part, i) =>
        regex.test(part) ? (
          <mark
            key={i}
            style={{
              background: "var(--accent-soft)",
              color: "var(--accent)",
              borderRadius: 2,
              padding: "0 1px",
            }}
          >
            {part}
          </mark>
        ) : (
          <span key={i}>{part}</span>
        )
      )}
    </span>
  );
}

function ResultCard({ result, query, apiUrl }) {
  const [open, setOpen] = useState(false);

  const score      = Math.round((result.score || 0) * 100);
  const scoreColor = score >= 75 ? "var(--teal)" : score >= 45 ? "var(--amber)" : "var(--rose)";
  const fname      = baseName(result.path);
  const preview    = open
    ? (result.text || "")
    : ((result.text || "").slice(0, 200) + ((result.text || "").length > 200 ? "…" : ""));

  return (
    <div
      className={"search-result-card" + (open ? " open" : "")}
      onClick={() => setOpen((v) => !v)}
    >
      {/* Top row */}
      <div className="search-result-top">
        <span className="search-result-icon">{getIcon(result.path, result.type)}</span>

        <div className="search-result-meta">
          <span className="search-result-file" title={fname}>{fname}</span>
          {result.page != null && (
            <span className="search-result-page">p.{result.page}</span>
          )}
          {result.chunk_id != null && (
            <span className="search-result-chunk">#{result.chunk_id}</span>
          )}
        </div>

        <div className="search-result-score-wrap">
          <div className="search-result-score-bar">
            <div
              className="search-result-score-fill"
              style={{ width: score + "%", background: scoreColor }}
            />
          </div>
          <span className="search-result-score-val" style={{ color: scoreColor }}>
            {score}%
          </span>
        </div>

        <span className={"search-result-chevron" + (open ? " open" : "")}>▼</span>
      </div>

      {/* Text preview */}
      <div className="search-result-preview">
        <HighlightText text={preview} query={query} />
      </div>

      {/* Footer when expanded */}
      {open && (
        <div
          className="search-result-footer"
          onClick={(e) => e.stopPropagation()}
          style={{ paddingLeft: 14 }}
        >
          <a
            href={apiUrl + "/document/" + fname}
            target="_blank"
            rel="noopener noreferrer"
            className="citation-link"
          >
            ⬇ Open document
          </a>
        </div>
      )}
    </div>
  );
}

export default function SearchPanel({ apiUrl, files }) {
  const [query, setQuery]       = useState("");
  const [results, setResults]   = useState([]);
  const [loading, setLoading]   = useState(false);
  const [searched, setSearched] = useState(false);
  const [kValue, setKValue]     = useState(10);
  const [fileFilter, setFileFilter] = useState("");
  const inputRef = useRef();

  const search = useCallback(async () => {
    const q = query.trim();
    if (!q) return;

    setLoading(true);
    setSearched(false);
    setResults([]);

    try {
      const form = new URLSearchParams();
      form.append("q", q);
      form.append("k", String(kValue));
      if (fileFilter) form.append("file_filter", fileFilter);

      const res = await fetch(`${apiUrl}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: form.toString(),
      });
      if (!res.ok) throw new Error("HTTP " + res.status);
      const data = await res.json();
      setResults(data.results || []);
    } catch (e) {
      setResults([]);
    } finally {
      setLoading(false);
      setSearched(true);
    }
  }, [query, kValue, fileFilter, apiUrl]);

  const onKey = (e) => {
    if (e.key === "Enter") search();
  };

  const getFileIcon = (f) => {
    const ext = f.split(".").pop()?.toLowerCase();
    return EXT_ICONS[ext] || "📁";
  };

  return (
    <div className="search-panel">
      {/* Header */}
      <div className="search-header">
        <div className="search-title">
          <span className="search-icon-large">⌕</span>
          <div>
            <div className="search-title-text">Semantic Search</div>
            <div className="search-subtitle">
              Search chunks directly — no LLM, instant results
            </div>
          </div>
        </div>
      </div>

      {/* Search bar */}
      <div className="search-bar-wrap">
        <div className="search-input-row">
          <div className="search-input-wrap">
            <span className="search-prefix">⌕</span>
            <input
              ref={inputRef}
              className="search-input"
              placeholder="Search your documents…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={onKey}
              autoFocus
            />
            {query && (
              <button
                className="search-clear"
                onClick={() => {
                  setQuery("");
                  setResults([]);
                  setSearched(false);
                  inputRef.current?.focus();
                }}
              >
                ✕
              </button>
            )}
          </div>
          <button
            className="search-go-btn"
            onClick={search}
            disabled={!query.trim() || loading}
          >
            {loading ? "…" : "Search"}
          </button>
        </div>

        {/* Controls */}
        <div className="search-controls">
          <div className="depth-row">
            <span className="depth-label">Results</span>
            <input
              type="range"
              className="depth-slider"
              min={3} max={20} value={kValue}
              onChange={(e) => setKValue(Number(e.target.value))}
            />
            <span className="depth-val">{kValue}</span>
          </div>

          {files && files.length > 1 && (
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span className="depth-label">File</span>
              <select
                className="search-file-select"
                value={fileFilter}
                onChange={(e) => setFileFilter(e.target.value)}
              >
                <option value="">All files</option>
                {files.map((f) => (
                  <option key={f.file} value={f.file}>
                    {getFileIcon(f.file)} {f.file}
                  </option>
                ))}
              </select>
            </div>
          )}
        </div>
      </div>

      {/* Results area */}
      <div className="search-results">
        {/* Loading */}
        {loading && (
          <div className="search-empty">
            <div className="thinking"><span /><span /><span /></div>
            <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 12 }}>
              Searching…
            </div>
          </div>
        )}

        {/* No results */}
        {!loading && searched && results.length === 0 && (
          <div className="search-empty">
            <div style={{ fontSize: 36 }}>∅</div>
            <div style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 8 }}>
              No matching chunks found for "{query}"
            </div>
          </div>
        )}

        {/* Results */}
        {!loading && results.length > 0 && (
          <>
            <div className="search-results-header">
              <span className="search-results-count">{results.length} chunks</span>
              <span className="search-results-query">matching "{query}"</span>
            </div>
            {results.map((r, i) => (
              <ResultCard key={i} result={r} query={query} apiUrl={apiUrl} />
            ))}
          </>
        )}

        {/* Initial state */}
        {!searched && !loading && (
          <div className="search-empty">
            <div className="search-empty-glyph">⌕</div>
            <div style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 10, letterSpacing: 1 }}>
              Type a query and press Enter
            </div>
          </div>
        )}
      </div>
    </div>
  );
}