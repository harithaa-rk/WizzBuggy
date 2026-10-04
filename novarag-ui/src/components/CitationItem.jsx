import { useState } from "react";
import DocViewer from "./Docviewer";

function baseName(p) {
  if (!p) return "Unknown";
  return p.split(/[/\\]/).pop();
}

const EXT_ICONS = {
  pdf: "📄", docx: "📝", png: "🖼", jpg: "🖼", jpeg: "🖼",
  wav: "🎵", mp3: "🎵", m4a: "🎵",
};

export default function CitationItem({ citation, apiUrl, userQ }) {
  const [open, setOpen]             = useState(false);
  const [tooltipVisible, setTooltip] = useState(false);
  const [viewerOpen, setViewerOpen]  = useState(false);

  const fname   = baseName(citation.path);
  const ext     = fname.split(".").pop()?.toLowerCase();
  const icon    = EXT_ICONS[ext] || "📁";
  const score   = Math.min(Math.max(Number(citation.score) || 0, 0), 1);
  const isImage = citation.type === "image" || ["png","jpg","jpeg"].includes(ext);
  const isAudio = ["mp3","wav","m4a"].includes(ext);

  let snippet = citation.snippet || "";
  if (userQ && snippet) {
    const safe = userQ.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    snippet = snippet.replace(new RegExp("(" + safe + ")", "gi"), "**$1**");
  }

  const tooltipText = (citation.snippet || "").slice(0, 200) +
    ((citation.snippet || "").length > 200 ? "…" : "");

  return (
    <>
      <div className="citation-item">
        <div className="citation-header" onClick={() => setOpen((v) => !v)}>
          <span className="citation-icon">{icon}</span>
          <span className="citation-file" title={fname}>{fname}</span>
          {citation.page && (
            <span className="citation-page">p.{citation.page}</span>
          )}

          {/* Score bar with hover tooltip */}
          <div
            className="citation-score-wrap"
            onMouseEnter={() => setTooltip(true)}
            onMouseLeave={() => setTooltip(false)}
          >
            <div className="citation-score">
              <div
                className="citation-score-fill"
                style={{ width: (score * 100) + "%" }}
              />
            </div>
            <span style={{ fontSize: 10, color: "var(--text-muted)", minWidth: 28 }}>
              {Math.round(score * 100)}%
            </span>
            {tooltipVisible && tooltipText && (
              <div className="chunk-tooltip">{tooltipText}</div>
            )}
          </div>

          <span className={"citation-chevron" + (open ? " open" : "")}>▼</span>
        </div>

        {open && (
          <div className="citation-body">
            {isImage && (
              <img
                src={apiUrl + "/document/" + fname}
                alt={fname}
                style={{ maxWidth: "100%", borderRadius: 6, cursor: "zoom-in" }}
                onClick={() => setViewerOpen(true)}
              />
            )}
            {isAudio && (
              <audio controls src={apiUrl + "/document/" + fname} style={{ width: "100%" }} />
            )}
            {citation.snippet && (
              <div className="citation-snippet">
                {snippet.split(/(\*\*.*?\*\*)/).map((p, i) =>
                  p.startsWith("**") ? (
                    <mark key={i} style={{ background:"var(--accent-soft)", color:"var(--accent)", borderRadius:2 }}>
                      {p.slice(2,-2)}
                    </mark>
                  ) : p
                )}
              </div>
            )}
            <div style={{ display: "flex", gap: 8, marginTop: 4 }}>
              <button
                className="action-btn"
                onClick={() => setViewerOpen(true)}
              >
                👁 View document
              </button>
              <a
                href={apiUrl + "/document/" + fname}
                target="_blank"
                rel="noopener noreferrer"
                className="citation-link"
              >
                ⬇ Download
              </a>
            </div>
          </div>
        )}
      </div>

      {/* Document Viewer modal */}
      {viewerOpen && (
        <DocViewer
          apiUrl={apiUrl}
          filename={fname}
          page={citation.page}
          onClose={() => setViewerOpen(false)}
        />
      )}
    </>
  );
}