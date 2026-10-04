import { useState, useEffect, useRef } from "react";

// Renders a document inline — PDF via iframe, images natively, audio with player
// No external library needed — uses browser's built-in PDF renderer via iframe

export default function DocViewer({ apiUrl, filename, page, onClose }) {
  const ext      = filename?.split(".").pop()?.toLowerCase();
  const url      = `${apiUrl}/document/${filename}`;
  const isImage  = ["png","jpg","jpeg"].includes(ext);
  const isAudio  = ["mp3","wav","m4a"].includes(ext);
  const isPDF    = ext === "pdf";
  const isDocx   = ext === "docx";

  // For PDF: append page anchor so browser jumps to it
  const pdfUrl = isPDF && page ? `${url}#page=${page}` : url;

  return (
    <div className="docviewer-overlay" onClick={onClose}>
      <div className="docviewer-modal" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="docviewer-header">
          <div className="docviewer-title">
            <span className="docviewer-icon">
              {isPDF ? "📄" : isImage ? "🖼" : isAudio ? "🎵" : "📝"}
            </span>
            <span className="docviewer-filename">{filename}</span>
            {page && <span className="docviewer-page">Page {page}</span>}
          </div>
          <div className="docviewer-actions">
            <a
              href={url}
              download={filename}
              className="action-btn"
              onClick={(e) => e.stopPropagation()}
            >
              ⬇ Download
            </a>
            <button className="action-btn" onClick={onClose}>✕ Close</button>
          </div>
        </div>

        {/* Content */}
        <div className="docviewer-body">
          {isPDF && (
            <iframe
              src={pdfUrl}
              className="docviewer-iframe"
              title={filename}
            />
          )}
          {isImage && (
            <div className="docviewer-image-wrap">
              <img src={url} alt={filename} className="docviewer-image" />
            </div>
          )}
          {isAudio && (
            <div className="docviewer-audio-wrap">
              <audio controls src={url} className="docviewer-audio" autoPlay />
              <div className="docviewer-audio-label">{filename}</div>
            </div>
          )}
          {isDocx && (
            <div className="docviewer-unsupported">
              <div style={{ fontSize: 40 }}>📝</div>
              <div style={{ marginTop: 12, fontSize: 13, color: "var(--text-secondary)" }}>
                Word documents can't be previewed inline.
              </div>
              <a href={url} download={filename} className="btn btn-primary" style={{ marginTop: 16, display: "inline-flex" }}>
                ⬇ Download to view
              </a>
            </div>
          )}
          {!isPDF && !isImage && !isAudio && !isDocx && (
            <div className="docviewer-unsupported">
              <div style={{ fontSize: 40 }}>📁</div>
              <div style={{ marginTop: 12, fontSize: 13, color: "var(--text-secondary)" }}>
                Preview not available for this file type.
              </div>
              <a href={url} download={filename} className="btn btn-primary" style={{ marginTop: 16, display: "inline-flex" }}>
                ⬇ Download
              </a>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}