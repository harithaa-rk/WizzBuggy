import { useState } from "react";

const MODELS = [
  { label: "Fast",     value: "fast",    desc: "Mistral quick mode" },
  { label: "Smart",    value: "mistral", desc: "Mistral deep mode"  },
  { label: "Creative", value: "llama3",  desc: "Llama3"             },
];

const CLASSIFICATIONS = [
  { value: "PUBLIC",       label: "🌐 Public",       color: "#30d0b0" },
  { value: "INTERNAL",     label: "🔵 Internal",     color: "#5b6af0" },
  { value: "CONFIDENTIAL", label: "🟡 Confidential", color: "#f0a030" },
  { value: "RESTRICTED",   label: "🔴 Restricted",   color: "#e05080" },
];

// Global state carriers (read by ChatPanel via DOM)
let _model       = "fast";
let _k           = 8;
let _fileFilter  = "";   // comma-separated filenames

export const getModel      = () => _model;
export const getK          = () => _k;
export const getFileFilter = () => _fileFilter;

export default function IngestPanel({ apiUrl, onIngested, toast, activeFiles, onFilterChange, token, userRole }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [model, setModel]               = useState("fast");
  const [kValue, setKValue]             = useState(8);
  const [loading, setLoading]           = useState(false);
  const [dragOver, setDragOver]         = useState(false);
  const [classification, setClassification] = useState("INTERNAL");

  const handleFile = (file) => { if (file) setSelectedFile(file); };

  const ingest = async () => {
    if (!selectedFile) return;
    setLoading(true);
    try {
      const form = new FormData();
      form.append("file", selectedFile);
      form.append("classification", classification);

      const headers = {};
      if (token) headers["Authorization"] = token;

      const res  = await fetch(`${apiUrl}/ingest`, { method: "POST", body: form, headers });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "HTTP " + res.status);
      }
      const data = await res.json();
      toast("✓ " + selectedFile.name + " — " + (data.chunks ?? "?") + " chunks indexed (" + (data.classification || classification) + ")", "success");
      onIngested();
      setSelectedFile(null);
    } catch (e) {
      toast("Failed to ingest: " + e.message, "error");
    } finally {
      setLoading(false);
    }
  };

  const [collapsed, setCollapsed]       = useState(false);

  const updateModel = (v) => { _model = v; setModel(v); };
  const updateK     = (v) => { _k = v;     setKValue(v); };

  return (
    <div className={"ingest-panel" + (collapsed ? " collapsed" : "")}>
      <div className="ingest-header-bar">
        <span className="ingest-header-title">
          📥 Ingest & Model Config
          {activeFiles && activeFiles.length > 0 && (
            <span className="ingest-file-count">({activeFiles.length} files indexed)</span>
          )}
        </span>
        <button
          type="button"
          className="ingest-toggle-btn"
          onClick={() => setCollapsed(!collapsed)}
          title={collapsed ? "Expand panel" : "Collapse panel"}
        >
          {collapsed ? "Expand ▼" : "Collapse ▲"}
        </button>
      </div>

      {!collapsed && (
        <>
          <div className="ingest-row">
        <div
          className={"drop-zone" + (dragOver ? " drag-over" : "")}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => { e.preventDefault(); setDragOver(false); handleFile(e.dataTransfer.files[0]); }}
        >
          <input
            type="file"
            accept=".pdf,.docx,.png,.jpg,.jpeg,.wav,.mp3,.m4a"
            onChange={(e) => handleFile(e.target.files[0])}
          />
          <span className="drop-icon">⬆</span>
          <div>
            {selectedFile ? (
              <div className="drop-selected">
                {"📎 " + selectedFile.name + " (" + (selectedFile.size / 1024 / 1024).toFixed(2) + " MB)"}
              </div>
            ) : (
              <>
                <div className="drop-text">Drop file or click to browse</div>
                <div className="drop-hint">PDF · DOCX · PNG · JPG · WAV · MP3 · M4A</div>
              </>
            )}
          </div>
        </div>
        <button className="btn btn-primary" onClick={ingest} disabled={!selectedFile || loading}>
          {loading ? "Indexing…" : "Ingest"}
        </button>
      </div>

      {loading && <div className="ingest-progress"><div className="ingest-progress-fill" /></div>}

      {/* Classification selector */}
      <div className="classification-row">
        <span className="model-label">Classification</span>
        {CLASSIFICATIONS.map((c) => (
          <button
            key={c.value}
            className={"model-btn" + (classification === c.value ? " active" : "")}
            onClick={() => setClassification(c.value)}
            title={c.label}
            style={classification === c.value ? { borderColor: c.color, color: c.color } : {}}
          >
            {c.label}
          </button>
        ))}
      </div>

      {/* Model + depth */}
      <div className="model-row">
        <span className="model-label">Model</span>
        {MODELS.map((m) => (
          <button
            key={m.value}
            className={"model-btn" + (model === m.value ? " active" : "")}
            onClick={() => updateModel(m.value)}
            title={m.desc}
          >
            {m.label}
          </button>
        ))}
        <span style={{ flex: 1 }} />
        <div className="depth-row">
          <span className="depth-label">Depth</span>
          <input
            type="range" className="depth-slider"
            min={3} max={15} value={kValue}
            onChange={(e) => updateK(Number(e.target.value))}
          />
          <span className="depth-val">{kValue}</span>
        </div>
      </div>

      {/* Multi-file filter */}
      {activeFiles && activeFiles.length > 1 && (
        <FileFilter files={activeFiles} onFilterChange={onFilterChange} />
      )}
        </>
      )}
    </div>
  );
}

/* ── Multi-file filter checkboxes ── */
function FileFilter({ files, onFilterChange }) {
  const [selected, setSelected] = useState(new Set());

  const toggle = (fname) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(fname)) next.delete(fname);
      else next.add(fname);
      const filter = [...next].join(",");
      _fileFilter = filter;
      onFilterChange(filter);
      return next;
    });
  };

  const clearAll = () => {
    setSelected(new Set());
    _fileFilter = "";
    onFilterChange("");
  };

  const getIcon = (f) => {
    const ext = f.split(".").pop()?.toLowerCase();
    return { pdf:"📄", docx:"📝", png:"🖼", jpg:"🖼", jpeg:"🖼", wav:"🎵", mp3:"🎵", m4a:"🎵" }[ext] || "📁";
  };

  return (
    <div className="file-filter-bar">
      <span className="file-filter-label">
        Filter by file
        {selected.size > 0 && (
          <button className="filter-clear-btn" onClick={clearAll}>
            clear
          </button>
        )}
      </span>
      <div className="file-filter-chips">
        {files.map((f) => (
          <button
            key={f.file}
            className={"filter-chip" + (selected.has(f.file) ? " active" : "")}
            onClick={() => toggle(f.file)}
            title={f.file}
          >
            {getIcon(f.file)} {f.file.length > 20 ? f.file.slice(0, 18) + "…" : f.file}
          </button>
        ))}
      </div>
    </div>
  );
}