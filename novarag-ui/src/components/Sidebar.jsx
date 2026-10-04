export default function Sidebar({
  open, files, onClear, onDeleteFile,
  sessions, activeId, onNewSession, onSwitch, onDelete,
}) {
  const getIcon = (filename) => {
    const ext = filename?.split(".").pop()?.toLowerCase();
    return { pdf:"📄", docx:"📝", png:"🖼", jpg:"🖼", jpeg:"🖼", wav:"🎵", mp3:"🎵", m4a:"🎵" }[ext] || "📁";
  };

  const formatDate = (ts) => new Date(ts).toLocaleDateString(undefined, {
    month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
  });

  return (
    <aside className={"sidebar" + (open ? "" : " closed")}>
      <div className="sidebar-header">
        <span className="sidebar-title">Indexed Files</span>
        <span style={{ fontSize: 11, color: "var(--accent)" }}>{files.length}</span>
      </div>

      <div className="sidebar-files">
        {files.length === 0 ? (
          <div className="no-files">No files indexed yet.</div>
        ) : (
          files.map((f) => (
            <div className="file-item" key={f.file}>
              <span className="file-icon">{getIcon(f.file)}</span>
              <div className="file-info">
                <div className="file-name" title={f.file}>{f.file}</div>
                <div className="file-chunks">{f.chunks} chunks</div>
              </div>
              <button
                className="file-del-btn"
                title={"Delete " + f.file}
                onClick={() => onDeleteFile(f.file)}
              >
                ✕
              </button>
            </div>
          ))
        )}
      </div>

      {/* Session history */}
      <div className="session-section">
        <div className="session-section-title">
          <span>Sessions</span>
          <span style={{ color: "var(--accent)" }}>{sessions.length}</span>
        </div>
        {sessions.map((s) => (
          <div
            key={s.id}
            className={"session-item" + (s.id === activeId ? " active" : "")}
            onClick={() => onSwitch(s.id)}
          >
            <span className="session-icon">💬</span>
            <div className="session-info">
              <div className="session-name" title={s.name}>{s.name}</div>
              <div className="session-meta">
                {formatDate(s.createdAt)} · {s.messages?.length || 0} msgs
              </div>
            </div>
            <button
              className="session-del"
              onClick={(e) => { e.stopPropagation(); onDelete(s.id); }}
            >✕</button>
          </div>
        ))}
        <button className="new-session-btn" onClick={onNewSession}>
          + New session
        </button>
      </div>

      <div className="sidebar-footer">
        <button className="btn btn-danger btn-full" onClick={onClear}>
          🧹 Clear Conversation
        </button>
      </div>
    </aside>
  );
}