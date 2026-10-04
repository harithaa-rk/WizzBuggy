import { useState, useEffect, useCallback } from "react";

const API_URL = "http://127.0.0.1:8000";

const CLEARANCE_COLORS = {
  PUBLIC:       "#30d0b0",
  INTERNAL:     "#5b6af0",
  CONFIDENTIAL: "#f0a030",
  RESTRICTED:   "#e05080",
};

const CLEARANCE_ICONS = {
  PUBLIC:       "🌐",
  INTERNAL:     "🔵",
  CONFIDENTIAL: "🟡",
  RESTRICTED:   "🔴",
};

const ROLE_ICONS = {
  admin:    "🛡️",
  manager:  "🏢",
  employee: "👤",
  auditor:  "🔍",
};

function Section({ title, icon, children }) {
  return (
    <div className="admin-section">
      <div className="admin-section-header">
        <span className="admin-section-icon">{icon}</span>
        <span className="admin-section-title">{title}</span>
      </div>
      {children}
    </div>
  );
}

function StatCard({ icon, label, value, color }) {
  return (
    <div className="admin-stat-card" style={{ "--stat-color": color }}>
      <div className="admin-stat-icon">{icon}</div>
      <div className="admin-stat-value">{value}</div>
      <div className="admin-stat-label">{label}</div>
    </div>
  );
}

export default function AdminDashboard({ user, token, onClose, toast, onUserUpdate }) {
  const [tab, setTab]               = useState("overview");
  const [users, setUsers]           = useState([]);
  const [documents, setDocuments]   = useState([]);
  const [auditLogs, setAuditLogs]   = useState([]);
  const [loading, setLoading]       = useState({});
  const [editUser, setEditUser]     = useState(null);
  const [showCreateUser, setShowCreateUser] = useState(false);
  const [auditFilter, setAuditFilter] = useState({ action: "", user: "" });

  const authHeaders = {
    "Content-Type": "application/json",
    "Authorization": token,
  };

  const withLoading = (key, fn) => async (...args) => {
    setLoading(l => ({ ...l, [key]: true }));
    try { await fn(...args); }
    finally { setLoading(l => ({ ...l, [key]: false })); }
  };

  const fetchUsers = useCallback(async () => {
    try {
      const r = await fetch(`${API_URL}/admin/users`, { headers: authHeaders });
      if (r.ok) setUsers((await r.json()).users || []);
    } catch {}
  }, [token]);

  const fetchDocuments = useCallback(async () => {
    try {
      const r = await fetch(`${API_URL}/admin/documents`, { headers: authHeaders });
      if (r.ok) setDocuments((await r.json()).documents || []);
    } catch {}
  }, [token]);

  const fetchAuditLogs = useCallback(async () => {
    try {
      const params = new URLSearchParams({ limit: 200 });
      if (auditFilter.action) params.append("action", auditFilter.action);
      if (auditFilter.user) params.append("user", auditFilter.user);
      const r = await fetch(`${API_URL}/admin/audit-logs?${params}`, { headers: authHeaders });
      if (r.ok) setAuditLogs((await r.json()).logs || []);
    } catch {}
  }, [token, auditFilter]);

  useEffect(() => { fetchUsers(); fetchDocuments(); fetchAuditLogs(); }, [fetchUsers, fetchDocuments, fetchAuditLogs]);
  useEffect(() => { if (tab === "audit") fetchAuditLogs(); }, [tab, fetchAuditLogs]);

  const updateUserRole = withLoading("updateUser", async (username, patch) => {
    const r = await fetch(`${API_URL}/admin/users/${username}`, {
      method: "PATCH",
      headers: authHeaders,
      body: JSON.stringify(patch),
    });
    const data = await r.json();
    if (!r.ok) { toast?.(data.detail || "Update failed", "error"); return; }
    toast?.(`User '${username}' updated`, "success");
    setEditUser(null);
    fetchUsers();
  });

  const toggleUserActive = withLoading("toggleActive", async (username, is_active) => {
    await updateUserRole(username, { is_active });
  });

  const deleteUser = withLoading("deleteUser", async (username) => {
    if (!window.confirm(`Delete user '${username}'? This cannot be undone.`)) return;
    const r = await fetch(`${API_URL}/admin/users/${username}`, {
      method: "DELETE",
      headers: authHeaders,
    });
    const data = await r.json();
    if (!r.ok) { toast?.(data.detail || "Delete failed", "error"); return; }
    toast?.(`User '${username}' deleted`, "success");
    fetchUsers();
  });

  const updateDocClassification = withLoading("docClass", async (filename, classification) => {
    const r = await fetch(`${API_URL}/admin/documents/${encodeURIComponent(filename)}/classification`, {
      method: "PATCH",
      headers: authHeaders,
      body: JSON.stringify({ classification }),
    });
    const data = await r.json();
    if (!r.ok) { toast?.(data.detail || "Update failed", "error"); return; }
    toast?.(`${filename} → ${classification}`, "success");
    fetchDocuments();
  });

  const clearAuditLogs = withLoading("clearAudit", async () => {
    if (!window.confirm("Clear ALL audit logs? This cannot be undone.")) return;
    const r = await fetch(`${API_URL}/admin/audit-logs/clear`, {
      method: "POST",
      headers: authHeaders,
    });
    if (r.ok) { toast?.("Audit trail cleared", "info"); fetchAuditLogs(); }
  });

  const resetPassword = withLoading("resetPw", async (username, pw) => {
    const r = await fetch(`${API_URL}/admin/users/${username}/reset-password`, {
      method: "POST",
      headers: authHeaders,
      body: JSON.stringify({ new_password: pw }),
    });
    const data = await r.json();
    if (!r.ok) { toast?.(data.detail || "Reset failed", "error"); return; }
    toast?.(`Password reset for '${username}'`, "success");
  });

  // Stats
  const activeUsers = users.filter(u => u.is_active).length;
  const adminCount  = users.filter(u => u.role === "admin").length;
  const classMap    = documents.reduce((acc, d) => {
    acc[d.classification] = (acc[d.classification] || 0) + 1; return acc;
  }, {});
  const restrictedDocs = (classMap.RESTRICTED || 0) + (classMap.CONFIDENTIAL || 0);

  const TABS = [
    { id: "overview",   label: "Overview",   icon: "📊" },
    { id: "users",      label: "Users",      icon: "👥" },
    { id: "documents",  label: "Documents",  icon: "📁" },
    { id: "audit",      label: "Audit Trail",icon: "📋" },
  ];

  return (
    <div className="admin-overlay" onClick={e => e.target === e.currentTarget && onClose()}>
      <div className="admin-panel">
        {/* Header */}
        <div className="admin-panel-header">
          <div className="admin-panel-title">
            <span>🛡️</span>
            <div>
              <div style={{ fontFamily: "'Syne', sans-serif", fontWeight: 800, fontSize: 20 }}>
                Admin Dashboard
              </div>
              <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 2 }}>
                Organization Management · {user.full_name}
              </div>
            </div>
          </div>
          <button className="admin-close-btn" onClick={onClose}>✕</button>
        </div>

        {/* Tab bar */}
        <div className="admin-tabs">
          {TABS.map(t => (
            <button
              key={t.id}
              className={"admin-tab" + (tab === t.id ? " active" : "")}
              onClick={() => setTab(t.id)}
            >
              {t.icon} {t.label}
            </button>
          ))}
        </div>

        {/* Tab content */}
        <div className="admin-body">

          {/* OVERVIEW */}
          {tab === "overview" && (
            <div className="admin-overview">
              <div className="admin-stats-row">
                <StatCard icon="👥" label="Active Users"    value={activeUsers}                    color="var(--accent)" />
                <StatCard icon="📄" label="Documents"       value={documents.length}               color="var(--teal)" />
                <StatCard icon="🔴" label="Sensitive Docs"  value={restrictedDocs}                 color="var(--rose)" />
                <StatCard icon="📋" label="Audit Events"    value={auditLogs.length}               color="var(--amber)" />
                <StatCard icon="🛡️" label="Admins"          value={adminCount}                     color="var(--rose)" />
              </div>

              <div className="admin-overview-grid">
                <Section title="Clearance Matrix" icon="🔐">
                  <div className="admin-clearance-list">
                    {["PUBLIC", "INTERNAL", "CONFIDENTIAL", "RESTRICTED"].map(c => (
                      <div key={c} className="admin-clearance-row">
                        <span style={{ color: CLEARANCE_COLORS[c] }}>{CLEARANCE_ICONS[c]} {c}</span>
                        <span className="admin-clearance-count">
                          {classMap[c] || 0} doc{classMap[c] !== 1 ? "s" : ""}
                        </span>
                      </div>
                    ))}
                  </div>
                </Section>

                <Section title="Recent Security Events" icon="⚡">
                  <div className="admin-recent-audit">
                    {auditLogs.slice(0, 8).map(log => (
                      <div key={log.id} className={"admin-audit-row" + (log.status === "DENIED" ? " denied" : "")}>
                        <span className={"audit-status-dot " + (log.status === "SUCCESS" ? "ok" : "fail")} />
                        <div>
                          <span className="audit-action">{log.action}</span>
                          <span className="audit-user"> — {log.username}</span>
                        </div>
                        <span className="audit-time">
                          {new Date(log.timestamp * 1000).toLocaleTimeString()}
                        </span>
                      </div>
                    ))}
                    {auditLogs.length === 0 && (
                      <div className="admin-empty">No events yet.</div>
                    )}
                  </div>
                </Section>
              </div>
            </div>
          )}

          {/* USERS */}
          {tab === "users" && (
            <Section title="Organization Members" icon="👥">
              <div className="admin-users-toolbar">
                <button
                  className="btn btn-primary"
                  style={{ fontSize: 11 }}
                  onClick={() => setShowCreateUser(v => !v)}
                >
                  + Add User
                </button>
                <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
                  {users.length} member{users.length !== 1 ? "s" : ""}
                </span>
              </div>

              {showCreateUser && (
                <CreateUserForm
                  token={token}
                  onCreated={() => { fetchUsers(); setShowCreateUser(false); }}
                  toast={toast}
                />
              )}

              <div className="admin-users-table">
                <div className="admin-table-header">
                  <div>User</div>
                  <div>Role</div>
                  <div>Clearance</div>
                  <div>Department</div>
                  <div>Status</div>
                  <div>Actions</div>
                </div>
                {users.map(u => (
                  <div key={u.id} className={"admin-table-row" + (editUser?.username === u.username ? " editing" : "")}>
                    <div className="admin-user-cell">
                      <div className="admin-user-name">{u.full_name}</div>
                      <div className="admin-user-email">{u.email}</div>
                    </div>
                    <div>
                      {editUser?.username === u.username ? (
                        <select
                          className="admin-select"
                          value={editUser.role}
                          onChange={e => setEditUser(eu => ({ ...eu, role: e.target.value }))}
                        >
                          <option value="employee">Employee</option>
                          <option value="manager">Manager</option>
                          <option value="admin">Admin</option>
                          <option value="auditor">Auditor</option>
                        </select>
                      ) : (
                        <span className="admin-role-chip">
                          {ROLE_ICONS[u.role]} {u.role}
                        </span>
                      )}
                    </div>
                    <div>
                      {editUser?.username === u.username ? (
                        <select
                          className="admin-select"
                          value={editUser.clearance_level}
                          onChange={e => setEditUser(eu => ({ ...eu, clearance_level: e.target.value }))}
                        >
                          <option value="PUBLIC">PUBLIC</option>
                          <option value="INTERNAL">INTERNAL</option>
                          <option value="CONFIDENTIAL">CONFIDENTIAL</option>
                          <option value="RESTRICTED">RESTRICTED</option>
                        </select>
                      ) : (
                        <span className="admin-clearance-chip"
                          style={{ color: CLEARANCE_COLORS[u.clearance_level], borderColor: CLEARANCE_COLORS[u.clearance_level] }}>
                          {CLEARANCE_ICONS[u.clearance_level]} {u.clearance_level}
                        </span>
                      )}
                    </div>
                    <div style={{ fontSize: 12, color: "var(--text-secondary)" }}>
                      {editUser?.username === u.username ? (
                        <input
                          className="admin-input-sm"
                          value={editUser.department}
                          onChange={e => setEditUser(eu => ({ ...eu, department: e.target.value }))}
                        />
                      ) : (
                        u.department || "—"
                      )}
                    </div>
                    <div>
                      <span className={"admin-status-badge " + (u.is_active ? "active" : "inactive")}>
                        {u.is_active ? "Active" : "Inactive"}
                      </span>
                    </div>
                    <div className="admin-user-actions">
                      {editUser?.username === u.username ? (
                        <>
                          <button
                            className="admin-action-btn ok"
                            onClick={() => updateUserRole(u.username, {
                              role: editUser.role,
                              clearance_level: editUser.clearance_level,
                              department: editUser.department,
                            })}
                          >✓</button>
                          <button className="admin-action-btn cancel" onClick={() => setEditUser(null)}>✕</button>
                        </>
                      ) : (
                        <>
                          <button
                            className="admin-action-btn edit"
                            title="Edit user"
                            onClick={() => setEditUser({ ...u })}
                          >✏</button>
                          <button
                            className={"admin-action-btn " + (u.is_active ? "deactivate" : "activate")}
                            title={u.is_active ? "Deactivate" : "Activate"}
                            disabled={u.username === "admin"}
                            onClick={() => toggleUserActive(u.username, !u.is_active)}
                          >{u.is_active ? "🚫" : "✅"}</button>
                          <PasswordResetBtn username={u.username} onReset={resetPassword} />
                          {u.username !== "admin" && (
                            <button
                              className="admin-action-btn danger"
                              title="Delete user"
                              onClick={() => deleteUser(u.username)}
                            >🗑</button>
                          )}
                        </>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </Section>
          )}

          {/* DOCUMENTS */}
          {tab === "documents" && (
            <Section title="Document Confidentiality Matrix" icon="📁">
              <div className="admin-docs-table">
                <div className="admin-table-header">
                  <div>Filename</div>
                  <div>Classification</div>
                  <div>Chunks</div>
                  <div>Uploaded By</div>
                  <div>Department</div>
                  <div>Change</div>
                </div>
                {documents.map(doc => (
                  <div key={doc.filename} className="admin-table-row">
                    <div className="admin-doc-name" title={doc.filename}>
                      {doc.filename.length > 32 ? doc.filename.slice(0, 30) + "…" : doc.filename}
                    </div>
                    <div>
                      <span
                        className="admin-clearance-chip"
                        style={{ color: CLEARANCE_COLORS[doc.classification], borderColor: CLEARANCE_COLORS[doc.classification] }}
                      >
                        {CLEARANCE_ICONS[doc.classification]} {doc.classification}
                      </span>
                    </div>
                    <div style={{ color: "var(--text-secondary)", fontSize: 12 }}>{doc.chunks}</div>
                    <div style={{ fontSize: 12, color: "var(--text-secondary)" }}>{doc.uploaded_by}</div>
                    <div style={{ fontSize: 12, color: "var(--text-muted)" }}>{doc.department}</div>
                    <div>
                      <ClassificationSelector
                        filename={doc.filename}
                        current={doc.classification}
                        onChange={updateDocClassification}
                        loading={loading.docClass}
                      />
                    </div>
                  </div>
                ))}
                {documents.length === 0 && (
                  <div className="admin-empty">No documents indexed yet.</div>
                )}
              </div>
            </Section>
          )}

          {/* AUDIT TRAIL */}
          {tab === "audit" && (
            <Section title="Security Audit Trail" icon="📋">
              <div className="admin-audit-toolbar">
                <input
                  className="admin-input-sm"
                  placeholder="Filter by user..."
                  value={auditFilter.user}
                  onChange={e => setAuditFilter(f => ({ ...f, user: e.target.value }))}
                  style={{ width: 180 }}
                />
                <select
                  className="admin-select"
                  value={auditFilter.action}
                  onChange={e => setAuditFilter(f => ({ ...f, action: e.target.value }))}
                >
                  <option value="">All actions</option>
                  <option value="LOGIN">LOGIN</option>
                  <option value="LOGOUT">LOGOUT</option>
                  <option value="REGISTER">REGISTER</option>
                  <option value="INGEST">INGEST</option>
                  <option value="QUERY">QUERY</option>
                  <option value="DELETE_FILE">DELETE_FILE</option>
                  <option value="SET_CLASSIFICATION">SET_CLASSIFICATION</option>
                  <option value="ACCESS_DENIED">ACCESS_DENIED</option>
                  <option value="DOCUMENT_ACCESS_DENIED">DOCUMENT_ACCESS_DENIED</option>
                  <option value="UPDATE_USER">UPDATE_USER</option>
                  <option value="DELETE_USER">DELETE_USER</option>
                </select>
                <button className="btn btn-ghost" style={{ fontSize: 11 }} onClick={fetchAuditLogs}>
                  🔄 Refresh
                </button>
                {user.role === "admin" && (
                  <button
                    className="btn btn-danger"
                    style={{ fontSize: 11, marginLeft: "auto" }}
                    onClick={clearAuditLogs}
                    disabled={loading.clearAudit}
                  >
                    🗑 Clear Logs
                  </button>
                )}
              </div>

              <div className="admin-audit-table">
                <div className="admin-table-header">
                  <div>Time</div>
                  <div>User</div>
                  <div>Action</div>
                  <div>Status</div>
                  <div>Details</div>
                </div>
                {auditLogs.map(log => (
                  <div key={log.id} className={"admin-table-row audit-entry" + (log.status !== "SUCCESS" ? " denied" : "")}>
                    <div className="audit-time-cell">
                      {new Date(log.timestamp * 1000).toLocaleString()}
                    </div>
                    <div className="audit-user-cell">{log.username}</div>
                    <div className="audit-action-cell">{log.action}</div>
                    <div>
                      <span className={"audit-badge " + (log.status === "SUCCESS" ? "ok" : "fail")}>
                        {log.status}
                      </span>
                    </div>
                    <div className="audit-details-cell" title={log.details}>{log.details}</div>
                  </div>
                ))}
                {auditLogs.length === 0 && (
                  <div className="admin-empty">No audit events match your filters.</div>
                )}
              </div>
            </Section>
          )}
        </div>
      </div>
    </div>
  );
}

/* Sub-components */

function ClassificationSelector({ filename, current, onChange, loading }) {
  const [val, setVal] = useState(current);
  const changed = val !== current;
  return (
    <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
      <select
        className="admin-select"
        value={val}
        onChange={e => setVal(e.target.value)}
      >
        <option value="PUBLIC">PUBLIC</option>
        <option value="INTERNAL">INTERNAL</option>
        <option value="CONFIDENTIAL">CONFIDENTIAL</option>
        <option value="RESTRICTED">RESTRICTED</option>
      </select>
      {changed && (
        <button
          className="admin-action-btn ok"
          disabled={loading}
          onClick={() => onChange(filename, val)}
        >✓</button>
      )}
    </div>
  );
}

function PasswordResetBtn({ username, onReset }) {
  const [open, setOpen] = useState(false);
  const [pw, setPw]     = useState("");
  return open ? (
    <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
      <input
        className="admin-input-sm"
        type="password"
        placeholder="New password"
        value={pw}
        onChange={e => setPw(e.target.value)}
        style={{ width: 120 }}
      />
      <button
        className="admin-action-btn ok"
        disabled={pw.length < 6}
        onClick={() => { onReset(username, pw); setOpen(false); setPw(""); }}
      >✓</button>
      <button className="admin-action-btn cancel" onClick={() => setOpen(false)}>✕</button>
    </div>
  ) : (
    <button className="admin-action-btn" title="Reset password" onClick={() => setOpen(true)}>🔑</button>
  );
}

function CreateUserForm({ token, onCreated, toast }) {
  const [form, setForm] = useState({
    username: "", email: "", password: "", full_name: "",
    department: "General", role: "employee", clearance_level: "INTERNAL",
  });
  const [loading, setLoading] = useState(false);

  const submit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const r = await fetch(`${API_URL}/admin/users/create`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "Authorization": token },
        body: JSON.stringify(form),
      });
      const data = await r.json();
      if (!r.ok) { toast?.(data.detail || "Failed", "error"); return; }
      toast?.(`User '${form.username}' created`, "success");
      onCreated();
    } catch (e) {
      toast?.(e.message, "error");
    } finally {
      setLoading(false);
    }
  };

  const f = (field) => ({
    value: form[field],
    onChange: e => setForm(prev => ({ ...prev, [field]: e.target.value })),
  });

  return (
    <form onSubmit={submit} className="admin-create-user-form">
      <div className="admin-create-user-grid">
        <div><label className="auth-label">Full Name</label><input className="admin-input-sm" placeholder="Jane Smith" required {...f("full_name")} /></div>
        <div><label className="auth-label">Username</label><input className="admin-input-sm" placeholder="janesmith" required {...f("username")} /></div>
        <div><label className="auth-label">Email</label><input className="admin-input-sm" type="email" placeholder="jane@org.local" required {...f("email")} /></div>
        <div><label className="auth-label">Password</label><input className="admin-input-sm" type="password" placeholder="min 6 chars" required {...f("password")} /></div>
        <div><label className="auth-label">Department</label><input className="admin-input-sm" placeholder="Engineering" {...f("department")} /></div>
        <div>
          <label className="auth-label">Role</label>
          <select className="admin-select" {...f("role")}>
            <option value="employee">Employee</option>
            <option value="manager">Manager</option>
            <option value="admin">Admin</option>
            <option value="auditor">Auditor</option>
          </select>
        </div>
        <div>
          <label className="auth-label">Clearance</label>
          <select className="admin-select" {...f("clearance_level")}>
            <option value="PUBLIC">PUBLIC</option>
            <option value="INTERNAL">INTERNAL</option>
            <option value="CONFIDENTIAL">CONFIDENTIAL</option>
            <option value="RESTRICTED">RESTRICTED</option>
          </select>
        </div>
      </div>
      <button type="submit" className="btn btn-primary" style={{ fontSize: 11 }} disabled={loading}>
        {loading ? "Creating…" : "✓ Create User"}
      </button>
    </form>
  );
}
