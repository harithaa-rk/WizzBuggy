import { useState } from "react";

const API_URL = "http://127.0.0.1:8000";

const ROLES = [
  { value: "employee", label: "Employee", icon: "👤", desc: "Standard access · INTERNAL clearance" },
  { value: "manager",  label: "Manager",  icon: "🏢", desc: "Elevated access · CONFIDENTIAL clearance" },
  { value: "admin",    label: "Admin",    icon: "🛡️", desc: "Full access · RESTRICTED clearance" },
  { value: "auditor",  label: "Auditor",  icon: "🔍", desc: "Read-only · INTERNAL clearance, audit logs" },
];

export default function AuthPage({ onAuth, toast }) {
  const [mode, setMode]         = useState("login"); // login | register
  const [loading, setLoading]   = useState(false);
  const [showPw, setShowPw]     = useState(false);
  const [error, setError]       = useState("");

  // Login fields
  const [loginUser, setLoginUser] = useState("");
  const [loginPw,   setLoginPw]   = useState("");

  // Register fields
  const [regName,   setRegName]   = useState("");
  const [regUser,   setRegUser]   = useState("");
  const [regEmail,  setRegEmail]  = useState("");
  const [regPw,     setRegPw]     = useState("");
  const [regDept,   setRegDept]   = useState("");
  const [regRole,   setRegRole]   = useState("employee");

  const doLogin = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const res = await fetch(`${API_URL}/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username: loginUser, password: loginPw }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Login failed");
      onAuth(data.user);
      toast?.(`Welcome back, ${data.user.full_name}! (${data.user.role})`, "success");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const doRegister = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const res = await fetch(`${API_URL}/auth/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          username: regUser,
          email: regEmail,
          password: regPw,
          full_name: regName,
          department: regDept || "General",
          role: regRole,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Registration failed");
      onAuth(data.user);
      toast?.(`Account created! Welcome, ${data.user.user?.full_name || regName}`, "success");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-page">
      {/* Animated background grid */}
      <div className="auth-bg-grid" aria-hidden="true">
        {Array.from({ length: 64 }).map((_, i) => (
          <div key={i} className="auth-bg-cell" style={{ animationDelay: `${(i * 0.07) % 4}s` }} />
        ))}
      </div>

      <div className="auth-container">
        {/* Logo */}
        <div className="auth-logo">
          <span className="auth-logo-nova">Nova</span>
          <span className="auth-logo-rag">RAG</span>
          <span className="auth-logo-badge">Enterprise</span>
        </div>

        <p className="auth-tagline">
          Confidential Knowledge Intelligence — secure, role-gated, multi-modal.
        </p>

        {/* Mode tabs */}
        <div className="auth-tabs">
          <button
            className={"auth-tab" + (mode === "login" ? " active" : "")}
            onClick={() => { setMode("login"); setError(""); }}
          >🔑 Sign In</button>
          <button
            className={"auth-tab" + (mode === "register" ? " active" : "")}
            onClick={() => { setMode("register"); setError(""); }}
          >+ Register</button>
        </div>

        {/* Card */}
        <div className="auth-card">
          {error && (
            <div className="auth-error">
              <span>⚠</span> {error}
            </div>
          )}

          {mode === "login" ? (
            <form onSubmit={doLogin} className="auth-form">
              <div className="auth-field">
                <label className="auth-label">Username or Email</label>
                <input
                  className="auth-input"
                  type="text"
                  value={loginUser}
                  onChange={e => setLoginUser(e.target.value)}
                  placeholder="admin"
                  required
                  autoFocus
                />
              </div>

              <div className="auth-field">
                <label className="auth-label">Password</label>
                <div className="auth-pw-wrap">
                  <input
                    className="auth-input"
                    type={showPw ? "text" : "password"}
                    value={loginPw}
                    onChange={e => setLoginPw(e.target.value)}
                    placeholder="••••••••"
                    required
                  />
                  <button
                    type="button"
                    className="auth-pw-toggle"
                    onClick={() => setShowPw(v => !v)}
                    tabIndex={-1}
                  >{showPw ? "🙈" : "👁️"}</button>
                </div>
              </div>

              <button
                type="submit"
                className="auth-btn-primary"
                disabled={loading || !loginUser || !loginPw}
              >
                {loading ? <span className="auth-spinner" /> : "🔐 Sign In"}
              </button>

              {/* Default accounts hint */}
              <div className="auth-hint">
                <div className="auth-hint-title">Default accounts</div>
                <div className="auth-hint-row"><code>admin</code> / <code>Admin@1234</code> — RESTRICTED</div>
                <div className="auth-hint-row"><code>manager</code> / <code>Manager@1234</code> — CONFIDENTIAL</div>
                <div className="auth-hint-row"><code>employee</code> / <code>Employee@1234</code> — INTERNAL</div>
              </div>
            </form>
          ) : (
            <form onSubmit={doRegister} className="auth-form">
              <div className="auth-field-row">
                <div className="auth-field">
                  <label className="auth-label">Full Name</label>
                  <input
                    className="auth-input"
                    type="text"
                    value={regName}
                    onChange={e => setRegName(e.target.value)}
                    placeholder="Jane Smith"
                    required
                    autoFocus
                  />
                </div>
                <div className="auth-field">
                  <label className="auth-label">Username</label>
                  <input
                    className="auth-input"
                    type="text"
                    value={regUser}
                    onChange={e => setRegUser(e.target.value)}
                    placeholder="janesmith"
                    required
                  />
                </div>
              </div>

              <div className="auth-field-row">
                <div className="auth-field">
                  <label className="auth-label">Email</label>
                  <input
                    className="auth-input"
                    type="email"
                    value={regEmail}
                    onChange={e => setRegEmail(e.target.value)}
                    placeholder="jane@org.local"
                    required
                  />
                </div>
                <div className="auth-field">
                  <label className="auth-label">Department</label>
                  <input
                    className="auth-input"
                    type="text"
                    value={regDept}
                    onChange={e => setRegDept(e.target.value)}
                    placeholder="Engineering"
                  />
                </div>
              </div>

              <div className="auth-field">
                <label className="auth-label">Password</label>
                <div className="auth-pw-wrap">
                  <input
                    className="auth-input"
                    type={showPw ? "text" : "password"}
                    value={regPw}
                    onChange={e => setRegPw(e.target.value)}
                    placeholder="Min 6 characters"
                    required
                  />
                  <button
                    type="button"
                    className="auth-pw-toggle"
                    onClick={() => setShowPw(v => !v)}
                    tabIndex={-1}
                  >{showPw ? "🙈" : "👁️"}</button>
                </div>
              </div>

              <div className="auth-field">
                <label className="auth-label">Role</label>
                <div className="auth-role-grid">
                  {ROLES.map(r => (
                    <button
                      key={r.value}
                      type="button"
                      className={"auth-role-card" + (regRole === r.value ? " selected" : "")}
                      onClick={() => setRegRole(r.value)}
                    >
                      <span className="auth-role-icon">{r.icon}</span>
                      <div>
                        <div className="auth-role-name">{r.label}</div>
                        <div className="auth-role-desc">{r.desc}</div>
                      </div>
                    </button>
                  ))}
                </div>
              </div>

              <button
                type="submit"
                className="auth-btn-primary"
                disabled={loading || !regName || !regUser || !regEmail || !regPw}
              >
                {loading ? <span className="auth-spinner" /> : "🚀 Create Account"}
              </button>
            </form>
          )}
        </div>

        {/* Security badge */}
        <div className="auth-security-row">
          <span className="auth-badge">🔒 AES-256 encrypted</span>
          <span className="auth-badge">🛡️ PBKDF2 auth</span>
          <span className="auth-badge">📋 Full audit trail</span>
        </div>
      </div>
    </div>
  );
}
