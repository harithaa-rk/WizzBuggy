import sqlite3
import hashlib
import secrets
import time
import os
import sys
from typing import Optional, List, Dict, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

DB_PATH = "novarag_memory.db"

# Clearance hierarchy: higher rank can access equal or lower rank documents
CLEARANCE_RANKS = {
    "PUBLIC": 1,
    "INTERNAL": 2,
    "CONFIDENTIAL": 3,
    "RESTRICTED": 4,
}

# Default clearance assigned to roles if not explicitly overridden
DEFAULT_ROLE_CLEARANCE = {
    "admin": "RESTRICTED",
    "manager": "CONFIDENTIAL",
    "employee": "INTERNAL",
    "auditor": "INTERNAL",
}

# Maximum classification a role is allowed to ingest
MAX_INGEST_CLEARANCE = {
    "admin": "RESTRICTED",
    "manager": "CONFIDENTIAL",
    "employee": "INTERNAL",
    "auditor": None,  # Auditors cannot ingest files
}

TOKEN_EXPIRATION_SECONDS = 7 * 24 * 3600  # 7 days


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
    """Hashes a password using PBKDF2-HMAC-SHA256 with 260,000 iterations."""
    if salt is None:
        salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        260000
    )
    return dk.hex(), salt


def verify_password(password: str, password_hash: str, salt: str) -> bool:
    """Verifies a password against the stored hash and salt."""
    test_hash, _ = hash_password(password, salt)
    return secrets.compare_digest(test_hash, password_hash)


def init_auth_db():
    """Initializes tables for RBAC, user management, confidentiality classifications, and audit logs."""
    with _conn() as c:
        # Users table
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                username        TEXT UNIQUE NOT NULL COLLATE NOCASE,
                email           TEXT UNIQUE NOT NULL COLLATE NOCASE,
                full_name       TEXT NOT NULL,
                password_hash   TEXT NOT NULL,
                salt            TEXT NOT NULL,
                role            TEXT NOT NULL DEFAULT 'employee',
                clearance_level TEXT NOT NULL DEFAULT 'INTERNAL',
                department      TEXT DEFAULT 'General',
                is_active       INTEGER NOT NULL DEFAULT 1,
                created_at      REAL NOT NULL,
                last_login      REAL
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")

        # Document confidentiality classifications
        c.execute("""
            CREATE TABLE IF NOT EXISTS document_classifications (
                filename        TEXT PRIMARY KEY COLLATE NOCASE,
                classification  TEXT NOT NULL DEFAULT 'INTERNAL',
                uploaded_by     TEXT NOT NULL,
                department      TEXT DEFAULT 'General',
                uploaded_at     REAL NOT NULL,
                description     TEXT DEFAULT ''
            )
        """)

        # Security & Confidentiality Audit Trail
        c.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       REAL NOT NULL,
                username        TEXT NOT NULL,
                action          TEXT NOT NULL,
                details         TEXT,
                status          TEXT NOT NULL,
                ip_address      TEXT
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_audit_time ON audit_logs(timestamp DESC)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_logs(username)")

        # User authentication tokens / sessions
        c.execute("""
            CREATE TABLE IF NOT EXISTS user_tokens (
                token           TEXT PRIMARY KEY,
                username        TEXT NOT NULL,
                created_at      REAL NOT NULL,
                expires_at      REAL NOT NULL
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_token_user ON user_tokens(username)")

        c.commit()

    # Pre-seed default organization accounts if no users exist
    _seed_default_accounts()


def _seed_default_accounts():
    """Creates default Admin, Manager, and Employee accounts if no accounts exist."""
    with _conn() as c:
        count = c.execute("SELECT COUNT(*) AS cnt FROM users").fetchone()["cnt"]
        if count == 0:
            print("🛡️ Initializing default enterprise organization accounts...")
            
            # Master Administrator
            admin_hash, admin_salt = hash_password("Admin@1234")
            c.execute("""
                INSERT INTO users (username, email, full_name, password_hash, salt, role, clearance_level, department, is_active, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
            """, ("admin", "admin@organization.local", "Security Administrator", admin_hash, admin_salt, "admin", "RESTRICTED", "Cybersecurity & Exec", time.time()))

            # Demo Operations Manager
            mgr_hash, mgr_salt = hash_password("Manager@1234")
            c.execute("""
                INSERT INTO users (username, email, full_name, password_hash, salt, role, clearance_level, department, is_active, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
            """, ("manager", "manager@organization.local", "Operations Manager", mgr_hash, mgr_salt, "manager", "CONFIDENTIAL", "Engineering & Operations", time.time()))

            # Demo Staff Analyst / Employee
            emp_hash, emp_salt = hash_password("Employee@1234")
            c.execute("""
                INSERT INTO users (username, email, full_name, password_hash, salt, role, clearance_level, department, is_active, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
            """, ("employee", "employee@organization.local", "Staff Analyst", emp_hash, emp_salt, "employee", "INTERNAL", "Data & Research", time.time()))

            c.commit()
            print("✅ Default organization accounts seeded: admin (RESTRICTED), manager (CONFIDENTIAL), employee (INTERNAL).")


# ── Audit Trail ─────────────────────────────────────────────────────────────

def log_audit(username: str, action: str, details: str = "", status: str = "SUCCESS", ip: str = "127.0.0.1"):
    """Records an access attempt or security event to the audit trail."""
    try:
        with _conn() as c:
            c.execute("""
                INSERT INTO audit_logs (timestamp, username, action, details, status, ip_address)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (time.time(), username, action, details, status, ip))
            c.commit()
    except Exception as e:
        print("Audit logging error:", e)


def get_audit_logs(limit: int = 150, action_filter: Optional[str] = None, user_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetches security audit events ordered by most recent first."""
    query = "SELECT id, timestamp, username, action, details, status, ip_address FROM audit_logs"
    params = []
    clauses = []
    if action_filter:
        clauses.append("action = ?")
        params.append(action_filter)
    if user_filter:
        clauses.append("username = ?")
        params.append(user_filter)

    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    with _conn() as c:
        rows = c.execute(query, params).fetchall()
        return [dict(r) for r in rows]


def clear_audit_logs():
    """Clears all audit logs (admin-only action)."""
    with _conn() as c:
        c.execute("DELETE FROM audit_logs")
        c.commit()


# ── Authentication & Token Management ───────────────────────────────────────

def register_user(
    username: str,
    email: str,
    password: str,
    full_name: str,
    department: str = "General",
    role: str = "employee",
    clearance_level: Optional[str] = None
) -> Dict[str, Any]:
    """Registers a new user account."""
    username = username.strip().lower()
    email = email.strip().lower()
    full_name = full_name.strip()
    department = department.strip() or "General"
    role = role.lower()

    if role not in DEFAULT_ROLE_CLEARANCE:
        role = "employee"
    if not clearance_level or clearance_level not in CLEARANCE_RANKS:
        clearance_level = DEFAULT_ROLE_CLEARANCE.get(role, "INTERNAL")

    if len(username) < 3:
        raise ValueError("Username must be at least 3 characters.")
    if len(password) < 6:
        raise ValueError("Password must be at least 6 characters.")

    with _conn() as c:
        existing = c.execute("SELECT id FROM users WHERE username = ? OR email = ?", (username, email)).fetchone()
        if existing:
            raise ValueError("Username or email already registered.")

        p_hash, salt = hash_password(password)
        now = time.time()
        c.execute("""
            INSERT INTO users (username, email, full_name, password_hash, salt, role, clearance_level, department, is_active, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
        """, (username, email, full_name, p_hash, salt, role, clearance_level, department, now))
        c.commit()

    log_audit(username, "REGISTER", f"Account registered with role: {role}, clearance: {clearance_level}", "SUCCESS")
    return get_user_by_username(username)


def authenticate_user(username_or_email: str, password: str, ip: str = "127.0.0.1") -> Dict[str, Any]:
    """Authenticates credentials, generates an active session token, and updates last_login."""
    query_str = username_or_email.strip().lower()

    with _conn() as c:
        row = c.execute("""
            SELECT id, username, email, full_name, password_hash, salt, role, clearance_level, department, is_active
            FROM users WHERE username = ? OR email = ?
        """, (query_str, query_str)).fetchone()

    if not row:
        log_audit(query_str, "LOGIN", "Account not found", "DENIED", ip)
        raise ValueError("Invalid username or password.")

    user = dict(row)
    if not user["is_active"]:
        log_audit(user["username"], "LOGIN", "Account is deactivated", "DENIED", ip)
        raise ValueError("Account is deactivated. Contact an organization administrator.")

    if not verify_password(password, user["password_hash"], user["salt"]):
        log_audit(user["username"], "LOGIN", "Incorrect password", "DENIED", ip)
        raise ValueError("Invalid username or password.")

    # Generate token
    token = secrets.token_urlsafe(36)
    now = time.time()
    expires_at = now + TOKEN_EXPIRATION_SECONDS

    with _conn() as c:
        c.execute("""
            INSERT INTO user_tokens (token, username, created_at, expires_at)
            VALUES (?, ?, ?, ?)
        """, (token, user["username"], now, expires_at))
        c.execute("UPDATE users SET last_login = ? WHERE id = ?", (now, user["id"]))
        c.commit()

    log_audit(user["username"], "LOGIN", f"Successful login ({user['role']}, clearance: {user['clearance_level']})", "SUCCESS", ip)

    del user["password_hash"]
    del user["salt"]
    user["token"] = token
    return user


def get_user_from_token(token: str) -> Optional[Dict[str, Any]]:
    """Retrieves the user associated with a valid token, or None if expired/invalid."""
    if not token:
        return None
    token = token.strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()

    now = time.time()
    with _conn() as c:
        t_row = c.execute("SELECT username, expires_at FROM user_tokens WHERE token = ?", (token,)).fetchone()
        if not t_row:
            return None
        if t_row["expires_at"] < now:
            c.execute("DELETE FROM user_tokens WHERE token = ?", (token,))
            c.commit()
            return None

        u_row = c.execute("""
            SELECT id, username, email, full_name, role, clearance_level, department, is_active, created_at, last_login
            FROM users WHERE username = ?
        """, (t_row["username"],)).fetchone()

    if not u_row or not u_row["is_active"]:
        return None

    return dict(u_row)


def revoke_token(token: str):
    """Revokes a session token upon logout."""
    if not token:
        return
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    with _conn() as c:
        c.execute("DELETE FROM user_tokens WHERE token = ?", (token,))
        c.commit()


# ── User Administration ──────────────────────────────────────────────────────

def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    """Returns safe user dict without password credentials."""
    with _conn() as c:
        row = c.execute("""
            SELECT id, username, email, full_name, role, clearance_level, department, is_active, created_at, last_login
            FROM users WHERE username = ?
        """, (username.strip().lower(),)).fetchone()
    return dict(row) if row else None


def list_all_users() -> List[Dict[str, Any]]:
    """Lists all registered organization users for the admin dashboard."""
    with _conn() as c:
        rows = c.execute("""
            SELECT id, username, email, full_name, role, clearance_level, department, is_active, created_at, last_login
            FROM users ORDER BY created_at ASC
        """).fetchall()
        return [dict(r) for r in rows]


def update_user_profile(
    target_username: str,
    role: Optional[str] = None,
    clearance_level: Optional[str] = None,
    department: Optional[str] = None,
    is_active: Optional[bool] = None,
    admin_username: str = "admin"
) -> Dict[str, Any]:
    """Updates user permissions or status (admin only)."""
    target = get_user_by_username(target_username)
    if not target:
        raise ValueError(f"User '{target_username}' not found.")

    updates = []
    params = []

    if role is not None:
        role = role.lower()
        if role not in DEFAULT_ROLE_CLEARANCE:
            raise ValueError(f"Invalid role: {role}")
        # Prevent de-admining master admin
        if target["username"] == "admin" and role != "admin":
            raise ValueError("The master admin role cannot be downgraded.")
        updates.append("role = ?")
        params.append(role)
        # If clearance was not explicitly provided, update to role's default
        if clearance_level is None:
            clearance_level = DEFAULT_ROLE_CLEARANCE.get(role, "INTERNAL")

    if clearance_level is not None:
        if clearance_level not in CLEARANCE_RANKS:
            raise ValueError(f"Invalid clearance level: {clearance_level}")
        updates.append("clearance_level = ?")
        params.append(clearance_level)

    if department is not None:
        updates.append("department = ?")
        params.append(department.strip())

    if is_active is not None:
        if target["username"] == "admin" and not is_active:
            raise ValueError("The master admin account cannot be deactivated.")
        updates.append("is_active = ?")
        params.append(1 if is_active else 0)

    if not updates:
        return target

    params.append(target["username"])
    query = f"UPDATE users SET {', '.join(updates)} WHERE username = ?"

    with _conn() as c:
        c.execute(query, params)
        c.commit()

    log_audit(
        admin_username,
        "UPDATE_USER",
        f"Modified user '{target_username}': {dict(role=role, clearance=clearance_level, active=is_active)}",
        "SUCCESS"
    )
    return get_user_by_username(target_username)


def admin_reset_password(target_username: str, new_password: str, admin_username: str = "admin"):
    """Admin resets a user's password."""
    if len(new_password) < 6:
        raise ValueError("Password must be at least 6 characters.")
    p_hash, salt = hash_password(new_password)
    with _conn() as c:
        c.execute("UPDATE users SET password_hash = ?, salt = ? WHERE username = ?", (p_hash, salt, target_username))
        c.commit()
    log_audit(admin_username, "RESET_PASSWORD", f"Reset password for '{target_username}'", "SUCCESS")


def delete_user(target_username: str, admin_username: str = "admin"):
    """Deletes a user account (admin only, master admin protected)."""
    if target_username.lower() == "admin":
        raise ValueError("Cannot delete master administrator account.")
    with _conn() as c:
        c.execute("DELETE FROM users WHERE username = ?", (target_username,))
        c.execute("DELETE FROM user_tokens WHERE username = ?", (target_username,))
        c.commit()
    log_audit(admin_username, "DELETE_USER", f"Deleted user '{target_username}'", "SUCCESS")


# ── Document Confidentiality & Classification Matrix ────────────────────────

def get_document_classification(filename: str) -> str:
    """Returns the confidentiality classification level of a document, default 'INTERNAL'."""
    with _conn() as c:
        row = c.execute("SELECT classification FROM document_classifications WHERE filename = ?", (filename,)).fetchone()
    if row:
        return row["classification"]
    return "INTERNAL"


def get_all_document_classifications() -> Dict[str, Dict[str, Any]]:
    """Returns a mapping of all document classification records."""
    with _conn() as c:
        rows = c.execute("SELECT filename, classification, uploaded_by, department, uploaded_at, description FROM document_classifications").fetchall()
        return {r["filename"]: dict(r) for r in rows}


def set_document_classification(
    filename: str,
    classification: str,
    uploaded_by: str,
    department: str = "General",
    description: str = ""
):
    """Sets or updates the confidentiality classification of an indexed file."""
    classification = classification.upper()
    if classification not in CLEARANCE_RANKS:
        classification = "INTERNAL"

    now = time.time()
    with _conn() as c:
        c.execute("""
            INSERT INTO document_classifications (filename, classification, uploaded_by, department, uploaded_at, description)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(filename) DO UPDATE SET
                classification = excluded.classification,
                uploaded_by = CASE WHEN excluded.uploaded_by != '' THEN excluded.uploaded_by ELSE document_classifications.uploaded_by END,
                department = CASE WHEN excluded.department != '' THEN excluded.department ELSE document_classifications.department END
        """, (filename, classification, uploaded_by, department, now, description))
        c.commit()

    log_audit(uploaded_by, "SET_CLASSIFICATION", f"Document '{filename}' classified as {classification}", "SUCCESS")


def delete_document_classification(filename: str):
    """Removes a document from the classification table upon deletion."""
    with _conn() as c:
        c.execute("DELETE FROM document_classifications WHERE filename = ?", (filename,))
        c.commit()


def sync_existing_files(filenames: List[str]):
    """Populates default INTERNAL classifications for any indexed files not yet recorded."""
    classifications = get_all_document_classifications()
    now = time.time()
    with _conn() as c:
        for fname in filenames:
            if fname not in classifications:
                c.execute("""
                    INSERT INTO document_classifications (filename, classification, uploaded_by, department, uploaded_at, description)
                    VALUES (?, 'INTERNAL', 'system', 'General', ?, 'Initial document')
                """, (fname, now))
        c.commit()


def check_clearance(user_clearance: str, doc_classification: str) -> bool:
    """Returns True if user_clearance meets or exceeds doc_classification."""
    u_rank = CLEARANCE_RANKS.get(user_clearance.upper(), 1)
    d_rank = CLEARANCE_RANKS.get(doc_classification.upper(), 2)
    return u_rank >= d_rank


def get_allowed_filenames_for_user(user: Dict[str, Any], all_filenames: List[str]) -> set[str]:
    """Returns the set of filenames that the given user has clearance to access."""
    if user.get("role") == "admin":
        return set(all_filenames)

    user_clearance = user.get("clearance_level", "INTERNAL")
    classifications = get_all_document_classifications()

    allowed = set()
    for fname in all_filenames:
        doc_class = classifications.get(fname, {}).get("classification", "INTERNAL")
        if check_clearance(user_clearance, doc_class):
            allowed.add(fname)
    return allowed


# Auto-initialize database tables and seed accounts
init_auth_db()
