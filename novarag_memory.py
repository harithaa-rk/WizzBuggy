# novarag_memory.py
# Drop-in replacement for the in-memory conversation_memory list.
# Stores turns in SQLite so they survive server restarts.

import sqlite3
import json
import time
import sys
import numpy as np
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

DB_PATH = "novarag_memory.db"
MAX_MEMORY_TURNS = 5

def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c

def init_db():
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS memory (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                session   TEXT    NOT NULL DEFAULT 'default',
                question  TEXT    NOT NULL,
                answer    TEXT    NOT NULL,
                embedding BLOB,           -- numpy float32 array as bytes
                file      TEXT,
                ts        REAL    NOT NULL
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_session ON memory(session, ts)")
        
        c.execute("""
            CREATE TABLE IF NOT EXISTS semantic_cache (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                query       TEXT    NOT NULL,
                embedding   BLOB    NOT NULL,
                answer      TEXT    NOT NULL,
                citations   TEXT    NOT NULL,
                confidence  REAL    NOT NULL,
                model       TEXT    NOT NULL DEFAULT 'fast',
                hits        INTEGER NOT NULL DEFAULT 0,
                created_at  REAL    NOT NULL,
                last_hit_at REAL    NOT NULL
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_cache_model ON semantic_cache(model)")
        c.commit()
    print("✅ SQLite memory DB ready:", DB_PATH)

init_db()

# ── Public API (mirrors the old list interface) ──────────────────

def append_turn(question: str, answer: str, embedding: np.ndarray | None, file: str | None, session: str = "default"):
    blob = embedding.tobytes() if embedding is not None else None
    with _conn() as c:
        c.execute(
            "INSERT INTO memory (session, question, answer, embedding, file, ts) VALUES (?,?,?,?,?,?)",
            (session, question, answer, blob, file, time.time())
        )
        c.commit()
    _trim(session)

def get_turns(session: str = "default") -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT question, answer, embedding, file FROM memory WHERE session=? ORDER BY ts DESC LIMIT ?",
            (session, MAX_MEMORY_TURNS)
        ).fetchall()
    turns = []
    for r in reversed(rows):  # oldest first
        emb = np.frombuffer(r["embedding"], dtype=np.float32) if r["embedding"] else None
        turns.append({
            "question":  r["question"],
            "answer":    r["answer"],
            "embedding": emb,
            "file":      r["file"],
        })
    return turns

def clear_session(session: str = "default"):
    with _conn() as c:
        c.execute("DELETE FROM memory WHERE session=?", (session,))
        c.commit()

def clear_all():
    with _conn() as c:
        c.execute("DELETE FROM memory")
        c.commit()

def _trim(session: str):
    """Keep only the most recent MAX_MEMORY_TURNS rows per session."""
    with _conn() as c:
        ids = c.execute(
            "SELECT id FROM memory WHERE session=? ORDER BY ts DESC LIMIT -1 OFFSET ?",
            (session, MAX_MEMORY_TURNS)
        ).fetchall()
        if ids:
            placeholders = ",".join("?" * len(ids))
            c.execute(f"DELETE FROM memory WHERE id IN ({placeholders})", [r["id"] for r in ids])
            c.commit()

# ── Semantic Caching ─────────────────────────────────────────────

def lookup_semantic_cache(
    query_embedding: np.ndarray,
    model: str = "fast",
    threshold: float = 0.90
) -> dict | None:
    """
    Finds the most semantically similar cached query using cosine similarity.
    Returns the cached entry if max similarity >= threshold, else None.
    """
    if query_embedding is None:
        return None

    # Normalize query embedding
    norm = float(np.linalg.norm(query_embedding))
    if norm <= 0:
        return None
    query_vec = (query_embedding / norm).astype(np.float32)

    with _conn() as c:
        rows = c.execute(
            "SELECT id, query, embedding, answer, citations, confidence, model, hits FROM semantic_cache WHERE model = ?",
            (model,)
        ).fetchall()

    if not rows:
        return None

    best_sim = -1.0
    best_row = None

    for row in rows:
        cached_blob = row["embedding"]
        if not cached_blob:
            continue
        cached_vec = np.frombuffer(cached_blob, dtype=np.float32)
        cached_norm = float(np.linalg.norm(cached_vec))
        if cached_norm <= 0:
            continue
        cached_unit = cached_vec / cached_norm
        sim = float(np.dot(query_vec, cached_unit))
        if sim > best_sim:
            best_sim = sim
            best_row = row

    if best_row is not None and best_sim >= threshold:
        # Increment hit count and update last_hit_at
        with _conn() as c:
            c.execute(
                "UPDATE semantic_cache SET hits = hits + 1, last_hit_at = ? WHERE id = ?",
                (time.time(), best_row["id"])
            )
            c.commit()

        try:
            citations = json.loads(best_row["citations"])
        except Exception:
            citations = []

        return {
            "cached": True,
            "cache_id": best_row["id"],
            "matched_query": best_row["query"],
            "similarity": round(best_sim, 4),
            "answer": best_row["answer"],
            "citations": citations,
            "confidence": best_row["confidence"],
            "hits": best_row["hits"] + 1,
        }

    return None

def store_semantic_cache(
    query: str,
    query_embedding: np.ndarray,
    answer: str,
    citations: list,
    confidence: float,
    model: str = "fast",
    max_cache_size: int = 500
):
    """Stores a successful, grounded answer and its embedding in semantic_cache."""
    if query_embedding is None or not answer or not query:
        return

    # Do not cache error responses or refusals
    refusals = [
        "The answer is not present in the provided sources.",
        "Local LLM not running.",
        "Error:",
    ]
    if any(r in answer for r in refusals):
        return

    blob = query_embedding.astype(np.float32).tobytes()
    citations_json = json.dumps(citations)
    now = time.time()

    with _conn() as c:
        c.execute("""
            INSERT INTO semantic_cache (query, embedding, answer, citations, confidence, model, hits, created_at, last_hit_at)
            VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)
        """, (query.strip(), blob, answer.strip(), citations_json, confidence, model, now, now))
        c.commit()

        # Trim old entries if cache exceeds max_cache_size
        count = c.execute("SELECT COUNT(*) FROM semantic_cache").fetchone()[0]
        if count > max_cache_size:
            c.execute("""
                DELETE FROM semantic_cache WHERE id IN (
                    SELECT id FROM semantic_cache ORDER BY hits ASC, last_hit_at ASC LIMIT ?
                )
            """, (count - max_cache_size,))
            c.commit()

def clear_semantic_cache():
    """Flushes the semantic cache (called when documents are ingested or deleted)."""
    with _conn() as c:
        c.execute("DELETE FROM semantic_cache")
        c.commit()
    print("🧹 Semantic cache cleared")

def get_semantic_cache_stats() -> dict:
    """Returns metadata and statistics about the semantic cache."""
    with _conn() as c:
        total_entries = c.execute("SELECT COUNT(*) FROM semantic_cache").fetchone()[0]
        total_hits = c.execute("SELECT COALESCE(SUM(hits), 0) FROM semantic_cache").fetchone()[0]
        top_queries = c.execute(
            "SELECT query, hits, confidence, model FROM semantic_cache ORDER BY hits DESC LIMIT 5"
        ).fetchall()
    return {
        "total_cached_queries": total_entries,
        "total_cache_hits": total_hits,
        "top_cached": [dict(r) for r in top_queries]
    }