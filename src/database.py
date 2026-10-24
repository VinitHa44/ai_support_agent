"""
SQLite Database module for Ampere AI Support Agent.
Manages:
- Seed accounts table (get_agent_status)
- Conversation history / memory
- Support tickets with deduplication
"""

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from src.config import DB_PATH

logger = logging.getLogger(__name__)

SEED_ACCOUNTS = [
    {
        "email": "priya@example.com",
        "status": "running",
        "diagnosis": "Healthy, nothing to fix.",
    },
    {
        "email": "rahul@example.com",
        "status": "offline",
        "diagnosis": "Missing API key.",
    },
]


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    path = db_path or DB_PATH
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[str] = None):
    """Initializes tables and seeds initial account data."""
    conn = get_connection(db_path)
    with conn:
        cursor = conn.cursor()

        # 1. Accounts Table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS accounts (
                email TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                diagnosis TEXT NOT NULL
            )
            """
        )

        # 2. Conversation Messages Table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_messages_conv 
            ON messages (conversation_id, id)
            """
        )

        # 3. Support Tickets Table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS tickets (
                id TEXT PRIMARY KEY,
                conversation_id TEXT UNIQUE NOT NULL,
                user_email TEXT,
                summary TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # Seed accounts if empty or missing
        for acc in SEED_ACCOUNTS:
            cursor.execute(
                """
                INSERT INTO accounts (email, status, diagnosis)
                VALUES (:email, :status, :diagnosis)
                ON CONFLICT(email) DO UPDATE SET
                    status=excluded.status,
                    diagnosis=excluded.diagnosis
                """,
                acc,
            )
    conn.close()
    logger.info("Database initialized and seeded.")


def get_agent_status(email: str, db_path: Optional[str] = None) -> Dict[str, str]:
    """
    Looks up agent status by registered email address.
    Returns:
        {
            "email": email,
            "status": "running" | "offline" | "not_found",
            "diagnosis": "..."
        }
    """
    normalized_email = (email or "").strip().lower()
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT email, status, diagnosis FROM accounts WHERE LOWER(email) = LOWER(?)",
        (normalized_email,),
    )
    row = cursor.fetchone()
    conn.close()

    if row:
        return {
            "email": row["email"],
            "status": row["status"],
            "diagnosis": row["diagnosis"],
        }
    else:
        return {
            "email": normalized_email,
            "status": "not_found",
            "diagnosis": "No account for that email.",
        }


def save_message(
    conversation_id: str,
    role: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    db_path: Optional[str] = None,
):
    """Persists a message in conversation memory."""
    conn = get_connection(db_path)
    meta_str = json.dumps(metadata) if metadata else None
    with conn:
        conn.execute(
            """
            INSERT INTO messages (conversation_id, role, content, metadata)
            VALUES (?, ?, ?, ?)
            """,
            (conversation_id, role, content, meta_str),
        )
    conn.close()


def get_conversation_messages(
    conversation_id: str, limit: int = 50, db_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Retrieves the most recent conversation history in chronological order."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT role, content, metadata, created_at
        FROM (
            SELECT id, role, content, metadata, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id DESC
            LIMIT ?
        )
        ORDER BY id ASC
        """,
        (conversation_id, limit),
    )
    rows = cursor.fetchall()
    conn.close()

    result = []
    for r in rows:
        meta = None
        if r["metadata"]:
            try:
                meta = json.loads(r["metadata"])
            except Exception:
                meta = {}
        result.append(
            {
                "role": r["role"],
                "content": r["content"],
                "metadata": meta,
                "created_at": r["created_at"],
            }
        )
    return result


def get_ticket_by_conversation(
    conversation_id: str, db_path: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Checks if a ticket already exists for this conversation (deduplication)."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, conversation_id, user_email, summary, status, created_at
        FROM tickets
        WHERE conversation_id = ?
        """,
        (conversation_id,),
    )
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None


def create_ticket(
    conversation_id: str,
    summary: str,
    status: str = "escalated",
    user_email: Optional[str] = None,
    db_path: Optional[str] = None,
    ticket_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Creates a new ticket, enforcing one ticket per conversation.

    Args:
        ticket_id: Optional pre-generated ID.  When provided the caller is
                   responsible for having already sent the same ID to Discord so
                   both records stay in sync.  If omitted a new UUID is generated.
    """
    existing = get_ticket_by_conversation(conversation_id, db_path)
    if existing:
        return existing

    ticket_id = ticket_id or f"tick-{uuid.uuid4().hex[:8]}"
    conn = get_connection(db_path)
    with conn:
        conn.execute(
            """
            INSERT INTO tickets (id, conversation_id, user_email, summary, status)
            VALUES (?, ?, ?, ?, ?)
            """,
            (ticket_id, conversation_id, user_email, summary, status),
        )
    conn.close()

    return {
        "id": ticket_id,
        "conversation_id": conversation_id,
        "user_email": user_email,
        "summary": summary,
        "status": status,
    }


def update_ticket_status(
    ticket_id: str, status: str, db_path: Optional[str] = None
):
    """Updates the status of a ticket (e.g. escalated or escalation_pending)."""
    conn = get_connection(db_path)
    with conn:
        conn.execute(
            "UPDATE tickets SET status = ? WHERE id = ?", (status, ticket_id)
        )
    conn.close()
