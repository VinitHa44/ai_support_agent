"""
FastAPI Server for Ampere AI Customer Support Agent.
Exposes:
- POST /chat: Core conversational endpoint
- GET /health: Healthcheck
- GET /api/tickets: List created support tickets
- GET /api/accounts: View seeded accounts
- GET /api/history/{conversation_id}: View message history
- Static files & Web UI for live testing
"""

import logging
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from src.agent import SupportAgent
from src.config import DB_PATH
from src.database import get_connection, init_db
from src.webhook import (
    clear_mock_webhook_log,
    get_mock_webhook_log,
    set_simulate_webhook_failure,
)

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Ampere AI Support Agent API",
    description="Backend AI customer support agent with RAG docs search, account diagnostics, and human escalation.",
    version="1.0.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- Request & Response Models ---------------- #

class ChatRequest(BaseModel):
    conversation_id: str = Field(..., description="Unique conversation identifier")
    message: str = Field(..., description="User message text")


class ChatResponse(BaseModel):
    conversation_id: str = Field(..., description="Unique conversation identifier")
    answer: str = Field(..., description="Agent answer to the user")
    doc_links: List[str] = Field(default_factory=list, description="Citations to official documentation")
    outcome: str = Field(..., description="Allowed: answered, needs_clarification, escalated, escalation_pending")
    ticket_id: Optional[str] = Field(default=None, description="Support ticket ID if an issue was escalated")


class SimulateFailureRequest(BaseModel):
    simulate_failure: bool


# ---------------- Startup Event ---------------- #

agent_instance = SupportAgent()


@app.on_event("startup")
def on_startup():
    init_db()
    logger.info("Database initialized on startup.")


# ---------------- Endpoints ---------------- #

@app.get("/health")
def health():
    return {"status": "ok", "service": "ampere-support-agent"}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest):
    """
    Main conversational endpoint:
    - Retains multi-turn conversation memory
    - Searches docs and provides source links
    - Performs SQLite account lookup for agent status
    - Files support tickets and notifies Discord webhook
    """
    if not payload.conversation_id or not payload.conversation_id.strip():
        raise HTTPException(status_code=400, detail="conversation_id cannot be empty")
    if not payload.message or not payload.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")

    response = agent_instance.process_message(
        conversation_id=payload.conversation_id.strip(),
        message=payload.message.strip(),
    )
    return response


@app.get("/api/accounts")
def get_accounts():
    """Returns all accounts in the database for reference/testing."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT email, status, diagnosis FROM accounts")
    rows = cursor.fetchall()
    conn.close()
    return {"accounts": [dict(r) for r in rows]}


@app.get("/api/conversations")
def get_conversations():
    """
    Returns all distinct conversation IDs with metadata for the sidebar history panel.
    Each item includes: conversation_id, message_count, preview (first user message), last_at.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            conversation_id,
            COUNT(*) AS message_count,
            MAX(created_at) AS last_at,
            MIN(CASE WHEN role = 'user' THEN content END) AS preview
        FROM messages
        GROUP BY conversation_id
        ORDER BY last_at DESC
        LIMIT 50
        """
    )
    rows = cursor.fetchall()
    conn.close()
    return {
        "conversations": [
            {
                "conversation_id": r["conversation_id"],
                "message_count": r["message_count"],
                "last_at": r["last_at"],
                "preview": (r["preview"] or "")[:60],
            }
            for r in rows
        ]
    }


@app.delete("/api/conversations")
def clear_all_conversations():
    """Deletes all conversation messages and tickets from the database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM messages")
    cursor.execute("DELETE FROM tickets")
    conn.commit()
    conn.close()
    return {"status": "success", "message": "All conversations and tickets deleted."}


@app.get("/api/presets")
def get_presets():
    """Dynamically generates prompt presets from scraped documentation topics and live database accounts."""
    import os
    import json
    from src.config import DOCS_PATH
    presets = []

    # 1. Dynamically generate presets from scraped documentation topics
    if os.path.exists(DOCS_PATH):
        try:
            with open(DOCS_PATH, "r", encoding="utf-8") as f:
                docs = json.load(f)
            for d in docs[:3]:
                title = d.get("title", "")
                if title and title != "Introduction":
                    presets.append({
                        "tag": "Docs RAG",
                        "prompt": f"Tell me about {title} in Ampere",
                    })
                elif title == "Introduction":
                    presets.append({
                        "tag": "Docs RAG",
                        "prompt": "What is Ampere and how do I add an agent?",
                    })
        except Exception as e:
            logger.debug(f"Error reading doc presets: {e}")

    # 2. Add troubleshooting presets from live database accounts
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT email, status FROM accounts ORDER BY email ASC")
        rows = cursor.fetchall()
        conn.close()
        for r in rows:
            presets.append({
                "tag": f"Account ({r['status']})",
                "prompt": f"My assistant is down, email: {r['email']}",
            })
    except Exception as e:
        logger.debug(f"Error reading account presets: {e}")

    # 3. Add dynamic escalation and out-of-scope inquiry presets
    presets.append({
        "tag": "Escalate",
        "prompt": "I need to talk to a human support agent",
    })
    presets.append({
        "tag": "Out-of-Scope",
        "prompt": "How do I bake a chocolate cake?",
    })

    return {"presets": presets}


@app.get("/api/tickets")
def get_tickets():
    """Returns all support tickets created in the database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, conversation_id, user_email, summary, status, created_at FROM tickets")
    rows = cursor.fetchall()
    conn.close()
    return {
        "tickets": [dict(r) for r in rows],
        "mock_dispatches": get_mock_webhook_log(),
    }


@app.get("/api/history/{conversation_id}")
def get_history(conversation_id: str):
    """Returns conversation message history for a given conversation_id."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT role, content, metadata, created_at FROM messages WHERE conversation_id = ? ORDER BY id ASC",
        (conversation_id,),
    )
    rows = cursor.fetchall()
    conn.close()
    return {"conversation_id": conversation_id, "messages": [dict(r) for r in rows]}


@app.post("/api/simulate-webhook-failure")
def simulate_webhook_failure(req: SimulateFailureRequest):
    """Endpoint for testing fault tolerance: forces Discord webhook failures."""
    set_simulate_webhook_failure(req.simulate_failure)
    return {"simulate_failure": req.simulate_failure}


# Mount static directory for frontend
import os
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
