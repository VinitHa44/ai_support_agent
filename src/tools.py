"""
Tool definitions and schemas for Ampere AI Support Agent.
Includes:
- get_agent_status(email)
- search_ampere_docs(query)
- escalate_to_human(conversation_id, summary, user_email)
"""

import logging
from typing import Any, Dict, List, Optional
from src.database import create_ticket, get_agent_status, get_ticket_by_conversation, update_ticket_status
from src.pii import strip_pii
from src.rag import get_knowledge_base
from src.webhook import post_to_discord

logger = logging.getLogger(__name__)


# ---------------- Tool Implementations ---------------- #

def execute_get_agent_status(email: str, db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Looks up an account status and diagnosis by registered email address.
    """
    logger.info(f"Tool call: get_agent_status for {email}")
    return get_agent_status(email=email, db_path=db_path)


def execute_search_docs(query: str, top_k: int = 3) -> Dict[str, Any]:
    """
    Searches official Ampere documentation for answers.
    """
    logger.info(f"Tool call: search_ampere_docs for '{query}'")
    kb = get_knowledge_base()
    results = kb.search(query, top_k=top_k)
    if not results:
        return {
            "found": False,
            "results": [],
            "message": "No relevant documentation found for the query.",
        }

    formatted = []
    doc_links = []
    for chunk, score in results:
        if chunk.url not in doc_links:
            doc_links.append(chunk.url)
        formatted.append({
            "title": chunk.title,
            "url": chunk.url,
            "snippet": chunk.text,
            "score": round(score, 2),
        })

    return {
        "found": True,
        "results": formatted,
        "doc_links": doc_links,
    }


def execute_escalate_to_human(
    conversation_id: str,
    summary: str,
    user_email: Optional[str] = None,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Creates a support ticket, scrubs PII, and posts to Discord webhook.
    Handles deduplication: returns existing ticket if one already exists for this conversation.
    """
    logger.info(f"Tool call: escalate_to_human for conv {conversation_id}")
    
    # Scrub PII from summary
    clean_summary = strip_pii(summary)

    # Check for existing ticket (deduplication)
    existing = get_ticket_by_conversation(conversation_id, db_path=db_path)
    if existing:
        logger.info(f"Existing ticket found for {conversation_id}: {existing['id']}")
        return {
            "ticket_id": existing["id"],
            "status": existing["status"],
            "summary": existing["summary"],
            "deduplicated": True,
        }

    # Generate the canonical ticket ID first so Discord alert and DB record share the same ID
    import uuid as _uuid
    canonical_ticket_id = f"tick-{_uuid.uuid4().hex[:8]}"

    # Attempt Discord webhook post with the canonical ticket ID
    success = post_to_discord(
        ticket_id=canonical_ticket_id,
        conversation_id=conversation_id,
        summary=clean_summary,
    )
    status = "escalated" if success else "escalation_pending"

    # Persist in DB — pass the same canonical_ticket_id so it matches Discord
    ticket = create_ticket(
        conversation_id=conversation_id,
        summary=clean_summary,
        status=status,
        user_email=user_email,
        db_path=db_path,
        ticket_id=canonical_ticket_id,
    )

    return {
        "ticket_id": ticket["id"],
        "status": ticket["status"],
        "summary": ticket["summary"],
        "deduplicated": False,
    }


# ---------------- OpenAI / Generic Function Schemas ---------------- #

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_agent_status",
            "description": "Look up the status and diagnosis of an Ampere assistant/bot by the user's registered account email. Only call this when an account problem or broken/down agent is being discussed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {
                        "type": "string",
                        "description": "The registered email address of the account.",
                    }
                },
                "required": ["email"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_ampere_docs",
            "description": "Search the official Ampere documentation at https://docs.ampere.sh/docs for product questions, features, setup guides, and pricing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query or keyword phrase to find in the docs.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": "Escalate the support issue to a human engineer when the user asks for a person or the issue cannot be resolved.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Summary of the user issue to send to support.",
                    },
                    "user_email": {
                        "type": "string",
                        "description": "Optional user email if known.",
                    },
                },
                "required": ["summary"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_clarification",
            "description": "Ask the user a clarifying question before proceeding (e.g. asking for their registered account email when an assistant is down/broken, or clarifying an ambiguous issue).",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The clarifying question to present to the user.",
                    }
                },
                "required": ["question"],
            },
        },
    },
]
