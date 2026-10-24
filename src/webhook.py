"""
Discord Webhook integration with mock mode and fault tolerance.
"""

import json
import logging
from typing import Any, Dict, List, Optional
import httpx
from src.config import DISCORD_WEBHOOK_URL
from src.pii import strip_pii

logger = logging.getLogger(__name__)

# In-memory history for mock mode / test inspection
MOCK_WEBHOOK_LOG: List[Dict[str, Any]] = []

# Optional global flag to simulate webhook failure in tests
SIMULATE_WEBHOOK_FAILURE = False


def set_simulate_webhook_failure(simulate: bool):
    global SIMULATE_WEBHOOK_FAILURE
    SIMULATE_WEBHOOK_FAILURE = simulate


def get_mock_webhook_log() -> List[Dict[str, Any]]:
    return list(MOCK_WEBHOOK_LOG)


def clear_mock_webhook_log():
    MOCK_WEBHOOK_LOG.clear()


def post_to_discord(
    ticket_id: str,
    conversation_id: str,
    summary: str,
    webhook_url: Optional[str] = None,
) -> bool:
    """
    Sends support ticket notification to Discord webhook with scrubbed PII.
    Returns:
        True if successfully sent/mocked, False if delivery failed.
    """
    global SIMULATE_WEBHOOK_FAILURE
    if SIMULATE_WEBHOOK_FAILURE:
        logger.warning(f"Simulating Discord webhook delivery failure for {ticket_id}")
        return False

    url = webhook_url if webhook_url is not None else DISCORD_WEBHOOK_URL
    clean_summary = strip_pii(summary)

    payload = {
        "embeds": [
            {
                "title": f"🚨 Human Support Escalation: {ticket_id}",
                "description": clean_summary,
                "color": 15158332,  # Red/Amber
                "fields": [
                    {"name": "Ticket ID", "value": ticket_id, "inline": True},
                    {"name": "Conversation ID", "value": conversation_id, "inline": True},
                ],
                "footer": {"text": "Ampere Support Agent System (PII Sanitized)"},
            }
        ]
    }

    # If webhook URL is empty, 'mock', or not a live discord endpoint -> run mock mode
    is_mock = (
        not url
        or url.strip().lower() in ("mock", "test", "none")
        or not (url.startswith("http://") or url.startswith("https://"))
    )

    if is_mock:
        logger.info(f"[MOCK DISCORD WEBHOOK] Ticket {ticket_id} posted: {clean_summary}")
        MOCK_WEBHOOK_LOG.append({
            "ticket_id": ticket_id,
            "conversation_id": conversation_id,
            "summary": clean_summary,
            "payload": payload,
        })
        return True

    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.post(url, json=payload)
            if resp.status_code in (200, 204):
                logger.info(f"Discord webhook delivered successfully for {ticket_id}")
                return True
            else:
                logger.error(
                    f"Discord webhook failed with status {resp.status_code}: {resp.text}"
                )
                return False
    except Exception as exc:
        logger.error(f"Discord webhook request failed: {exc}")
        return False
