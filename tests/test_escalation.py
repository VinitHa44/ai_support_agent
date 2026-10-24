"""
Tests for Human Escalation, Ticketing, Discord Webhook, and Deduplication.
"""

import pytest
from src.database import get_ticket_by_conversation, init_db
from src.tools import execute_escalate_to_human
from src.webhook import (
    clear_mock_webhook_log,
    get_mock_webhook_log,
    set_simulate_webhook_failure,
)


@pytest.fixture(autouse=True)
def setup_env(tmp_path):
    test_db = str(tmp_path / "test_escalation.db")
    init_db(test_db)
    clear_mock_webhook_log()
    set_simulate_webhook_failure(False)
    yield test_db
    clear_mock_webhook_log()
    set_simulate_webhook_failure(False)


def test_escalation_creates_ticket_and_mocks_discord(setup_env):
    db_path = setup_env
    conv_id = "conv-esc-001"
    summary = "User rahul@example.com cannot connect to assistant sk-1234567890abcdef1234567890."

    res = execute_escalate_to_human(
        conversation_id=conv_id,
        summary=summary,
        user_email="rahul@example.com",
        db_path=db_path,
    )

    assert res["status"] == "escalated"
    assert res["ticket_id"].startswith("tick-")
    assert not res["deduplicated"]

    # Verify PII was scrubbed in returned summary
    assert "rahul@example.com" not in res["summary"]
    assert "[REDACTED_EMAIL]" in res["summary"]
    assert "sk-1234567890abcdef1234567890" not in res["summary"]
    assert "[REDACTED_SECRET]" in res["summary"]

    # Verify mock webhook recorded the dispatch
    logs = get_mock_webhook_log()
    assert len(logs) == 1
    assert logs[0]["conversation_id"] == conv_id
    assert "rahul@example.com" not in logs[0]["summary"]


def test_escalation_deduplication(setup_env):
    db_path = setup_env
    conv_id = "conv-esc-002"

    # First escalation
    res1 = execute_escalate_to_human(
        conversation_id=conv_id,
        summary="User is stuck and needs human help",
        db_path=db_path,
    )
    assert not res1["deduplicated"]
    ticket_id_1 = res1["ticket_id"]

    # Second escalation attempt on the exact same conversation
    res2 = execute_escalate_to_human(
        conversation_id=conv_id,
        summary="User is asking again for a human",
        db_path=db_path,
    )
    assert res2["deduplicated"]
    assert res2["ticket_id"] == ticket_id_1

    # Verify DB only has 1 ticket for this conversation
    ticket = get_ticket_by_conversation(conv_id, db_path=db_path)
    assert ticket is not None
    assert ticket["id"] == ticket_id_1


def test_fault_tolerance_webhook_failure(setup_env):
    db_path = setup_env
    conv_id = "conv-esc-003"

    # Simulate Discord webhook failure
    set_simulate_webhook_failure(True)

    res = execute_escalate_to_human(
        conversation_id=conv_id,
        summary="User requires escalation but webhook network fails",
        db_path=db_path,
    )

    # Status must be escalation_pending
    assert res["status"] == "escalation_pending"
    assert res["ticket_id"].startswith("tick-")

    # DB status check
    ticket = get_ticket_by_conversation(conv_id, db_path=db_path)
    assert ticket["status"] == "escalation_pending"
