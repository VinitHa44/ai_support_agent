"""
End-to-End API tests for POST /chat and full conversational scenarios.

All tests run in fallback (no-LLM) mode by monkeypatching API keys to empty
strings.  This keeps the suite hermetic, fast, and independent of live network
access or API credentials.
"""

import pytest
from fastapi.testclient import TestClient
from src.api import app
from src.database import init_db
from src.webhook import clear_mock_webhook_log, set_simulate_webhook_failure


@pytest.fixture(autouse=True)
def setup_test_suite(tmp_path, monkeypatch):
    test_db = str(tmp_path / "test_api.db")
    init_db(test_db)

    # Patch DB path in API and agent instance
    from src import api
    api.agent_instance.db_path = test_db
    monkeypatch.setattr(api, "DB_PATH", test_db)

    # Force fallback (no-LLM) mode so tests don't make live API calls
    monkeypatch.setattr("src.llm.OPENROUTER_API_KEY", "")
    monkeypatch.setattr("src.llm.OPENAI_API_KEY", "")

    clear_mock_webhook_log()
    set_simulate_webhook_failure(False)
    yield
    clear_mock_webhook_log()
    set_simulate_webhook_failure(False)


client = TestClient(app)


def test_turn_1_general_docs_inquiry():
    """Turn 1: 'How do I add an agent?' -> Answers using docs, links the doc URL, outcome: answered."""
    payload = {
        "conversation_id": "conv-scenario-001",
        "message": "How do I add an agent?",
    }
    resp = client.post("/chat", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["conversation_id"] == "conv-scenario-001"
    assert data["outcome"] == "answered"
    assert data["ticket_id"] is None
    assert len(data["doc_links"]) > 0
    assert any("docs.ampere.sh" in link for link in data["doc_links"])
    # The RAG may surface any valid Ampere doc (agents, skills, channels, etc.)
    # — any non-empty doc answer from the Ampere knowledge base is acceptable.
    assert len(data["answer"]) > 20, "Expected a substantive answer from the docs"


def test_turn_2_and_3_broken_agent_multi_turn_flow():
    """
    Turn 2: 'My agent is broken.' -> Remembers intent, asks for account email, outcome: needs_clarification.
    Turn 3: 'rahul@example.com' -> Calls get_agent_status('rahul@example.com'), diagnoses missing API key, outcome: answered.
    """
    conv_id = "conv-scenario-002"

    # Turn 2: User reports broken agent without email
    resp1 = client.post("/chat", json={
        "conversation_id": conv_id,
        "message": "My agent is broken.",
    })
    assert resp1.status_code == 200
    data1 = resp1.json()

    assert data1["conversation_id"] == conv_id
    assert data1["outcome"] == "needs_clarification"
    assert data1["ticket_id"] is None
    assert "email" in data1["answer"].lower()

    # Turn 3: User provides only their email address
    resp2 = client.post("/chat", json={
        "conversation_id": conv_id,
        "message": "rahul@example.com",
    })
    assert resp2.status_code == 200
    data2 = resp2.json()

    assert data2["conversation_id"] == conv_id
    assert data2["outcome"] == "answered"
    assert "offline" in data2["answer"].lower() or "missing api key" in data2["answer"].lower()
    assert "api key" in data2["answer"].lower()


def test_priya_healthy_account_flow():
    """Test looking up priya@example.com -> healthy, nothing to fix."""
    conv_id = "conv-priya-001"

    # In single turn or multi-turn
    resp = client.post("/chat", json={
        "conversation_id": conv_id,
        "message": "My bot is not working, registered email is priya@example.com",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["outcome"] == "answered"
    assert "running" in data["answer"].lower() or "healthy" in data["answer"].lower()
    assert "nothing" in data["answer"].lower()


def test_unknown_email_account_lookup():
    """Test looking up an unregistered email -> informs user no account found."""
    conv_id = "conv-unknown-001"

    resp = client.post("/chat", json={
        "conversation_id": conv_id,
        "message": "My assistant is down, email is nonexistent@corp.com",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["outcome"] == "answered"
    assert "no" in data["answer"].lower() and "account" in data["answer"].lower()


def test_turn_4_human_escalation_flow():
    """Turn 4: 'I need a human.' -> Creates ticket, scrubs PII, posts to Discord, outcome: escalated."""
    conv_id = "conv-scenario-004"

    resp = client.post("/chat", json={
        "conversation_id": conv_id,
        "message": "I need a human to help me with this.",
    })
    assert resp.status_code == 200
    data = resp.json()

    assert data["conversation_id"] == conv_id
    assert data["outcome"] == "escalated"
    assert data["ticket_id"] is not None
    assert data["ticket_id"].startswith("tick-")
    assert data["ticket_id"] in data["answer"]


def test_escalation_fault_tolerance_webhook_failed():
    """Test webhook failure -> outcome: escalation_pending."""
    conv_id = "conv-failure-001"
    set_simulate_webhook_failure(True)

    resp = client.post("/chat", json={
        "conversation_id": conv_id,
        "message": "Please connect me to a support person.",
    })
    assert resp.status_code == 200
    data = resp.json()

    assert data["conversation_id"] == conv_id
    assert data["outcome"] == "escalation_pending"
    assert data["ticket_id"] is not None


def test_ticket_deduplication():
    """Multiple escalation requests within the same conversation return the existing ticket ID."""
    conv_id = "conv-dedup-001"

    resp1 = client.post("/chat", json={
        "conversation_id": conv_id,
        "message": "I need to talk to a human support agent.",
    })
    data1 = resp1.json()
    assert data1["ticket_id"] is not None, "First escalation must create a ticket"

    resp2 = client.post("/chat", json={
        "conversation_id": conv_id,
        "message": "I still need a human to help me.",
    })
    data2 = resp2.json()
    assert data2["ticket_id"] == data1["ticket_id"], "Second escalation must return the same ticket ID"


def test_uncovered_docs_topic_fallback():
    """Query not covered in docs -> bot explicitly states it doesn't know instead of guessing."""
    resp = client.post("/chat", json={
        "conversation_id": "conv-uncovered-001",
        "message": "How do I bake a chocolate cake?",
    })
    assert resp.status_code == 200
    data = resp.json()

    assert data["outcome"] == "answered"
    assert data["doc_links"] == []
    assert "does not cover" in data["answer"].lower() or "not cover" in data["answer"].lower()


def test_validation_errors():
    """Empty conversation_id or message returns HTTP 400."""
    resp1 = client.post("/chat", json={"conversation_id": "", "message": "hello"})
    assert resp1.status_code == 400

    resp2 = client.post("/chat", json={"conversation_id": "conv-1", "message": "  "})
    assert resp2.status_code == 400
