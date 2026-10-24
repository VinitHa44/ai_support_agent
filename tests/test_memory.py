"""
Tests for multi-turn conversation memory and context isolation.
"""

import pytest
from src.database import get_conversation_messages, init_db, save_message


@pytest.fixture(autouse=True)
def setup_env(tmp_path):
    test_db = str(tmp_path / "test_memory.db")
    init_db(test_db)
    return test_db


def test_conversation_history_persistence(setup_env):
    db_path = setup_env
    conv_id = "conv-mem-001"

    save_message(conv_id, "user", "My agent is broken.", db_path=db_path)
    save_message(
        conv_id,
        "assistant",
        "Could you please share your account email?",
        metadata={"awaiting": "email"},
        db_path=db_path,
    )
    save_message(conv_id, "user", "rahul@example.com", db_path=db_path)

    messages = get_conversation_messages(conv_id, db_path=db_path)
    assert len(messages) == 3
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "My agent is broken."
    assert messages[1]["role"] == "assistant"
    assert messages[1]["metadata"]["awaiting"] == "email"
    assert messages[2]["role"] == "user"
    assert messages[2]["content"] == "rahul@example.com"


def test_conversation_isolation(setup_env):
    db_path = setup_env
    conv_a = "conv-isolation-A"
    conv_b = "conv-isolation-B"

    save_message(conv_a, "user", "Message for A", db_path=db_path)
    save_message(conv_b, "user", "Message for B", db_path=db_path)

    msgs_a = get_conversation_messages(conv_a, db_path=db_path)
    msgs_b = get_conversation_messages(conv_b, db_path=db_path)

    assert len(msgs_a) == 1
    assert msgs_a[0]["content"] == "Message for A"

    assert len(msgs_b) == 1
    assert msgs_b[0]["content"] == "Message for B"
