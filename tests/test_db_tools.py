"""
Tests for SQLite Database, seed accounts, and get_agent_status tool.
"""

import os
import pytest
from src.database import get_agent_status, init_db


@pytest.fixture(autouse=True)
def setup_db(tmp_path):
    test_db = str(tmp_path / "test_support.db")
    init_db(test_db)
    return test_db


def test_priya_account_status(setup_db):
    res = get_agent_status("priya@example.com", db_path=setup_db)
    assert res["status"] == "running"
    assert "Healthy" in res["diagnosis"]
    assert "priya@example.com" in res["email"]


def test_rahul_account_status(setup_db):
    res = get_agent_status("rahul@example.com", db_path=setup_db)
    assert res["status"] == "offline"
    assert "Missing API key" in res["diagnosis"]
    assert "rahul@example.com" in res["email"]


def test_unknown_account_status(setup_db):
    res = get_agent_status("unknown_user@domain.com", db_path=setup_db)
    assert res["status"] == "not_found"
    assert "No account" in res["diagnosis"]


def test_case_and_whitespace_insensitivity(setup_db):
    res = get_agent_status("  RAHUL@EXAMPLE.COM  ", db_path=setup_db)
    assert res["status"] == "offline"
    assert "Missing API key" in res["diagnosis"]
