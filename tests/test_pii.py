"""
Tests for PII sanitization and redaction.
"""

from src.pii import strip_pii


def test_strip_email():
    text = "User rahul@example.com reported an issue with their bot."
    sanitized = strip_pii(text)
    assert "rahul@example.com" not in sanitized
    assert "[REDACTED_EMAIL]" in sanitized


def test_strip_api_key():
    text = "Here is my secret sk-1234567890abcdef1234567890 please check it"
    sanitized = strip_pii(text)
    assert "sk-1234567890abcdef1234567890" not in sanitized
    assert "[REDACTED_SECRET]" in sanitized


def test_strip_password():
    text = "User login with password: MySuperSecretPassword123! failed"
    sanitized = strip_pii(text)
    assert "MySuperSecretPassword123!" not in sanitized
    assert "[REDACTED_PASSWORD]" in sanitized


def test_strip_multiple_pii_elements():
    text = (
        "Contact me at priya@test.com or +1 555 123 4567. "
        "My key is sk-abcdef123456789012345678 and pwd: secretPassword"
    )
    sanitized = strip_pii(text)
    assert "priya@test.com" not in sanitized
    assert "sk-abcdef123456789012345678" not in sanitized
    assert "secretPassword" not in sanitized
