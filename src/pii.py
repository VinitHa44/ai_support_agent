"""
PII (Personally Identifiable Information) Redaction & Sanitization.
Ensures emails, passwords, API keys, phone numbers, and credentials
are scrubbed from ticket summaries before transmission to external services.
"""

import re

# Regex patterns for sensitive data
EMAIL_REGEX = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
)
PHONE_REGEX = re.compile(
    r"(\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b"
)
API_KEY_REGEX = re.compile(
    r"(?i)\b(sk-[a-zA-Z0-9_\-]{20,}|ghp_[a-zA-Z0-9]{30,}|key-[a-zA-Z0-9]{16,}|(?:api[_-]?key|secret|token|bearer)[\s:=]+['\"]?([a-zA-Z0-9_\-\.]{12,})['\"]?)\b"
)
PASSWORD_REGEX = re.compile(
    r"(?i)\b(password|passwd|pwd)[\s:=]+['\"]?([^\s,;'\"]+)['\"]?"
)
CREDIT_CARD_REGEX = re.compile(
    r"\b(?:\d{4}[ -]?){3}\d{4}\b"
)


def strip_pii(text: str) -> str:
    """
    Scrubs emails, secrets, API keys, passwords, phone numbers, and payment cards
    from a given text string.
    """
    if not text:
        return ""

    sanitized = text

    # Redact API keys / tokens
    sanitized = API_KEY_REGEX.sub("[REDACTED_SECRET]", sanitized)

    # Redact passwords
    sanitized = PASSWORD_REGEX.sub(r"\1: [REDACTED_PASSWORD]", sanitized)

    # Redact email addresses
    sanitized = EMAIL_REGEX.sub("[REDACTED_EMAIL]", sanitized)

    # Redact credit card numbers
    sanitized = CREDIT_CARD_REGEX.sub("[REDACTED_CARD]", sanitized)

    # Redact phone numbers (careful not to over-redact small numbers)
    sanitized = PHONE_REGEX.sub("[REDACTED_PHONE]", sanitized)

    return sanitized
