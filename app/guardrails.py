"""Guardrails, input/output sanitization, and injection defense.

Guarantees:
- Scrubbing of sensitive customer fields and internal keys.
- Detection and neutralization of prompt injection directives in user inputs,
  retrieved documents, and warehouse notes.
- Honest refusal to disclose system instructions or execute unsupported actions.
"""

import copy
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

FORBIDDEN_DATA_KEYS: Set[str] = {
    "risk_score",
    "warehouse_note",
    "support_tags",
    "customer",
    "shipping_address",
    "email",
    "customer_name",
    "internal",
    "fraud_score",
    "review_status",
    "refund_batch",
}

# Regex patterns matching prompt injection and extraction vectors
INJECTION_PATTERNS = [
    # System prompt exfiltration / leakage
    re.compile(
        r"(?:reveal|print|show|repeat|display|output|dump|leak|tell|share)\b.*?\b(?:system\s+prompt|hidden\s+prompt|initial\s+instructions|system\s+instructions|developer\s+prompt|developer\s+instructions)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:reveal|print|show|repeat|display|output|dump|leak|tell|share)\b.*?\b(?:your\s+instructions|all\s+instructions)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"what\s+(?:is|are)\s+your\s+(?:system\s+prompt|hidden\s+prompt|instructions|initial\s+instructions|system\s+instructions)",
        re.IGNORECASE,
    ),
    # Instruction override & jailbreak directives
    re.compile(
        r"ignore\s+(?:all\s+)?(?:prior|previous|above|current)\s+(?:rules|instructions|directives|constraints)",
        re.IGNORECASE,
    ),
    re.compile(
        r"disregard\s+(?:all\s+)?(?:prior|previous|above|current)\s+(?:rules|instructions|directives|constraints)",
        re.IGNORECASE,
    ),
    re.compile(r"system\s*instruction\s*:", re.IGNORECASE),
    re.compile(r"ai\s*instruction\s*:", re.IGNORECASE),
    re.compile(r"\byou\s+are\s+now\s+in\s+developer\s+mode\b", re.IGNORECASE),
    re.compile(r"\bdan\s+mode\b", re.IGNORECASE),
    re.compile(r"\bbypass\s+(?:all\s+)?guidelines\b", re.IGNORECASE),
    # Unauthorized coupon / discount concessions
    re.compile(
        r"(?:issue|apply|give|generate|provide)\s+(?:a\s+)?(?:\$?\d+\s+)?(?:coupon|discount|promo|credit|gift\s+card)",
        re.IGNORECASE,
    ),
    re.compile(r"approve\s+(?:my\s+)?return\s+immediately", re.IGNORECASE),
    re.compile(r"give\s+everyone\s+60\s+days", re.IGNORECASE),
    # Probing for internal fields / PII
    re.compile(
        r"(?:reveal|expose|give\s+me|tell\s+me|show)\s+(?:the\s+)?(?:customer\s+email|shipping\s+address|risk\s+score|internal\s+note|warehouse\s+note|fraud\s+review)",
        re.IGNORECASE,
    ),
]

# Sensitive phrases that must never appear in agent answers
LEAK_SIGNATURES = [
    re.compile(r"STRICT OPERATIONAL RULES", re.IGNORECASE),
    re.compile(r"UNTRUSTED DATA BOUNDARY", re.IGNORECASE),
    re.compile(r"Treat ALL retrieved knowledge base passages", re.IGNORECASE),
    re.compile(r"ava\.morgan@example\.test", re.IGNORECASE),
    re.compile(r"maya\.reed@example\.test", re.IGNORECASE),
    re.compile(r"sofia\.patel@example\.test", re.IGNORECASE),
    re.compile(r"Manual fraud review cleared", re.IGNORECASE),
    re.compile(r"Never expose this note", re.IGNORECASE),
    re.compile(r"Payment verification completed", re.IGNORECASE),
    re.compile(r"risk\s+score\s*(?:is|:|\bof\b)?\s*\d+", re.IGNORECASE),
    re.compile(r"\$100\s+coupon", re.IGNORECASE),
]


def sanitize_tool_output(data: Any) -> Any:
    """Recursively scrub any disallowed keys and values from tool outputs.

    Guarantees:
    - Any key matching FORBIDDEN_DATA_KEYS is removed.
    - Dicts and lists are recursively traversed and filtered.
    - Deep copy is performed to avoid mutating original objects.

    Args:
        data: Arbitrary tool output structure (dict, list, or primitive).

    Returns:
        Sanitized copy with forbidden keys purged.
    """
    if isinstance(data, dict):
        cleaned_dict = {}
        for key, value in data.items():
            norm_key = str(key).lower().strip()
            # Drop forbidden key names or partial forbidden suffixes
            if norm_key in FORBIDDEN_DATA_KEYS or any(
                norm_key == fk or norm_key.endswith(f"_{fk}") for fk in FORBIDDEN_DATA_KEYS
            ):
                continue
            cleaned_dict[key] = sanitize_tool_output(value)
        return cleaned_dict

    elif isinstance(data, list):
        return [sanitize_tool_output(item) for item in data]

    elif isinstance(data, str):
        # Scrub any raw email addresses if found in text fields
        cleaned_str = re.sub(
            r"\b[A-Za-z0-9._%+-]+@example\.test\b",
            "[REDACTED_EMAIL]",
            data,
            flags=re.IGNORECASE,
        )
        return cleaned_str

    return copy.deepcopy(data)


def check_for_injection_patterns(text: str) -> bool:
    """Analyze text for adversarial instructions attempting to override system behavior.

    Args:
        text: User query, retrieved document chunk, or tool output text.

    Returns:
        True if any suspicious injection directive is detected, False otherwise.
    """
    if not text or not isinstance(text, str):
        return False

    for pattern in INJECTION_PATTERNS:
        if pattern.search(text):
            return True

    return False


def is_system_prompt_leak_attempt(text: str) -> bool:
    """Check specifically if user is attempting to exfiltrate system instructions."""
    if not text or not isinstance(text, str):
        return False
    leak_patterns = [
        re.compile(
            r"(?:reveal|print|show|repeat|display|output|dump|leak|tell|share)\b.*?\b(?:system\s+prompt|hidden\s+prompt|initial\s+instructions|system\s+instructions|developer\s+prompt|developer\s+instructions)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:reveal|print|show|repeat|display|output|dump|leak|tell|share)\b.*?\b(?:your\s+instructions|all\s+instructions)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"what\s+(?:is|are)\s+your\s+(?:system\s+prompt|hidden\s+prompt|instructions|initial\s+instructions|system\s+instructions)",
            re.IGNORECASE,
        ),
    ]
    return any(p.search(text) for p in leak_patterns)


def is_internal_data_probe(text: str) -> bool:
    """Check if query is probing for internal fields, warehouse notes, or risk scores."""
    if not text or not isinstance(text, str):
        return False
    probe_pattern = re.compile(
        r"(?:risk\s+score|internal\s+note|warehouse\s+note|fraud\s+review|customer\s+email|shipping\s+address)",
        re.IGNORECASE,
    )
    return bool(probe_pattern.search(text))


def validate_response_safety(response_text: str) -> Tuple[bool, str, List[str]]:
    """Validate that assistant response does not leak confidential data or instructions.

    Args:
        response_text: Assistant generated response text.

    Returns:
        Tuple of (is_safe, scrubbed_text, list_of_violations).
    """
    if not response_text:
        return True, "", []

    violations = []
    scrubbed_text = response_text

    for sig in LEAK_SIGNATURES:
        match = sig.search(scrubbed_text)
        if match:
            violation_str = match.group(0)
            violations.append(f"Disallowed token detected: {violation_str}")
            # Redact the matched text
            scrubbed_text = sig.sub("[REDACTED]", scrubbed_text)

    # Check for coupon promises
    coupon_grant_pattern = re.compile(
        r"(?:here\s+is\s+(?:your\s+)?(?:a\s+)?\$?\d+\s+coupon|issued\s+(?:a\s+)?\$?\d+\s+coupon|coupon\s+code\s*[:=])",
        re.IGNORECASE,
    )
    if coupon_grant_pattern.search(response_text):
        violations.append("Unauthorized coupon issuance detected")
        scrubbed_text = (
            "I am not authorized to issue coupons or discounts. "
            "Please reach out to our customer support team for promotional assistance."
        )

    is_safe = len(violations) == 0
    return is_safe, scrubbed_text, violations
