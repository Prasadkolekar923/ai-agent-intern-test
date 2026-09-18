"""Structured observability and JSON logging utilities.

Emits structured JSON logs per interaction turn containing timestamps,
retrieved context citations, tool invocations, and handoff flags without
ever leaking secrets or sensitive credentials.
"""

import copy
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("aster_row_agent")

# Sensitive key names and value patterns to scrub
SECRET_KEY_NAMES = {
    "api_key",
    "groq_api_key",
    "anthropic_api_key",
    "secret",
    "token",
    "authorization",
    "password",
    "access_token",
}

SECRET_VALUE_PATTERNS = [
    re.compile(r"\bgsk_[A-Za-z0-9_-]{20,}\b", re.IGNORECASE),
    re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b", re.IGNORECASE),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b", re.IGNORECASE),
    re.compile(r"\bBearer\s+[A-Za-z0-9_.-]{10,}\b", re.IGNORECASE),
]


def scrub_secrets(obj: Any) -> Any:
    """Recursively redact secrets and API keys from arbitrary data structures.

    Args:
        obj: Dictionary, list, string, or primitive to scrub.

    Returns:
        Scrubbed copy of data structure with secrets redacted.
    """
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            norm_k = str(k).lower().strip()
            if norm_k in SECRET_KEY_NAMES or any(sk in norm_k for sk in ("api_key", "secret_key", "auth_token")):
                cleaned[k] = "[REDACTED_SECRET]"
            else:
                cleaned[k] = scrub_secrets(v)
        return cleaned

    elif isinstance(obj, list):
        return [scrub_secrets(item) for item in obj]

    elif isinstance(obj, str):
        cleaned_str = obj
        for pattern in SECRET_VALUE_PATTERNS:
            cleaned_str = pattern.sub("[REDACTED_SECRET]", cleaned_str)
        return cleaned_str

    return copy.deepcopy(obj)


def format_turn_log(
    session_id: str,
    user_message: str,
    resolved_focus: Optional[Dict[str, Any]],
    retrieved_chunks: List[Dict[str, Any]],
    tool_calls: List[Dict[str, Any]],
    final_response: str,
    handoff_flag: bool,
    handoff_reason: Optional[str] = None,
    errors: Optional[List[str]] = None,
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """Format interaction turn details into structured dictionary per PRD spec.

    Fields:
    - session_id
    - timestamp (ISO-8601 UTC)
    - user_message
    - resolved_focus
    - retrieved_chunks (file + heading + score)
    - tool_calls (name + args + sanitized result)
    - final_response
    - handoff_flag
    - errors

    Args:
        session_id: Unique session identifier.
        user_message: Raw user message.
        resolved_focus: Session focus slot dict (last_order_id, last_topic).
        retrieved_chunks: List of chunk metadata dicts (file, heading, score).
        tool_calls: List of tool invocation dicts (name, args, result).
        final_response: Generated assistant answer.
        handoff_flag: Boolean indicating if human handoff is recommended.
        handoff_reason: Optional reason string for handoff.
        errors: Optional list of error messages.
        timestamp: Optional explicit ISO timestamp.

    Returns:
        Structured dictionary safe from secret leakage.
    """
    ts = timestamp or datetime.now(timezone.utc).isoformat()

    # Format chunks to guaranteed {file, heading, score}
    formatted_chunks = []
    for chunk in retrieved_chunks:
        formatted_chunks.append({
            "file": chunk.get("file") or chunk.get("filename") or "",
            "heading": chunk.get("heading") or "",
            "score": chunk.get("score") if chunk.get("score") is not None else chunk.get("relevance_score", 0.0),
        })

    # Format tool calls to guaranteed {name, args, result}
    formatted_tool_calls = []
    for tc in tool_calls:
        formatted_tool_calls.append({
            "name": str(tc.get("name", "")),
            "args": tc.get("args", {}),
            "result": tc.get("result", {}),
        })

    raw_entry = {
        "session_id": str(session_id),
        "timestamp": ts,
        "user_message": str(user_message),
        "resolved_focus": resolved_focus or {},
        "retrieved_chunks": formatted_chunks,
        "tool_calls": formatted_tool_calls,
        "final_response": str(final_response),
        "handoff_flag": bool(handoff_flag),
        "handoff_reason": handoff_reason,
        "errors": [str(e) for e in (errors or [])],
    }

    # Guarantee secret scrubbing
    return scrub_secrets(raw_entry)


def format_turn_log_json(
    session_id: str,
    user_message: str,
    resolved_focus: Optional[Dict[str, Any]],
    retrieved_chunks: List[Dict[str, Any]],
    tool_calls: List[Dict[str, Any]],
    final_response: str,
    handoff_flag: bool,
    handoff_reason: Optional[str] = None,
    errors: Optional[List[str]] = None,
    timestamp: Optional[str] = None,
) -> str:
    """Format interaction turn details into a single structured JSON log line.

    Args:
        Same as format_turn_log.

    Returns:
        Single-line JSON string.
    """
    entry = format_turn_log(
        session_id=session_id,
        user_message=user_message,
        resolved_focus=resolved_focus,
        retrieved_chunks=retrieved_chunks,
        tool_calls=tool_calls,
        final_response=final_response,
        handoff_flag=handoff_flag,
        handoff_reason=handoff_reason,
        errors=errors,
        timestamp=timestamp,
    )
    return json.dumps(entry, ensure_ascii=False, default=str)


def log_turn(log_entry_or_json: Any) -> None:
    """Emit the structured turn log line via standard logging.

    Args:
        log_entry_or_json: Dict from format_turn_log or JSON string.
    """
    if isinstance(log_entry_or_json, dict):
        line = json.dumps(log_entry_or_json, ensure_ascii=False, default=str)
    else:
        line = str(log_entry_or_json)

    logger.info(line)
