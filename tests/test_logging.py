"""Tests for structured observability and JSON logging (Phase 6).

Guarantees tested:
- Generation of one structured JSON log line per interaction turn.
- Required fields present: session_id, timestamp, user_message, resolved_focus,
  retrieved_chunks (file+heading+score), tool_calls (name+args+sanitized result),
  final_response, handoff_flag, errors.
- Strict redacting of API keys, tokens, and secrets.
- Agent integration and --debug output mode.
"""

import json
from unittest.mock import MagicMock
import pytest

from app.agent import AgentResponse, SupportAgent
from app.kb_loader import KBChunk, load_and_chunk_documents
from app.logging_utils import (
    format_turn_log,
    format_turn_log_json,
    log_turn,
    scrub_secrets,
)
from app.orders_tool import OrdersTool
from app.retrieval import KBRetriever
from app.config import KNOWLEDGE_BASE_DIR


# =====================================================================
# Unit Tests: scrub_secrets
# =====================================================================

def test_scrub_secrets_purges_api_key_patterns():
    """Verify that Groq, Anthropic, and generic API keys in strings are redacted."""
    sensitive_payload = {
        "user_message": "My key is gsk_testsecretkey1234567890abcdef123456 and token sk-ant-api03-abcdef1234567890123456",
        "nested": {
            "auth_header": "Bearer secret_token_value_here_12345",
            "normal_field": "Ridge Daypack",
        },
    }

    cleaned = scrub_secrets(sensitive_payload)

    assert "gsk_testsecretkey" not in cleaned["user_message"]
    assert "sk-ant-api03" not in cleaned["user_message"]
    assert "[REDACTED_SECRET]" in cleaned["user_message"]
    assert "secret_token_value" not in cleaned["nested"]["auth_header"]
    assert "[REDACTED_SECRET]" in cleaned["nested"]["auth_header"]
    assert cleaned["nested"]["normal_field"] == "Ridge Daypack"


def test_scrub_secrets_redacts_sensitive_dict_keys():
    """Verify that dictionary keys associated with secrets are redacted."""
    payload = {
        "api_key": "gsk_dummy",
        "groq_api_key": "any_secret",
        "password": "mypassword123",
        "session_id": "session_123",
    }

    cleaned = scrub_secrets(payload)

    assert cleaned["api_key"] == "[REDACTED_SECRET]"
    assert cleaned["groq_api_key"] == "[REDACTED_SECRET]"
    assert cleaned["password"] == "[REDACTED_SECRET]"
    assert cleaned["session_id"] == "session_123"


# =====================================================================
# Unit Tests: format_turn_log & format_turn_log_json
# =====================================================================

def test_format_turn_log_contains_all_required_spec_fields():
    """Verify all 9 PRD-mandated fields are present and structured properly."""
    log_entry = format_turn_log(
        session_id="sess_001",
        user_message="Where is ORD-1007?",
        resolved_focus={"last_order_id": "ORD-1007", "last_topic": None},
        retrieved_chunks=[
            {"file": "01-returns.md", "heading": "Window", "score": 0.85},
        ],
        tool_calls=[
            {"name": "lookup_order", "args": {"order_id": "ORD-1007"}, "result": {"status": "shipped"}},
        ],
        final_response="Order ORD-1007 is shipped.",
        handoff_flag=False,
        handoff_reason=None,
        errors=[],
        timestamp="2026-08-15T12:00:00Z",
    )

    # Assert mandatory top-level fields
    assert log_entry["session_id"] == "sess_001"
    assert log_entry["timestamp"] == "2026-08-15T12:00:00Z"
    assert log_entry["user_message"] == "Where is ORD-1007?"
    assert log_entry["resolved_focus"]["last_order_id"] == "ORD-1007"
    assert len(log_entry["retrieved_chunks"]) == 1
    assert log_entry["retrieved_chunks"][0]["file"] == "01-returns.md"
    assert log_entry["retrieved_chunks"][0]["heading"] == "Window"
    assert log_entry["retrieved_chunks"][0]["score"] == 0.85
    assert len(log_entry["tool_calls"]) == 1
    assert log_entry["tool_calls"][0]["name"] == "lookup_order"
    assert log_entry["tool_calls"][0]["args"] == {"order_id": "ORD-1007"}
    assert log_entry["tool_calls"][0]["result"] == {"status": "shipped"}
    assert log_entry["final_response"] == "Order ORD-1007 is shipped."
    assert log_entry["handoff_flag"] is False
    assert log_entry["errors"] == []


def test_format_turn_log_json_produces_single_line_valid_json():
    """Verify format_turn_log_json produces a single-line, valid JSON string."""
    json_line = format_turn_log_json(
        session_id="sess_json",
        user_message="Hello",
        resolved_focus={},
        retrieved_chunks=[],
        tool_calls=[],
        final_response="Hi there!",
        handoff_flag=False,
    )

    # Must be valid single line JSON
    assert "\n" not in json_line
    parsed = json.loads(json_line)
    assert parsed["session_id"] == "sess_json"
    assert parsed["final_response"] == "Hi there!"


# =====================================================================
# Integration Tests: SupportAgent Observability Logging
# =====================================================================

def test_agent_populates_last_turn_log_on_deterministic_turn():
    """Agent records structured log even on fast-path deterministic turns."""
    agent = SupportAgent(orders_tool=OrdersTool())
    agent.handle_message("Where is my order?", session_id="test_turn_1")

    assert agent.last_turn_log is not None
    assert agent.last_turn_log["session_id"] == "test_turn_1"
    assert agent.last_turn_log["user_message"] == "Where is my order?"
    assert "Could you please provide your order ID" in agent.last_turn_log["final_response"]
    assert agent.last_turn_log["tool_calls"] == []
    assert agent.last_turn_log["retrieved_chunks"] == []
    assert agent.last_turn_log["handoff_flag"] is False

    # Check JSON counterpart
    assert agent.last_turn_log_json is not None
    parsed = json.loads(agent.last_turn_log_json)
    assert parsed["session_id"] == "test_turn_1"


def test_agent_records_tool_calls_and_chunks_in_turn_log():
    """Verify tool execution and retrieval provenance are recorded in structured log."""
    mock_client = MagicMock()

    # Turn with search_kb
    tc = MagicMock()
    tc.id = "tc_obs_1"
    tc.function.name = "search_kb"
    tc.function.arguments = json.dumps({"query": "standard return window"})

    resp_msg1 = MagicMock(content="", tool_calls=[tc])
    resp_msg2 = MagicMock(
        content="Standard returns are accepted within 30 days [01-returns-policy-current.md > Standard return window].",
        tool_calls=None,
    )

    mock_client.chat.completions.create.side_effect = [
        MagicMock(choices=[MagicMock(message=resp_msg1)]),
        MagicMock(choices=[MagicMock(message=resp_msg2)]),
    ]

    chunks = load_and_chunk_documents(KNOWLEDGE_BASE_DIR)
    retriever = KBRetriever(chunks)
    agent = SupportAgent(retriever=retriever, client=mock_client)

    agent.handle_message("What is the return window?", session_id="sess_obs_kb")

    log_entry = agent.last_turn_log
    assert log_entry is not None
    assert log_entry["session_id"] == "sess_obs_kb"
    assert len(log_entry["tool_calls"]) == 1
    assert log_entry["tool_calls"][0]["name"] == "search_kb"
    assert log_entry["tool_calls"][0]["args"]["query"] == "standard return window"
    # Chunks recorded with provenance
    assert len(log_entry["retrieved_chunks"]) > 0
    assert log_entry["retrieved_chunks"][0]["file"] == "01-returns-policy-current.md"
    assert "score" in log_entry["retrieved_chunks"][0]


def test_agent_never_logs_api_keys():
    """Ensure that even if a secret appears in query or error, it is scrubbed from the log."""
    agent = SupportAgent(orders_tool=OrdersTool())
    agent.handle_message(
        "Here is my key gsk_supersecretvalue12345678901234567890 for order",
        session_id="secret_scrub_sess",
    )

    log_json = agent.last_turn_log_json
    assert "gsk_supersecretvalue" not in log_json
    assert "[REDACTED_SECRET]" in log_json


def test_agent_debug_flag_prints_structured_json(capsys):
    """When debug=True, agent prints the structured JSON log line to stdout."""
    agent = SupportAgent(orders_tool=OrdersTool(), debug=True)
    agent.handle_message("Where is my order?", session_id="debug_sess")

    captured = capsys.readouterr()
    assert agent.last_turn_log_json in captured.out
    # Assert captured line is valid JSON
    parsed = json.loads(agent.last_turn_log_json)
    assert parsed["session_id"] == "debug_sess"
