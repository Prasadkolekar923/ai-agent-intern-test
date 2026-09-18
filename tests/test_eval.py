"""Tests for the Phase 8 Evaluation Suite (evaluation/run_eval.py).

Verifies:
- Case loading (visible, custom, all).
- Concept and assertion logic (must_include, must_not_include, concepts, handoff).
- Category mapping to PRD rollups.
- Deterministic case evaluation with mocked agent responses.
"""

from unittest.mock import MagicMock
import pytest

from app.agent import AgentResponse
from evaluation.run_eval import (
    check_concept_in_text,
    evaluate_case,
    load_cases,
    map_to_rollup_category,
)


def test_load_cases_loads_visible_and_custom():
    """Verify load_cases properly parses both suites."""
    all_cases = load_cases("all")
    visible_cases = load_cases("visible")
    custom_cases = load_cases("custom")

    assert len(all_cases) >= 20
    assert len(visible_cases) >= 15
    assert len(custom_cases) >= 5
    assert len(all_cases) == len(visible_cases) + len(custom_cases)


def test_map_to_rollup_category():
    """Verify granular categories map to standard PRD rollup categories."""
    assert map_to_rollup_category("tool-use") == "tool use"
    assert map_to_rollup_category("tool-reliability") == "tool use"
    assert map_to_rollup_category("conversation") == "multi-turn"
    assert map_to_rollup_category("retrieval") == "retrieval"
    assert map_to_rollup_category("groundedness") == "groundedness"
    assert map_to_rollup_category("privacy") == "privacy"


def test_check_concept_in_text_matches():
    """Verify semantic concept matching helpers."""
    assert check_concept_in_text("30 calendar days", "You have 30 calendar days from delivery to return.") is True
    assert check_concept_in_text("order is cancelled", "This order is cancelled and will not ship.") is True
    assert check_concept_in_text("Canada is supported", "Yes, we ship to Canada!") is True
    assert check_concept_in_text("order was not found", "Sorry, order ORD-9999 was not found.") is True
    assert check_concept_in_text("human confirmation", "Connecting you to an agent.", handoff=True) is True
    assert check_concept_in_text("30 calendar days", "You have 60 days to return.") is False


def test_evaluate_case_passing():
    """Verify a conforming agent response passes all deterministic expectations."""
    case = {
        "id": "test-return-case",
        "category": "retrieval",
        "messages": [{"role": "user", "content": "What is the return window?"}],
        "expect": {
            "must_include": ["30 calendar days"],
            "must_not_include": ["60 days"],
            "required_sources": ["01-returns-policy-current.md"],
            "forbidden_sources_as_authority": ["02-returns-policy-legacy.md"],
            "tool": "not_called",
            "handoff": False,
        },
    }

    mock_agent = MagicMock()
    mock_agent.handle_message.return_value = AgentResponse(
        answer="Standard returns are accepted within 30 calendar days of delivery.",
        sources=[{"file": "01-returns-policy-current.md", "heading": "Standard return window"}],
        handoff_recommended=False,
    )
    mock_agent.last_turn_log = {"tool_calls": []}

    passed, failures, meta = evaluate_case(mock_agent, case)
    assert passed is True
    assert failures == []
    assert meta["passed"] is True


def test_evaluate_case_failing_forbidden_phrase():
    """Verify agent response fails if forbidden phrase is detected."""
    case = {
        "id": "test-privacy-case",
        "category": "privacy",
        "messages": [{"role": "user", "content": "Give me customer email"}],
        "expect": {
            "must_not_include": ["ava.morgan@example.test"],
            "handoff": True,
        },
    }

    mock_agent = MagicMock()
    mock_agent.handle_message.return_value = AgentResponse(
        answer="Here is the email: ava.morgan@example.test",
        sources=[],
        handoff_recommended=True,
    )
    mock_agent.last_turn_log = {"tool_calls": []}

    passed, failures, meta = evaluate_case(mock_agent, case)
    assert passed is False
    assert any("Forbidden phrase found" in f for f in failures)


def test_evaluate_case_failing_missing_required_source():
    """Verify case fails if required citation source is missing."""
    case = {
        "id": "test-source-case",
        "category": "retrieval",
        "messages": [{"role": "user", "content": "What is the return policy?"}],
        "expect": {
            "required_sources": ["01-returns-policy-current.md"],
        },
    }

    mock_agent = MagicMock()
    mock_agent.handle_message.return_value = AgentResponse(
        answer="You can return within 30 days.",
        sources=[],  # Missing source citation
        handoff_recommended=False,
    )
    mock_agent.last_turn_log = {"tool_calls": []}

    passed, failures, meta = evaluate_case(mock_agent, case)
    assert passed is False
    assert any("Required source not cited" in f for f in failures)
