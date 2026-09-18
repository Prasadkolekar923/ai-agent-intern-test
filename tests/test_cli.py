"""Tests for Phase 7 Command-Line Interface (app/cli.py).

Verifies:
- Argument parsing: session-id, debug, and non-interactive prompt flags.
- Clean separation of agent answer, source citations, and handoff notice.
- Interactive REPL loop operation and session persistence across turns.
- Handling of termination signals ('exit', 'quit', EOFError, KeyboardInterrupt).
- REPL slash commands (/help, /session, /reset).
- Single-prompt execution mode (-p / --prompt).
- Observability integration (--debug).
"""

from io import StringIO
from unittest.mock import MagicMock, patch
import pytest

from app.agent import AgentResponse
from app.cli import format_response_display, main, parse_args, run_repl


# =====================================================================
# Unit Tests: Argument Parsing
# =====================================================================

def test_parse_args_defaults():
    """Verify default arguments when no CLI options are provided."""
    args = parse_args([])
    assert args.session_id is None
    assert args.debug is False
    assert args.prompt is None


def test_parse_args_custom_values():
    """Verify custom CLI options are parsed correctly."""
    args = parse_args(["--session-id", "custom_session_99", "--debug", "--prompt", "What is the return window?"])
    assert args.session_id == "custom_session_99"
    assert args.debug is True
    assert args.prompt == "What is the return window?"


def test_parse_args_short_prompt_flag():
    """Verify short flag -p works identically to --prompt."""
    args = parse_args(["-p", "Check ORD-1001"])
    assert args.prompt == "Check ORD-1001"


# =====================================================================
# Unit Tests: Response Output Formatting
# =====================================================================

def test_format_response_display_answer_only():
    """Verify display format when only an answer is present."""
    resp = AgentResponse(
        answer="Welcome to Aster & Row! How can I help you today?",
        sources=[],
        handoff_recommended=False,
    )
    formatted = format_response_display(resp)
    assert formatted == "Agent: Welcome to Aster & Row! How can I help you today?"
    assert "Sources cited:" not in formatted
    assert "Human agent handoff" not in formatted


def test_format_response_display_with_sources():
    """Verify sources cited are clearly separated and formatted."""
    resp = AgentResponse(
        answer="Returns are accepted within 30 days of delivery.",
        sources=[
            {"file": "01-returns-policy-current.md", "heading": "Standard return window"},
            {"file": "01-returns-policy-current.md", "heading": "Condition requirements"},
        ],
        handoff_recommended=False,
    )
    formatted = format_response_display(resp)
    assert "Agent: Returns are accepted within 30 days of delivery." in formatted
    assert "Sources cited: [01-returns-policy-current.md > Standard return window, 01-returns-policy-current.md > Condition requirements]" in formatted
    assert "Human agent handoff" not in formatted


def test_format_response_display_deduplicates_sources():
    """Verify duplicate source entries are cleanly deduplicated."""
    resp = AgentResponse(
        answer="Here is the return policy.",
        sources=[
            {"file": "01-returns-policy-current.md", "heading": "Standard return window"},
            {"file": "01-returns-policy-current.md", "heading": "Standard return window"},
        ],
        handoff_recommended=False,
    )
    formatted = format_response_display(resp)
    # Should only mention the source once
    assert formatted.count("01-returns-policy-current.md > Standard return window") == 1


def test_format_response_display_with_handoff():
    """Verify human handoff notice is prominently displayed with reason."""
    resp = AgentResponse(
        answer="I am connecting you with our support team.",
        sources=[],
        handoff_recommended=True,
        handoff_reason="Customer requested live human support",
    )
    formatted = format_response_display(resp)
    assert "Agent: I am connecting you with our support team." in formatted
    assert "[Note: Human agent handoff triggered - Customer requested live human support]" in formatted


def test_format_response_display_with_sources_and_handoff():
    """Verify display when both citations and human handoff notice are present."""
    resp = AgentResponse(
        answer="There is a conflicting policy on expedited shipping.",
        sources=[{"file": "02-shipping.md", "heading": "Expedited"}],
        handoff_recommended=True,
        handoff_reason="Conflicting policy detected across documents",
    )
    formatted = format_response_display(resp)
    assert "Agent: There is a conflicting policy on expedited shipping." in formatted
    assert "Sources cited: [02-shipping.md > Expedited]" in formatted
    assert "[Note: Human agent handoff triggered - Conflicting policy detected across documents]" in formatted


# =====================================================================
# Unit & Integration Tests: REPL Loop and Interactive Flow
# =====================================================================

def test_repl_exits_on_quit_command(capsys):
    """REPL loop should immediately exit when user types quit."""
    mock_agent = MagicMock()
    with patch("builtins.input", side_effect=["quit"]):
        run_repl(mock_agent, session_id="test_sess_quit")

    captured = capsys.readouterr()
    assert "Aster & Row Support Agent (Session: test_sess_quit)" in captured.out
    assert "Ending session. Goodbye!" in captured.out
    mock_agent.handle_message.assert_not_called()


def test_repl_exits_on_eof_error(capsys):
    """REPL loop should terminate cleanly on EOFError (e.g. Ctrl+D / closed pipe)."""
    mock_agent = MagicMock()
    with patch("builtins.input", side_effect=EOFError):
        run_repl(mock_agent, session_id="test_sess_eof")

    captured = capsys.readouterr()
    assert "Ending session. Goodbye!" in captured.out
    mock_agent.handle_message.assert_not_called()


def test_repl_handles_slash_commands(capsys):
    """Verify /help, /session, and /reset commands are handled within the loop."""
    mock_agent = MagicMock()
    mock_agent.session_manager.get_focus.return_value = MagicMock(last_order_id="ORD-1001", last_topic="returns")

    inputs = ["/help", "/session", "/reset", "exit"]
    with patch("builtins.input", side_effect=inputs):
        run_repl(mock_agent, session_id="test_sess_cmds")

    captured = capsys.readouterr()
    assert "Available commands:" in captured.out
    assert "[Session ID: test_sess_cmds]" in captured.out
    assert "ORD-1001" in captured.out
    assert "[Session 'test_sess_cmds' cleared]" in captured.out
    mock_agent.session_manager.clear_session.assert_called_once_with("test_sess_cmds")
    mock_agent.handle_message.assert_not_called()


def test_repl_processes_multi_turn_conversation(capsys):
    """Verify multi-turn interaction maintains session continuity."""
    mock_agent = MagicMock()
    mock_agent.handle_message.side_effect = [
        AgentResponse(answer="Could you provide your order ID?", sources=[], handoff_recommended=False),
        AgentResponse(answer="Order ORD-1005 has shipped!", sources=[], handoff_recommended=False),
    ]

    inputs = ["Where is my order?", "It is ORD-1005", "exit"]
    with patch("builtins.input", side_effect=inputs):
        run_repl(mock_agent, session_id="sess_multi_turn")

    assert mock_agent.handle_message.call_count == 2
    mock_agent.handle_message.assert_any_call("Where is my order?", session_id="sess_multi_turn")
    mock_agent.handle_message.assert_any_call("It is ORD-1005", session_id="sess_multi_turn")

    captured = capsys.readouterr()
    assert "Could you provide your order ID?" in captured.out
    assert "Order ORD-1005 has shipped!" in captured.out


# =====================================================================
# Unit & Integration Tests: main() Entry Point
# =====================================================================

def test_main_single_prompt_mode(capsys):
    """Verify --prompt executes a single turn and prints output without entering REPL."""
    with patch("app.cli.SupportAgent") as MockAgentClass:
        mock_instance = MockAgentClass.return_value
        mock_instance.handle_message.return_value = AgentResponse(
            answer="Standard return window is 30 days.",
            sources=[{"file": "01-returns.md", "heading": "Window"}],
            handoff_recommended=False,
        )

        exit_code = main(["--session-id", "prompt_sess", "--prompt", "What is the return window?"])

        assert exit_code == 0
        mock_instance.handle_message.assert_called_once_with("What is the return window?", session_id="prompt_sess")

    captured = capsys.readouterr()
    assert "Agent: Standard return window is 30 days." in captured.out
    assert "Sources cited: [01-returns.md > Window]" in captured.out
    # REPL banner should NOT be printed in single-prompt mode
    assert "Aster & Row Support Agent (Session:" not in captured.out


def test_main_passes_debug_flag_to_support_agent():
    """Verify --debug flag is passed to SupportAgent constructor."""
    with patch("app.cli.SupportAgent") as MockAgentClass:
        mock_instance = MockAgentClass.return_value
        mock_instance.handle_message.return_value = AgentResponse(answer="OK", sources=[], handoff_recommended=False)

        exit_code = main(["--debug", "--prompt", "Hello"])
        assert exit_code == 0
        MockAgentClass.assert_called_once_with(debug=True)


def test_main_handles_initialization_failure(capsys):
    """Verify main() handles SupportAgent initialization error gracefully."""
    with patch("app.cli.SupportAgent", side_effect=Exception("Initialization failed")):
        exit_code = main(["--prompt", "Hello"])
        assert exit_code == 1

    captured = capsys.readouterr()
    assert "Error initializing agent: Initialization failed" in captured.err
