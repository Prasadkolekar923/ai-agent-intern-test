"""Command-line interface for the Aster & Row Support Agent (Phase 7).

Provides an interactive REPL loop or single-prompt runner that:
- Maintains a consistent session ID across turns (or accepts a custom --session-id).
- Clearly separates and formats the final answer, source citations, and human handoff notices.
- Supports a --debug flag to output Phase 6 structured JSON observability logs.
- Gracefully handles exit commands, EOF, and interruptions.
"""

import argparse
import sys
import uuid
from typing import List, Optional

from app.agent import AgentResponse, SupportAgent


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        args: Optional list of argument strings (defaults to sys.argv[1:]).

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Aster & Row AI Support Agent CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Session identifier for multi-turn conversations (defaults to a random 8-character ID)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print structured observability logs for each turn",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        default=None,
        help="Run a single prompt non-interactively and exit",
    )
    return parser.parse_args(args)


def format_response_display(response: AgentResponse) -> str:
    """Format an agent response with answer, sources, and handoff clearly separated.

    Args:
        response: AgentResponse object from SupportAgent.

    Returns:
        Cleanly formatted multiline string for user display.
    """
    lines: List[str] = [f"Agent: {response.answer}"]

    if response.sources:
        # Deduplicate sources while preserving insertion order
        seen = set()
        unique_sources = []
        for s in response.sources:
            file_name = s.get("file") or s.get("filename") or ""
            heading = s.get("heading") or ""
            formatted = f"{file_name} > {heading}" if heading else file_name
            if formatted and formatted not in seen:
                seen.add(formatted)
                unique_sources.append(formatted)

        if unique_sources:
            lines.append(f"\nSources cited: [{', '.join(unique_sources)}]")

    if response.handoff_recommended:
        reason = response.handoff_reason or "Customer support representative requested"
        lines.append(f"\n[Note: Human agent handoff triggered - {reason}]")

    return "\n".join(lines)


def run_repl(agent: SupportAgent, session_id: str) -> None:
    """Run interactive REPL loop for multi-turn customer support.

    Args:
        agent: Initialized SupportAgent instance.
        session_id: Active session identifier.
    """
    print("=" * 60)
    print(f"Aster & Row Support Agent (Session: {session_id})")
    print("Powered by Groq")
    print("Commands: 'exit' or 'quit' to end, '/help' for commands.")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("Customer: ").strip()
            if not user_input:
                continue

            lowered = user_input.lower()
            if lowered in ("exit", "quit"):
                print("Ending session. Goodbye!")
                break

            if lowered == "/help":
                print("\nAvailable commands:")
                print("  exit, quit   - Terminate the chat session")
                print("  /session     - Display active session ID and focused entities")
                print("  /reset       - Clear conversation history and entity focus")
                print("  /help        - Show this help message\n")
                continue

            if lowered == "/session":
                focus = agent.session_manager.get_focus(session_id)
                print(f"\n[Session ID: {session_id}]")
                if focus:
                    print(f"[Focus: Order ID={focus.last_order_id or 'None'}, Topic={focus.last_topic or 'None'}]\n")
                continue

            if lowered == "/reset":
                agent.session_manager.clear_session(session_id)
                print(f"\n[Session '{session_id}' cleared]\n")
                continue

            # Process turn
            response = agent.handle_message(user_input, session_id=session_id)

            print()
            print(format_response_display(response))
            print("-" * 60 + "\n")

        except (KeyboardInterrupt, EOFError):
            print("\nEnding session. Goodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}\n")


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point.

    Args:
        argv: Optional list of command line argument strings.

    Returns:
        Exit code (0 for success, non-zero on fatal initialization error).
    """
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    args = parse_args(argv)
    session_id = args.session_id or str(uuid.uuid4())[:8]

    try:
        agent = SupportAgent(debug=args.debug)
    except Exception as e:
        print(f"Error initializing agent: {e}", file=sys.stderr)
        return 1

    # Single-prompt mode
    if args.prompt:
        response = agent.handle_message(args.prompt, session_id=session_id)
        print(format_response_display(response))
        return 0

    # Interactive REPL mode
    run_repl(agent, session_id=session_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
