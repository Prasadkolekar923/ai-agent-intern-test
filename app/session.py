"""Session and multi-turn state management.

Tracks recent conversation turns and focused entities (order ID, active topic)
per session to resolve follow-up questions cleanly without cross-session leakage.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class FocusSlot:
    """Current focus topic/entities for the active session."""
    last_order_id: Optional[str] = None
    last_topic: Optional[str] = None


@dataclass
class SessionState:
    """Session history and focused entity tracking strictly scoped to a session_id."""
    session_id: str
    turns: List[Dict[str, str]] = field(default_factory=list)
    focus: FocusSlot = field(default_factory=FocusSlot)


class SessionManager:
    """In-memory session manager handling multi-turn sessions."""

    def __init__(self, max_turns: int = 6):
        self.max_turns = max_turns
        self.sessions: Dict[str, SessionState] = {}

    def get_or_create_session(self, session_id: str) -> SessionState:
        """Retrieve existing session or instantiate a new one strictly keyed by session_id."""
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionState(session_id=session_id)
        return self.sessions[session_id]

    def clear_session(self, session_id: str) -> None:
        """Clear conversation history and focus for a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]

    def add_turn(self, session_id: str, user_message: str, assistant_response: str) -> None:
        """Add user/assistant exchange to session history."""
        session = self.get_or_create_session(session_id)
        session.turns.append({"role": "user", "content": user_message})
        session.turns.append({"role": "assistant", "content": assistant_response})
        # Keep recent bounded turns
        if len(session.turns) > self.max_turns * 2:
            session.turns = session.turns[-self.max_turns * 2:]

    def update_focus(
        self,
        session_id: str,
        order_id: Optional[str] = None,
        topic: Optional[str] = None,
    ) -> None:
        """Update focus slot for follow-up resolution."""
        session = self.get_or_create_session(session_id)
        if order_id:
            session.focus.last_order_id = order_id.strip().upper()
        if topic:
            session.focus.last_topic = topic.strip()

    def get_focus(self, session_id: str) -> FocusSlot:
        """Retrieve active focus slots for a session."""
        session = self.get_or_create_session(session_id)
        return session.focus

    def resolve_elliptical_query(
        self,
        session_id: str,
        user_message: str,
    ) -> Tuple[str, Optional[str], Optional[str]]:
        """Resolve elliptical questions using active focus slots.

        Detects elliptical order follow-ups (e.g., 'When will it arrive?', 'Where is it?')
        and elliptical topic follow-ups (e.g., 'What about Canada?', 'How about exchanges?').

        Returns:
            Tuple of (resolved_user_message, resolved_order_id, resolved_topic).
        """
        session = self.get_or_create_session(session_id)
        focus = session.focus

        resolved_msg = user_message.strip()
        resolved_order_id = None
        resolved_topic = None

        has_ord_id = bool(re.search(r"\bORD-\d+\b", user_message, re.IGNORECASE))

        # Check for order follow-up when an order is already in focus
        if focus.last_order_id and not has_ord_id:
            order_followup_patterns = [
                r"^\s*(?:when\s+(?:will\s+it|does\s+it|will\s+my\s+order)\s+arrive\??)\s*$",
                r"^\s*(?:where\s+is\s+it\??)\s*$",
                r"^\s*(?:status\s+(?:of\s+it|update)\??)\s*$",
                r"^\s*(?:has\s+it\s+shipped\??)\s*$",
                r"^\s*(?:can\s+i\s+track\s+it\??)\s*$",
                r"^\s*(?:is\s+it\s+delayed\??)\s*$",
                r"^\s*(?:track\s+it\??)\s*$",
                r"^\s*(?:when\s+will\s+i\s+get\s+it\??)\s*$",
            ]
            is_order_followup = any(re.search(pat, user_message, re.IGNORECASE) for pat in order_followup_patterns)
            general_pronoun_order = (
                re.search(r"\b(?:it|my order|the package|the shipment)\b", user_message, re.IGNORECASE)
                and re.search(r"\b(?:arrive|deliver|delivery|status|ship|shipped|tracking|where|when|cancel)\b", user_message, re.IGNORECASE)
            )

            if is_order_followup or general_pronoun_order:
                resolved_order_id = focus.last_order_id
                resolved_msg = f"{user_message} (Referring to order {focus.last_order_id})"

        # Check for topic follow-up when a topic is in focus
        if focus.last_topic:
            topic_followup_patterns = [
                r"^\s*(?:what\s+about|how\s+about|and\s+for|what\s+if)\s+([a-zA-Z\s]+)\??\s*$",
                r"^\s*(?:is\s+that\s+true\s+for|does\s+that\s+apply\s+to)\s+([a-zA-Z\s]+)\??\s*$",
            ]
            for pat in topic_followup_patterns:
                match = re.search(pat, user_message, re.IGNORECASE)
                if match:
                    sub_topic = match.group(1).strip()
                    resolved_topic = focus.last_topic
                    resolved_msg = f"{user_message} (Regarding {focus.last_topic}: {sub_topic})"
                    break

        return resolved_msg, resolved_order_id, resolved_topic

    def get_focus_prompt_snippet(self, session_id: str) -> Optional[str]:
        """Generate a concise focus context string for lean prompt injection."""
        session = self.get_or_create_session(session_id)
        focus = session.focus
        parts = []
        if focus.last_order_id:
            parts.append(f"Active order ID in focus: {focus.last_order_id}")
        if focus.last_topic:
            parts.append(f"Active conversation topic in focus: {focus.last_topic}")
        if parts:
            return "SESSION CONTEXT: " + " | ".join(parts)
        return None
