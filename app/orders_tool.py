"""Order status lookup tool for Aster & Row.

Loads data/orders.json once at startup and provides sanitized, allowlisted
order information adhering to customer privacy rules and status precedence.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from app.config import ORDERS_FILE

logger = logging.getLogger(__name__)

# Strict allowlist of top-level fields permitted to be returned
ALLOWLISTED_FIELDS = [
    "order_id",
    "membership_tier",
    "placed_at",
    "status",
    "status_updated_at",
    "shipped_at",
    "delivered_at",
    "carrier",
    "tracking_number",
    "estimated_delivery",
    "customer_safe_message",
]

# Strict allowlist of item fields permitted to be returned
ALLOWLISTED_ITEM_FIELDS = [
    "name",
    "quantity",
    "final_sale",
]


class OrdersTool:
    """Manages order dataset and sanitized lookup operations."""

    def __init__(self, orders_path: Optional[Path] = None):
        """Initialize orders tool and load dataset from disk.

        Args:
            orders_path: Path to orders.json file. Defaults to config ORDERS_FILE.
        """
        self.orders_path = Path(orders_path) if orders_path else ORDERS_FILE
        self._orders_by_id: Dict[str, Dict[str, Any]] = {}
        self.snapshot_at: Optional[str] = None
        self.load_error: Optional[str] = None
        self._load_orders()

    def _load_orders(self) -> None:
        """Load and cache orders from disk once at startup."""
        if not self.orders_path.exists():
            self.load_error = f"Orders file not found at {self.orders_path}"
            logger.error(self.load_error)
            return

        try:
            with open(self.orders_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.snapshot_at = data.get("snapshot_at")
            for order in data.get("orders", []):
                oid = order.get("order_id", "").strip().upper()
                if oid:
                    self._orders_by_id[oid] = order
        except Exception as e:
            self.load_error = f"Failed to load orders dataset: {str(e)}"
            logger.error(self.load_error)

    def normalize_id(self, order_id: Any) -> str:
        """Normalize order ID by stripping whitespace and common punctuation.

        Handles variations like 'ord-1003', '#ORD-1003', ' ORD-1003 ', etc.

        Args:
            order_id: Raw user input.

        Returns:
            Normalized uppercase order ID or empty string if invalid.
        """
        if not order_id or not isinstance(order_id, str):
            return ""
        
        cleaned = order_id.strip()
        # Remove common leading punctuation such as #
        cleaned = re.sub(r"^[#\s]+", "", cleaned)
        # Remove trailing punctuation (periods, commas, semicolons)
        cleaned = re.sub(r"[.,;!?\s]+$", "", cleaned)
        # Standardize uppercase
        cleaned = cleaned.upper()
        return cleaned

    def lookup(self, order_id: Any) -> Dict[str, Any]:
        """Perform sanitized order lookup with status precedence applied.

        Guarantees:
        - Never leaks customer details (name, email, shipping address)
        - Never leaks internal fields (risk_score, warehouse_notes, support_tags)
        - Applies status precedence rules (clears stale transit data for cancelled/returned)
        - Flags exception orders for human review

        Args:
            order_id: User-provided order ID string.

        Returns:
            Dict containing sanitized order fields or not-found status.
        """
        if self.load_error:
            return {
                "found": False,
                "error": f"Order database unavailable: {self.load_error}",
            }

        norm_id = self.normalize_id(order_id)
        if not norm_id:
            return {
                "found": False,
                "error": "Invalid or missing order ID. Please provide a valid order ID.",
            }

        raw_order = self._orders_by_id.get(norm_id)
        if not raw_order:
            return {
                "found": False,
                "order_id": norm_id,
                "error": f"Order {norm_id} was not found in our records.",
            }

        # Build sanitized output adhering strictly to allowlisted fields
        sanitized: Dict[str, Any] = {"found": True}

        for field_name in ALLOWLISTED_FIELDS:
            sanitized[field_name] = raw_order.get(field_name)

        # Sanitize items list
        raw_items = raw_order.get("items", [])
        sanitized_items: List[Dict[str, Any]] = []
        for item in raw_items:
            sanitized_item = {}
            for item_field in ALLOWLISTED_ITEM_FIELDS:
                if item_field in item:
                    sanitized_item[item_field] = item[item_field]
            sanitized_items.append(sanitized_item)
        sanitized["items"] = sanitized_items

        # Apply status precedence logic
        status = (sanitized.get("status") or "").lower()

        # Rule 1: Stale transit data in cancelled/returned orders must be cleared
        if status in {"cancelled", "returned"}:
            sanitized["carrier"] = None
            sanitized["tracking_number"] = None
            sanitized["estimated_delivery"] = None

        # Rule 2: Shipped with null ETA - keep null, do not invent date
        elif status == "shipped" and sanitized.get("estimated_delivery") is None:
            sanitized["estimated_delivery"] = None

        # Rule 3: Exception status requires human review
        if status == "exception":
            sanitized["requires_human_review"] = True

        return sanitized
