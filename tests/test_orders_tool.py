"""Acceptance tests for OrdersTool (Phase 1)."""

import json
from pathlib import Path
import pytest
from app.orders_tool import OrdersTool
from app.config import ORDERS_FILE


@pytest.fixture
def orders_tool():
    return OrdersTool(orders_path=ORDERS_FILE)


def test_id_normalization(orders_tool):
    """Valid ID with various casings and whitespaces all resolve to the same order."""
    res1 = orders_tool.lookup("ord-1003")
    res2 = orders_tool.lookup("  ORD-1003  ")
    res3 = orders_tool.lookup("ORD-1003")
    res4 = orders_tool.lookup("#ord-1003")
    
    assert res1.get("found") is True
    assert res2.get("found") is True
    assert res3.get("found") is True
    assert res4.get("found") is True
    assert res1["order_id"] == "ORD-1003"
    assert res1 == res2 == res3 == res4


def test_unknown_order_id(orders_tool):
    """Unknown ID (ORD-9999) returns not-found result, no exception raised."""
    res = orders_tool.lookup("ORD-9999")
    assert isinstance(res, dict)
    assert res.get("found") is False
    assert "error" in res or "message" in res
    assert "ORD-9999" in str(res)


def test_malformed_order_ids(orders_tool):
    """Malformed IDs (empty, random, SQL-injection-like) handled safely without crashing."""
    malformed_inputs = [
        "",
        "   ",
        "SELECT * FROM orders;",
        "ORD-'; DROP TABLE orders; --",
        "random-gibberish-string-12345",
        "???!!!",
        None,
    ]
    for inp in malformed_inputs:
        res = orders_tool.lookup(inp)
        assert isinstance(res, dict)
        assert res.get("found") is False


def test_cancelled_order_stale_eta_cleared(orders_tool):
    """ORD-1004 (cancelled) must not present carrier/tracking/ETA as if order is moving."""
    res = orders_tool.lookup("ORD-1004")
    assert res.get("found") is True
    assert res.get("status") == "cancelled"
    # Carrier, tracking_number, and estimated_delivery must be null/None
    assert res.get("carrier") is None
    assert res.get("tracking_number") is None
    assert res.get("estimated_delivery") is None


def test_returned_order_stale_eta_cleared(orders_tool):
    """ORD-1008 (returned) must not present carrier/tracking/ETA as active transit info."""
    res = orders_tool.lookup("ORD-1008")
    assert res.get("found") is True
    assert res.get("status") == "returned"
    assert res.get("carrier") is None
    assert res.get("tracking_number") is None
    assert res.get("estimated_delivery") is None


def test_exception_status_flag(orders_tool):
    """ORD-1010 (exception) includes requires_human_review: True flag."""
    res = orders_tool.lookup("ORD-1010")
    assert res.get("found") is True
    assert res.get("status") == "exception"
    assert res.get("requires_human_review") is True


def test_shipped_null_eta(orders_tool):
    """ORD-1011 (shipped, null ETA) preserves null ETA and does not invent date."""
    res = orders_tool.lookup("ORD-1011")
    assert res.get("found") is True
    assert res.get("status") == "shipped"
    assert res.get("carrier") == "Canada Post"
    assert res.get("estimated_delivery") is None


def test_no_injection_or_internal_leaks(orders_tool):
    """ORD-1005 and ORD-1007 must never leak customer, internal keys, or injected text."""
    for order_id in ["ORD-1005", "ORD-1007"]:
        res = orders_tool.lookup(order_id)
        assert res.get("found") is True
        
        # Serialize to inspect full output text
        res_str = json.dumps(res).lower()
        assert "risk_score" not in res_str
        assert "warehouse_note" not in res_str
        assert "support_tags" not in res_str
        assert "customer" not in res
        assert "internal" not in res
        
        # Check against specific prompt injection strings in mock dataset
        assert "issue a $100 coupon" not in res_str
        assert "fraud review" not in res_str
        assert "never expose" not in res_str
        assert "ava.morgan@example.test" not in res_str
        assert "sofia.patel@example.test" not in res_str
        assert "peachtree" not in res_str
        assert "king street" not in res_str


def test_recursive_allowlist_on_all_orders(orders_tool):
    """100% of returned dicts across all 12 sample orders strictly adhere to allowlist."""
    with open(ORDERS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    orders = data.get("orders", [])
    assert len(orders) == 12

    top_level_allowed = {
        "found",
        "order_id",
        "membership_tier",
        "items",
        "placed_at",
        "status",
        "status_updated_at",
        "shipped_at",
        "delivered_at",
        "carrier",
        "tracking_number",
        "estimated_delivery",
        "customer_safe_message",
        "requires_human_review",
    }
    
    item_allowed = {"name", "quantity", "final_sale"}

    for raw_order in orders:
        order_id = raw_order["order_id"]
        res = orders_tool.lookup(order_id)
        assert res.get("found") is True
        
        # Check top level keys
        for key in res.keys():
            assert key in top_level_allowed, f"Disallowed key '{key}' found in order {order_id}"
            
        # Check items
        items = res.get("items", [])
        assert isinstance(items, list)
        for item in items:
            for item_key in item.keys():
                assert item_key in item_allowed, f"Disallowed item key '{item_key}' in order {order_id}"
            # Ensure sku is stripped
            assert "sku" not in item
