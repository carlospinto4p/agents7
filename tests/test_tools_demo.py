"""Unit tests for `agents7.tools.demo`'s pure business logic.

The generic tool-calling loop (schema derivation, JSON
serialization, error feedback, the `ToolAgent` round-trip) is
already covered by `test_tool_agent.py`, `test_tool_registry.py`,
and `test_tool_schema.py` against their own fixture tools — this
file only covers what is specific to these functions: catalog
matching, stock lookup, and order costing.
"""

import pytest
from agents7.tools.demo import (
    PRODUCTS,
    STOCK,
    OrderTotal,
    Product,
    StockLevel,
    check_stock,
    find_product,
    order_total,
)

# ── find_product ─────────────────────────────────────────────


def test_find_product_matches_by_word():
    """A partial name matches the product."""
    found = find_product("blue mug")
    assert [p.sku for p in found.matches] == ["MUG-01"]


def test_find_product_is_case_insensitive():
    """Matching ignores case on both sides."""
    assert find_product("BLUE MUG").matches[0].sku == "MUG-01"


def test_find_product_matches_a_plural_query():
    """A natural plural still finds the singular product."""
    assert find_product("blue mugs").matches[0].sku == "MUG-01"


def test_find_product_requires_every_word():
    """All query words must appear, not just one."""
    assert find_product("blue teapot").matches == []


def test_find_product_returns_multiple_matches():
    """A query matching several products returns them all."""
    skus = {p.sku for p in find_product("ceramic mugs").matches}
    assert skus == {"MUG-01", "MUG-02"}


def test_find_product_no_match_is_empty():
    """An unknown product yields no matches, not an error."""
    assert find_product("hovercraft").matches == []


def test_find_product_blank_query_is_empty():
    """A blank query does not match the whole catalog."""
    assert find_product("   ").matches == []


def test_find_product_echoes_the_query():
    """The result carries the query that produced it."""
    assert find_product("teapot").query == "teapot"


def test_find_product_returns_products():
    """Matches are `Product` models carrying a price."""
    product = find_product("teapot").matches[0]
    assert isinstance(product, Product)
    assert product.price_cents == 4599


# ── check_stock ──────────────────────────────────────────────


def test_check_stock_returns_level():
    """A known SKU reports its units and warehouse."""
    level = check_stock("MUG-01")
    assert isinstance(level, StockLevel)
    assert level.units == 42
    assert level.warehouse == "EU-1"


def test_check_stock_reports_zero_units():
    """An out-of-stock SKU reports zero rather than raising."""
    assert check_stock("MUG-02").units == 0


def test_check_stock_unknown_sku_raises():
    """An unknown SKU raises `LookupError`."""
    with pytest.raises(LookupError, match="NOPE-99"):
        check_stock("NOPE-99")


# ── order_total ──────────────────────────────────────────────


def test_order_total_applies_tax_and_shipping():
    """A small order pays tax plus standard shipping."""
    total = order_total(1250, 3)
    assert isinstance(total, OrderTotal)
    assert total.subtotal_cents == 3750
    assert total.tax_cents == 712
    assert total.shipping_cents == 499
    assert total.total_cents == 4961


def test_order_total_waives_shipping_above_threshold():
    """A subtotal at the free-shipping threshold ships free."""
    total = order_total(2500, 2)
    assert total.subtotal_cents == 5000
    assert total.shipping_cents == 0


def test_order_total_express_always_charges():
    """Express shipping is charged even above the threshold."""
    total = order_total(2500, 2, express=True)
    assert total.shipping_cents == 1299


def test_order_total_rejects_zero_quantity():
    """A non-positive quantity raises `ValueError`."""
    with pytest.raises(ValueError, match="quantity"):
        order_total(1250, 0)


def test_order_total_rejects_negative_price():
    """A negative unit price raises `ValueError`."""
    with pytest.raises(ValueError, match="unit_price_cents"):
        order_total(-1, 1)


# ── Catalog integrity ────────────────────────────────────────


def test_every_product_has_stock_entry():
    """Stock data covers exactly the catalog's SKUs."""
    assert {p.sku for p in PRODUCTS} == set(STOCK)
