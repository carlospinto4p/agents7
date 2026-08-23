"""Demo tools for a tiny in-memory store, used by config-based samples.

Used to demonstrate `ToolAgent`'s tool-calling round-trip: the three
functions below cooperate as the tools of one `ToolAgent`, called in
sequence to answer a customer question (find a product, check its
stock, cost an order) rather than recalling facts from the model's
weights.

Mirrors `agents7.models.demo`: a sample resolves its `tools` (and,
for structured-output agents, its `output_model`) to a dotted path a
`config.yml`-only sample can reference without owning a package of
its own. Nothing here imports the framework — a tool is just a
well-annotated function, and `ToolAgent` derives its JSON schema
from the signature and its description from the first docstring
line.

A tool that raises is not an error the agent dies on:
`ToolRegistry.execute` catches the exception and feeds
``"Error: <message>"`` back to the model, which can then correct
itself or explain the problem to the customer. `check_stock`
demonstrates this by raising `LookupError` on an unknown SKU.

Deliberately a small literal dataset: the demo is the tool-calling
round-trip, not a database. Keeping the data in process makes every
tool pure and deterministic. Money is handled in integer cents
throughout — no floats — so a total is exactly reproducible across
runs.
"""

import re

from pydantic import BaseModel


class Product(BaseModel):
    """A single product in the demo store catalog."""

    sku: str
    name: str
    price_cents: int


class ProductSearch(BaseModel):
    """The result of a catalog search.

    A wrapper rather than a bare ``list[Product]`` so the return is
    a single Pydantic model: `Tool.run` serializes those with
    `model_dump_json`, whereas a list falls back to `json.dumps`
    with ``default=str`` and would hand the model a Python ``repr``
    of each product instead of JSON.
    """

    query: str
    matches: list[Product]


class StockLevel(BaseModel):
    """Availability of one product in one warehouse."""

    sku: str
    units: int
    warehouse: str


class OrderTotal(BaseModel):
    """The costed breakdown of one order line."""

    subtotal_cents: int
    tax_cents: int
    shipping_cents: int
    total_cents: int


#: Sales tax applied to an order's subtotal, as a percentage.
TAX_PERCENT = 19

#: Flat shipping costs, in cents.
STANDARD_SHIPPING_CENTS = 499
EXPRESS_SHIPPING_CENTS = 1299

#: Subtotal at or above which standard shipping is free.
FREE_SHIPPING_CENTS = 5000

#: Every product the demo store knows about.
PRODUCTS: tuple[Product, ...] = (
    Product(sku="MUG-01", name="Blue ceramic mug", price_cents=1250),
    Product(sku="MUG-02", name="Green ceramic mug", price_cents=1250),
    Product(sku="TEA-07", name="Loose leaf green tea", price_cents=890),
    Product(sku="POT-03", name="Cast iron teapot", price_cents=4599),
    Product(sku="TRY-11", name="Bamboo serving tray", price_cents=2150),
)

#: Units on hand per SKU, and the warehouse holding them.
STOCK: dict[str, tuple[int, str]] = {
    "MUG-01": (42, "EU-1"),
    "MUG-02": (0, "EU-1"),
    "TEA-07": (7, "EU-2"),
    "POT-03": (3, "EU-1"),
    "TRY-11": (18, "EU-2"),
}


def find_product(query: str) -> ProductSearch:
    """Search the store catalog for products matching a query.

    :param query: Words to match against product names, case
        insensitive. Every word must appear in the name.
    :return: The search result, with an empty `matches` when
        nothing matches.
    """
    wanted = _tokens(query)
    matches = (
        [p for p in PRODUCTS if wanted <= _tokens(p.name)] if wanted else []
    )
    return ProductSearch(query=query, matches=matches)


def _tokens(text: str) -> set[str]:
    """Split text into normalized, singularized word tokens.

    Matching on tokens rather than raw substrings means a natural
    plural finds the product: a model asking for "blue mugs" should
    find the "Blue ceramic mug". Deliberately naive — a trailing
    ``s`` is the only inflection handled, which is enough for this
    catalog.

    :param text: The text to tokenize.
    :return: The set of normalized tokens.
    """
    return {
        _singular(word) for word in re.findall(r"[a-z0-9]+", text.lower())
    }


def _singular(word: str) -> str:
    """Strip a trailing plural ``s`` from a word.

    :param word: A lowercased word token.
    :return: The word without its trailing ``s``, when long enough
        that dropping it is safe.
    """
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


def check_stock(sku: str) -> StockLevel:
    """Report how many units of a SKU are currently in stock.

    :param sku: The product's stock-keeping unit, as returned by
        `find_product`.
    :return: The stock level held for `sku`.
    :raises LookupError: If `sku` is not in the catalog.
    """
    if sku not in STOCK:
        raise LookupError(f"Unknown SKU `{sku}`.")
    units, warehouse = STOCK[sku]
    return StockLevel(sku=sku, units=units, warehouse=warehouse)


def order_total(
    unit_price_cents: int,
    quantity: int,
    express: bool = False,
) -> OrderTotal:
    """Cost an order line, including sales tax and shipping.

    :param unit_price_cents: Price of a single unit, in cents.
    :param quantity: How many units are being ordered.
    :param express: Whether to use express shipping instead of
        standard.
    :return: The costed breakdown of the line.
    :raises ValueError: If `quantity` is below one, or
        `unit_price_cents` is negative.
    """
    if quantity < 1:
        raise ValueError("`quantity` must be at least 1.")
    if unit_price_cents < 0:
        raise ValueError("`unit_price_cents` must not be negative.")
    subtotal = unit_price_cents * quantity
    tax = subtotal * TAX_PERCENT // 100
    shipping = _shipping_cents(subtotal, express)
    return OrderTotal(
        subtotal_cents=subtotal,
        tax_cents=tax,
        shipping_cents=shipping,
        total_cents=subtotal + tax + shipping,
    )


def _shipping_cents(subtotal_cents: int, express: bool) -> int:
    """Return the shipping cost for a subtotal, in cents.

    :param subtotal_cents: The line subtotal before tax.
    :param express: Whether express shipping was requested.
    :return: The shipping cost in cents.
    """
    if express:
        return EXPRESS_SHIPPING_CENTS
    if subtotal_cents >= FREE_SHIPPING_CENTS:
        return 0
    return STANDARD_SHIPPING_CENTS
