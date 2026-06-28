import asyncio
from loguru import logger
from config import supabase


#  price cache

_PRICES: dict[tuple[str, str], int] = {}
_prices_loaded = False


async def _ensure_prices() -> None:
    """Load menu prices from Supabase once, cache in memory."""
    global _PRICES, _prices_loaded
    if _prices_loaded:
        return
    try:
        def _q():
            return supabase.table("menu_items").select("pizza_name,pizza_size,price").execute().data or []
        rows = await asyncio.to_thread(_q)
        _PRICES = {(r["pizza_name"], r["pizza_size"]): int(r["price"]) for r in rows}
        _prices_loaded = True
        logger.info(f"[DB] Loaded {len(_PRICES)} menu prices from Supabase")
    except Exception as e:
        logger.exception(f"[DB] Failed to load menu prices: {e}")


def _price(pizza_name: str, pizza_size: str) -> int:
    return _PRICES.get((pizza_name, pizza_size), 299)


def _safe_int(val, default: int = 1) -> int:
    """Convert quantity safely — handles 'one', 'two', floats, None."""
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        # LLM sometimes writes "one", "two" etc.
        words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
        return words.get(str(val).strip().lower(), default)


# ── user ──────────────────────────────────────────────────────────────────────

async def db_get_user(phone: str) -> dict | None:
    def _q():
        r = (
            supabase.table("users")
            .select("phone_number,full_name,primary_delivery_address,refund_eligibility,user_tier")
            .eq("phone_number", phone)
            .single()
            .execute()
        )
        return r.data
    try:
        return await asyncio.to_thread(_q)
    except Exception:
        return None


async def db_ensure_user(phone: str, name: str = "Walk-in Customer") -> None:
    """
    Upsert a user row. Uses upsert with ignore_duplicates=True so it is safe
    to call for both new and existing customers without a separate SELECT first.
    Logs but does NOT raise on failure — a missing user row should not block
    the order; db_create_order will catch any FK error and log it clearly.
    """
    def _q():
        supabase.table("users").upsert(
            {
                "phone_number": phone,
                "full_name": name,
                "primary_delivery_address": "Lucknow, Uttar Pradesh",
                "refund_eligibility": True,
                "user_tier": "Standard",
            },
            on_conflict="phone_number",
            ignore_duplicates=True,
        ).execute()
        logger.info(f"[DB] db_ensure_user OK: phone={phone} name={name}")

    try:
        await asyncio.to_thread(_q)
    except Exception as e:
        logger.exception(f"[DB] db_ensure_user failed for phone={phone}: {e}")
        # Do NOT re-raise — let db_create_order proceed; FK will catch truly missing users


# ── orders ────────────────────────────────────────────────────────────────────

async def db_get_latest_order(phone: str) -> dict | None:
    def _q():
        r = (
            supabase.table("orders")
            .select(
                "order_id,customer_name,customer_phone,order_status,total_amount,"
                "payment_method,delivery_locality,created_at,"
                "order_items(pizza_name,pizza_size,quantity,unit_price,special_instructions)"
            )
            .eq("user_phone", phone)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        return r.data[0] if r.data else None
    try:
        return await asyncio.to_thread(_q)
    except Exception:
        return None


async def db_create_order(
    phone: str,
    items: list[dict],
    locality: str,
    payment: str,
    customer_name: str = "Walk-in Customer",
    customer_phone: str = "",
) -> dict | None:
    # Load prices before entering the sync thread (prices loading is async)
    await _ensure_prices()

    def _q():
        total = sum(
            _price(i["pizza_name"], i["pizza_size"]) * _safe_int(i.get("quantity", 1))
            for i in items
        )
        logger.info(f"[DB] Inserting order: phone={phone} name={customer_name} total={total}")

        order_res = supabase.table("orders").insert({
            "user_phone": phone,
            "customer_name": customer_name,
            "customer_phone": customer_phone or phone,
            "order_status": "Received",
            "total_amount": total,
            "payment_method": payment,
            "delivery_locality": locality,
        }).execute()

        if not order_res.data:
            logger.error(f"[DB] orders.insert returned no data: {order_res}")
            return None

        order_id = order_res.data[0]["order_id"]
        logger.info(f"[DB] Order row created: order_id={order_id}")

        item_rows = [
            {
                "order_id": order_id,
                "pizza_name": i["pizza_name"],
                "pizza_size": i["pizza_size"],
                "quantity": _safe_int(i.get("quantity", 1)),
                "unit_price": _price(i["pizza_name"], i["pizza_size"]),
                "special_instructions": i.get("special_instructions", ""),
            }
            for i in items
        ]
        supabase.table("order_items").insert(item_rows).execute()
        logger.info(f"[DB] Order items inserted for order_id={order_id}")
        return {"order_id": order_id, "total": total}

    try:
        return await asyncio.to_thread(_q)
    except Exception as e:
        # Was silently swallowed before — now we see the real Supabase error
        logger.exception(f"[DB] db_create_order failed: {e}")
        return None


# ── FAQ ───────────────────────────────────────────────────────────────────────

async def db_search_faq(query: str, limit: int = 3) -> list[dict]:
    def _q():
        r = (
            supabase.table("pizza_hut_knowledge_base")
            .select("category,context_question,context_answer")
            .or_(
                f"context_question.ilike.%{query}%,"
                f"context_answer.ilike.%{query}%,"
                f"category.ilike.%{query}%"
            )
            .limit(limit)
            .execute()
        )
        return r.data or []
    try:
        return await asyncio.to_thread(_q)
    except Exception:
        return []


__all__ = [
    "db_get_user",
    "db_get_latest_order",
    "db_search_faq",
    "db_ensure_user",
    "db_create_order",
]