import asyncio
import json
import re
from typing import Annotated, TypedDict

from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from config import GROQ_KEY  # ensure env side-effects are consistent
from db import db_create_order, db_ensure_user, db_get_latest_order, db_get_user, db_search_faq
from redis_store import redis_append

# ── models ────────────────────────────────────────────────────────────────────

_FAST  = "llama-3.1-8b-instant"       # 128k context — triage & FAQ
_SMART = "llama-3.3-70b-versatile"    # 128k context — order collection & refund

_triage_llm = ChatGroq(model=_FAST,  temperature=0.0)
_order_llm  = ChatGroq(model=_SMART, temperature=0.3)
_refund_llm = ChatGroq(model=_SMART, temperature=0.2)
_faq_llm    = ChatGroq(model=_FAST,  temperature=0.1)
 
 
# agent sstate 
class PizzaState(TypedDict):
    messages:Annotated[list[BaseMessage], add_messages]
    active_agent:str
    phone_number: str
    session_id:str
    order_draft:dict 
    confirmed:    bool
 
 
#routing tokens 
 
_ROUTING_TOKENS = {"NEW_ORDER", "ORDER_STATUS", "REFUND", "FAQ", "ORDER", "", "TRIAGE"}
 
 
#triage
async def triage_node(state: PizzaState) -> dict:
    sys = SystemMessage(
        content=(
            "You are the triage agent at Pizza Hut Lucknow. "
            "Read the customer message and output EXACTLY one token:\n"
            "  NEW_ORDER    — customer wants to place a new order\n"
            "  ORDER_STATUS — customer asking about an existing order / tracking\n"
            "  REFUND       — customer wants a refund or disputes a charge\n"
            "  FAQ          — any other question (hours, menu, delivery area, etc.)\n"
            "Output only that token, nothing else."
        )
    )
    last_human = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    msgs = [last_human] if last_human else state["messages"]
    r = await _triage_llm.ainvoke([sys] + msgs)
    token = r.content.strip().upper()
 
    if "NEW_ORDER"    in token: return {"active_agent": "order_collect", "messages": [r]}
    if "ORDER_STATUS" in token: return {"active_agent": "order_status",  "messages": [r]}
    if "REFUND"       in token: return {"active_agent": "refund",        "messages": [r]}
    return {"active_agent": "faq", "messages": [r]}
 
 
# order collection
 
_ORDER_SYS = """You are a friendly Polite order-taking agent at Pizza Hut Lucknow (Hazratganj).
You are in a VOICE conversation — keep every reply SHORT and natural (1-2 sentences max).
 
Collect ALL of the following through natural conversation:
  1. Customer's full name
  2. Customer's phone number (10-digit Indian mobile, e.g. 9876543210)
  3. Pizza name & size (Small / Medium / Large)
  4. Quantity (default 1 if not mentioned)
  5. Special instructions (ask ONCE; "none" is fine)
  6. Delivery locality (area in Lucknow, e.g. Hazratganj, Gomti Nagar)
  7. Payment method: Cash on Delivery or UPI
 
CRITICAL RULES:
- Extract info from the FULL conversation — never re-ask something already provided.
- Ask for ONLY ONE missing detail at a time.
- Collect name and phone BEFORE asking about pizza details if they haven't been given yet.
- Accept fuzzy pizza names (Margarita → Margherita, Tandoori Panir → Tandoori Paneer, etc.).
- When ALL 7 details are collected, give a warm spoken confirmation summary, then on a NEW LINE output EXACTLY:
  ##ORDER_COMPLETE##
  {{"customer_name":"Rahul Singh","customer_phone":"9876543210","pizza_name":"Pepperoni","pizza_size":"Medium","quantity":1,"special_instructions":"none","locality":"Gomti Nagar","payment":"UPI"}}
 
Available pizzas: Margherita, Pepperoni, Veggie Supreme, Tandoori Paneer, BBQ Chicken, Garlic Breadstix
Sizes: Small, Medium, Large  (Garlic Breadstix → Standard only)
Payment: Cash on Delivery, UPI
 
Currently collected: {draft}
"""
 
 
async def order_collect_node(state: PizzaState) -> dict:
    draft = state.get("order_draft") or {}
    sys   = SystemMessage(content=_ORDER_SYS.format(draft=json.dumps(draft, ensure_ascii=False)))
 
    # Full conversation, minus internal routing tokens
    conv = [
        m for m in state["messages"]
        if not (isinstance(m, AIMessage) and m.content.strip().upper() in _ROUTING_TOKENS)
    ]
 
    r= await _order_llm.ainvoke([sys] + conv)
    text = r.content
    new_draft = dict(draft)
 
    if "##ORDER_COMPLETE##" in text:
        try:
            json_part = text.split("##ORDER_COMPLETE##")[-1].strip()
            match = re.search(r"\{.*\}", json_part, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                new_draft.update(parsed)
 
                # Normalise phone: strip spaces/dashes, ensure +91 prefix
                raw_phone = str(parsed.get("customer_phone", "")).strip().replace(" ", "").replace("-", "")
                if raw_phone and not raw_phone.startswith("+"):
                    raw_phone = "+91" + raw_phone.lstrip("0")
                customer_phone = raw_phone or state["phone_number"]
                customer_name  = parsed.get("customer_name", "Walk-in Customer")
 
                # Ensure user row exists — does not raise, errors are logged in db.py
                await db_ensure_user(customer_phone, name=customer_name)
 
                result = await db_create_order(
                    phone=customer_phone,
                    items=[{
                        "pizza_name":           parsed.get("pizza_name", "Margherita"),
                        "pizza_size":           parsed.get("pizza_size", "Medium"),
                        "quantity":             parsed.get("quantity", 1),   # _safe_int handled in db.py
                        "special_instructions": parsed.get("special_instructions", ""),
                    }],
                    locality=parsed.get("locality", "Hazratganj"),
                    payment=parsed.get("payment", "Cash on Delivery"),
                    customer_name=customer_name,
                    customer_phone=customer_phone,
                )
 
                if result:
                    spoken = text.split("##ORDER_COMPLETE##")[0].strip()
                    spoken += (
                        f" Your order ID is {result['order_id']} "
                        f"and the total is {result['total']} rupees. Thank you, {customer_name}!"
                    )
                    await redis_append(
                        state["session_id"],
                        f"ORDER_PLACED order_id={result['order_id']} name={customer_name} phone={customer_phone}",
                    )
                    return {
                        "messages":     [AIMessage(content=spoken)],
                        "active_agent": "done",
                        "order_draft":  {},
                        "confirmed":    True,
                    }
                else:
                    # db_create_order returned None — real error already logged in db.py
                    from loguru import logger
                    logger.error(f"[order_collect] db_create_order returned None for phone={customer_phone}")
                    return {
                        "messages": [AIMessage(content=(
                            "I'm sorry, I wasn't able to save your order right now. "
                            "Please try again in a moment."
                        ))],
                        "active_agent": "order_collect",
                        "order_draft":  new_draft,
                        "confirmed":    False,
                    }
        except Exception as exc:
            from loguru import logger
            logger.exception(f"[order_collect] unexpected error: {exc}")
            return {
                "messages":     [r],
                "active_agent": "order_collect",
                "order_draft":  new_draft,
                "confirmed":    False,
            }
 
    return {
        "messages":     [r],
        "active_agent": "order_collect",
        "order_draft":  new_draft,
        "confirmed":    False,
    }
 
 
#  order status
 
async def order_status_node(state: PizzaState) -> dict:
    phone = state["phone_number"]
    order = await db_get_latest_order(phone)
 
    if order:
        items = ", ".join(
            f"{i['quantity']}x {i['pizza_name']} ({i['pizza_size']})"
            for i in (order.get("order_items") or [])
        )
        ctx = (
            f"Order #{order['order_id']} — "
            f"Customer: {order.get('customer_name', 'N/A')} — "
            f"Status: {order['order_status']} — "
            f"Items: {items or 'N/A'} — "
            f"Total: {order['total_amount']} rupees — "
            f"Area: {order.get('delivery_locality', 'Lucknow')}"
        )
    else:
        ctx = "No orders found for this number."
 
    sys = SystemMessage(
        content=(
            "You are the order-tracking agent at Pizza Hut Lucknow. "
            f"Live data: {ctx}. "
            "ALWAYS state the order ID number explicitly in your reply. "
            "Give a warm spoken update in 1-2 short sentences."
        )
    )
    last_human = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    msgs = [last_human] if last_human else state["messages"]
    r = await _order_llm.ainvoke([sys] + msgs)
    return {"messages": [r], "active_agent": "triage"}
 
 
# refund 
async def refund_node(state: PizzaState) -> dict:
    phone = state["phone_number"]
    user  = await db_get_user(phone)
    order = await db_get_latest_order(phone)
 
    eligible   = user.get("refund_eligibility", False) if user else False
    tier       = user.get("user_tier", "Standard")     if user else "Standard"
    name       = user.get("full_name", "customer")     if user else "customer"
    order_ctx  = (
        f"Order #{order['order_id']}, status: {order['order_status']}, {order['total_amount']} rupees"
        if order else "No recent order."
    )
 
    sys = SystemMessage(
        content=(
            "You are the billing agent at Pizza Hut Lucknow. "
            f"Customer: {name}, tier: {tier}, refund eligible: {eligible}. "
            f"Last order: {order_ctx}. "
            "If eligible, confirm refund in 3-5 business days. "
            "If not, apologise briefly. Reply in 1-2 short spoken sentences."
        )
    )
    last_human = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    msgs = [last_human] if last_human else state["messages"]
    r = await _refund_llm.ainvoke([sys] + msgs)
    return {"messages": [r], "active_agent": "triage"}
 
 
#faq 
 
async def faq_node(state: PizzaState) -> dict:
    last = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), ""
    )
    faqs    = await db_search_faq(last or "hours delivery menu")
    faq_ctx = (
        "\n".join(f"[{f['category']}] Q: {f['context_question']} A: {f['context_answer']}" for f in faqs)
        if faqs else "No FAQ data available."
    )
 
    sys = SystemMessage(
        content=(
            "You are the info-desk at Pizza Hut Lucknow (Hazratganj). "
            f"FAQ data:\n{faq_ctx}\n\n"
            "Answer in 1 warm spoken sentence. "
            "If the answer isn't in the data, say you'll connect them with the store."
        )
    )
    last_human = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    msgs = [last_human] if last_human else state["messages"]
    r = await _faq_llm.ainvoke([sys] + msgs)
    return {"messages": [r], "active_agent": "triage"}
 
 
# agent ── graph
def _build_graph():
    wf = StateGraph(PizzaState)
    wf.add_node("triage",        triage_node)
    wf.add_node("order_collect", order_collect_node)
    wf.add_node("order_status",  order_status_node)
    wf.add_node("refund",        refund_node)
    wf.add_node("faq",           faq_node)
 
    wf.add_conditional_edges(
        START,
        lambda s: "order_collect" if s.get("active_agent") == "order_collect" else "triage",
        {"triage": "triage", "order_collect": "order_collect"},
    )
    wf.add_conditional_edges(
        "triage",
        lambda s: s["active_agent"],
        {"order_collect": "order_collect", "order_status": "order_status",
         "refund": "refund", "faq": "faq"},
    )
    wf.add_edge("order_collect", END)
    wf.add_edge("order_status",  END)
    wf.add_edge("refund",        END)
    wf.add_edge("faq",           END)
    return wf.compile()
 
 
pizza_graph = _build_graph()
 
__all__ = ["PizzaState", "_ROUTING_TOKENS", "pizza_graph"]
 