import asyncio
from typing import Optional

from loguru import logger
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    LLMContextFrame,
)
try:
    from pipecat.frames.frames import StartInterruptionFrame
except ImportError:
    StartInterruptionFrame = None

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from llm_graph import PizzaState, _ROUTING_TOKENS, pizza_graph


def _sanitize(t: str) -> str:
    return (
        t.replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2026", "...")
        .replace("\u00a0", " ")
    )


def _is_routing_token(msg: BaseMessage) -> bool:
    """Return True if this message is an internal LangGraph routing token, not real dialogue."""
    if not isinstance(msg, AIMessage):
        return False
    return msg.content.strip().upper() in _ROUTING_TOKENS


class LangGraphProcessor(FrameProcessor):
    """
    Bridge between Pipecat LLMContextFrame and LangGraph.
    Maintains clean conversation history (no routing tokens) across turns.
    """

    def __init__(self, *, graph=pizza_graph, phone_number: str, session_id: str, **kw):
        super().__init__(**kw)
        self._graph    = graph
        self._phone    = phone_number
        self._session  = session_id
        self._task: Optional[asyncio.Task] = None

        self._active_agent = "triage"
        self._order_draft  = {}
        self._confirmed    = False

        # Only real dialogue lives here — routing tokens are never stored
        self._history: list[BaseMessage] = []

    def _latest_user_text(self, ctx: LLMContext) -> str:
        for msg in reversed(ctx.get_messages() or []):
            if msg.get("role") == "user":
                c = msg.get("content", "")
                return c if isinstance(c, str) else str(c)
        return ""

    async def _run_and_stream(self, ctx: LLMContext) -> None:
        from redis_store import redis_append

        user_text = self._latest_user_text(ctx)
        if not user_text:
            return

        logger.debug(f"[LG] user={user_text!r}  agent={self._active_agent}  history_len={len(self._history)}")

        # Add the new user turn to history
        self._history.append(HumanMessage(content=user_text))

        state: PizzaState = {
            "messages":     list(self._history),   # clean history — no routing tokens
            "active_agent": self._active_agent,
            "phone_number": self._phone,
            "session_id":   self._session,
            "order_draft":  self._order_draft,
            "confirmed":    self._confirmed,
        }

        try:
            result = await self._graph.ainvoke(state)

            # Persist routing state
            self._active_agent = result.get("active_agent", "triage")
            if self._active_agent == "done":
                self._active_agent = "triage"
            self._order_draft = result.get("order_draft", {})
            self._confirmed   = result.get("confirmed", False)

            # Find the last real spoken reply — skip routing tokens
            spoken = ""
            for msg in reversed(result.get("messages", [])):
                c = getattr(msg, "content", "").strip()
                if not c or c.upper() in _ROUTING_TOKENS:
                    continue
                if "##ORDER_COMPLETE##" in c:
                    c = c.split("##ORDER_COMPLETE##")[0].strip()
                spoken = c
                break

            logger.debug(f"[LG] reply={spoken[:80]!r}  next={self._active_agent}  draft={self._order_draft}")

            if spoken:
                # Only store real dialogue in history — never routing tokens
                self._history.append(AIMessage(content=spoken))

                await self.push_frame(LLMFullResponseStartFrame())
                for word in _sanitize(spoken).split(" "):
                    await self.push_frame(LLMTextFrame(word + " "))
                    await asyncio.sleep(0)
                await self.push_frame(LLMFullResponseEndFrame())

                await redis_append(
                    self._session,
                    f"User: {user_text} | Agent: {spoken[:120]}",
                )
            else:
                # Graph returned nothing speakable — pop the user message so it
                # doesn't pollute history and cause a duplicate next turn
                self._history.pop()
                logger.warning("[LG] Graph returned no speakable reply")

        except asyncio.CancelledError:
            # Pipecat interrupted us — pop the user turn we just added
            self._history.pop()
            logger.debug("[LG] turn cancelled by interruption")

        except Exception as e:
            # Pop the user turn so it isn't replayed as a duplicate next time
            self._history.pop()
            logger.exception(f"[LG] error processing turn: {e}")
            await self.push_frame(LLMFullResponseStartFrame())
            await self.push_frame(
                LLMTextFrame("Sorry, I had a little trouble with that. Could you please repeat? ")
            )
            await self.push_frame(LLMFullResponseEndFrame())

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame):
            if self._task and not self._task.done():
                self._task.cancel()
            self._task = asyncio.create_task(self._run_and_stream(frame.context))

        elif StartInterruptionFrame and isinstance(frame, StartInterruptionFrame):
            if self._task and not self._task.done():
                self._task.cancel()
            await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)


__all__ = ["LangGraphProcessor"]