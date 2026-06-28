import aiohttp
from loguru import logger

from config import UPSTASH_URL, UPSTASH_TOKEN

async def redis_append(session_id: str, turn: str) -> None:
    if not UPSTASH_URL or not UPSTASH_TOKEN:
        return

    headers = {"Authorization": f"Bearer {UPSTASH_TOKEN}"}
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(f"{UPSTASH_URL}/get/{session_id}", headers=headers) as r:
                existing = (await r.json()).get("result") or ""
            updated = f"{existing}\n{turn}".strip()
            await s.post(f"{UPSTASH_URL}/set/{session_id}", headers=headers, json=updated)
    except Exception as e:
        logger.warning(f"[Redis] {e}")

__all__ = ["redis_append"]
