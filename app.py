import asyncio
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.transports.base_transport import BaseTransport

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.services.sarvam.tts import SarvamTTSService

from pipecat.frames.frames import (
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.aggregators.llm_context import LLMContext

from processor import LangGraphProcessor

load_dotenv(Path(__file__).parent / ".env")

from config import get_transport_params, SARVAM_API_KEY


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    DEFAULT_PHONE = "+919876543210" # phone_number is used as default parameter here

    phone = getattr(runner_args, "session_id", None) or DEFAULT_PHONE
    body = getattr(runner_args, "body", None) or {}
    if isinstance(body, dict):
        phone = body.get("phone", DEFAULT_PHONE)

    session_id = getattr(runner_args, "session_id", None) or str(id(runner_args))

    stt = SarvamSTTService(
        api_key=SARVAM_API_KEY,
        settings=SarvamSTTService.Settings(model="saaras:v3"),
    )

    tts = None
    for candidate_model in ("bulbul:v3", "bulbul:v3-beta", "bulbul:v2"):
        try:
            tts = SarvamTTSService(
                api_key=SARVAM_API_KEY,
                settings=SarvamTTSService.Settings(model=candidate_model),
            )
            logger.info(f"[SarvamTTS] Using model={candidate_model}")
            break
        except Exception as e:
            logger.warning(f"[SarvamTTS] Failed model={candidate_model}: {e}")

    if tts is None:
        raise RuntimeError("SarvamTTSService: unable to initialize any supported bulbul model")

    llm = LangGraphProcessor(
        phone_number=phone,
        session_id=session_id,
    )

    context = LLMContext(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a warm, professional voice agent for Pizza Hut Lucknow, "
                    "Hazratganj. Keep all replies brief and conversational."
                ),
            }
        ]
    )
    ctx_pair = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            ctx_pair.user(),
            llm,
            tts,
            transport.output(),
            ctx_pair.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"[Voice] Client connected  phone={phone}  session={session_id}")
        greeting = (
            "Hello! Welcome to Pizza Hut Lucknow, Hazratganj. "
            "I can help you place a new order, track an existing order, "
            "process a refund, or answer any questions. How can I help you today?"
        )
        await task.queue_frames(
            [
                LLMFullResponseStartFrame(),
                LLMTextFrame(greeting),
                LLMFullResponseEndFrame(),
            ]
        )

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("[Voice] Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    transport_params = get_transport_params()
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
