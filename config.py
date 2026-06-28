import os
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client, Client
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.runner.utils import create_transport
from pipecat.transports.base_transport import TransportParams

load_dotenv(Path(__file__).parent / ".env")

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
DEEPGRAM_KEY = os.getenv("DEEPGRAM_API_KEY", "")
GROQ_KEY = os.getenv("GROQ_API_KEY", "")
UPSTASH_URL = os.getenv("UPSTASH_REDIS_REST_URL", "")
UPSTASH_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN", "")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

os.environ["GROQ_API_KEY"] = GROQ_KEY

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

transport_params = {
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    )
}

def get_transport_params():
    # convenience alias for app entrypoint
    return transport_params

# Expose for callers that use create_transport directly
__all__ = [
    "supabase",
    "SUPABASE_URL",
    "SUPABASE_KEY",
    "DEEPGRAM_KEY",
    "GROQ_KEY",
    "UPSTASH_URL",
    "UPSTASH_TOKEN",
    "SARVAM_API_KEY",
    "get_transport_params",
    "create_transport",
]
