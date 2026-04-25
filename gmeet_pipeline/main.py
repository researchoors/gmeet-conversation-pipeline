"""Entry point — loads config, wires components, runs uvicorn."""

import sys
import logging
import uvicorn

from typing import Optional

from .config import GmeetSettings
from .state import BotRegistry
from .ws_manager import ConnectionManager
from .server import GmeetServer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("gmeet_pipeline.main")


def create_app(settings: Optional[GmeetSettings] = None) -> GmeetServer:
    """Create and wire the full application from settings.
    
    This is the composition root — all dependency injection happens here.
    """
    if settings is None:
        settings = GmeetSettings.load()

    # Ensure audio dir exists
    from pathlib import Path
    Path(settings.audio_dir).mkdir(parents=True, exist_ok=True)

    # Shared components
    registry = BotRegistry()
    ws_manager = ConnectionManager()

    # Transport
    if settings.recall_api_key:
        from .transports.recall import RecallTransport
        transport = RecallTransport(
            api_key=settings.recall_api_key,
            base_url=settings.recall_base,
            service_url=settings.service_url,
        )
    else:
        logger.warning("No RECALL_API_KEY — transport will not work")
        from .transports.base import BaseTransport
        transport = None  # type: ignore

    # Memory (optional, for voice gateway routing)
    memory_snapshot = None
    if settings.llm_routing == "voice_gateway":
        from .memory import MemorySnapshot
        memory_snapshot = MemorySnapshot(
            memory_file=Path(settings.memory_file),
            user_file=Path(settings.user_file),
        )
        memory_snapshot.build()
        logger.info(f"Memory: {len(memory_snapshot.entries)} entries loaded")

    # LLM
    if settings.llm_routing == "voice_gateway":
        from .llm.openrouter import VoiceGatewayLLM
        llm = VoiceGatewayLLM(
            api_key=settings.openrouter_key,
            fast_model=settings.fast_model,
            standard_model=settings.standard_model,
            deep_model=settings.deep_model,
            service_url=settings.service_url,
            memory_snapshot=memory_snapshot,
        )
        logger.info(
            f"LLM routing: fast={settings.fast_model} "
            f"standard={settings.standard_model} deep={settings.deep_model}"
        )
    else:
        from .llm.openrouter import SimpleOpenRouterLLM
        llm = SimpleOpenRouterLLM(
            api_key=settings.openrouter_key,
            model=settings.llm_model,
            service_url=settings.service_url,
        )
        logger.info(f"LLM: {settings.llm_model}")

    # TTS
    if settings.tts_backend == "local":
        from .tts.local import LocalTTS
        tts = LocalTTS(
            rvc_model_path=settings.rvc_model_path,
            rvc_exp_dir=settings.rvc_exp_dir,
            rvc_repo_dir=settings.rvc_repo_dir,
            rvc_f0_method=settings.rvc_f0_method,
            rvc_f0_up_key=settings.rvc_f0_up_key,
            rvc_index_rate=settings.rvc_index_rate,
            kokoro_voice=settings.kokoro_voice,
            audio_dir=settings.audio_dir,
        )
        logger.info("TTS: Local Kokoro+RVC")
    else:
        from .tts.elevenlabs import ElevenLabsTTS
        tts = ElevenLabsTTS(
            api_key=settings.elevenlabs_key,
            voice_id=settings.elevenlabs_voice,
            model_id=settings.elevenlabs_model,
            audio_dir=settings.audio_dir,
            ws_manager=ws_manager,
        )
        logger.info(f"TTS: ElevenLabs (voice={settings.elevenlabs_voice})")

    # Wire the webhook handler's ws_manager reference for ElevenLabs mode
    # (the TTS object already has it, webhook needs it for broadcast)

    server = GmeetServer(
        settings=settings,
        transport=transport,
        llm=llm,
        tts=tts,
        ws_manager=ws_manager,
        registry=registry,
        memory_snapshot=memory_snapshot,
    )

    # Update webhook handler with proper ws_manager reference
    server.webhook_handler.ws_manager = ws_manager

    return server


def main():
    """CLI entry point."""
    settings = GmeetSettings.load()

    if not settings.recall_api_key:
        print("ERROR: RECALL_API_KEY not found")
        sys.exit(1)
    if not settings.openrouter_key:
        print("ERROR: OPENROUTER_API_KEY not found")
        sys.exit(1)
    if settings.tts_backend == "elevenlabs" and not settings.elevenlabs_key:
        print("ERROR: ELEVENLABS_API_KEY not found")
        sys.exit(1)

    server = create_app(settings)

    print(f"Starting Hank Bob Meeting Agent on port {settings.port}")
    print(f"Public URL: {settings.service_url}")
    uvicorn.run(server.app, host="0.0.0.0", port=settings.port)


if __name__ == "__main__":
    main()
