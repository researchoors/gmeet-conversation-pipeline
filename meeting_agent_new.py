"""Legacy entry point — thin wrapper that runs with ElevenLabs + simple LLM routing."""

from gmeet_pipeline.config import GmeetSettings
from gmeet_pipeline.main import create_app


def main():
    settings = GmeetSettings.load()
    settings.tts_backend = "elevenlabs"
    settings.llm_routing = "simple"

    server = create_app(settings)

    import uvicorn
    print(f"Starting Hank Bob Meeting Agent on port {settings.port}")
    print(f"Public URL: {settings.service_url}")
    print(f"LLM: {settings.llm_model}")
    print(f"Voice: {settings.elevenlabs_voice}")
    print(f"Audio: PCM streaming via WebSocket")
    uvicorn.run(server.app, host="0.0.0.0", port=settings.port)


if __name__ == "__main__":
    main()
