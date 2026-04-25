"""Legacy entry point — thin wrapper that runs with local Kokoro+RVC + voice gateway routing."""

from gmeet_pipeline.config import GmeetSettings
from gmeet_pipeline.main import create_app


def main():
    settings = GmeetSettings.load()
    settings.tts_backend = "local"
    settings.llm_routing = "voice_gateway"

    server = create_app(settings)

    import uvicorn
    print(f"Starting Hank Bob Meeting Agent on port {settings.port}")
    print(f"Public URL: {settings.service_url}")
    print(f"LLM routing: fast={settings.fast_model} standard={settings.standard_model} deep={settings.deep_model}")
    print(f"TTS: Local Kokoro+RVC")
    print(f"Audio: WAV → decodeAudioData → AudioBufferSourceNode")
    uvicorn.run(server.app, host="0.0.0.0", port=settings.port)


if __name__ == "__main__":
    main()
