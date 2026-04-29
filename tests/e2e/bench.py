"""E2E benchmark runner — measures pipeline latency with mocked backends.

Simulates realistic conversation scenarios and records:
  - webhook → audio-queue latency (end-to-end)
  - LLM generation time
  - TTS generation time
  - throughput under concurrent load

Usage:
    python -m tests.e2e.bench --iterations 20
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gmeet_pipeline.config import GmeetSettings
from gmeet_pipeline.llm.base import BaseLLM
from gmeet_pipeline.tts.base import BaseTTS
from gmeet_pipeline.transports.base import BaseTransport
from gmeet_pipeline.state import BotRegistry
from gmeet_pipeline.ws_manager import ConnectionManager
from gmeet_pipeline.server import GmeetServer

from httpx import ASGITransport, AsyncClient

from . import (
    make_transcript_data,
    MockLLM,
    MockTTS,
    MockTransport,
)


# ---------------------------------------------------------------------------
# Configurable backends with latency simulation
# ---------------------------------------------------------------------------

class BenchLLM(BaseLLM):
    """LLM with configurable latency distribution.

    Simulates real OpenRouter response times with optional jitter.
    """

    def __init__(self, base_ms: int = 100, jitter_ms: int = 30):
        self._base = base_ms / 1000.0
        self._jitter = jitter_ms / 1000.0
        self.timings: list[float] = []

    async def generate(self, conversation: list, message: str, bot_state=None) -> Optional[str]:
        import random
        delay = self._base + random.uniform(0, self._jitter)
        t0 = time.monotonic()
        await asyncio.sleep(delay)
        elapsed = time.monotonic() - t0
        self.timings.append(elapsed)
        return "I tell ya what, that's a mighty fine question right there."


class BenchTTS(BaseTTS):
    """TTS with configurable latency that writes a real WAV."""

    def __init__(self, audio_dir: str, base_ms: int = 20, jitter_ms: int = 10):
        self._audio_dir = audio_dir
        self._base = base_ms / 1000.0
        self._jitter = jitter_ms / 1000.0
        self.timings: list[float] = []

    async def generate(self, text: str, bot_id: str) -> Optional[str]:
        import struct, random
        from pathlib import Path

        delay = self._base + random.uniform(0, self._jitter)
        t0 = time.monotonic()
        await asyncio.sleep(delay)

        sample_rate = 22050
        num_samples = sample_rate // 10
        filename = f"bench_{uuid.uuid4().hex[:8]}.wav"
        filepath = Path(self._audio_dir) / filename

        data = b"\x00\x00" * num_samples
        fmt_chunk = b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
        data_chunk = b"data" + struct.pack("<I", len(data)) + data
        riff = b"RIFF" + struct.pack("<I", 4 + len(fmt_chunk) + len(data_chunk)) + b"WAVE"
        filepath.write_bytes(riff + fmt_chunk + data_chunk)

        elapsed = time.monotonic() - t0
        self.timings.append(elapsed)
        return filename


# ---------------------------------------------------------------------------
# Benchmark scenarios
# ---------------------------------------------------------------------------

SCENARIOS = {
    "single_utterance": {
        "utterances": [
            ("Alice", "hey hank, how's it going?"),
        ],
    },
    "multi_speaker": {
        "utterances": [
            ("Alice", "hey hank!"),
            ("Bob", "what do you think about the new design?"),
            ("Charlie", "yeah I'm curious too"),
        ],
    },
    "rapid_fire": {
        "utterances": [
            ("Alice", "quick question"),
            ("Alice", "actually two questions"),
            ("Alice", "maybe three"),
            ("Bob", "slow down!"),
        ],
    },
    "long_utterance": {
        "utterances": [
            ("Alice", "I was thinking about the architecture and how we could improve the latency by moving the TTS inference to a separate worker process with a queue-based system instead of the current thread pool approach"),
        ],
    },
    "mixed_flow": {
        "utterances": [
            ("Alice", "hey hank"),
            # (partial transcript — sent but should be ignored)
            ("Bob", "so about the project"),
            ("Charlie", "I have concerns about the timeline"),
        ],
    },
}


async def run_benchmark(
    client: AsyncClient,
    bot_id: str,
    scenario: dict,
) -> list[dict]:
    """Run a single scenario and return per-utterance timing records."""
    results = []
    utterances = scenario["utterances"]

    for speaker, text in utterances:
        payload = make_transcript_data(bot_id, speaker, text)

        t_send = time.monotonic()
        resp = await client.post("/webhook/recall", json=payload)
        t_webhook = time.monotonic()

        assert resp.status_code == 200

        # Poll until audio appears (with timeout)
        audio_filename = None
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            aq_resp = await client.get("/api/audio-queue")
            items = aq_resp.json().get("items", [])
            if items:
                audio_filename = items[-1]["filename"]
                break
            await asyncio.sleep(0.02)

        t_audio = time.monotonic()

        results.append({
            "speaker": speaker,
            "text": text[:50],
            "webhook_ms": round((t_webhook - t_send) * 1000, 1),
            "e2e_ms": round((t_audio - t_send) * 1000, 1),
            "audio_generated": audio_filename is not None,
        })

        # Brief pause between utterances
        await asyncio.sleep(0.1)

    return results


async def run_concurrency_benchmark(
    client: AsyncClient,
    bot_ids: list[str],
    num_concurrent: int,
) -> dict:
    """Test pipeline under concurrent transcript webhooks to different bots."""

    async def single_bot(bot_id: str) -> dict:
        payload = make_transcript_data(bot_id, "Alice", "concurrent test message")
        t0 = time.monotonic()
        resp = await client.post("/webhook/recall", json=payload)
        t1 = time.monotonic()
        assert resp.status_code == 200
        return {"bot_id": bot_id, "webhook_ms": round((t1 - t0) * 1000, 1)}

    tasks = [single_bot(bid) for bid in bot_ids[:num_concurrent]]
    t_start = time.monotonic()
    results = await asyncio.gather(*tasks)
    t_end = time.monotonic()

    return {
        "concurrent_bots": num_concurrent,
        "total_ms": round((t_end - t_start) * 1000, 1),
        "per_bot_ms": [r["webhook_ms"] for r in results],
    }


async def main(iterations: int = 20, output: Optional[str] = None):
    """Run the full benchmark suite."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_dir = str(Path(tmpdir) / "audio")
        settings = GmeetSettings(
            recall_api_key="bench-key",
            openrouter_key="bench-or-key",
            tts_backend="local",
            llm_routing="simple",
            audio_dir=audio_dir,
        )

        llm = BenchLLM(base_ms=100, jitter_ms=30)
        tts = BenchTTS(audio_dir=audio_dir, base_ms=20, jitter_ms=10)
        transport = MockTransport()
        registry = BotRegistry()
        ws_manager = ConnectionManager()

        server = GmeetServer(
            settings=settings,
            transport=transport,
            llm=llm,
            tts=tts,
            ws_manager=ws_manager,
            registry=registry,
        )

        transport_client = ASGITransport(app=server.app)
        async with AsyncClient(transport=transport_client, base_url="http://bench") as client:
            all_results = {}

            for scenario_name, scenario in SCENARIOS.items():
                print(f"\n{'='*60}")
                print(f"Scenario: {scenario_name} ({iterations} iterations)")
                print(f"{'='*60}")

                scenario_results = []

                for i in range(iterations):
                    # Create a fresh bot for each iteration
                    resp = await client.post("/api/bot/join", json={
                        "meeting_url": f"bench://{scenario_name}/{i}",
                        "bot_name": "Hank Bob",
                    })
                    bot_id = resp.json()["bot_id"]

                    # Clear audio queue from previous iterations
                    server.audio_queue.clear()

                    iter_results = await run_benchmark(client, bot_id, scenario)
                    scenario_results.extend(iter_results)

                # Aggregate
                e2e_times = [r["e2e_ms"] for r in scenario_results if r["audio_generated"]]
                webhook_times = [r["webhook_ms"] for r in scenario_results]

                if e2e_times:
                    stats = {
                        "iterations": iterations,
                        "total_utterances": len(scenario_results),
                        "e2e_mean_ms": round(statistics.mean(e2e_times), 1),
                        "e2e_median_ms": round(statistics.median(e2e_times), 1),
                        "e2e_p95_ms": round(sorted(e2e_times)[int(len(e2e_times) * 0.95)], 1),
                        "e2e_min_ms": round(min(e2e_times), 1),
                        "e2e_max_ms": round(max(e2e_times), 1),
                        "e2e_stdev_ms": round(statistics.stdev(e2e_times), 1) if len(e2e_times) > 1 else 0,
                        "webhook_mean_ms": round(statistics.mean(webhook_times), 1),
                        "audio_success_rate": round(
                            len(e2e_times) / len(scenario_results) * 100, 1
                        ),
                    }
                else:
                    stats = {"iterations": iterations, "total_utterances": len(scenario_results), "audio_success_rate": 0}

                all_results[scenario_name] = stats

                print(f"  Utterances: {stats['total_utterances']}")
                print(f"  Audio success: {stats.get('audio_success_rate', 0)}%")
                if e2e_times:
                    print(f"  E2E latency: mean={stats['e2e_mean_ms']}ms median={stats['e2e_median_ms']}ms p95={stats['e2e_p95_ms']}ms")
                    print(f"  E2E range: {stats['e2e_min_ms']}–{stats['e2e_max_ms']}ms (stdev={stats['e2e_stdev_ms']}ms)")
                    print(f"  Webhook overhead: {stats['webhook_mean_ms']}ms")

            # Concurrency benchmark
            print(f"\n{'='*60}")
            print("Concurrency benchmark")
            print(f"{'='*60}")

            for concurrency in [1, 2, 4, 8]:
                # Create bots
                bot_ids = []
                for _ in range(concurrency):
                    resp = await client.post("/api/bot/join", json={
                        "meeting_url": f"bench://concurrent/{concurrency}",
                        "bot_name": "Hank Bob",
                    })
                    bot_ids.append(resp.json()["bot_id"])

                server.audio_queue.clear()
                conc_result = await run_concurrency_benchmark(client, bot_ids, concurrency)
                all_results[f"concurrency_{concurrency}"] = conc_result
                print(f"  {concurrency} concurrent bots: total={conc_result['total_ms']}ms")

            # LLM/TTS internal timings
            print(f"\n{'='*60}")
            print("Backend latency (internal)")
            print(f"{'='*60}")
            if llm.timings:
                llm_ms = [t * 1000 for t in llm.timings]
                print(f"  LLM: mean={round(statistics.mean(llm_ms), 1)}ms median={round(statistics.median(llm_ms), 1)}ms n={len(llm_ms)}")
            if tts.timings:
                tts_ms = [t * 1000 for t in tts.timings]
                print(f"  TTS: mean={round(statistics.mean(tts_ms), 1)}ms median={round(statistics.median(tts_ms), 1)}ms n={len(tts_ms)}")

            # Output
            if output:
                Path(output).write_text(json.dumps(all_results, indent=2))
                print(f"\nResults written to {output}")

            print(f"\n{'='*60}")
            print("SUMMARY")
            print(f"{'='*60}")
            for name, stats in all_results.items():
                if "e2e_mean_ms" in stats:
                    print(f"  {name}: e2e_mean={stats['e2e_mean_ms']}ms success={stats.get('audio_success_rate', 0)}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E2E pipeline benchmarks")
    parser.add_argument("--iterations", type=int, default=20, help="Iterations per scenario")
    parser.add_argument("--output", type=str, default=None, help="Write JSON results to file")
    args = parser.parse_args()
    asyncio.run(main(iterations=args.iterations, output=args.output))
