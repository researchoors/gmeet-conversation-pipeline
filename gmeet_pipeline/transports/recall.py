import logging
import httpx
from typing import Optional

from .base import BaseTransport

logger = logging.getLogger("gmeet_pipeline.transports.recall")


class RecallTransport(BaseTransport):
    def __init__(self, api_key: str, base_url: str, service_url: str, webhook_secret: str = ""):
        self.api_key = api_key
        self.base_url = base_url
        self.service_url = service_url
        self.webhook_secret = webhook_secret

    async def join(self, meeting_url: str, bot_name: str = "Hank Bob", **kwargs) -> dict:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{self.base_url}/bot/",
                headers={
                    "Authorization": f"Token {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "meeting_url": meeting_url,
                    "bot_name": bot_name,
                    "variant": {"google_meet": "web_4_core"},
                    "output_media": {
                        "camera": {
                            "kind": "webpage",
                            "config": {"url": self.service_url},
                        }
                    },
                    "recording_config": {
                        "transcript": {
                            "provider": {
                                "recallai_streaming": {
                                    "mode": "prioritize_low_latency",
                                    "language_code": "en",
                                }
                            },
                            "diarization": {
                                "use_separate_streams_when_available": True
                            },
                        },
                        "realtime_endpoints": [
                            {
                                "type": "webhook",
                                "url": f"{self.service_url}/webhook/recall/{self.webhook_secret}" if self.webhook_secret else f"{self.service_url}/webhook/recall",
                                "events": [
                                    "transcript.data",
                                    "transcript.partial_data",
                                    "participant_events.join",
                                    "participant_events.leave",
                                ],
                            }
                        ],
                    },
                },
            )
            if resp.status_code not in (200, 201):
                logger.error(f"Recall API error: {resp.status_code} {resp.text}")
                raise Exception(f"Recall API error: {resp.status_code} {resp.text}")
            return resp.json()

    async def leave(self, bot_id: str) -> dict:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f"{self.base_url}/bot/{bot_id}/leave/",
                headers={"Authorization": f"Token {self.api_key}"},
            )
        return {"status": "leaving"}

    async def get_status(self, bot_id: str) -> Optional[str]:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{self.base_url}/bot/{bot_id}/",
                headers={"Authorization": f"Token {self.api_key}"},
            )
            if resp.status_code == 200:
                return resp.json().get("status", {}).get("code", "unknown")
        return None
