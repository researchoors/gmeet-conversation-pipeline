"""Additional tests for gmeet_pipeline.agent_page — local Hank Hill puppet page."""

import re

import pytest

from gmeet_pipeline.agent_page import (
    get_agent_html,
    AGENT_DISPLAY_NAME,
    AGENT_INITIALS,
)


class TestAgentPageConstants:
    """Test module-level constants."""

    def test_display_name(self):
        assert AGENT_DISPLAY_NAME == "Hank Bob"

    def test_initials(self):
        assert AGENT_INITIALS == "HB"


class TestLocalAgentPageStructure:
    """Test the local agent page HTML structure."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.html = get_agent_html("local")

    def test_is_valid_html5(self):
        assert "<!DOCTYPE html>" in self.html
        assert "</html>" in self.html

    def test_has_canvas_element(self):
        assert 'id="avatarCanvas"' in self.html
        assert "<canvas" in self.html

    def test_has_pipeline_debug_panel(self):
        assert 'id="pipelineBar"' in self.html
        assert 'id="step-idle"' in self.html
        assert 'id="step-llm"' in self.html
        assert 'id="step-tts"' in self.html
        assert 'id="step-speaking"' in self.html

    def test_has_queue_depth_display(self):
        assert 'id="queueDepth"' in self.html

    def test_has_latency_display(self):
        assert 'id="latencyInfo"' in self.html

    def test_has_participant_list(self):
        assert 'id="participantList"' in self.html

    def test_has_last_transcript_display(self):
        assert 'id="lastTranscript"' in self.html

    def test_has_activation_badge(self):
        assert 'id="activationStatus"' in self.html

    def test_has_response_text_display(self):
        assert 'id="responseText"' in self.html

    def test_has_status_display(self):
        assert 'id="status"' in self.html

    def test_kokoro_badge(self):
        assert "Kokoro" in self.html

    def test_flash_badge(self):
        assert "Flash" in self.html

    def test_gemini_flash_label(self):
        assert "Gemini Flash" in self.html


class TestLocalAgentPageJavaScript:
    """Test that the local agent page has correct JS functions."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.html = get_agent_html("local")

    def test_draw_hank_function(self):
        assert "function drawHank(" in self.html

    def test_puppet_loop_function(self):
        assert "function puppetLoop(" in self.html

    def test_poll_audio_function(self):
        assert "async function pollAudio(" in self.html

    def test_poll_session_state_function(self):
        assert "async function pollSessionState(" in self.html

    def test_play_audio_function(self):
        assert "async function playAudio(" in self.html

    def test_init_audio_function(self):
        assert "async function initAudio(" in self.html

    def test_update_pipeline_ui_function(self):
        assert "function updatePipelineUI(" in self.html

    def test_update_activation_ui_function(self):
        assert "function updateActivationUI(" in self.html

    def test_format_latency_function(self):
        assert "function formatLatency(" in self.html

    def test_audio_polling_interval(self):
        # Audio polling every 500ms
        assert "setInterval(pollAudio, 500)" in self.html

    def test_session_state_polling_interval(self):
        # Session state polling every 1000ms
        assert "setInterval(pollSessionState, 1000)" in self.html

    def test_analyser_node_creation(self):
        assert "audioCtx.createAnalyser()" in self.html

    def test_stop_all_audio_function(self):
        assert "function stopAllAudio(" in self.html

    def test_request_animation_frame_for_puppet(self):
        assert "requestAnimationFrame(puppetLoop)" in self.html

    def test_breathing_animation(self):
        assert "breathOffset" in self.html
        assert "Math.sin" in self.html

    def test_blink_animation(self):
        assert "isBlinking" in self.html
        assert "blinkTimer" in self.html

    def test_mouth_animation_with_amplitude(self):
        assert "targetMouth" in self.html
        assert "mouthOpenness" in self.html


class TestLocalAgentPageApiEndpoints:
    """Test that the local page polls the correct API endpoints."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.html = get_agent_html("local")

    def test_polls_audio_queue(self):
        assert "/api/audio-queue" in self.html

    def test_polls_session_state(self):
        assert "/api/session-state" in self.html

    def test_fetches_audio_files(self):
        assert "/audio/" in self.html

    def test_sends_debug_messages(self):
        assert "/api/debug" in self.html


class TestElevenlabsAgentPageStructure:
    """Test the ElevenLabs agent page HTML structure."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.html = get_agent_html("elevenlabs")

    def test_is_valid_html5(self):
        assert "<!DOCTYPE html>" in self.html
        assert "</html>" in self.html

    def test_has_websocket_connection(self):
        assert "new WebSocket(" in self.html

    def test_has_pcm_playback(self):
        assert "playCompletePCM" in self.html

    def test_has_fallback_audio(self):
        assert "playFallbackAudio" in self.html

    def test_has_transcript_polling(self):
        assert "/api/transcript" in self.html

    def test_has_avatar_initials(self):
        assert "HB" in self.html

    def test_has_ping_keepalive(self):
        assert "ping" in self.html
        assert "setInterval" in self.html

    def test_ws_reconnection_logic(self):
        assert "wsReconnectDelay" in self.html
        assert "WS_MAX_DELAY" in self.html