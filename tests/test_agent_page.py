"""Tests for gmeet_pipeline.agent_page."""

from gmeet_pipeline.agent_page import get_agent_html, AGENT_DISPLAY_NAME


class TestGetAgentHtmlElevenlabs:
    """Test get_agent_html('elevenlabs') returns correct HTML."""

    def test_returns_html_string(self):
        html = get_agent_html("elevenlabs")
        assert isinstance(html, str)
        assert len(html) > 100

    def test_contains_websocket_code(self):
        html = get_agent_html("elevenlabs")
        assert "WebSocket" in html or "ws" in html

    def test_contains_hank_bob(self):
        html = get_agent_html("elevenlabs")
        assert "Hank Bob" in html


class TestGetAgentHtmlLocal:
    """Test get_agent_html('local') returns correct HTML."""

    def test_returns_html_string(self):
        html = get_agent_html("local")
        assert isinstance(html, str)
        assert len(html) > 100

    def test_contains_pollaudio_code(self):
        html = get_agent_html("local")
        assert "pollAudio" in html

    def test_contains_hank_bob(self):
        html = get_agent_html("local")
        assert "Hank Bob" in html


class TestAgentPageShared:
    """Test shared properties across both agent pages."""

    def test_display_name_constant(self):
        assert AGENT_DISPLAY_NAME == "Hank Bob"

    def test_both_contain_hank_bob(self):
        html_11 = get_agent_html("elevenlabs")
        html_local = get_agent_html("local")
        assert "Hank Bob" in html_11
        assert "Hank Bob" in html_local

    def test_default_is_elevenlabs(self):
        """Default tts_backend argument should return elevenlabs page."""
        html = get_agent_html()
        assert "WebSocket" in html or "ws" in html
