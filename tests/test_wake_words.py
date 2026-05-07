"""Comprehensive tests for gmeet_pipeline.wake_words."""

import pytest

from gmeet_pipeline.wake_words import (
    classify_wake_command,
    normalize_text,
    WakeCommand,
    _BOT_ALIASES,
    _SILENCE_PHRASES,
    _WAKE_PHRASES,
    _contains_bot_name,
)


# ── normalize_text ────────────────────────────────────────────────────


class TestNormalizeText:
    """Test text normalization for phrase matching."""

    def test_strips_punctuation(self):
        assert normalize_text("Hank Bob, quiet!") == "hank bob quiet"

    def test_strips_multiple_punctuation(self):
        assert normalize_text("Hey... Hank Bob???") == "hey hank bob"

    def test_collapses_whitespace(self):
        assert normalize_text("Hank   Bob   quiet") == "hank bob quiet"

    def test_strips_leading_trailing_whitespace(self):
        assert normalize_text("  Hank Bob  ") == "hank bob"

    def test_lowercases(self):
        assert normalize_text("HANK BOB WAKE UP") == "hank bob wake up"

    def test_preserves_digits(self):
        assert normalize_text("Hank Bob 2.0") == "hank bob 2 0"

    def test_empty_string(self):
        assert normalize_text("") == ""

    def test_only_punctuation(self):
        assert normalize_text("!!!???") == ""


# ── classify_wake_command — empty / no-match ──────────────────────────


class TestClassifyNoMatch:
    """Test that non-commands are not classified."""

    def test_empty_string_returns_none(self):
        result = classify_wake_command("")
        assert result.kind is None
        assert result.confidence == 0.0

    def test_unrelated_text_returns_none(self):
        result = classify_wake_command("we should benchmark the router after the call")
        assert result.kind is None
        assert result.confidence == 0.0

    def test_silence_phrase_without_bot_name(self):
        """Bare silence phrases without bot name are NOT classified."""
        for phrase in _SILENCE_PHRASES:
            result = classify_wake_command(phrase)
            assert result.kind is None, f"'{phrase}' without bot name should not classify"

    def test_wake_phrase_without_bot_name(self):
        """Bare wake phrases without bot name are NOT classified."""
        for phrase in _WAKE_PHRASES:
            result = classify_wake_command(phrase)
            assert result.kind is None, f"'{phrase}' without bot name should not classify"

    def test_long_utterance_with_name_not_wake(self):
        """Long utterances with name mention are NOT wake commands (>4 words)."""
        result = classify_wake_command(
            "Hank Bob I was wondering if you could tell me about the benchmarks"
        )
        assert result.kind is None

    def test_medium_utterance_with_name_and_silence(self):
        """Silence phrase with name IS classified even in longer utterance."""
        result = classify_wake_command("Hank Bob could you please shut up now")
        assert result.kind == "silence"


# ── classify_wake_command — silence commands ──────────────────────────


class TestClassifySilence:
    """Test silence command classification."""

    def test_basic_silence(self):
        result = classify_wake_command("Hank Bob, quiet")
        assert result.kind == "silence"
        assert result.confidence >= 0.8

    def test_shut_up(self):
        result = classify_wake_command("Hank Bob shut up")
        assert result.kind == "silence"
        assert result.confidence >= 0.7

    def test_stop(self):
        result = classify_wake_command("Hank Bob stop")
        assert result.kind == "silence"

    def test_not_now(self):
        result = classify_wake_command("Hank Bob not now")
        assert result.kind == "silence"

    def test_mute(self):
        result = classify_wake_command("Hank Bob mute")
        assert result.kind == "silence"

    def test_silence_keyword(self):
        result = classify_wake_command("Hank Bob silence")
        assert result.kind == "silence"

    def test_go_silent(self):
        result = classify_wake_command("Hank Bob go silent")
        assert result.kind == "silence"

    def test_wake_down(self):
        result = classify_wake_command("Hank Bob wake down")
        assert result.kind == "silence"

    def test_shut_it(self):
        result = classify_wake_command("Hank Bob shut it")
        assert result.kind == "silence"

    def test_not_right_now(self):
        result = classify_wake_command("Hank Bob not right now")
        assert result.kind == "silence"

    def test_be_quiet(self):
        result = classify_wake_command("Hank Bob be quiet")
        assert result.kind == "silence"

    def test_stop_talking(self):
        result = classify_wake_command("Hank Bob stop talking")
        assert result.kind == "silence"

    def test_mute_yourself(self):
        result = classify_wake_command("Hank Bob mute yourself")
        assert result.kind == "silence"


class TestAllSilencePhrasesWithBotName:
    """All _SILENCE_PHRASES should classify as silence when prefixed with bot name."""

    @pytest.mark.parametrize("phrase", _SILENCE_PHRASES)
    def test_silence_with_name(self, phrase):
        result = classify_wake_command(f"Hank Bob {phrase}")
        assert result.kind == "silence", f"'Hank Bob {phrase}' should be silence"


# ── classify_wake_command — wake commands ─────────────────────────────


class TestClassifyWake:
    """Test wake command classification."""

    def test_hey_hank_bob(self):
        result = classify_wake_command("hey Hank Bob")
        assert result.kind == "wake"
        assert result.confidence >= 0.8

    def test_hey_hank_bob_short_context(self):
        """'hey hank bob' with short context is wake."""
        result = classify_wake_command("hey Hank Bob")
        assert result.kind == "wake"

    def test_wake_up(self):
        result = classify_wake_command("Hank Bob wake up")
        assert result.kind == "wake"

    def test_come_back(self):
        result = classify_wake_command("Hank Bob come back")
        assert result.kind == "wake"

    def test_listen(self):
        result = classify_wake_command("Hank Bob listen")
        assert result.kind == "wake"

    def test_you_there(self):
        result = classify_wake_command("Hank Bob you there")
        assert result.kind == "wake"

    def test_bare_name_is_wake(self):
        """Just the bot name alone is a wake command."""
        result = classify_wake_command("Hank Bob")
        assert result.kind == "wake"

    def test_name_with_hey_prefix_is_wake(self):
        """'hey Hank Bob' is wake."""
        result = classify_wake_command("hey Hank Bob")
        assert result.kind == "wake"


class TestAllWakePhrasesWithBotName:
    """All _WAKE_PHRASES should classify as wake when prefixed with bot name."""

    @pytest.mark.parametrize("phrase", _WAKE_PHRASES)
    def test_wake_with_name(self, phrase):
        result = classify_wake_command(f"Hank Bob {phrase}")
        assert result.kind == "wake", f"'Hank Bob {phrase}' should be wake"


# ── classify_wake_command — bot aliases ───────────────────────────────


class TestAllBotAliases:
    """All _BOT_ALIASES should be recognized as the bot name."""

    @pytest.mark.parametrize("alias", _BOT_ALIASES)
    def test_alias_is_wake(self, alias):
        """Each alias alone should classify as wake (≤4 words)."""
        result = classify_wake_command(alias)
        # Some aliases are fuzzy matches (like "hank bop", "hang bob") —
        # they should still match via fuzzy matching
        if alias in ("hank bob", "hey hank bob"):
            assert result.kind == "wake", f"Exact alias '{alias}' should be wake"
        else:
            # Fuzzy aliases — still should match
            assert result.kind is not None, f"Alias '{alias}' should be recognized"


# ── classify_wake_command — fuzzy matching ────────────────────────────


class TestFuzzyMatching:
    """Test fuzzy matching for misspelled names."""

    def test_hang_bob(self):
        result = classify_wake_command("hang bob quiet")
        assert result.kind == "silence"

    def test_hank_bop(self):
        result = classify_wake_command("hank bop quiet")
        assert result.kind == "silence"

    def test_hank_barb(self):
        result = classify_wake_command("hank barb quiet")
        assert result.kind == "silence"

    def test_hank_bab(self):
        result = classify_wake_command("hank bab quiet")
        assert result.kind == "silence"

    def test_hankbot_wake(self):
        result = classify_wake_command("hankbot wake up")
        assert result.kind == "wake"

    def test_hank_bot_wake(self):
        result = classify_wake_command("hank bot wake up")
        assert result.kind == "wake"


# ── classify_wake_command — confidence scores ─────────────────────────


class TestConfidenceScores:
    """Test that confidence scores are reasonable."""

    def test_exact_name_has_high_confidence(self):
        result = classify_wake_command("Hank Bob quiet")
        assert result.confidence >= 0.9

    def test_fuzzy_name_has_reasonable_confidence(self):
        result = classify_wake_command("hang bob quiet")
        assert result.confidence >= 0.7

    def test_no_match_has_zero_confidence(self):
        result = classify_wake_command("something unrelated")
        assert result.confidence == 0.0

    def test_wake_command_confidence(self):
        result = classify_wake_command("hey Hank Bob")
        assert result.confidence >= 0.8

    def test_silence_command_confidence(self):
        result = classify_wake_command("Hank Bob stop")
        assert result.confidence >= 0.8


# ── classify_wake_command — matched_phrase ────────────────────────────


class TestMatchedPhrase:
    """Test that matched_phrase is populated correctly."""

    def test_silence_matched_phrase(self):
        result = classify_wake_command("Hank Bob quiet")
        assert result.matched_phrase != ""

    def test_wake_matched_phrase(self):
        result = classify_wake_command("Hank Bob wake up")
        assert result.matched_phrase != ""

    def test_no_match_has_empty_matched_phrase(self):
        result = classify_wake_command("something unrelated")
        assert result.matched_phrase == ""


# ── WakeCommand dataclass ─────────────────────────────────────────────


class TestWakeCommandDataclass:
    """Test the WakeCommand frozen dataclass."""

    def test_default_values(self):
        cmd = WakeCommand(kind=None)
        assert cmd.kind is None
        assert cmd.confidence == 0.0
        assert cmd.matched_phrase == ""

    def test_frozen(self):
        cmd = WakeCommand(kind="wake", confidence=0.9, matched_phrase="hank bob")
        with pytest.raises(AttributeError):
            cmd.kind = "silence"

    def test_with_values(self):
        cmd = WakeCommand(kind="silence", confidence=0.85, matched_phrase="hank bob")
        assert cmd.kind == "silence"
        assert cmd.confidence == 0.85
        assert cmd.matched_phrase == "hank bob"


# ── _contains_bot_name ────────────────────────────────────────────────


class TestContainsBotName:
    """Test the internal _contains_bot_name helper."""

    def test_exact_match(self):
        has, score, span = _contains_bot_name("hank bob")
        assert has is True
        assert score == 1.0

    def test_no_match(self):
        has, score, span = _contains_bot_name("alice said hello")
        assert has is False

    def test_fuzzy_match(self):
        has, score, span = _contains_bot_name("hang bob")
        assert has is True
        assert score >= 0.78

    def test_name_in_sentence(self):
        has, score, span = _contains_bot_name("hey hank bob stop")
        assert has is True
