"""Tests for Hank Bob wake/silence phrase classification."""

from gmeet_pipeline.wake_words import classify_wake_command


def test_classifies_silence_command():
    result = classify_wake_command("Hank Bob, quiet")

    assert result.kind == "silence"
    assert result.confidence >= 0.8


def test_classifies_phonetic_silence_variant():
    result = classify_wake_command("hang bob shut up")

    assert result.kind == "silence"
    assert result.confidence >= 0.7


def test_classifies_wake_command():
    result = classify_wake_command("hey Hank Bob")

    assert result.kind == "wake"
    assert result.confidence >= 0.8


def test_ignores_unrelated_text():
    result = classify_wake_command("we should benchmark the router after the call")

    assert result.kind is None
    assert result.confidence == 0.0
