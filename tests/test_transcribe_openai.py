"""Tests for transcribe_openai module."""

from src.transcribe_openai import transcribe


def test_transcribe_function_exists():
  """Test that transcribe function exists and is callable."""
  assert callable(transcribe)
