"""Tests for unified transcribe module."""

from src.backends import (
  transcribe_faster_whisper,
  transcribe_openai_whisper,
  transcribe_whispercpp,
)


def test_transcribe_functions_exist():
  """Test that transcribe functions exist and are callable."""
  assert callable(transcribe_faster_whisper)
  assert callable(transcribe_openai_whisper)
  assert callable(transcribe_whispercpp)
