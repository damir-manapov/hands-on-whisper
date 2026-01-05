"""Tests for transcribe_whispercpp module."""

from src.transcribe_whispercpp import transcribe


def test_transcribe_function_exists():
  """Test that transcribe function exists and is callable."""
  assert callable(transcribe)
