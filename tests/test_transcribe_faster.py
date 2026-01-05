"""Tests for transcribe_faster module."""

from src.transcribe_faster import transcribe


def test_transcribe_function_exists():
  """Test that transcribe function exists and is callable."""
  assert callable(transcribe)
