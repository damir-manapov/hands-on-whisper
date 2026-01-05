"""Transcription using faster-whisper."""

from faster_whisper import WhisperModel


def transcribe(audio_path: str, model_size: str = "base") -> str:
  """
  Transcribe audio file using faster-whisper.

  Args:
      audio_path: Path to the audio file
      model_size: Whisper model size (tiny, base, small, medium, large-v3)

  Returns:
      Transcribed text
  """
  model = WhisperModel(model_size, device="cpu", compute_type="int8")
  segments, _info = model.transcribe(audio_path, beam_size=5)
  return " ".join(segment.text.strip() for segment in segments)


if __name__ == "__main__":
  import sys

  if len(sys.argv) < 2:
    print("Usage: python src/transcribe_faster.py <audio_file>")
    sys.exit(1)

  audio_file = sys.argv[1]
  result = transcribe(audio_file)
  print(result)
