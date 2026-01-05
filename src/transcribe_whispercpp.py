"""Transcription using whisper.cpp via pywhispercpp."""

from pywhispercpp.model import Model


def transcribe(audio_path: str, model_path: str) -> str:
  """
  Transcribe audio file using whisper.cpp.

  Args:
      audio_path: Path to the audio file (16-bit WAV, 16kHz recommended)
      model_path: Path to the ggml model file

  Returns:
      Transcribed text
  """
  model = Model(model_path)
  segments = model.transcribe(audio_path)
  return " ".join(segment.text.strip() for segment in segments)


if __name__ == "__main__":
  import sys

  if len(sys.argv) < 3:
    print("Usage: python src/transcribe_whispercpp.py <model_path> <audio_file>")
    print("Example: python src/transcribe_whispercpp.py models/ggml-base.en.bin audio.wav")
    sys.exit(1)

  model_file = sys.argv[1]
  audio_file = sys.argv[2]
  result = transcribe(audio_file, model_file)
  print(result)
