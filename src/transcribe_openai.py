"""Transcription using OpenAI whisper."""

import whisper


def transcribe(audio_path: str, model_size: str = "base") -> str:
  """
  Transcribe audio file using OpenAI whisper.

  Args:
      audio_path: Path to the audio file
      model_size: Whisper model size (tiny, base, small, medium, large)

  Returns:
      Transcribed text
  """
  model = whisper.load_model(model_size)
  result = model.transcribe(audio_path)
  return result["text"].strip()


if __name__ == "__main__":
  import sys

  if len(sys.argv) < 2:
    print("Usage: python src/transcribe_openai.py <audio_file>")
    sys.exit(1)

  audio_file = sys.argv[1]
  result = transcribe(audio_file)
  print(result)
