"""Download all whisper models for offline use."""

import argparse


def download_faster_whisper_models() -> None:
  """Download faster-whisper models."""
  from faster_whisper import WhisperModel

  models = ["tiny", "base", "small", "medium", "large-v3", "distil-large-v3"]
  for model_name in models:
    print(f"Downloading faster-whisper: {model_name}...")
    WhisperModel(model_name, device="cpu", compute_type="int8")
    print(f"  Done: {model_name}")


def download_openai_whisper_models() -> None:
  """Download OpenAI whisper models."""
  import whisper

  models = ["tiny", "base", "small", "medium", "large-v3"]
  for model_name in models:
    print(f"Downloading openai-whisper: {model_name}...")
    whisper.load_model(model_name, device="cpu")
    print(f"  Done: {model_name}")


def download_whispercpp_models() -> None:
  """Download whisper.cpp ggml models (f16 and quantized)."""
  import urllib.request
  from pathlib import Path

  models_dir = Path("models")
  models_dir.mkdir(exist_ok=True)

  # Define which quantizations are available for each model
  # Main models from https://huggingface.co/ggerganov/whisper.cpp/tree/main
  model_quantizations = {
    "tiny": ["", "-q8_0"],
    "base": ["", "-q8_0"],
    "small": ["", "-q8_0"],
    "medium": ["", "-q8_0"],
    "large-v3": ["", "-q5_0"],  # No q8_0 available, only q5_0
    "large-v3-turbo": ["", "-q8_0"],
  }
  base_url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"

  for model_name, quantizations in model_quantizations.items():
    for quant in quantizations:
      filename = f"ggml-{model_name}{quant}.bin"
      filepath = models_dir / filename
      if filepath.exists():
        print(f"whisper.cpp: {filename} already exists")
        continue
      print(f"Downloading whisper.cpp: {filename}...")
      url = f"{base_url}/{filename}"
      try:
        urllib.request.urlretrieve(url, filepath)
        print(f"  Done: {filepath}")
      except Exception as e:
        print(f"  Failed: {e}")
        # Remove partial download
        if filepath.exists():
          filepath.unlink()

  # distil-large-v3 is in a separate repo: distil-whisper/distil-large-v3-ggml
  distil_url = "https://huggingface.co/distil-whisper/distil-large-v3-ggml/resolve/main"
  distil_filename = "ggml-distil-large-v3.bin"
  distil_filepath = models_dir / distil_filename
  if distil_filepath.exists():
    print(f"whisper.cpp: {distil_filename} already exists")
  else:
    print(f"Downloading whisper.cpp: {distil_filename}...")
    try:
      urllib.request.urlretrieve(f"{distil_url}/{distil_filename}", distil_filepath)
      print(f"  Done: {distil_filepath}")
    except Exception as e:
      print(f"  Failed: {e}")
      if distil_filepath.exists():
        distil_filepath.unlink()


def main() -> None:
  parser = argparse.ArgumentParser(description="Download whisper models for offline use")
  parser.add_argument(
    "--backend",
    "-b",
    choices=["faster-whisper", "openai", "whispercpp", "all"],
    default="all",
    help="Which backend to download models for (default: all)",
  )
  args = parser.parse_args()

  if args.backend in ("faster-whisper", "all"):
    download_faster_whisper_models()

  if args.backend in ("openai", "all"):
    download_openai_whisper_models()

  if args.backend in ("whispercpp", "all"):
    download_whispercpp_models()

  print("\nAll models downloaded!")


if __name__ == "__main__":
  main()
