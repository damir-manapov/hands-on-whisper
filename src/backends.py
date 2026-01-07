"""Backend transcription functions for different Whisper implementations."""

from __future__ import annotations

import json
import time
from pathlib import Path

# Cloud backends and their models
CLOUD_BACKENDS = ["yandex", "openai-api", "deepgram"]
OPENAI_API_MODELS = ["whisper-1", "gpt-4o-transcribe", "gpt-4o-mini-transcribe"]
DEEPGRAM_MODELS = [
  # Latest models
  "nova-3",
  "nova-3-general",
  "nova-3-medical",
  # Nova-2 variants
  "nova-2",
  "nova-2-general",
  "nova-2-meeting",
  "nova-2-phonecall",
  "nova-2-finance",
  "nova-2-conversationalai",
  "nova-2-voicemail",
  "nova-2-video",
  "nova-2-medical",
  "nova-2-drivethru",
  "nova-2-automotive",
  "nova-2-atc",
  # Whisper Cloud (hosted OpenAI Whisper)
  "whisper-tiny",
  "whisper-base",
  "whisper-small",
  "whisper-medium",
  "whisper-large",
  # Legacy models
  "nova",
  "nova-general",
  "nova-phonecall",
  "nova-medical",
  "enhanced",
  "enhanced-general",
  "enhanced-meeting",
  "enhanced-phonecall",
  "enhanced-finance",
  "base",
  "base-general",
  "base-meeting",
  "base-phonecall",
  "base-finance",
  "base-conversationalai",
  "base-voicemail",
  "base-video",
]

# Default search space for optimization
LOCAL_BACKENDS = ["faster-whisper", "openai", "whispercpp"]
ALL_MODELS = ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"]
ALL_COMPUTE_TYPES = ["int8", "float16", "float32"]

# Optional parameter fields shown in reports per backend type.
LOCAL_BACKEND_PARAMS = [
  "beam_size",
  "temperature",
  "compute_type",
  "condition_on_prev",
  "batch_size",
]
CLOUD_BACKEND_PARAMS: dict[str, list[str]] = {
  "yandex": [],  # No tunable params
  "openai-api": ["temperature"],  # Supports temperature
  "deepgram": ["temperature"],  # Supports temperature
}


def get_backend_params(backend: str) -> list[str]:
  """Get the list of optional parameter fields to show for a backend."""
  if backend in CLOUD_BACKEND_PARAMS:
    return CLOUD_BACKEND_PARAMS[backend]
  return LOCAL_BACKEND_PARAMS


def resolve_whispercpp_model_path(model: str, compute_type: str) -> str:
  """Resolve whisper.cpp model path from model name and compute type."""
  if compute_type == "int8":
    if model == "large-v3":
      suffix = "-q5_0"
    elif model == "distil-large-v3":
      suffix = ""
    else:
      suffix = "-q8_0"
  else:
    suffix = ""
  return f"models/ggml-{model}{suffix}.bin"


def transcribe_faster_whisper(  # noqa: PLR0913
  audio_path: str,
  model_size: str,
  language: str | None,
  runtime: str,
  beam_size: int,
  temperature: float,
  compute_type: str,
  condition_on_prev: bool = True,
  batch_size: int = 0,
) -> str:
  """Transcribe using faster-whisper."""
  from faster_whisper import WhisperModel

  if compute_type == "auto":
    ct = "float16" if runtime == "cuda" else "int8"
  elif compute_type == "float32":
    ct = "float32"
  elif compute_type == "float16":
    ct = "float16"
  elif compute_type == "int8":
    ct = "int8"
  else:
    ct = compute_type

  model = WhisperModel(model_size, device=runtime, compute_type=ct)

  if batch_size > 0:
    from faster_whisper import BatchedInferencePipeline

    batched_model = BatchedInferencePipeline(model)
    segments, _info = batched_model.transcribe(
      audio_path,
      batch_size=batch_size,
      language=language,
    )
  else:
    segments, _info = model.transcribe(
      audio_path,
      beam_size=beam_size,
      language=language,
      temperature=temperature,
      condition_on_previous_text=condition_on_prev,
    )
  return " ".join(segment.text.strip() for segment in segments)


def transcribe_openai_whisper(  # noqa: PLR0913
  audio_path: str,
  model_size: str,
  language: str | None,
  runtime: str,
  beam_size: int,
  temperature: float,
  compute_type: str,
  condition_on_prev: bool = True,
) -> str:
  """Transcribe using OpenAI whisper."""
  import whisper

  fp16 = compute_type != "float32" and runtime == "cuda"

  if model_size == "distil-large-v3":
    model_path = "models/distil-large-v3-openai/model.bin"
    model = whisper.load_model(model_path, device=runtime)
  else:
    model = whisper.load_model(model_size, device=runtime)

  result = model.transcribe(
    audio_path,
    language=language,
    beam_size=beam_size,
    temperature=temperature,
    fp16=fp16,
    condition_on_previous_text=condition_on_prev,
  )
  return result["text"].strip()


def transcribe_whispercpp(  # noqa: PLR0913
  audio_path: str,
  model_path: str,
  language: str | None,
  beam_size: int,
  temperature: float,
  compute_type: str,  # noqa: ARG001 - used for run_id
) -> str:
  """Transcribe using whisper.cpp."""
  from pywhispercpp.model import Model

  model = Model(model_path)
  segments = model.transcribe(
    audio_path,
    language=language or "",
    beam_search={"beam_size": beam_size, "patience": -1.0},
    temperature=temperature,
  )
  return " ".join(segment.text.strip() for segment in segments)


def transcribe_openai_api(
  audio_path: str,
  model: str = "whisper-1",
  language: str | None = None,
  temperature: float = 0.0,
  api_key: str | None = None,
) -> str:
  """Transcribe using OpenAI Whisper API."""
  import os

  import requests

  api_key = api_key or os.environ.get("OPENAI_API_KEY")
  if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable required")

  url = "https://api.openai.com/v1/audio/transcriptions"
  headers = {"Authorization": f"Bearer {api_key}"}

  with Path(audio_path).open("rb") as f:
    files = {"file": (Path(audio_path).name, f, "audio/mpeg")}
    data = {"model": model, "temperature": str(temperature)}
    if language:
      data["language"] = language.split("-")[0] if "-" in language else language

    response = requests.post(url, headers=headers, files=files, data=data, timeout=300)

  if not response.ok:
    raise ValueError(f"OpenAI API error {response.status_code}: {response.text}")

  return response.json().get("text", "")


def transcribe_yandex(
  audio_path: str,
  language: str | None,
  api_key: str | None = None,
  folder_id: str | None = None,
) -> str:
  """Transcribe using Yandex SpeechKit."""
  import os

  api_key = api_key or os.environ.get("YANDEX_API_KEY")
  folder_id = folder_id or os.environ.get("YANDEX_FOLDER_ID")

  if not api_key:
    raise ValueError("YANDEX_API_KEY environment variable or --yandex-api-key required")
  if not folder_id:
    raise ValueError("YANDEX_FOLDER_ID environment variable or --yandex-folder-id required")

  audio_duration = _get_audio_duration(audio_path)

  if audio_duration <= 30:
    return _yandex_sync_recognize(audio_path, language, api_key, folder_id)
  else:
    return _yandex_async_recognize(audio_path, language, api_key, folder_id)


def _get_audio_duration(audio_path: str) -> float:
  """Get audio duration in seconds using ffprobe."""
  import subprocess

  result = subprocess.run(
    [
      "ffprobe",
      "-v",
      "error",
      "-show_entries",
      "format=duration",
      "-of",
      "default=noprint_wrappers=1:nokey=1",
      audio_path,
    ],
    capture_output=True,
    text=True,
    check=True,
  )
  return float(result.stdout.strip())


def transcribe_deepgram(
  audio_path: str,
  model: str = "nova-2",
  language: str | None = None,
  temperature: float = 0.0,
  api_key: str | None = None,
) -> str:
  """Transcribe using Deepgram API."""
  import os

  import requests

  api_key = api_key or os.environ.get("DEEPGRAM_API_KEY")
  if not api_key:
    raise ValueError("DEEPGRAM_API_KEY environment variable required")

  url = "https://api.deepgram.com/v1/listen"
  headers = {"Authorization": f"Token {api_key}"}

  params = {
    "model": model,
    "punctuate": "true",
    "utterances": "false",
  }

  if language:
    params["language"] = language.split("-")[0] if "-" in language else language

  if temperature > 0:
    params["temperature"] = str(temperature)

  with Path(audio_path).open("rb") as f:
    response = requests.post(
      url,
      headers=headers,
      params=params,
      data=f,
      timeout=300,
    )

  if not response.ok:
    raise ValueError(f"Deepgram API error {response.status_code}: {response.text}")

  result = response.json()
  transcript = (
    result.get("results", {})
    .get("channels", [{}])[0]
    .get("alternatives", [{}])[0]
    .get("transcript", "")
  )
  return transcript.strip()


def _yandex_sync_recognize(
  audio_path: str,
  language: str | None,
  api_key: str,
  folder_id: str,
) -> str:
  """Synchronous recognition for short audio (up to 30 seconds)."""
  import requests

  audio_data = Path(audio_path).read_bytes()

  url = "https://stt.api.cloud.yandex.net/speech/v1/stt:recognize"
  headers = {"Authorization": f"Api-Key {api_key}"}
  params = {
    "folderId": folder_id,
    "lang": language or "ru-RU",
  }

  response = requests.post(url, headers=headers, params=params, data=audio_data, timeout=60)
  response.raise_for_status()
  return response.json().get("result", "")


def _yandex_async_recognize(  # noqa: PLR0912
  audio_path: str,
  language: str | None,
  api_key: str,
  folder_id: str,
) -> str:
  """Async recognition for long audio using Yandex SpeechKit v3 API."""
  import base64

  import requests

  audio_path = Path(audio_path)
  audio_data = audio_path.read_bytes()

  ext = audio_path.suffix.lower()
  container_type = {".mp3": "MP3", ".wav": "WAV", ".ogg": "OGG_OPUS"}.get(ext, "WAV")

  url = "https://stt.api.cloud.yandex.net/stt/v3/recognizeFileAsync"
  headers = {
    "Authorization": f"Api-Key {api_key}",
    "x-folder-id": folder_id,
    "Content-Type": "application/json",
  }

  payload = {
    "content": base64.b64encode(audio_data).decode("utf-8"),
    "recognitionModel": {
      "model": "general",
      "audioFormat": {"containerAudio": {"containerAudioType": container_type}},
      "textNormalization": {
        "textNormalization": "TEXT_NORMALIZATION_ENABLED",
        "profanityFilter": False,
        "literatureText": True,
      },
      "languageRestriction": {
        "restrictionType": "WHITELIST",
        "languageCode": [language or "ru-RU"],
      },
    },
  }

  response = requests.post(url, headers=headers, json=payload, timeout=120)
  if not response.ok:
    raise ValueError(f"Yandex API error {response.status_code}: {response.text}")

  operation_id = response.json().get("id")
  if not operation_id:
    raise ValueError(f"No operation ID in response: {response.json()}")

  result_url = f"https://operation.api.cloud.yandex.net/operations/{operation_id}"
  poll_headers = {"Authorization": f"Api-Key {api_key}"}

  for _ in range(120):
    time.sleep(5)
    result = requests.get(result_url, headers=poll_headers, timeout=30)
    result.raise_for_status()
    data = result.json()

    if data.get("done"):
      if "error" in data:
        raise ValueError(f"Yandex recognition error: {data['error']}")

      get_result_url = (
        f"https://stt.api.cloud.yandex.net/stt/v3/getRecognition?operationId={operation_id}"
      )
      result_response = requests.get(get_result_url, headers=poll_headers, timeout=60)
      if not result_response.ok:
        raise ValueError(
          f"Failed to get results: {result_response.status_code}: {result_response.text}"
        )

      texts = []
      for line in result_response.text.strip().split("\n"):
        if not line.strip():
          continue
        chunk = json.loads(line)
        result_data = chunk.get("result", {})
        if "finalRefinement" in result_data:
          alternatives = (
            result_data["finalRefinement"].get("normalizedText", {}).get("alternatives", [])
          )
        elif "final" in result_data:
          alternatives = result_data["final"].get("alternatives", [])
        else:
          continue
        if alternatives:
          texts.append(alternatives[0].get("text", ""))

      if texts:
        return " ".join(texts)

      raise ValueError(f"Could not extract text from results: {result_response.text[:500]}")

  raise TimeoutError("Yandex recognition timed out after 10 minutes")


# Backend dispatch - maps backend name to transcription function
BACKEND_HANDLERS = {
  "faster-whisper": transcribe_faster_whisper,
  "openai": transcribe_openai_whisper,
  "whispercpp": transcribe_whispercpp,
  "yandex": transcribe_yandex,
  "openai-api": transcribe_openai_api,
}
