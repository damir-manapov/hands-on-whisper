"""Core transcription runner and utilities."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backends import (
  CLOUD_BACKENDS,
  resolve_whispercpp_model_path,
  transcribe_deepgram,
  transcribe_faster_whisper,
  transcribe_openai_api,
  transcribe_openai_whisper,
  transcribe_whispercpp,
  transcribe_yandex,
)
from report import generate_report


def get_gpu_name() -> str | None:
  """Get GPU name if CUDA is available."""
  try:
    import torch

    if torch.cuda.is_available():
      return torch.cuda.get_device_name(0)
  except ImportError:
    pass
  return None


def generate_run_id(  # noqa: PLR0913
  backend: str,
  model: str,
  language: str | None,
  runtime: str,
  beam_size: int,
  temperature: float,
  compute_type: str,
  condition_on_prev: bool = True,
  batch_size: int = 0,
  smart_format: bool = True,
  diarize: bool = False,
) -> str:
  """Generate a unique ID based on settings."""
  if backend == "yandex":
    parts = [backend, model, language]
  elif backend == "openai-api":
    parts = [backend, model, language, f"temp{temperature}"]
  elif backend == "deepgram":
    sf = "smartfmt" if smart_format else "nosmartfmt"
    dia = "diarize" if diarize else "nodiarize"
    parts = [backend, model, language, f"temp{temperature}", sf, dia]
  else:
    cond = "cond" if condition_on_prev else "nocond"
    batch = f"batch{batch_size}" if batch_size > 0 else "seq"
    parts = [
      backend,
      model,
      language,
      runtime,
      f"beam{beam_size}",
      f"temp{temperature}",
      compute_type,
      cond,
      batch,
    ]
  settings = ":".join(str(p) for p in parts)
  return hashlib.sha256(settings.encode()).hexdigest()[:12]


def find_existing_run(data: dict[str, Any], run_id: str) -> dict[str, Any] | None:
  """Find an existing run by ID in the data."""
  return next((r for r in data.get("runs", []) if r.get("id") == run_id), None)


def load_reference(data: dict[str, Any]) -> str | None:
  """Load reference text file based on audio path in JSON data."""
  audio_path = Path(data.get("audio", ""))
  ref_path = audio_path.with_suffix(".txt")
  if ref_path.exists():
    return ref_path.read_text(encoding="utf-8").strip()
  return None


def save_results(data: dict[str, Any], json_path: Path) -> None:
  """Save JSON data and regenerate MD report."""
  json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

  reference = load_reference(data)
  report = generate_report(data, reference)
  md_path = json_path.with_suffix(".md")
  md_path.write_text(report, encoding="utf-8")


def get_result_suffix(backend: str, runtime: str) -> str:
  """Get the result file suffix based on backend and runtime."""
  if backend in CLOUD_BACKENDS:
    return f"_{backend}"
  elif runtime == "cuda":
    return "_gpu"
  else:
    return "_cpu"


def run_transcription(  # noqa: PLR0913
  audio: str,
  backend: str,
  model: str,
  language: str | None,
  runtime: str,
  beam_size: int,
  temperature: float,
  compute_type: str,
  condition_on_prev: bool,
  batch_size: int = 0,
  smart_format: bool = True,
  diarize: bool = False,
  user: str | None = None,
) -> dict:
  """Run transcription and return run record with timing/memory stats."""
  import os

  import psutil

  run_id = generate_run_id(
    backend,
    model,
    language,
    runtime,
    beam_size,
    temperature,
    compute_type,
    condition_on_prev,
    batch_size,
    smart_format,
    diarize,
  )

  process = psutil.Process()
  mem_before = process.memory_info().rss
  start_time = time.perf_counter()

  if backend == "faster-whisper":
    result = transcribe_faster_whisper(
      audio,
      model,
      language,
      runtime,
      beam_size,
      temperature,
      compute_type,
      condition_on_prev,
      batch_size,
    )
  elif backend == "openai":
    result = transcribe_openai_whisper(
      audio, model, language, runtime, beam_size, temperature, compute_type, condition_on_prev
    )
  elif backend == "whispercpp":
    if model.endswith(".bin"):
      model_path = model
    else:
      model_path = resolve_whispercpp_model_path(model, compute_type)
    result = transcribe_whispercpp(
      audio, model_path, language, beam_size, temperature, compute_type
    )
  elif backend == "yandex":
    result = transcribe_yandex(
      audio,
      language,
      api_key=os.environ.get("YANDEX_API_KEY"),
      folder_id=os.environ.get("YANDEX_FOLDER_ID"),
    )
  elif backend == "openai-api":
    openai_model_map = {
      "whisper-1": "whisper-1",
      "gpt-4o": "gpt-4o-transcribe",
      "gpt-4o-mini": "gpt-4o-mini-transcribe",
    }
    api_model = openai_model_map.get(model, "whisper-1")
    result = transcribe_openai_api(audio, api_model, language, temperature)
  elif backend == "deepgram":
    result = transcribe_deepgram(audio, model, language, temperature, smart_format, diarize)
  else:
    msg = f"Unknown backend: {backend}"
    raise ValueError(msg)

  duration = time.perf_counter() - start_time
  mem_after = process.memory_info().rss
  mem_used_mb = round((mem_after - mem_before) / 1024 / 1024, 1)
  mem_peak_mb = round(mem_after / 1024 / 1024, 1)

  gpu_name = get_gpu_name() if runtime == "cuda" else None

  return {
    "id": run_id,
    "timestamp": datetime.now(UTC).isoformat(),
    "user": user,
    "duration_seconds": round(duration, 2),
    "memory_delta_mb": mem_used_mb,
    "memory_peak_mb": mem_peak_mb,
    "backend": backend,
    "model": model,
    "language": language,
    "runtime": runtime,
    "gpu_name": gpu_name,
    "beam_size": beam_size,
    "temperature": temperature,
    "compute_type": compute_type,
    "condition_on_prev": condition_on_prev,
    "batch_size": batch_size,
    "smart_format": smart_format,
    "diarize": diarize,
    "text": result,
  }


def run_single(  # noqa: PLR0913
  audio: str,
  backend: str,
  model: str,
  language: str | None,
  runtime: str,
  beam_size: int,
  temperature: float,
  compute_type: str,
  condition_on_prev: bool,
  batch_size: int,
  data: dict,
  user: str | None = None,
) -> None:
  """Run a single transcription and update data (used by cmd_transcribe)."""
  run_id = generate_run_id(
    backend,
    model,
    language,
    runtime,
    beam_size,
    temperature,
    compute_type,
    condition_on_prev,
    batch_size,
  )

  if find_existing_run(data, run_id):
    if backend in CLOUD_BACKENDS:
      print(f"\n[{run_id}] {backend} / lang={language} - skipped (exists)")
    else:
      print(f"\n[{run_id}] {backend} / {model} / lang={language} / {runtime} - skipped (exists)")
    return

  if backend in CLOUD_BACKENDS:
    print(f"\n[{run_id}] {backend} / lang={language}")
  else:
    print(f"\n[{run_id}] {backend} / {model} / lang={language} / {runtime}")
    cond_str = "" if condition_on_prev else ", no_cond_prev"
    batch_str = f", batch={batch_size}" if batch_size > 0 else ""
    print(f"  beam={beam_size}, temp={temperature}, compute={compute_type}{cond_str}{batch_str}")

  run_record = run_transcription(
    audio,
    backend,
    model,
    language,
    runtime,
    beam_size,
    temperature,
    compute_type,
    condition_on_prev,
    batch_size,
    user=user,
  )
  data["runs"].append(run_record)

  duration = run_record["duration_seconds"]
  mem_used_mb = run_record["memory_delta_mb"]
  mem_peak_mb = run_record["memory_peak_mb"]
  result = run_record["text"]

  print(f"  Done ({duration:.2f}s, mem: +{mem_used_mb}MB, peak: {mem_peak_mb}MB)")
  print(f"  Text: {result[:100]}..." if len(result) > 100 else f"  Text: {result}")
