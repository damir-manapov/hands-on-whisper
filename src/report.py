"""Report generation for transcription results."""

from __future__ import annotations

from typing import Any

from backends import get_backend_params


def normalize_text(text: str) -> str:
  """Normalize text for WER comparison: lowercase, remove punctuation, collapse whitespace."""
  import re

  text = text.lower()
  text = re.sub(r"[^\w\s]", "", text)
  text = re.sub(r"\s+", " ", text).strip()
  return text


def calculate_metrics(reference: str, hypothesis: str) -> tuple[float, float]:
  """Calculate WER and CER between reference and hypothesis (normalizes both)."""
  from jiwer import cer, wer

  ref_norm = normalize_text(reference)
  hyp_norm = normalize_text(hypothesis)
  wer_score = wer(ref_norm, hyp_norm) * 100
  cer_score = cer(ref_norm, hyp_norm) * 100
  return wer_score, cer_score


def generate_report(data: dict[str, Any], reference: str | None = None) -> str:
  """Generate a markdown report from transcription data."""
  lines = ["# Transcription Report", ""]
  lines.append(f"**Audio file:** `{data.get('audio', 'unknown')}`")
  lines.append("")

  runs = data.get("runs", [])
  if not runs:
    lines.append("No transcription runs found.")
    return "\n".join(lines)

  def get_wer(run: dict) -> float:
    if reference:
      wer_score, _ = calculate_metrics(reference, run.get("text", ""))
      return wer_score
    return run.get("duration_seconds", 0)

  sorted_runs = sorted(runs, key=get_wer)

  lines.append(f"**Total runs:** {len(runs)}")
  if reference:
    lines.append(f"**Reference:** {len(reference)} chars, {len(reference.split())} words")
  lines.append("")

  # Summary table
  lines.append("## Performance Summary")
  lines.append("")
  hdr = (
    "| # | Backend | Model | GPU | Compute | Beam | Temp | Cond | Batch "
    "| Lang | Dur(s) | MemΔ | Peak |"
  )
  sep = (
    "|---|---------|-------|-----|---------|------|------|------|"
    "-------|------|--------|------|------|"
  )
  if reference:
    lines.append(f"{hdr} WER% | CER% |")
    lines.append(f"{sep}------|------|")
  else:
    lines.append(hdr)
    lines.append(sep)

  for i, run in enumerate(sorted_runs, 1):
    backend = run.get("backend", "?")
    model = run.get("model", "?")
    gpu_name = run.get("gpu_name") or "-"
    if gpu_name != "-":
      gpu_name = gpu_name.replace("NVIDIA ", "").replace("GeForce ", "")

    params = get_backend_params(backend)
    compute = run.get("compute_type") or "-" if "compute_type" in params else "-"
    beam = run.get("beam_size", 5) if "beam_size" in params else "-"
    temp = f"{run.get('temperature', 0.0):.2f}" if "temperature" in params else "-"
    cond_prev = (
      ("Y" if run.get("condition_on_prev", True) else "N") if "condition_on_prev" in params else "-"
    )
    batch_size = run.get("batch_size", 0)
    batch_str = (str(batch_size) if batch_size > 0 else "-") if "batch_size" in params else "-"

    lang = run.get("language") or "auto"
    duration = run.get("duration_seconds", 0)
    mem_delta = run.get("memory_delta_mb", 0)
    mem_peak = run.get("memory_peak_mb", 0)
    row = (
      f"| {i} | {backend} | {model} | {gpu_name} | {compute} | {beam} | {temp} "
      f"| {cond_prev} | {batch_str} | {lang} | {duration:.1f} | {mem_delta} | {mem_peak} |"
    )
    if reference:
      wer_score, cer_score = calculate_metrics(reference, run.get("text", ""))
      lines.append(f"{row} {wer_score:.2f} | {cer_score:.2f} |")
    else:
      lines.append(row)

  lines.append("")

  # Detailed results
  _append_detailed_results(lines, sorted_runs, reference)

  return "\n".join(lines)


def _append_detailed_results(
  lines: list[str], sorted_runs: list[dict], reference: str | None
) -> None:
  """Append detailed transcription results to report lines."""
  lines.append("## Transcription Results")
  lines.append("")

  for i, run in enumerate(sorted_runs, 1):
    run_id = run.get("id", "?")
    backend = run.get("backend", "?")
    model = run.get("model", "?")
    lang = run.get("language") or "auto"
    runtime = run.get("runtime", run.get("device", "?"))  # backward compat
    gpu_name = run.get("gpu_name")
    duration = run.get("duration_seconds", 0)
    mem_delta = run.get("memory_delta_mb", 0)
    mem_peak = run.get("memory_peak_mb", 0)
    beam_size = run.get("beam_size", 5)
    temperature = run.get("temperature", 0.0)
    compute_type = run.get("compute_type", "auto")
    condition_on_prev = run.get("condition_on_prev", True)
    batch_size = run.get("batch_size", 0)
    timestamp = run.get("timestamp", "?")
    text = run.get("text", "")

    lines.append(f"### {i}. {backend} / {model}")
    lines.append("")
    lines.append(f"- **ID:** `{run_id}`")
    lines.append(f"- **Language:** {lang}")
    runtime_str = f"{runtime} ({gpu_name})" if gpu_name else runtime
    lines.append(f"- **Runtime:** {runtime_str}")
    lines.append(f"- **Duration:** {duration:.2f}s")
    lines.append(f"- **Memory:** Δ {mem_delta} MB, peak {mem_peak} MB")

    params = get_backend_params(backend)
    if "beam_size" in params:
      lines.append(f"- **Beam size:** {beam_size}")
    if "temperature" in params:
      lines.append(f"- **Temperature:** {temperature:.2f}")
    if "compute_type" in params:
      lines.append(f"- **Compute type:** {compute_type}")
    if "condition_on_prev" in params:
      lines.append(f"- **Condition on prev:** {condition_on_prev}")
    if "batch_size" in params:
      lines.append(f"- **Batch size:** {batch_size}")
    if reference:
      wer_score, cer_score = calculate_metrics(reference, text)
      lines.append(f"- **WER:** {wer_score:.2f}%")
      lines.append(f"- **CER:** {cer_score:.2f}%")
    lines.append(f"- **Timestamp:** {timestamp}")
    lines.append("")
    lines.append("**Text:**")
    lines.append("")
    lines.append(f"> {text}")
    lines.append("")
