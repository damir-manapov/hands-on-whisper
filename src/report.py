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


def generate_report(data: dict[str, Any], reference: str | None = None) -> str:  # noqa: PLR0912, PLR0915
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

  # Check if all runs are cloud backends
  from backends import CLOUD_BACKENDS

  all_cloud = all(run.get("backend") in CLOUD_BACKENDS for run in sorted_runs)

  if all_cloud:
    # Simplified table for cloud backends - build dynamic columns based on params present
    all_params = set()
    for run in sorted_runs:
      backend = run.get("backend", "?")
      all_params.update(get_backend_params(backend))

    # Build header dynamically
    hdr_parts = ["| #", "Backend", "Model"]
    sep_parts = ["|---", "---------", "-------"]

    if "temperature" in all_params:
      hdr_parts.append("Temp")
      sep_parts.append("------")
    if "smart_format" in all_params:
      hdr_parts.append("SmartFmt")
      sep_parts.append("----------")
    if "diarize" in all_params:
      hdr_parts.append("Diarize")
      sep_parts.append("---------")

    hdr_parts.extend(["Lang", "Dur(s)", "MemΔ", "Peak |"])
    sep_parts.extend(["------", "--------|", "------|", "------|"])

    hdr = " | ".join(hdr_parts)
    sep = "|".join(sep_parts)

    if reference:
      lines.append(f"{hdr} WER% | CER% |")
      lines.append(f"{sep}------|------|")
    else:
      lines.append(hdr)
      lines.append(sep)
  else:
    # Full table for local backends
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
    lang = run.get("language") or "auto"
    duration = run.get("duration_seconds", 0)
    mem_delta = run.get("memory_delta_mb", 0)
    mem_peak = run.get("memory_peak_mb", 0)

    if all_cloud:
      # Simplified row for cloud backends
      params = get_backend_params(backend)

      row_parts = [f"| {i}", backend, model]

      if "temperature" in all_params:
        temp = f"{run.get('temperature', 0.0):.2f}" if "temperature" in params else "-"
        row_parts.append(temp)
      if "smart_format" in all_params:
        smart_format = (
          ("Y" if run.get("smart_format", True) else "N") if "smart_format" in params else "-"
        )
        row_parts.append(smart_format)
      if "diarize" in all_params:
        diarize = ("Y" if run.get("diarize", False) else "N") if "diarize" in params else "-"
        row_parts.append(diarize)

      row_parts.extend([lang, f"{duration:.1f}", str(mem_delta), f"{mem_peak} |"])
      row = " | ".join(row_parts)
    else:
      # Full row for local backends
      gpu_name = run.get("gpu_name") or "-"
      if gpu_name != "-":
        gpu_name = gpu_name.replace("NVIDIA ", "").replace("GeForce ", "")

      params = get_backend_params(backend)
      compute = run.get("compute_type") or "-" if "compute_type" in params else "-"
      beam = run.get("beam_size", 5) if "beam_size" in params else "-"
      temp = f"{run.get('temperature', 0.0):.2f}" if "temperature" in params else "-"
      cond_prev = (
        ("Y" if run.get("condition_on_prev", True) else "N")
        if "condition_on_prev" in params
        else "-"
      )
      batch_size = run.get("batch_size", 0)
      batch_str = (str(batch_size) if batch_size > 0 else "-") if "batch_size" in params else "-"
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


def _append_detailed_results(  # noqa: PLR0915
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
    runtime = run.get("runtime", "?")
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

    from backends import CLOUD_BACKENDS

    lines.append(f"### {i}. {backend} / {model}")
    lines.append("")
    lines.append(f"- **ID:** `{run_id}`")
    lines.append(f"- **Language:** {lang}")

    # Only show runtime for local backends
    if backend not in CLOUD_BACKENDS:
      runtime_str = f"{runtime} ({gpu_name})" if gpu_name else runtime
      lines.append(f"- **Runtime:** {runtime_str}")

    lines.append(f"- **Duration:** {duration:.2f}s")
    lines.append(f"- **Memory:** Δ {mem_delta} MB, peak {mem_peak} MB")

    params = get_backend_params(backend)
    if "beam_size" in params:
      lines.append(f"- **Beam size:** {beam_size}")
    if "temperature" in params:
      lines.append(f"- **Temperature:** {temperature:.2f}")
    if "smart_format" in params:
      smart_format_val = run.get("smart_format", True)
      lines.append(f"- **Smart format:** {smart_format_val}")
    if "diarize" in params:
      diarize_val = run.get("diarize", False)
      lines.append(f"- **Diarize:** {diarize_val}")
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
