#!/usr/bin/env python3
"""Migrate existing JSON files to regenerate run IDs."""

import json
import sys
from pathlib import Path

from transcribe import generate_report, generate_run_id, load_reference


def migrate_json(json_path: Path) -> int:
  """Migrate a single JSON file, regenerating all run IDs."""
  data = json.loads(json_path.read_text(encoding="utf-8"))
  migrated = 0

  for run in data.get("runs", []):
    old_id = run.get("id", "")

    new_id = generate_run_id(
      backend=run.get("backend", ""),
      model=run.get("model", ""),
      language=run.get("language"),
      runtime=run.get("runtime", run.get("device", "cpu")),
      beam_size=run.get("beam_size", 5),
      temperature=run.get("temperature", 0.0),
      compute_type=run.get("compute_type", "auto"),
      condition_on_prev=run.get("condition_on_prev", True),
      batch_size=run.get("batch_size", 0),
    )
    if old_id != new_id:
      print(f"  {old_id} -> {new_id}")
      run["id"] = new_id
      migrated += 1

  if migrated > 0:
    # Save JSON
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    # Regenerate report
    reference = load_reference(data)
    report = generate_report(data, reference)
    md_path = json_path.with_suffix(".md")
    md_path.write_text(report, encoding="utf-8")

  return migrated


def main() -> None:
  """Migrate all JSON files in calls/ directory."""
  calls_dir = Path(__file__).parent.parent / "calls"
  if not calls_dir.exists():
    print("No calls/ directory found")
    sys.exit(1)

  json_files = list(calls_dir.glob("*.json"))
  if not json_files:
    print("No JSON files found in calls/")
    sys.exit(0)

  total_migrated = 0
  for json_path in sorted(json_files):
    print(f"\n{json_path.name}:")
    migrated = migrate_json(json_path)
    if migrated == 0:
      print("  (no changes)")
    total_migrated += migrated

  print(f"\nTotal: {total_migrated} runs migrated")


if __name__ == "__main__":
  main()
