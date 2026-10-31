"""Run real ASR over the prepared evaluation manifest and save transcripts."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import asr
from src.utils import is_mock_mode, load_config

LOGGER = logging.getLogger("run_asr_inference")


def _project_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def run_asr_inference(
    test_cases_csv: str,
    transcripts_dir: str,
    metadata_path: str,
) -> int:
    if is_mock_mode():
        raise RuntimeError(
            "Refusing to record MOCK output as model evidence. Set MOCK_MODELS=0."
        )

    manifest_path = _project_path(test_cases_csv)
    output_dir = _project_path(transcripts_dir)
    metadata_file = _project_path(metadata_path)
    with manifest_path.open(newline="", encoding="utf-8") as manifest:
        rows = list(csv.DictReader(manifest))
    if not rows:
        raise ValueError(f"No evaluation rows found in {manifest_path}.")

    missing_audio = [
        row["id"]
        for row in rows
        if not _project_path(row["audio_file"]).is_file()
    ]
    missing_references = [row["id"] for row in rows if not row["reference_text"].strip()]
    if missing_audio or missing_references:
        details = []
        if missing_audio:
            details.append(f"missing audio for {', '.join(missing_audio[:8])}")
        if missing_references:
            details.append(
                f"missing reference text for {', '.join(missing_references[:8])}"
            )
        raise FileNotFoundError(
            f"Evaluation manifest is incomplete: {'; '.join(details)}. "
            "Prepare the dataset before running inference."
        )

    config = load_config()
    memory_config = config.setdefault("memory", {})
    previous_free_setting = memory_config.get("free_gpu_between_stages", True)
    memory_config["free_gpu_between_stages"] = False
    output_dir.mkdir(parents=True, exist_ok=True)
    latencies: list[float] = []
    backend = ""
    try:
        for index, row in enumerate(rows, start=1):
            audio_path = _project_path(row["audio_file"])
            LOGGER.info(
                "[%d/%d] %s (%s)", index, len(rows), row["id"], row["language"]
            )
            try:
                result = asr.transcribe(audio_path, row["language"])
            except Exception as exc:
                raise RuntimeError(
                    f"ASR failed for {row['id']} using {audio_path}."
                ) from exc
            (output_dir / f"{row['id']}.txt").write_text(
                result["full_text"], encoding="utf-8"
            )
            latencies.append(float(result["latency_s"]))
        backend = asr.get_active_backend()
        model_ids = {
            "indicconformer": config["models"]["asr_primary"],
            "shrutam2": config["models"]["asr_fallback"],
        }
        if backend not in model_ids:
            raise RuntimeError(f"Unknown active ASR backend after inference: {backend!r}")
    finally:
        memory_config["free_gpu_between_stages"] = previous_free_setting
        asr.free_gpu_memory()

    languages = Counter(row["language"] for row in rows)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inference_mode": "real",
        "asr_backend": backend,
        "asr_model_used": model_ids[backend],
        "sample_count": len(rows),
        "samples_by_language": dict(sorted(languages.items())),
        "mean_asr_latency_seconds": round(sum(latencies) / len(latencies), 3),
        "dataset_id": rows[0].get("dataset_id", ""),
        "dataset_revision": rows[0].get("dataset_revision", ""),
        "dataset_license": rows[0].get("dataset_license", ""),
        "dataset_split": rows[0].get("dataset_split", ""),
    }
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    metadata_file.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    LOGGER.info(
        "Saved %d real transcripts with backend %s.",
        len(rows),
        metadata["asr_backend"],
    )
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe all audio in data/test_cases.csv with real ASR."
    )
    parser.add_argument("--test-cases", default="data/test_cases.csv")
    parser.add_argument("--transcripts-dir", default="results/transcripts")
    parser.add_argument(
        "--metadata", default="results/asr_inference_metadata.json"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        count = run_asr_inference(
            args.test_cases, args.transcripts_dir, args.metadata
        )
    except Exception:
        LOGGER.exception("Real ASR evaluation failed.")
        raise
    print(f"Transcribed {count} samples; see results/transcripts and the metadata JSON.")


if __name__ == "__main__":
    main()
