from __future__ import annotations

import csv
import json

import pytest

from eval.wer_eval import run_wer_eval
from src.utils import normalize_text


def test_normalization_preserves_combining_marks_in_indic_scripts():
    assert normalize_text("தமிழ், ಕನ್ನಡ।") == "தமிழ் ಕನ್ನಡ"


def _write_manifest(path, rows):
    with path.open("w", newline="", encoding="utf-8") as manifest:
        writer = csv.DictWriter(
            manifest, fieldnames=["id", "language", "reference_text"]
        )
        writer.writeheader()
        writer.writerows(rows)


def test_wer_eval_rejects_partial_hypothesis_set(tmp_path):
    manifest_path = tmp_path / "test_cases.csv"
    _write_manifest(
        manifest_path,
        [
            {"id": "sample-1", "language": "hi", "reference_text": "hello"},
            {"id": "sample-2", "language": "kn", "reference_text": "world"},
        ],
    )
    transcripts_dir = tmp_path / "transcripts"
    transcripts_dir.mkdir()
    (transcripts_dir / "sample-1.txt").write_text("hello", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="sample-2"):
        run_wer_eval(
            str(manifest_path),
            str(transcripts_dir),
            str(tmp_path / "wer_results.csv"),
        )
    assert not (tmp_path / "wer_results.csv").exists()


def test_wer_eval_writes_corpus_and_per_language_rates(tmp_path):
    manifest_path = tmp_path / "test_cases.csv"
    _write_manifest(
        manifest_path,
        [
            {
                "id": "sample-1",
                "language": "hi",
                "reference_text": "नमस्ते दुनिया",
            },
            {"id": "sample-2", "language": "kn", "reference_text": "hello world"},
        ],
    )
    transcripts_dir = tmp_path / "transcripts"
    transcripts_dir.mkdir()
    (transcripts_dir / "sample-1.txt").write_text("नमस्ते दुनिया", encoding="utf-8")
    (transcripts_dir / "sample-2.txt").write_text("hello", encoding="utf-8")
    output_path = tmp_path / "wer_results.csv"

    run_wer_eval(str(manifest_path), str(transcripts_dir), str(output_path))

    summary = json.loads((tmp_path / "wer_summary.json").read_text(encoding="utf-8"))
    assert summary["sample_count"] == 2
    assert summary["by_language"]["hi"] == {
        "sample_count": 1,
        "wer": 0.0,
        "cer": 0.0,
    }
    assert summary["by_language"]["kn"]["wer"] == 0.5
    assert output_path.is_file()
