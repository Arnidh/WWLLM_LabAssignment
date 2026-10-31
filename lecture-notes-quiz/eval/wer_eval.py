"""
eval/wer_eval.py
----------------
Computes Word Error Rate (WER) and Character Error Rate (CER) for
ASR outputs using the `jiwer` library.

Text normalisation (documented here and in README):
  1. Unicode NFC normalisation
  2. Strip punctuation (keep Devanagari / Latin letters, digits, whitespace)
  3. Collapse whitespace
  4. Lowercase

Usage
-----
    python eval/wer_eval.py --test-cases data/test_cases.csv \\
                            --transcripts-dir results/transcripts/
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import normalize_text, get_logger

logger = get_logger("wer_eval")


def compute_wer_cer(
    reference: str,
    hypothesis: str,
) -> dict[str, float]:
    """
    Compute WER and CER after normalisation.

    Parameters
    ----------
    reference : str
        Human-written reference transcript.
    hypothesis : str
        ASR-produced transcript.

    Returns
    -------
    dict with keys "wer", "cer", "ref_words", "hyp_words".
    """
    try:
        import jiwer
    except ImportError as exc:
        raise ImportError("Install jiwer: pip install jiwer") from exc

    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)

    wer = jiwer.wer(ref_norm, hyp_norm)
    cer = jiwer.cer(ref_norm, hyp_norm)

    return {
        "wer": round(wer, 4),
        "cer": round(cer, 4),
        "ref_words": len(ref_norm.split()),
        "hyp_words": len(hyp_norm.split()),
        "ref_norm": ref_norm[:120],
        "hyp_norm": hyp_norm[:120],
    }


def run_wer_eval(
    test_cases_csv: str,
    transcripts_dir: str,
    output_csv: str = "results/wer_results.csv",
) -> None:
    """
    Evaluate ASR quality on all test cases that have a reference transcript.

    Expects `test_cases.csv` to have columns:
        id, audio_file, language, test_type, reference_text, notes

    Transcripts are read from `transcripts_dir/<id>.txt`.
    """
    rows = []
    with open(test_cases_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("reference_text", "").strip():
                continue  # skip rows without a reference transcript
            row_id = row["id"]
            hyp_path = Path(transcripts_dir) / f"{row_id}.txt"
            if not hyp_path.exists():
                logger.warning("Hypothesis file not found: %s (skipping)", hyp_path)
                continue

            hyp = hyp_path.read_text(encoding="utf-8")
            ref = row["reference_text"]
            metrics = compute_wer_cer(ref, hyp)

            rows.append(
                {
                    "id": row_id,
                    "language": row.get("language", ""),
                    "test_type": row.get("test_type", ""),
                    **metrics,
                }
            )
            logger.info(
                "  %s | WER=%.1f%% | CER=%.1f%%",
                row_id,
                metrics["wer"] * 100,
                metrics["cer"] * 100,
            )

    if not rows:
        logger.warning("No rows evaluated. Check that reference_text is filled in.")
        return

    # Write CSV
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    avg_wer = sum(r["wer"] for r in rows) / len(rows)
    avg_cer = sum(r["cer"] for r in rows) / len(rows)
    logger.info(
        "WER Evaluation complete: %d samples | avg WER=%.1f%% | avg CER=%.1f%%",
        len(rows),
        avg_wer * 100,
        avg_cer * 100,
    )
    print(f"\nResults written to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WER/CER evaluation for ASR outputs")
    parser.add_argument("--test-cases", default="data/test_cases.csv")
    parser.add_argument("--transcripts-dir", default="results/transcripts")
    parser.add_argument("--output-csv", default="results/wer_results.csv")
    args = parser.parse_args()
    run_wer_eval(args.test_cases, args.transcripts_dir, args.output_csv)

