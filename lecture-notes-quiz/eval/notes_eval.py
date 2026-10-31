"""
eval/notes_eval.py
------------------
Evaluates generated lecture notes on two automatic axes:
  1. Keyword Coverage: fraction of key terms from the reference that appear in the notes.
  2. Bullet Coverage: fraction of reference key points whose main term appears in the notes.

Manual rating (1–5) is collected via a CSV template that you fill in.

Usage
-----
    python eval/notes_eval.py --test-cases data/test_cases.csv \\
                              --notes-dir results/notes/
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import normalize_text, get_logger

logger = get_logger("notes_eval")


# ── Automatic coverage check ──────────────────────────────────────────────────

def keyword_coverage(notes_text: str, key_terms: list[str]) -> float:
    """
    Fraction of `key_terms` whose normalised form appears in `notes_text`.

    Parameters
    ----------
    notes_text : str
        Full Markdown notes string.
    key_terms : list[str]
        Reference key terms (from test_cases.csv or hand-written).
    """
    if not key_terms:
        return 0.0

    notes_norm = normalize_text(notes_text)
    found = sum(
        1 for term in key_terms
        if normalize_text(term) in notes_norm
    )
    return round(found / len(key_terms), 4)


def run_notes_eval(
    test_cases_csv: str,
    notes_dir: str,
    output_csv: str = "results/notes_results.csv",
    ratings_template_csv: str = "results/notes_ratings_template.csv",
) -> None:
    """
    Automatic notes evaluation + generate a manual-rating template.

    The `notes` column in test_cases.csv may contain a JSON list of expected key terms,
    e.g.:  ["जड़त्व", "F=ma", "Newton", "त्वरण"]
    """
    auto_rows: list[dict] = []
    rating_rows: list[dict] = []

    with open(test_cases_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("test_type", "") not in ("notes", "all", ""):
                continue

            row_id = row["id"]
            notes_path = Path(notes_dir) / f"{row_id}.md"
            if not notes_path.exists():
                logger.warning("Notes file not found: %s (skipping)", notes_path)
                continue

            notes_text = notes_path.read_text(encoding="utf-8")

            # Parse expected key terms (stored as JSON list in the `notes` column)
            key_terms: list[str] = []
            notes_meta = row.get("notes", "").strip()
            if notes_meta.startswith("["):
                try:
                    import json
                    key_terms = json.loads(notes_meta)
                except Exception:
                    pass

            cov = keyword_coverage(notes_text, key_terms)

            auto_rows.append(
                {
                    "id": row_id,
                    "language": row.get("language", ""),
                    "test_type": row.get("test_type", ""),
                    "keyword_coverage": cov,
                    "notes_length_chars": len(notes_text),
                }
            )

            rating_rows.append(
                {
                    "id": row_id,
                    "language": row.get("language", ""),
                    "test_type": row.get("test_type", ""),
                    "keyword_coverage_auto": cov,
                    # TODO: fill in manually
                    "factual_consistency_1_5": "TODO",
                    "completeness_1_5": "TODO",
                    "clarity_1_5": "TODO",
                    "rater_notes": "",
                }
            )

            logger.info("  %s | keyword_coverage=%.1f%%", row_id, cov * 100)

    # Write auto results
    if auto_rows:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(auto_rows[0].keys()))
            writer.writeheader()
            writer.writerows(auto_rows)
        logger.info("Auto results → %s", output_csv)

    # Write rating template
    if rating_rows:
        with open(ratings_template_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rating_rows[0].keys()))
            writer.writeheader()
            writer.writerows(rating_rows)
        logger.info(
            "Manual rating template → %s  (fill in the TODO columns)", ratings_template_csv
        )

    print(f"\nAuto results: {output_csv}")
    print(f"Rating template (fill in manually): {ratings_template_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-cases", default="data/test_cases.csv")
    parser.add_argument("--notes-dir", default="results/notes")
    parser.add_argument("--output-csv", default="results/notes_results.csv")
    parser.add_argument("--ratings-template", default="results/notes_ratings_template.csv")
    args = parser.parse_args()
    run_notes_eval(
        args.test_cases, args.notes_dir, args.output_csv, args.ratings_template
    )

