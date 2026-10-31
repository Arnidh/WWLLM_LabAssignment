"""
eval/mcq_eval.py
----------------
Evaluates generated MCQs on:
  - Automatic schema validity (4 unique options, answer in A-D, evidence in transcript)
  - Manual rating template (clarity, correctness, distractor quality: 1–5 each)

Usage
-----
    python eval/mcq_eval.py --test-cases data/test_cases.csv \\
                            --quiz-dir results/quiz/
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import get_logger, normalize_text

logger = get_logger("mcq_eval")


# ── Automatic validators ──────────────────────────────────────────────────────

def validate_mcq(mcq: dict, transcript: str = "") -> dict[str, bool | str]:
    """
    Automatically validate one MCQ dict.

    Returns a dict with keys:
      schema_ok, unique_options, answer_valid, evidence_found, issues
    """
    issues: list[str] = []

    # Schema completeness
    required = {"question", "options", "answer", "explanation", "evidence"}
    missing = required - set(mcq.keys())
    if missing:
        issues.append(f"Missing keys: {missing}")

    options: dict = mcq.get("options", {})

    # 4 options exist
    if set(options.keys()) != {"A", "B", "C", "D"}:
        issues.append(f"Options keys are {set(options.keys())}, expected A B C D")

    # All options distinct
    vals = list(options.values())
    if len(set(vals)) < len(vals):
        issues.append("Duplicate option values")

    # Answer valid
    answer = str(mcq.get("answer", "")).strip().upper()
    if answer not in ("A", "B", "C", "D"):
        issues.append(f"Invalid answer '{answer}'")

    answer_valid = answer in options

    # Evidence found in transcript (soft)
    evidence: str = mcq.get("evidence", "")
    if transcript:
        evidence_found = normalize_text(evidence[:60]) in normalize_text(transcript)
    else:
        evidence_found = bool(evidence.strip())

    return {
        "schema_ok": len(missing) == 0,
        "unique_options": len(set(vals)) == 4,
        "answer_valid": answer_valid,
        "evidence_found": evidence_found,
        "issues": "; ".join(issues) if issues else "none",
    }


def run_mcq_eval(
    test_cases_csv: str,
    quiz_dir: str,
    transcripts_dir: str = "results/transcripts",
    output_csv: str = "results/mcq_results.csv",
    ratings_template_csv: str = "results/mcq_ratings_template.csv",
) -> None:
    """Validate all quiz JSON files and generate a manual rating template."""
    auto_rows: list[dict] = []
    rating_rows: list[dict] = []

    with open(test_cases_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_id = row["id"]
            quiz_path = Path(quiz_dir) / f"{row_id}.json"
            if not quiz_path.exists():
                logger.warning("Quiz file not found: %s", quiz_path)
                continue

            try:
                mcqs: list[dict] = json.loads(quiz_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                logger.error("Invalid JSON in %s: %s", quiz_path, exc)
                continue

            # Load transcript for evidence checking
            trans_path = Path(transcripts_dir) / f"{row_id}.txt"
            transcript = trans_path.read_text(encoding="utf-8") if trans_path.exists() else ""

            # Validate
            n_valid = 0
            for i, mcq in enumerate(mcqs):
                v = validate_mcq(mcq, transcript)
                all_ok = v["schema_ok"] and v["unique_options"] and v["answer_valid"]
                n_valid += int(all_ok)

                auto_rows.append(
                    {
                        "id": row_id,
                        "q_index": i + 1,
                        "language": row.get("language", ""),
                        "schema_ok": v["schema_ok"],
                        "unique_options": v["unique_options"],
                        "answer_valid": v["answer_valid"],
                        "evidence_found": v["evidence_found"],
                        "issues": v["issues"],
                    }
                )

                rating_rows.append(
                    {
                        "id": row_id,
                        "q_index": i + 1,
                        "question_preview": mcq.get("question", "")[:80],
                        "auto_valid": all_ok,
                        "clarity_1_5": "",
                        "correctness_1_5": "",
                        "distractor_quality_1_5": "",
                        "rater_notes": "",
                    }
                )

            logger.info("  %s | %d/5 MCQs auto-valid", row_id, n_valid)

    if auto_rows:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(auto_rows[0].keys()))
            writer.writeheader()
            writer.writerows(auto_rows)

    if rating_rows:
        with open(ratings_template_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rating_rows[0].keys()))
            writer.writeheader()
            writer.writerows(rating_rows)

    print(f"\nAuto results: {output_csv}")
    print(f"Rating template: {ratings_template_csv}  (rating fields are intentionally blank)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-cases", default="data/test_cases.csv")
    parser.add_argument("--quiz-dir", default="results/quiz")
    parser.add_argument("--transcripts-dir", default="results/transcripts")
    parser.add_argument("--output-csv", default="results/mcq_results.csv")
    parser.add_argument("--ratings-template", default="results/mcq_ratings_template.csv")
    args = parser.parse_args()
    run_mcq_eval(
        args.test_cases,
        args.quiz_dir,
        args.transcripts_dir,
        args.output_csv,
        args.ratings_template,
    )
