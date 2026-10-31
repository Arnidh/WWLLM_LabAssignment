"""
eval/run_all.py
---------------
Master evaluation script.
Runs WER, notes, and MCQ evaluations, then produces a summary table
and 3 matplotlib charts.

Usage
-----
    python eval/run_all.py
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.wer_eval import run_wer_eval
from eval.notes_eval import run_notes_eval
from eval.mcq_eval import run_mcq_eval
from src.utils import get_logger

logger = get_logger("run_all")

RESULTS = Path("results")


def _ensure_dirs() -> None:
    for d in [RESULTS, RESULTS / "figures", RESULTS / "transcripts",
              RESULTS / "notes", RESULTS / "quiz"]:
        d.mkdir(parents=True, exist_ok=True)


def _make_charts() -> None:
    """Generate charts for evaluation result files that contain samples."""
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        logger.warning("matplotlib/pandas not installed – skipping charts")
        return

    # ── Chart 1: WER/CER by test type ────────────────────────────────────
    wer_csv = RESULTS / "wer_results.csv"
    if wer_csv.exists() and wer_csv.stat().st_size > 0:
        df = pd.read_csv(wer_csv)
        if not df.empty and "test_type" in df.columns:
            agg = df.groupby("test_type")[["wer", "cer"]].mean() * 100
            fig, ax = plt.subplots(figsize=(8, 4))
            agg.plot(kind="bar", ax=ax, colormap="Set2")
            ax.set_title("WER and CER by Test Type")
            ax.set_ylabel("Error Rate (%)")
            ax.set_xlabel("Test Type")
            ax.legend(["WER", "CER"])
            plt.tight_layout()
            plt.savefig(RESULTS / "figures" / "wer_by_type.png", dpi=150)
            plt.close()
            logger.info("Chart 1 saved: wer_by_type.png")

    # ── Chart 2: MCQ auto-validity per sample ────────────────────────────
    mcq_csv = RESULTS / "mcq_results.csv"
    if mcq_csv.exists() and mcq_csv.stat().st_size > 0:
        df = pd.read_csv(mcq_csv)
        if not df.empty and "id" in df.columns:
            val_rate = (
                df.groupby("id")[["schema_ok", "unique_options", "answer_valid"]]
                .mean()
                .reset_index()
            )
            fig, ax = plt.subplots(figsize=(10, 4))
            x = range(len(val_rate))
            ax.bar(x, val_rate["schema_ok"], label="Schema OK", alpha=0.7)
            ax.bar(x, val_rate["unique_options"], label="Unique Options", alpha=0.7)
            ax.bar(x, val_rate["answer_valid"], label="Answer Valid", alpha=0.7)
            ax.set_xticks(list(x))
            ax.set_xticklabels(val_rate["id"], rotation=45, ha="right", fontsize=8)
            ax.set_title("MCQ Validity Rates per Sample")
            ax.set_ylabel("Fraction of 5 MCQs passing")
            ax.legend()
            plt.tight_layout()
            plt.savefig(RESULTS / "figures" / "mcq_validity.png", dpi=150)
            plt.close()
            logger.info("Chart 2 saved: mcq_validity.png")

    # ── Chart 3: Notes keyword coverage distribution ──────────────────────
    notes_csv = RESULTS / "notes_results.csv"
    if notes_csv.exists() and notes_csv.stat().st_size > 0:
        df = pd.read_csv(notes_csv)
        if not df.empty and "keyword_coverage" in df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(df["keyword_coverage"] * 100, bins=10, color="#5C85D6", edgecolor="white")
            ax.set_title("Notes Keyword Coverage Distribution")
            ax.set_xlabel("Coverage (%)")
            ax.set_ylabel("# Samples")
            plt.tight_layout()
            plt.savefig(RESULTS / "figures" / "notes_coverage_hist.png", dpi=150)
            plt.close()
            logger.info("Chart 3 saved: notes_coverage_hist.png")


def _write_summary_table() -> None:
    """Print a Markdown summary table from the results CSVs."""
    import csv as _csv

    lines = ["# Evaluation Summary\n"]

    for csv_file, title in [
        (RESULTS / "wer_results.csv", "ASR (WER / CER)"),
        (RESULTS / "notes_results.csv", "Notes (Keyword Coverage)"),
        (RESULTS / "mcq_results.csv", "MCQ (Auto-Validity)"),
    ]:
        if not csv_file.exists():
            continue
        with open(csv_file, newline="", encoding="utf-8") as f:
            rows = list(_csv.DictReader(f))
        if not rows:
            continue
        lines.append(f"\n## {title}\n")
        header = list(rows[0].keys())
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
        for row in rows[:10]:  # first 10 rows
            lines.append("| " + " | ".join(str(row[k]) for k in header) + " |")

    summary_md = RESULTS / "summary.md"
    summary_md.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Summary table written to results/summary.md")


def main() -> None:
    _ensure_dirs()

    logger.info("=== Running WER evaluation ===")
    run_wer_eval(
        "data/test_cases.csv",
        str(RESULTS / "transcripts"),
        str(RESULTS / "wer_results.csv"),
    )

    if any((RESULTS / "notes").glob("*.md")):
        logger.info("=== Running Notes evaluation ===")
        run_notes_eval(
            "data/test_cases.csv",
            str(RESULTS / "notes"),
            str(RESULTS / "notes_results.csv"),
            str(RESULTS / "notes_ratings_template.csv"),
        )
    else:
        logger.info(
            "Skipping Notes evaluation: no generated notes or gold note labels."
        )

    if any((RESULTS / "quiz").glob("*.json")):
        logger.info("=== Running MCQ evaluation ===")
        run_mcq_eval(
            "data/test_cases.csv",
            str(RESULTS / "quiz"),
            str(RESULTS / "transcripts"),
            str(RESULTS / "mcq_results.csv"),
            str(RESULTS / "mcq_ratings_template.csv"),
        )
    else:
        logger.info("Skipping MCQ evaluation: no generated quiz files.")

    _make_charts()
    _write_summary_table()

    logger.info("All evaluations done. Check results/")


if __name__ == "__main__":
    main()
