from __future__ import annotations

from pathlib import Path

from src.pdf_export import save_notes_pdf, save_quiz_pdf


def _assert_valid_pdf(path: str) -> None:
    pdf_path = Path(path)
    try:
        content = pdf_path.read_bytes()
        assert content.startswith(b"%PDF-")
        assert b"%%EOF" in content[-2048:]
    finally:
        pdf_path.unlink(missing_ok=True)


def test_notes_export_creates_pdf():
    path = save_notes_pdf(
        "## Newton's Laws\n\n"
        "- **First law**: Objects remain at rest unless acted on by force.\n\n"
        "## Summary\nThe lecture explains Newton's laws."
    )

    _assert_valid_pdf(path)


def test_quiz_export_creates_pdf_with_answers():
    path = save_quiz_pdf(
        [
            {
                "question": "What is force?",
                "options": {
                    "A": "Mass times acceleration",
                    "B": "Velocity times distance",
                },
                "answer": "A",
                "explanation": "The second law defines force.",
                "evidence": "Force equals mass times acceleration.",
            }
        ]
    )

    _assert_valid_pdf(path)
