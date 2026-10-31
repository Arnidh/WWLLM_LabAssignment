"""PDF exports for generated notes and quizzes."""
from __future__ import annotations

import hashlib
import html
import io
import re
import tempfile
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

_LATIN_FONT = "Helvetica"


def _font_for_text(text: str) -> str:
    candidates = [
        Path("C:/Windows/Fonts/Nirmala.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf"),
        Path("/usr/share/fonts/truetype/noto/NotoSansKannada-Regular.ttf"),
        Path("/usr/share/fonts/truetype/noto/NotoSansTamil-Regular.ttf"),
        Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansDevanagari-Regular.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansKannada-Regular.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansTamil-Regular.ttf"),
    ]
    for font_path in candidates:
        if font_path.is_file():
            font_name = "LectureExport_" + hashlib.sha1(
                str(font_path).encode("utf-8")
            ).hexdigest()[:10]
            if font_name in pdfmetrics.getRegisteredFontNames():
                font = pdfmetrics.getFont(font_name)
            else:
                font = TTFont(
                    font_name, str(font_path), subfontIndex=0, shapable=True
                )
                pdfmetrics.registerFont(font)
            if all(
                ord(character) in font.face.charWidths
                for character in text
                if not character.isspace()
            ):
                return font_name

    if all(ord(character) <= 255 for character in text):
        return _LATIN_FONT
    raise RuntimeError(
        "PDF export needs a Unicode font that covers the generated language. "
        "Install Nirmala UI on Windows or a matching Noto Sans font on Linux."
    )


def _styles(font_name: str) -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ExportTitle",
            parent=base["Title"],
            fontName=font_name,
            fontSize=20,
            leading=26,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#172554"),
            spaceAfter=18,
        ),
        "heading": ParagraphStyle(
            "ExportHeading",
            parent=base["Heading2"],
            fontName=font_name,
            fontSize=14,
            leading=19,
            textColor=colors.HexColor("#1d4ed8"),
            spaceBefore=10,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "ExportBody",
            parent=base["BodyText"],
            fontName=font_name,
            fontSize=10.5,
            leading=16,
            textColor=colors.HexColor("#1f2937"),
            spaceAfter=6,
            wordWrap="CJK",
        ),
        "option": ParagraphStyle(
            "ExportOption",
            parent=base["BodyText"],
            fontName=font_name,
            fontSize=10,
            leading=15,
            leftIndent=12,
            textColor=colors.HexColor("#1f2937"),
            spaceAfter=3,
            wordWrap="CJK",
        ),
        "answer": ParagraphStyle(
            "ExportAnswer",
            parent=base["BodyText"],
            fontName=font_name,
            fontSize=10,
            leading=15,
            textColor=colors.HexColor("#166534"),
            spaceBefore=5,
            spaceAfter=5,
            wordWrap="CJK",
        ),
        "evidence": ParagraphStyle(
            "ExportEvidence",
            parent=base["BodyText"],
            fontName=font_name,
            fontSize=9,
            leading=13,
            leftIndent=12,
            textColor=colors.HexColor("#4b5563"),
            spaceAfter=12,
            wordWrap="CJK",
        ),
    }


def _markup(text: str) -> str:
    escaped = html.escape(text, quote=False)
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)


def _build_pdf(title: str, story: list[Any]) -> bytes:
    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=45,
        title=title,
        author="Lecture Notes & Quiz Generator",
    )
    document.build(story)
    return buffer.getvalue()


def _save_pdf(prefix: str, pdf_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(
        delete=False, prefix=prefix, suffix=".pdf"
    ) as pdf_file:
        pdf_file.write(pdf_bytes)
        return pdf_file.name


def save_notes_pdf(notes_markdown: str) -> str:
    """Render generated Markdown notes to a downloadable PDF."""
    styles = _styles(_font_for_text(notes_markdown))
    story: list[Any] = [
        Paragraph("Lecture Notes", styles["title"]),
        Spacer(1, 4),
    ]

    for line in notes_markdown.splitlines():
        stripped = line.strip()
        if not stripped or stripped == "---":
            continue
        heading = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading:
            story.append(Paragraph(_markup(heading.group(2)), styles["heading"]))
        elif stripped.startswith("- "):
            story.append(
                ListFlowable(
                    [
                        ListItem(
                            Paragraph(_markup(stripped[2:]), styles["body"]),
                            leftIndent=12,
                        )
                    ],
                    bulletType="bullet",
                    start="circle",
                    leftIndent=18,
                    bulletFontName=styles["body"].fontName,
                )
            )
            story.append(Spacer(1, 3))
        else:
            story.append(Paragraph(_markup(stripped), styles["body"]))

    return _save_pdf("lecture_notes_", _build_pdf("Lecture Notes", story))


def save_quiz_pdf(mcqs: list[dict[str, Any]]) -> str:
    """Render generated MCQs, answers, and evidence to a downloadable PDF."""
    all_text = "\n".join(
        str(value)
        for question in mcqs
        for value in [
            question.get("question", ""),
            question.get("answer", ""),
            question.get("explanation", ""),
            question.get("evidence", ""),
            *question.get("options", {}).values(),
        ]
    )
    styles = _styles(_font_for_text(all_text))
    story: list[Any] = [Paragraph("Lecture Quiz", styles["title"])]

    for index, question in enumerate(mcqs, start=1):
        story.append(
            Paragraph(
                f"Q{index}. {_markup(str(question.get('question', '')))}",
                styles["heading"],
            )
        )
        for letter, option in question.get("options", {}).items():
            story.append(
                Paragraph(
                    f"<b>{_markup(str(letter))}.</b> {_markup(str(option))}",
                    styles["option"],
                )
            )
        answer = _markup(str(question.get("answer", "")))
        story.append(
            Paragraph(f"<b>Correct answer: {answer}</b>", styles["answer"])
        )
        explanation = question.get("explanation")
        if explanation:
            story.append(Paragraph(_markup(str(explanation)), styles["body"]))
        evidence = question.get("evidence")
        if evidence:
            story.append(
                Paragraph(
                    f"<b>Evidence:</b> {_markup(str(evidence))}",
                    styles["evidence"],
                )
            )

    return _save_pdf("lecture_quiz_", _build_pdf("Lecture Quiz", story))
