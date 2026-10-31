"""
src/pipeline.py
---------------
Orchestrates all stages: audio → ASR → notes → quiz → [translation].
Returns a single result dict consumed by the Gradio frontend.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .asr import TranscriptChunk, TranscriptResult, transcribe
from .notes import free_llm_memory, make_notes
from .quiz import QuizResult, make_mcqs
from .translate import TranslationResult, translate
from .utils import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    """All outputs from one end-to-end pipeline run."""

    # Stage outputs
    transcript: TranscriptResult | None = None
    notes: dict[str, Any] | None = None
    quiz: QuizResult | None = None
    translation: TranslationResult | None = None

    # Timing
    stage_latencies: dict[str, float] = field(default_factory=dict)
    total_latency_s: float = 0.0

    # Status
    success: bool = True
    error_stage: str | None = None   # "asr" | "notes" | "quiz" | "translation"
    error_message: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialise to a plain dict for the Gradio UI."""
        return {
            "transcript": {
                "full_text": self.transcript.full_text if self.transcript else "",
                "chunks": [
                    {
                        "index": c.index,
                        "start_s": c.start_s,
                        "end_s": c.end_s,
                        "text": c.text,
                    }
                    for c in (self.transcript.chunks if self.transcript else [])
                ],
                "latency_s": self.transcript.latency_s if self.transcript else 0,
            },
            "notes": {
                "markdown": self.notes.get("markdown", "") if self.notes else "",
                "latency_s": self.notes.get("latency_s", 0) if self.notes else 0,
                "used_map_reduce": (
                    self.notes.get("used_map_reduce", False) if self.notes else False
                ),
                "structured": (
                    {
                        key: value
                        for key, value in self.notes.items()
                        if key not in {"markdown", "latency_s", "used_map_reduce"}
                    }
                    if self.notes
                    else {}
                ),
            },
            "quiz": {
                "mcqs": self.quiz.mcqs if self.quiz else [],
                "attempts": self.quiz.attempts if self.quiz else 0,
                "latency_s": self.quiz.latency_s if self.quiz else 0,
                "warnings": self.quiz.validation_warnings if self.quiz else [],
            },
            "translation": {
                "text": self.translation.translated_text if self.translation else "",
                "src": self.translation.src_lang if self.translation else "",
                "tgt": self.translation.tgt_lang if self.translation else "",
                "latency_s": self.translation.latency_s if self.translation else 0,
            },
            "timings": self.stage_latencies,
            "total_latency_s": self.total_latency_s,
            "success": self.success,
            "error_stage": self.error_stage,
            "error_message": self.error_message,
            "warnings": self.warnings,
        }


def run_pipeline(
    audio_path: str | Path | None,
    lang: str = "hi",
    translate_to: str | None = None,
    transcript_override: str | None = None,
    progress_callback: Callable[[str, float], None] | None = None,
) -> PipelineResult:
    """
    Run the full lecture → notes → quiz pipeline.

    Parameters
    ----------
    audio_path : str, Path, or None
        Path to the audio file. If None, `transcript_override` must be given.
    lang : str
        BCP-47 source language code.
    translate_to : str or None
        If provided, translate the transcript and generate notes and quiz in
        this BCP-47 target language.
    transcript_override : str or None
        Skip ASR and use this string as the transcript directly (useful for testing).

    Returns
    -------
    PipelineResult
        Structured result with all stage outputs and timing.
    """
    result = PipelineResult()
    wall_start = time.perf_counter()

    def report_progress(stage: str, fraction: float) -> None:
        if progress_callback is not None:
            progress_callback(stage, fraction)

    def fail(stage: str, message: str) -> PipelineResult:
        result.success = False
        result.error_stage = stage
        result.error_message = message
        result.total_latency_s = time.perf_counter() - wall_start
        return result

    # ── Stage 1: ASR ──────────────────────────────────────────────────────
    report_progress("ASR", 0.0)
    if transcript_override:
        logger.info("Using provided transcript override (skipping ASR)")
        result.transcript = TranscriptResult(
            full_text=transcript_override,
            chunks=[TranscriptChunk(0, 0.0, 0.0, transcript_override)],
            latency_s=0.0,
        )
        result.stage_latencies["asr"] = 0.0
    else:
        if audio_path is None:
            return fail("asr", "No audio file and no transcript override provided.")

        stage_start = time.perf_counter()
        try:
            logger.info("=== Stage 1: ASR ===")
            asr_output = transcribe(str(audio_path), lang=lang)
            result.transcript = TranscriptResult(
                full_text=asr_output["full_text"],
                chunks=[
                    TranscriptChunk(
                        index=chunk["index"],
                        start_s=chunk["start_s"],
                        end_s=chunk["end_s"],
                        text=chunk["text"],
                    )
                    for chunk in asr_output["chunks"]
                ],
                latency_s=asr_output["latency_s"],
            )
            result.stage_latencies["asr"] = result.transcript.latency_s
        except Exception as exc:
            logger.error("ASR failed: %s", exc)
            result.stage_latencies["asr"] = time.perf_counter() - stage_start
            return fail("asr", str(exc))
    report_progress("ASR complete", 0.35)

    transcript_text = result.transcript.full_text
    content_text = transcript_text
    content_lang = lang

    # ── Stage 2: Transcript translation (optional) ────────────────────────
    if translate_to:
        report_progress("Transcript translation", 0.35)
        stage_start = time.perf_counter()
        try:
            logger.info("=== Stage 2: Translation (%s → %s) ===", lang, translate_to)
            result.translation = translate(
                transcript_text, src_lang=lang, tgt_lang=translate_to
            )
            result.stage_latencies["translation"] = result.translation.latency_s
            content_text = result.translation.translated_text
            content_lang = translate_to
        except Exception as exc:
            logger.error("Transcript translation failed: %s", exc)
            result.stage_latencies["translation"] = time.perf_counter() - stage_start
            return fail("translation", str(exc))
    report_progress("Translation complete", 0.5)

    try:
        # ── Stage 3: Notes ────────────────────────────────────────────────
        report_progress("Notes generation", 0.5)
        stage_start = time.perf_counter()
        try:
            logger.info("=== Stage 3: Notes generation ===")
            result.notes = make_notes(content_text, lang=content_lang)
            result.stage_latencies["notes"] = result.notes["latency_s"]
        except Exception as exc:
            logger.error("Notes generation failed: %s", exc)
            result.stage_latencies["notes"] = time.perf_counter() - stage_start
            return fail("notes", str(exc))

        # ── Stage 4: MCQ generation ───────────────────────────────────────
        report_progress("MCQ generation", 0.75)
        stage_start = time.perf_counter()
        try:
            logger.info("=== Stage 4: MCQ generation ===")
            result.quiz = make_mcqs(
                result.notes,
                lang=content_lang,
                transcript=content_text,
            )
            result.stage_latencies["quiz"] = result.quiz.latency_s
        except Exception as exc:
            logger.error("MCQ generation failed: %s", exc)
            result.stage_latencies["quiz"] = time.perf_counter() - stage_start
            return fail("quiz", str(exc))
    finally:
        free_llm_memory()

    report_progress("Quiz complete", 0.95)

    result.total_latency_s = time.perf_counter() - wall_start
    report_progress("Pipeline complete", 1.0)
    logger.info("=== Pipeline complete in %.2f s ===", result.total_latency_s)
    return result
