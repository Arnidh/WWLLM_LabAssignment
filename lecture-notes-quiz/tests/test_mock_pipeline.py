"""
tests/test_mock_pipeline.py
---------------------------
pytest suite that validates the full pipeline in MOCK mode.
All tests pass without a GPU or HF token.

Run with:
    MOCK_MODELS=1 pytest tests/ -v
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Force MOCK mode for all tests in this file
os.environ["MOCK_MODELS"] = "1"

from src.pipeline import run_pipeline, PipelineResult
from src import pipeline as pipeline_module
from src.asr import transcribe, MOCK_TRANSCRIPT
from src.notes import make_notes, MOCK_NOTES
from src.quiz import make_mcqs, MCQItem, MCQList, MOCK_MCQS
from src.translate import translate, MOCK_TRANSLATION
from src.utils import (
    normalize_text,
    safe_parse_json,
    extract_json_block,
    repair_json_truncated,
)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalize:
    def test_lowercase(self):
        assert normalize_text("Hello World") == "hello world"

    def test_strips_punctuation(self):
        assert "," not in normalize_text("hello, world!")

    def test_collapses_whitespace(self):
        result = normalize_text("one   two\tthree")
        assert result == "one two three"

    def test_devanagari_preserved(self):
        text = "नमस्ते दुनिया"
        result = normalize_text(text)
        assert "नमस्ते" in result
        assert "दुनिया" in result

    def test_unicode_nfc(self):
        # NFC normalisation: precomposed vs decomposed
        composed = "\u00e9"        # é (precomposed)
        decomposed = "e\u0301"     # e + combining accent
        assert normalize_text(composed) == normalize_text(decomposed)


class TestJsonUtils:
    def test_extract_fenced_json(self):
        text = "Here is the result:\n```json\n{\"a\": 1}\n```\nDone."
        block = extract_json_block(text)
        assert block == '{"a": 1}'

    def test_safe_parse_json_clean(self):
        parsed = safe_parse_json('{"x": 42}')
        assert parsed["x"] == 42

    def test_safe_parse_json_fenced(self):
        text = "```json\n[1, 2, 3]\n```"
        parsed = safe_parse_json(text)
        assert parsed == [1, 2, 3]

    def test_repair_truncated_json(self):
        truncated = '[{"question": "What is force?", "answer": "B"'
        result = repair_json_truncated(truncated)
        assert isinstance(result, list)
        assert result[0]["answer"] == "B"


# ─────────────────────────────────────────────────────────────────────────────
# ASR
# ─────────────────────────────────────────────────────────────────────────────

class TestASRMock:
    def test_returns_transcript_dict(self):
        result = transcribe("fake_audio.wav", lang="hi")
        assert result["full_text"] == MOCK_TRANSCRIPT
        assert len(result["chunks"]) == 1
        assert result["latency_s"] >= 0

    def test_chunk_fields(self):
        result = transcribe("fake_audio.wav", lang="hi")
        chunk = result["chunks"][0]
        assert chunk["index"] == 0
        assert chunk["start_s"] == 0.0
        assert isinstance(chunk["text"], str) and len(chunk["text"]) > 0

    def test_output_schema(self):
        result = transcribe("fake_audio.wav", lang="hi")
        assert set(result) == {"full_text", "chunks", "latency_s"}
        assert set(result["chunks"][0]) == {
            "index", "start_s", "end_s", "text"
        }


# ─────────────────────────────────────────────────────────────────────────────
# Notes
# ─────────────────────────────────────────────────────────────────────────────

class TestNotesMock:
    def test_returns_notes_result(self):
        result = make_notes(MOCK_TRANSCRIPT, lang="hi")
        assert isinstance(result["markdown"], str)
        assert len(result["markdown"]) > 0

    def test_notes_has_required_sections(self):
        result = make_notes(MOCK_TRANSCRIPT, lang="hi")
        md = result["markdown"]
        assert "## Title" in md
        assert "## Key Points" in md
        assert "## Key Terms" in md
        assert "## Summary" in md

    def test_notes_returns_structured_fields(self):
        result = make_notes(MOCK_TRANSCRIPT, lang="hi")
        assert len(result["key_points"]) >= 5
        assert {
            "title",
            "key_points",
            "key_terms",
            "summary",
            "missing_information",
        }.issubset(result)

    def test_latency_non_negative(self):
        result = make_notes(MOCK_TRANSCRIPT, lang="hi")
        assert result["latency_s"] >= 0


# ─────────────────────────────────────────────────────────────────────────────
# Quiz / MCQ schema validation
# ─────────────────────────────────────────────────────────────────────────────

class TestMCQSchema:
    def test_returns_quiz_result(self):
        result = make_mcqs(MOCK_NOTES, lang="hi")
        assert isinstance(result.mcqs, list)

    def test_exactly_five_mcqs(self):
        result = make_mcqs(MOCK_NOTES, lang="hi")
        assert len(result.mcqs) == 5

    def test_mcq_required_keys(self):
        result = make_mcqs(MOCK_NOTES, lang="hi")
        required = {"question", "options", "answer", "explanation", "evidence"}
        for mcq in result.mcqs:
            assert required.issubset(mcq.keys()), f"Missing keys in {mcq}"

    def test_options_have_abcd(self):
        result = make_mcqs(MOCK_NOTES, lang="hi")
        for mcq in result.mcqs:
            assert set(mcq["options"].keys()) == {"A", "B", "C", "D"}

    def test_options_are_distinct(self):
        result = make_mcqs(MOCK_NOTES, lang="hi")
        for mcq in result.mcqs:
            vals = list(mcq["options"].values())
            assert len(set(vals)) == 4, f"Duplicate options in: {mcq}"

    def test_answer_in_options(self):
        result = make_mcqs(MOCK_NOTES, lang="hi")
        for mcq in result.mcqs:
            answer = mcq["answer"]
            assert answer in ("A", "B", "C", "D")
            assert answer in mcq["options"]

    def test_pydantic_validation(self):
        """Run Pydantic MCQList schema on mock output."""
        result = make_mcqs(MOCK_NOTES, lang="hi")
        mcq_list = MCQList(items=result.mcqs)
        assert len(mcq_list.items) == 5

    def test_evidence_non_empty(self):
        result = make_mcqs(MOCK_NOTES, lang="hi")
        for mcq in result.mcqs:
            assert mcq.get("evidence", "").strip() != ""


# ─────────────────────────────────────────────────────────────────────────────
# Translation
# ─────────────────────────────────────────────────────────────────────────────

class TestTranslateMock:
    def test_returns_translation_result(self):
        result = translate(MOCK_NOTES, src_lang="hi", tgt_lang="en")
        assert isinstance(result.translated_text, str)
        assert len(result.translated_text) > 0

    def test_identity_translation(self):
        """Same src and tgt → return unchanged text (even in MOCK mode)."""
        result = translate("Hello", src_lang="en", tgt_lang="en")
        # In MOCK mode translate() returns the input unchanged when src == tgt
        assert result.translated_text == "Hello"
        assert result.src_lang == "en"
        assert result.tgt_lang == "en"

    def test_latency_non_negative(self):
        result = translate(MOCK_NOTES, src_lang="hi", tgt_lang="en")
        assert result.latency_s >= 0


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineMock:
    def test_pipeline_with_transcript_override(self):
        result = run_pipeline(
            audio_path=None,
            lang="hi",
            transcript_override=MOCK_TRANSCRIPT,
        )
        assert result.success
        assert result.transcript is not None
        assert result.notes is not None
        assert result.quiz is not None

    def test_pipeline_to_dict(self):
        result = run_pipeline(
            audio_path=None,
            lang="hi",
            transcript_override=MOCK_TRANSCRIPT,
        )
        d = result.to_dict()
        assert "transcript" in d
        assert "notes" in d
        assert "quiz" in d
        assert isinstance(d["quiz"]["mcqs"], list)

    def test_pipeline_with_translation(self):
        result = run_pipeline(
            audio_path=None,
            lang="hi",
            translate_to="en",
            transcript_override=MOCK_TRANSCRIPT,
        )
        assert result.success
        assert result.translation is not None
        assert result.translation.translated_text != ""

    def test_pipeline_fails_gracefully_no_audio(self):
        """No audio and no transcript override → graceful failure."""
        result = run_pipeline(audio_path=None, lang="hi")
        assert not result.success
        assert result.error_stage == "asr"

    def test_pipeline_timings(self):
        result = run_pipeline(
            audio_path=None,
            lang="hi",
            transcript_override=MOCK_TRANSCRIPT,
        )
        d = result.to_dict()
        assert d["total_latency_s"] >= 0
        assert isinstance(d["timings"], dict)

    def test_pipeline_progress_callback(self):
        progress = []
        result = run_pipeline(
            audio_path=None,
            lang="hi",
            transcript_override=MOCK_TRANSCRIPT,
            progress_callback=lambda stage, fraction: progress.append(
                (stage, fraction)
            ),
        )
        assert result.success
        assert progress[-1] == ("Pipeline complete", 1.0)

    def test_pipeline_preserves_transcript_when_notes_fail(self, monkeypatch):
        freed = []
        monkeypatch.setattr(
            pipeline_module, "free_llm_memory", lambda: freed.append(True)
        )
        monkeypatch.setattr(
            pipeline_module,
            "transcribe",
            lambda *args, **kwargs: {
                "full_text": "recognized words",
                "chunks": [
                    {
                        "index": 0,
                        "start_s": 0.0,
                        "end_s": 2.0,
                        "text": "recognized words",
                    }
                ],
                "latency_s": 0.25,
            },
        )

        def fail_notes(*args, **kwargs):
            raise RuntimeError("downstream model is unavailable")

        monkeypatch.setattr(pipeline_module, "make_notes", fail_notes)
        result = pipeline_module.run_pipeline(audio_path="lecture.wav")

        assert not result.success
        assert result.error_stage == "notes"
        assert result.transcript is not None
        assert result.transcript.full_text == "recognized words"
        assert result.stage_latencies["asr"] == 0.25
        assert result.to_dict()["transcript"]["chunks"][0]["end_s"] == 2.0
        assert freed == [True]

    def test_pipeline_reports_nonfatal_translation_error(self, monkeypatch):
        def fail_translation(*args, **kwargs):
            raise RuntimeError("translation model unavailable")

        monkeypatch.setattr(pipeline_module, "translate", fail_translation)
        result = pipeline_module.run_pipeline(
            audio_path=None,
            lang="hi",
            translate_to="en",
            transcript_override=MOCK_TRANSCRIPT,
        )

        assert result.success
        assert result.warnings == [
            "Translation failed (non-fatal): translation model unavailable"
        ]
        assert result.stage_latencies["translation"] >= 0
