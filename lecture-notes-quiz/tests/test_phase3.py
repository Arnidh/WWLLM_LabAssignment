from __future__ import annotations

import json
import os

import pytest

os.environ.setdefault("MOCK_MODELS", "1")

from src import quiz as quiz_module
from src import pipeline as pipeline_module
from src.asr import MOCK_TRANSCRIPT
from src.notes import NotesDocument, split_text_by_tokens
from src.quiz import MOCK_MCQS, QuizResult, validate_mcq_output


TRANSCRIPT = (
    "Force is mass times acceleration. "
    "Newton's first law describes inertia. "
    "Newton's second law relates force and acceleration. "
    "Newton's third law describes equal and opposite reactions. "
    "The lecture discusses rocket science."
)


def valid_mcqs() -> list[dict]:
    evidence = [
        "Force is mass times acceleration.",
        "Newton's first law describes inertia.",
        "Newton's second law relates force and acceleration.",
        "Newton's third law describes equal and opposite reactions.",
        "The lecture discusses rocket science.",
    ]
    return [
        {
            "question": f"Question {index}?",
            "options": {
                "A": f"Choice {index}A",
                "B": f"Choice {index}B",
                "C": f"Choice {index}C",
                "D": f"Choice {index}D",
            },
            "answer": "A",
            "explanation": "The source states this fact.",
            "evidence": sentence,
        }
        for index, sentence in enumerate(evidence, start=1)
    ]


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        return text.split()

    def decode(
        self, token_ids: list[str], skip_special_tokens: bool = True
    ) -> str:
        return " ".join(token_ids)

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        words = sum(len(message["content"].split()) for message in conversation)
        return list(range(words))


def test_quiz_schema_accepts_exactly_five_grounded_questions():
    parsed = validate_mcq_output(
        json.dumps(valid_mcqs()),
        TRANSCRIPT,
    )
    assert len(parsed) == 5
    assert set(parsed[0]["options"]) == {"A", "B", "C", "D"}


def test_canned_mock_quiz_has_transcript_evidence():
    assert len(validate_mcq_output(json.dumps(MOCK_MCQS), MOCK_TRANSCRIPT)) == 5


def test_quiz_schema_rejects_wrong_question_count():
    with pytest.raises(ValueError, match="exactly 5"):
        validate_mcq_output(json.dumps(valid_mcqs()[:4]), TRANSCRIPT)


def test_quiz_schema_rejects_duplicate_options():
    questions = valid_mcqs()
    questions[0]["options"]["D"] = questions[0]["options"]["A"]
    with pytest.raises(ValueError, match="distinct"):
        validate_mcq_output(json.dumps(questions), TRANSCRIPT)


def test_quiz_schema_rejects_invalid_answer():
    questions = valid_mcqs()
    questions[0]["answer"] = "E"
    with pytest.raises(ValueError, match="A, B, C, or D"):
        validate_mcq_output(json.dumps(questions), TRANSCRIPT)


def test_quiz_schema_rejects_unsupported_evidence():
    questions = valid_mcqs()
    questions[0]["evidence"] = "The moon is made of cheese."
    with pytest.raises(ValueError, match="not supported"):
        validate_mcq_output(json.dumps(questions), TRANSCRIPT)


def test_quiz_schema_rejects_partial_sentence_as_evidence():
    questions = valid_mcqs()
    questions[0]["evidence"] = "Force is mass times"
    with pytest.raises(ValueError, match="not supported"):
        validate_mcq_output(json.dumps(questions), TRANSCRIPT)


def test_quiz_evidence_allows_small_fuzzy_difference():
    questions = valid_mcqs()
    questions[0]["evidence"] = "Force is mass times acceleratio."
    assert len(validate_mcq_output(json.dumps(questions), TRANSCRIPT)) == 5


def test_notes_schema_rejects_too_few_key_points():
    with pytest.raises(ValueError):
        NotesDocument.model_validate(
            {
                "title": "Lecture",
                "key_points": ["Only one"],
                "key_terms": [],
                "summary": "A short summary.",
            }
        )


def test_generation_retries_invalid_json(monkeypatch):
    outputs = iter(["not JSON", json.dumps(valid_mcqs())])
    prompts: list[str] = []

    monkeypatch.setattr(quiz_module, "is_mock_mode", lambda: False)
    monkeypatch.setattr(quiz_module, "_load_llm", lambda: None)
    monkeypatch.setattr(quiz_module, "load_config", lambda: {
        "supported_languages": {"en": "English"},
        "generation": {"mcq_retry_limit": 2, "temperature": 0.2},
        "context": {"max_input_tokens": 6000},
    })
    monkeypatch.setattr(quiz_module.notes_module, "_llm_tokenizer", FakeTokenizer())

    def generate(prompt: str, **kwargs: object) -> str:
        prompts.append(prompt)
        return next(outputs)

    monkeypatch.setattr(quiz_module, "_generate", generate)
    result = quiz_module.make_mcqs(TRANSCRIPT, lang="en")

    assert result.attempts == 2
    assert result.latency_s >= 0
    assert len(result.mcqs) == 5
    assert len(prompts) == 2
    assert "failed validation" in prompts[1]


def test_generation_stops_after_initial_plus_two_repairs(monkeypatch):
    calls = 0

    monkeypatch.setattr(quiz_module, "is_mock_mode", lambda: False)
    monkeypatch.setattr(quiz_module, "_load_llm", lambda: None)
    monkeypatch.setattr(quiz_module, "load_config", lambda: {
        "supported_languages": {"en": "English"},
        "generation": {"mcq_retry_limit": 2, "temperature": 0.2},
        "context": {"max_input_tokens": 6000},
    })
    monkeypatch.setattr(quiz_module.notes_module, "_llm_tokenizer", FakeTokenizer())

    def invalid_output(*args: object, **kwargs: object) -> str:
        nonlocal calls
        calls += 1
        return "not JSON"

    monkeypatch.setattr(quiz_module, "_generate", invalid_output)
    with pytest.raises(RuntimeError, match="after 3 attempt"):
        quiz_module.make_mcqs(TRANSCRIPT, lang="en")
    assert calls == 3


def test_repair_prompt_truncates_previous_response_to_context_budget(monkeypatch):
    monkeypatch.setattr(quiz_module.notes_module, "_llm_tokenizer", FakeTokenizer())
    prompt = quiz_module._build_repair_prompt(
        "invalid " * 1000,
        "The response was not valid JSON.",
        "Force equals mass times acceleration.",
        TRANSCRIPT,
        "English",
        max_input_tokens=300,
    )
    assert quiz_module._prompt_token_count(prompt) <= 300


def test_map_reduce_splitter_respects_token_budget():
    parts = split_text_by_tokens(
        "one two three four five",
        token_budget=2,
        tokenizer=FakeTokenizer(),
    )
    assert parts == ["one two", "three four", "five"]


def test_pipeline_passes_transcript_for_evidence_and_frees_model(monkeypatch):
    captured: dict[str, object] = {}
    freed = []
    transcript = "A transcript sentence."
    notes = {
        "title": "Lecture",
        "key_points": ["point"],
        "key_terms": [],
        "summary": "summary",
        "missing_information": [],
        "markdown": "## Lecture",
        "latency_s": 0.1,
        "used_map_reduce": False,
    }

    monkeypatch.setattr(pipeline_module, "make_notes", lambda *args, **kwargs: notes)

    def make_quiz(source: object, **kwargs: object) -> QuizResult:
        captured["source"] = source
        captured.update(kwargs)
        return QuizResult([], "[]", 1, 0.1)

    monkeypatch.setattr(pipeline_module, "make_mcqs", make_quiz)
    monkeypatch.setattr(
        pipeline_module, "free_llm_memory", lambda: freed.append(True)
    )
    result = pipeline_module.run_pipeline(
        audio_path=None,
        lang="en",
        transcript_override=transcript,
    )

    assert result.success
    assert captured["source"] is notes
    assert captured["transcript"] == transcript
    assert freed == [True]


def test_pipeline_translates_before_generating_notes_and_quiz(monkeypatch):
    from src.translate import TranslationResult

    calls: list[tuple[str, str]] = []
    original = "Original-language transcript."
    translated = "Translated transcript."
    notes = {
        "title": "Lecture",
        "key_points": ["point"],
        "key_terms": [],
        "summary": "summary",
        "missing_information": [],
        "markdown": "## Lecture",
        "latency_s": 0.1,
        "used_map_reduce": False,
    }

    def translate_text(text: str, src_lang: str, tgt_lang: str) -> TranslationResult:
        calls.append(("translate", text))
        assert src_lang == "hi"
        assert tgt_lang == "en"
        return TranslationResult(translated, src_lang, tgt_lang, 0.1)

    def make_notes(text: str, lang: str) -> dict:
        calls.append(("notes", text))
        assert lang == "en"
        return notes

    def make_quiz(source: object, **kwargs: object) -> QuizResult:
        calls.append(("quiz", str(kwargs["transcript"])))
        assert kwargs["lang"] == "en"
        assert kwargs["transcript"] == translated
        return QuizResult([], "[]", 1, 0.1)

    monkeypatch.setattr(pipeline_module, "translate", translate_text)
    monkeypatch.setattr(pipeline_module, "make_notes", make_notes)
    monkeypatch.setattr(pipeline_module, "make_mcqs", make_quiz)
    monkeypatch.setattr(pipeline_module, "free_llm_memory", lambda: None)
    result = pipeline_module.run_pipeline(
        audio_path=None,
        lang="hi",
        translate_to="en",
        transcript_override=original,
    )

    assert result.success
    assert result.transcript is not None
    assert result.transcript.full_text == original
    assert calls == [
        ("translate", original),
        ("notes", translated),
        ("quiz", translated),
    ]
