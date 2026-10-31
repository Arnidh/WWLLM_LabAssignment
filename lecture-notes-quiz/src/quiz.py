"""
Grounded five-question MCQ generation with strict schema and evidence checks.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from . import notes as notes_module
from .notes import (
    _count_tokens,
    _generate,
    _load_llm,
    free_llm_memory,
    make_notes,
    notes_to_markdown,
)
from .prompts import SYSTEM_GROUNDING, mcq_prompt, mcq_repair_prompt
from .utils import get_logger, is_mock_mode, load_config, safe_parse_json, timer

logger = get_logger(__name__)

MOCK_MCQS = [
    {
        "question": "बाहरी बल न लगने पर वस्तु क्या करती है?",
        "options": {
            "A": "विरामावस्था में रहती है",
            "B": "हमेशा तेज़ होती है",
            "C": "दिशा बदलती है",
            "D": "रुकने के बजाय नष्ट हो जाती है",
        },
        "answer": "A",
        "explanation": "व्याख्यान में कहा गया है कि वस्तु बाहरी बल न लगने तक विरामावस्था में रहती है।",
        "evidence": "न्यूटन का पहला नियम कहता है कि कोई वस्तु तब तक विरामावस्था में रहती है जब तक उस पर कोई बाहरी बल न लगे।",
    },
    {
        "question": "दूसरे नियम के अनुसार बल किन राशियों का गुणनफल है?",
        "options": {
            "A": "द्रव्यमान और त्वरण",
            "B": "वेग और दूरी",
            "C": "ऊर्जा और समय",
            "D": "क्षेत्रफल और घनत्व",
        },
        "answer": "A",
        "explanation": "व्याख्यान दूसरे नियम में बल, द्रव्यमान और त्वरण का संबंध बताता है।",
        "evidence": "दूसरा नियम बताता है कि बल, द्रव्यमान और त्वरण का गुणनफल होता है।",
    },
    {
        "question": "न्यूटन का तीसरा नियम क्या कहता है?",
        "options": {
            "A": "बल = द्रव्यमान × त्वरण",
            "B": "वस्तु जड़त्व बनाए रखती है",
            "C": "प्रत्येक क्रिया की समान और विपरीत प्रतिक्रिया होती है",
            "D": "ऊर्जा नष्ट नहीं होती",
        },
        "answer": "C",
        "explanation": "तीसरा नियम क्रिया-प्रतिक्रिया का नियम है।",
        "evidence": "तीसरा नियम: प्रत्येक क्रिया के लिए समान और विपरीत प्रतिक्रिया होती है।",
    },
    {
        "question": "इन नियमों का व्यावहारिक उपयोग कहाँ होता है?",
        "options": {
            "A": "केवल प्रयोगशाला में",
            "B": "रॉकेट विज्ञान, वाहन डिज़ाइन और खेल-कूद में",
            "C": "केवल खगोल विज्ञान में",
            "D": "केवल रसायन शास्त्र में",
        },
        "answer": "B",
        "explanation": "व्याख्यान में रॉकेट विज्ञान, वाहन और खेल का उल्लेख किया गया।",
        "evidence": "इन नियमों का उपयोग रॉकेट विज्ञान, वाहन डिज़ाइन और खेल-कूद में होता है।",
    },
    {
        "question": "रॉकेट विज्ञान के अलावा किन उपयोगों का उल्लेख किया गया?",
        "options": {
            "A": "वाहन डिज़ाइन और खेल-कूद",
            "B": "केवल रसायन शास्त्र",
            "C": "केवल संगीत",
            "D": "केवल भोजन पकाने",
        },
        "answer": "A",
        "explanation": "व्याख्यान में वाहन डिज़ाइन और खेल-कूद का उपयोग भी बताया गया है।",
        "evidence": "इन नियमों का उपयोग रॉकेट विज्ञान, वाहन डिज़ाइन और खेल-कूद में होता है।",
    },
]

MOCK_MCQS_EN = [
    {
        "question": "What happens to an object when no external force acts on it?",
        "options": {
            "A": "It remains at rest",
            "B": "It always speeds up",
            "C": "It changes direction",
            "D": "It disappears",
        },
        "answer": "A",
        "explanation": "Newton's first law says an object remains at rest unless an external force acts on it.",
        "evidence": "Newton's first law states that an object remains at rest unless an external force acts on it.",
    },
    {
        "question": "What does Newton's second law say force equals?",
        "options": {
            "A": "Mass times acceleration",
            "B": "Velocity times distance",
            "C": "Energy divided by time",
            "D": "Area times density",
        },
        "answer": "A",
        "explanation": "The second law relates force to mass and acceleration.",
        "evidence": "Newton's second law states that force equals mass times acceleration.",
    },
    {
        "question": "What does Newton's third law describe?",
        "options": {
            "A": "Force equals mass times acceleration",
            "B": "An object's inertia",
            "C": "Equal and opposite reactions",
            "D": "Conservation of energy",
        },
        "answer": "C",
        "explanation": "The third law describes action and reaction.",
        "evidence": "Newton's third law states that every action has an equal and opposite reaction.",
    },
    {
        "question": "Which application of Newton's laws is mentioned?",
        "options": {
            "A": "Cooking",
            "B": "Rocket science",
            "C": "Music",
            "D": "Chemistry only",
        },
        "answer": "B",
        "explanation": "The lecture mentions rocket science as an application.",
        "evidence": "These laws are used in rocket science.",
    },
    {
        "question": "Besides rocket science, where do these laws apply?",
        "options": {
            "A": "Vehicle design and sports",
            "B": "Music and cooking",
            "C": "Chemistry and painting",
            "D": "Only in laboratories",
        },
        "answer": "A",
        "explanation": "The lecture also mentions vehicle design and sports.",
        "evidence": "They also apply to vehicle design and sports.",
    },
]


class MCQOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    A: str
    B: str
    C: str
    D: str

    @field_validator("A", "B", "C", "D")
    @classmethod
    def require_nonempty_option(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Options cannot be empty.")
        return value

    @model_validator(mode="after")
    def all_options_distinct(self) -> "MCQOptions":
        normalized = {
            re.sub(r"\s+", " ", value).casefold()
            for value in (self.A, self.B, self.C, self.D)
        }
        if len(normalized) != 4:
            raise ValueError("All four options must be distinct.")
        return self


class MCQItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    options: MCQOptions
    answer: str
    explanation: str
    evidence: str

    @field_validator("question", "explanation", "evidence")
    @classmethod
    def require_nonempty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question, explanation, and evidence cannot be empty.")
        return value

    @field_validator("answer")
    @classmethod
    def answer_in_options(cls, value: str) -> str:
        value = value.strip().upper()
        if value not in {"A", "B", "C", "D"}:
            raise ValueError(f"Answer '{value}' is not A, B, C, or D.")
        return value


class MCQList(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[MCQItem]

    @model_validator(mode="after")
    def exactly_five(self) -> "MCQList":
        if len(self.items) != 5:
            raise ValueError(f"Expected exactly 5 MCQs, got {len(self.items)}.")
        return self


@dataclass
class QuizResult:
    mcqs: list[dict[str, Any]]
    raw_json: str
    attempts: int
    latency_s: float
    validation_warnings: list[str] = field(default_factory=list)


def _normalize_evidence(text: str) -> str:
    text = unicodedata.normalize("NFC", text).casefold()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def evidence_is_supported(evidence: str, transcript: str) -> bool:
    """Require a full normalized sentence match or a conservative fuzzy match."""
    candidate = _normalize_evidence(evidence)
    if len(candidate) < 8 or not transcript.strip():
        return False

    sentences = [
        _normalize_evidence(sentence)
        for sentence in re.split(r"(?<=[.!?।])\s+|\n+", transcript)
        if sentence.strip()
    ]
    return any(
        candidate == sentence
        or SequenceMatcher(None, candidate, sentence).ratio() >= 0.9
        for sentence in sentences
        if sentence
    )


def validate_mcq_output(raw_output: str, transcript: str) -> list[dict[str, Any]]:
    """Parse and strictly validate one generated JSON array against its source."""
    try:
        parsed = safe_parse_json(raw_output)
        if not isinstance(parsed, list):
            raise ValueError(
                f"Expected a JSON array, got {type(parsed).__name__}."
            )
        validated = MCQList(items=parsed)
    except Exception as exc:
        raise ValueError(f"MCQ JSON/schema validation failed: {exc}") from exc

    unsupported = [
        index
        for index, item in enumerate(validated.items, start=1)
        if not evidence_is_supported(item.evidence, transcript)
    ]
    if unsupported:
        indexes = ", ".join(str(index) for index in unsupported)
        raise ValueError(
            f"MCQ evidence is not supported by the transcript for question(s): {indexes}."
        )
    return [item.model_dump() for item in validated.items]


def _source_as_text(source: str | dict[str, Any]) -> str:
    if isinstance(source, dict):
        return notes_to_markdown(source)
    return source


def _sentence_list(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?।])\s+|\n+", text)
        if sentence.strip()
    ]


def _select_evidence_bank(transcript: str, token_budget: int) -> str:
    if _count_tokens(transcript) <= token_budget:
        return transcript

    sentences = _sentence_list(transcript)
    if not sentences:
        return ""

    step = max(1, (len(sentences) + 31) // 32)
    sampled = sentences[::step]
    chosen: list[str] = []
    used = 0
    for sentence in sampled:
        cost = _count_tokens(sentence)
        if used + cost <= token_budget:
            chosen.append(sentence)
            used += cost
    if not chosen:
        return ""
    return "\n".join(chosen)


def _prompt_token_count(prompt: str) -> int:
    tokenizer = notes_module._llm_tokenizer
    if tokenizer is None:
        raise RuntimeError("Param-1 tokenizer is not loaded.")
    messages = [
        {"role": "system", "content": SYSTEM_GROUNDING},
        {"role": "user", "content": prompt},
    ]
    tokens = tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    return len(tokens)


def _build_repair_prompt(
    bad_output: str,
    error: str,
    question_source: str,
    evidence_bank: str,
    lang_name: str,
    max_input_tokens: int,
) -> str:
    tokenizer = notes_module._llm_tokenizer
    if tokenizer is None:
        raise RuntimeError("Param-1 tokenizer is not loaded.")

    base_prompt = mcq_repair_prompt(
        "", error, question_source, evidence_bank, lang_name
    )
    remaining = max_input_tokens - _prompt_token_count(base_prompt) - 32
    if remaining < 0:
        raise ValueError(
            "The MCQ repair instructions and source exceed "
            "context.max_input_tokens; shorten the notes or increase the budget."
        )

    output_tokens = tokenizer.encode(bad_output, add_special_tokens=False)
    while remaining >= 0:
        truncated_output = tokenizer.decode(
            output_tokens[:remaining],
            skip_special_tokens=True,
        )
        prompt = mcq_repair_prompt(
            truncated_output,
            error,
            question_source,
            evidence_bank,
            lang_name,
        )
        token_count = _prompt_token_count(prompt)
        if token_count <= max_input_tokens:
            return prompt
        remaining -= max(1, token_count - max_input_tokens)

    raise ValueError(
        "The MCQ repair prompt exceeds context.max_input_tokens even without "
        "the previous response."
    )


def make_mcqs(
    transcript_or_notes: str | dict[str, Any],
    lang: str = "hi",
    *,
    notes: dict[str, Any] | None = None,
    transcript: str | None = None,
) -> QuizResult:
    """Generate exactly five MCQs and reject unsupported evidence."""
    if is_mock_mode():
        logger.info("MOCK MODE: returning canned MCQs")
        mock_mcqs = MOCK_MCQS_EN if lang == "en" else MOCK_MCQS
        return QuizResult(
            mcqs=[dict(item) for item in mock_mcqs],
            raw_json=json.dumps(mock_mcqs, ensure_ascii=False, indent=2),
            attempts=1,
            latency_s=0.01,
        )

    cfg = load_config()
    generation_cfg = cfg.get("generation", {})
    context_cfg = cfg.get("context", {})
    max_input_tokens = int(context_cfg.get("max_input_tokens", 6000))
    lang_name = cfg.get("supported_languages", {}).get(lang, lang)
    source_text = _source_as_text(transcript_or_notes)
    evidence_source = transcript or source_text
    if not source_text.strip() or not evidence_source.strip():
        raise ValueError("Cannot generate a quiz from empty notes or transcript.")

    _load_llm()
    if notes is not None:
        question_source = notes_to_markdown(notes)
    elif _count_tokens(source_text) > max_input_tokens // 2:
        notes_result = make_notes(source_text, lang=lang)
        question_source = notes_result["markdown"]
    else:
        question_source = source_text

    evidence_budget = max(1, max_input_tokens // 3)
    evidence_bank = _select_evidence_bank(evidence_source, evidence_budget)
    if not evidence_bank:
        raise ValueError(
            "No transcript evidence fits the configured context.max_input_tokens."
        )

    current_prompt = mcq_prompt(question_source, lang_name, evidence_bank)
    while _prompt_token_count(current_prompt) > max_input_tokens:
        evidence_budget //= 2
        if evidence_budget < 1:
            raise ValueError(
                "Quiz instructions and source exceed context.max_input_tokens; "
                "increase the budget or shorten the notes."
            )
        evidence_bank = _select_evidence_bank(evidence_source, evidence_budget)
        current_prompt = mcq_prompt(question_source, lang_name, evidence_bank)

    retry_limit = min(2, max(0, int(generation_cfg.get("mcq_retry_limit", 2))))
    if generation_cfg.get("mcq_retry_limit", 2) > 2:
        logger.warning("Capping MCQ repair attempts at the supported maximum of 2.")

    raw_output = ""
    last_error = ""
    mcqs: list[dict[str, Any]] | None = None
    successful_attempt = 0
    with timer("MCQ generation", logger) as timing:
        for attempt in range(1, retry_limit + 2):
            raw_output = _generate(
                current_prompt,
                max_new_tokens=generation_cfg.get("max_new_tokens_mcq", 1200),
                temperature=min(
                    0.3,
                    float(generation_cfg.get("temperature", 0.2)),
                ),
            )
            try:
                mcqs = validate_mcq_output(raw_output, evidence_source)
                successful_attempt = attempt
                break
            except ValueError as exc:
                last_error = str(exc)
                logger.warning(
                    "MCQ validation failed on attempt %d/%d: %s",
                    attempt,
                    retry_limit + 1,
                    last_error,
                )
                if attempt <= retry_limit:
                    current_prompt = _build_repair_prompt(
                        raw_output,
                        last_error,
                        question_source,
                        evidence_bank,
                        lang_name,
                        max_input_tokens,
                    )

    if mcqs is not None:
        return QuizResult(
            mcqs=mcqs,
            raw_json=raw_output,
            attempts=successful_attempt,
            latency_s=timing.elapsed,
        )
    raise RuntimeError(
        f"MCQ generation failed after {retry_limit + 1} attempt(s): {last_error}"
    )


# Kept as a compatibility alias for callers from the earlier scaffold.
_free_llm = free_llm_memory
