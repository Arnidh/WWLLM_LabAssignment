"""
Structured lecture notes using the verified BharatGen Param-1 instruct model.
"""
from __future__ import annotations

import gc
import math
import os
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .prompts import (
    SYSTEM_GROUNDING,
    notes_chunk_summary_prompt,
    notes_combine_prompt,
    notes_prompt,
)
from .utils import get_logger, is_mock_mode, load_config, safe_parse_json, timer

logger = get_logger(__name__)

MOCK_NOTES = """\
## Title
न्यूटन के गति के नियम (Newton's Laws of Motion)

## Key Points
- न्यूटन का पहला नियम (जड़त्व का नियम): कोई वस्तु तब तक अपनी अवस्था नहीं बदलती जब तक बाहरी बल न लगे।
- न्यूटन का दूसरा नियम: बल = द्रव्यमान × त्वरण (F = ma)।
- न्यूटन का तीसरा नियम: प्रत्येक क्रिया की समान और विपरीत प्रतिक्रिया होती है।
- इन नियमों का उपयोग रॉकेट विज्ञान में किया जाता है।
- वाहन डिज़ाइन और खेल-कूद में भी ये नियम लागू होते हैं।

## Key Terms
- **बल (Force)**: वह प्रभाव जो किसी वस्तु की गति या आकार को बदलता है।
- **द्रव्यमान (Mass)**: किसी वस्तु में पदार्थ की मात्रा।
- **त्वरण (Acceleration)**: वेग में परिवर्तन की दर।
- **जड़त्व (Inertia)**: वस्तु की अपनी अवस्था बनाए रखने की प्रवृत्ति।

## Summary
इस व्याख्यान में न्यूटन के तीनों गति के नियमों की व्याख्या की गई। पहला नियम जड़त्व, दूसरा F=ma और तीसरा क्रिया-प्रतिक्रिया पर आधारित है। इन नियमों का व्यावहारिक उपयोग रॉकेट विज्ञान, वाहन और खेल में होता है।
"""

MOCK_NOTES_DATA: dict[str, Any] = {
    "title": "न्यूटन के गति के नियम",
    "key_points": [
        "न्यूटन का पहला नियम जड़त्व का नियम है।",
        "न्यूटन का दूसरा नियम बल, द्रव्यमान और त्वरण को जोड़ता है।",
        "न्यूटन का तीसरा नियम क्रिया और प्रतिक्रिया के बारे में है।",
        "इन नियमों का उपयोग रॉकेट विज्ञान में होता है।",
        "इन नियमों का उपयोग वाहन डिज़ाइन और खेल-कूद में भी होता है।",
    ],
    "key_terms": [
        {"term": "बल (Force)", "definition": "व्याख्यान इसे गति के नियमों से जोड़ता है।"},
        {"term": "द्रव्यमान (Mass)", "definition": "यह दूसरे नियम में बल और त्वरण के साथ आता है।"},
        {"term": "त्वरण (Acceleration)", "definition": "यह दूसरे नियम में बल और द्रव्यमान के साथ आता है।"},
        {"term": "जड़त्व (Inertia)", "definition": "यह न्यूटन के पहले नियम का विषय है।"},
    ],
    "summary": (
        "व्याख्यान में न्यूटन के तीन गति के नियमों का परिचय दिया गया। "
        "इन नियमों के उपयोग रॉकेट विज्ञान, वाहन डिज़ाइन और खेल-कूद में बताए गए।"
    ),
    "missing_information": [],
}

MOCK_NOTES_DATA_EN: dict[str, Any] = {
    "title": "Newton's Laws of Motion",
    "key_points": [
        "Newton's first law is also called the law of inertia.",
        "An object remains at rest unless an external force acts on it.",
        "Newton's second law states that force equals mass times acceleration.",
        "Newton's third law describes equal and opposite reactions.",
        "The laws apply to rocket science, vehicle design, and sports.",
    ],
    "key_terms": [
        {"term": "Force", "definition": "An influence that changes an object's motion or shape."},
        {"term": "Mass", "definition": "The amount of matter in an object."},
        {"term": "Acceleration", "definition": "The rate of change of velocity."},
        {"term": "Inertia", "definition": "An object's tendency to maintain its state of motion."},
    ],
    "summary": (
        "The lecture explains Newton's three laws of motion and their "
        "applications in rocket science, vehicle design, and sports."
    ),
    "missing_information": [],
}


class KeyTerm(BaseModel):
    model_config = ConfigDict(extra="forbid")

    term: str
    definition: str

    @field_validator("term", "definition")
    @classmethod
    def require_nonempty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Key-term names and definitions cannot be empty.")
        return value


class NotesDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    key_points: list[str] = Field(min_length=5, max_length=10)
    key_terms: list[KeyTerm]
    summary: str
    missing_information: list[str] = Field(default_factory=list)

    @field_validator("title", "summary")
    @classmethod
    def require_nonempty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Title and summary cannot be empty.")
        return value

    @field_validator("key_points", "missing_information")
    @classmethod
    def clean_string_lists(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("List items cannot be empty.")
        return cleaned


_llm_model = None
_llm_tokenizer = None


def _load_llm() -> None:
    """Load one shared Param-1 model lazily using the configured device/precision."""
    global _llm_model, _llm_tokenizer
    if _llm_model is not None:
        return

    cfg = load_config()
    model_id = cfg["models"]["llm"]
    quantization = str(cfg.get("memory", {}).get("llm_quantization", "fp16")).lower()
    if quantization not in {"fp16", "int8", "int4", "none"}:
        raise ValueError(
            f"Unsupported memory.llm_quantization '{quantization}'. "
            "Choose fp16, int8, int4, or none."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and quantization in {"int4", "int8"}:
        raise RuntimeError(
            f"memory.llm_quantization={quantization} requires CUDA and "
            "BitsAndBytes; use fp16 (CPU runs as float32) on this device."
        )

    if cfg.get("memory", {}).get("free_asr_before_llm", True):
        from .asr import free_gpu_memory

        free_gpu_memory()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_token = os.getenv("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=hf_token,
        trust_remote_code=False,
    )
    load_kwargs: dict[str, Any] = {
        "token": hf_token,
        "trust_remote_code": True,
    }

    if quantization in {"int4", "int8"}:
        from transformers import BitsAndBytesConfig

        if quantization == "int4":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        load_kwargs["device_map"] = "auto"
    elif device == "cuda":
        load_kwargs["torch_dtype"] = (
            torch.float16 if quantization == "fp16" else torch.float32
        )
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["torch_dtype"] = torch.float32

    logger.info(
        "Loading Param-1 '%s' with %s precision on %s",
        model_id,
        quantization,
        device,
    )
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    if device == "cpu":
        model.to("cpu")
    model.eval()
    _llm_model = model
    _llm_tokenizer = tokenizer


def free_llm_memory() -> None:
    """Release the shared Param-1 model and its cached GPU memory."""
    global _llm_model, _llm_tokenizer
    _llm_model = None
    _llm_tokenizer = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Param-1 model freed from memory.")


def _count_tokens(text: str) -> int:
    if _llm_tokenizer is not None:
        return len(_llm_tokenizer.encode(text, add_special_tokens=False))
    return max(1, math.ceil(len(text) / 4))


def split_text_by_tokens(
    text: str,
    token_budget: int,
    tokenizer: Any | None = None,
) -> list[str]:
    """Split text into bounded, ordered groups using a tokenizer when available."""
    if token_budget < 1:
        raise ValueError("token_budget must be at least one.")
    if not text.strip():
        return []

    if tokenizer is not None:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        return [
            tokenizer.decode(
                token_ids[start : start + token_budget],
                skip_special_tokens=True,
            )
            for start in range(0, len(token_ids), token_budget)
        ]

    words = text.split()
    groups: list[str] = []
    current: list[str] = []
    used = 0
    for word in words:
        word_tokens = max(1, math.ceil(len(word) / 4))
        if current and used + word_tokens > token_budget:
            groups.append(" ".join(current))
            current = []
            used = 0
        current.append(word)
        used += word_tokens
    if current:
        groups.append(" ".join(current))
    return groups


def notes_to_markdown(notes: dict[str, Any]) -> str:
    """Render structured notes for the Gradio Markdown panel and downloads."""
    lines = [
        "## Title",
        str(notes["title"]),
        "",
        "## Key Points",
        *[f"- {point}" for point in notes["key_points"]],
        "",
        "## Key Terms",
    ]
    lines.extend(
        f"- **{term['term']}**: {term['definition']}"
        for term in notes["key_terms"]
    )
    lines.extend(["", "## Summary", str(notes["summary"])])
    missing = notes.get("missing_information", [])
    if missing:
        lines.extend(
            ["", "## Missing or Unclear Information"]
            + [f"- {item}" for item in missing]
        )
    return "\n".join(lines)


def _chat_messages(user_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_GROUNDING},
        {"role": "user", "content": user_prompt},
    ]


def _generate(
    prompt: str,
    *,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
) -> str:
    """Generate a completion using the model-card's tokenizer chat template."""
    if _llm_model is None or _llm_tokenizer is None:
        raise RuntimeError("Param-1 is not loaded; call _load_llm() before generation.")

    cfg = load_config()
    generation_cfg = cfg.get("generation", {})
    context_cfg = cfg.get("context", {})
    output_limit = max_new_tokens or generation_cfg.get("max_new_tokens_notes", 800)
    context_limit = int(context_cfg.get("model_max_tokens", 8192))
    input_limit = int(context_cfg.get("max_input_tokens", 6000))
    messages = _chat_messages(prompt)

    input_ids = _llm_tokenizer.apply_chat_template(
        conversation=messages,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    input_tokens = int(input_ids.shape[-1])
    if input_tokens > input_limit:
        raise ValueError(
            f"Param-1 prompt has {input_tokens} tokens, exceeding "
            f"context.max_input_tokens={input_limit}; reduce the input or "
            "increase the configured prompt budget."
        )
    available_output = context_limit - input_tokens
    if available_output < 1:
        raise ValueError(
            f"Param-1 prompt uses {input_tokens} tokens, exceeding the "
            f"configured model context of {context_limit}."
        )
    output_limit = min(int(output_limit), available_output)
    configured_temperature = (
        generation_cfg.get("temperature", 0.2)
        if temperature is None
        else temperature
    )
    safe_temperature = min(0.3, max(0.0, float(configured_temperature)))
    device = getattr(_llm_model, "device", torch.device("cpu"))
    input_ids = input_ids.to(device)

    with torch.inference_mode():
        generated = _llm_model.generate(
            input_ids,
            max_new_tokens=output_limit,
            temperature=safe_temperature,
            top_p=generation_cfg.get("top_p", 0.85),
            repetition_penalty=generation_cfg.get("repetition_penalty", 1.1),
            do_sample=safe_temperature > 0,
            pad_token_id=(
                _llm_tokenizer.pad_token_id
                if _llm_tokenizer.pad_token_id is not None
                else _llm_tokenizer.eos_token_id
            ),
            eos_token_id=_llm_tokenizer.eos_token_id,
            use_cache=False,
        )
    completion = generated[0][input_ids.shape[-1] :]
    return _llm_tokenizer.decode(
        completion,
        skip_special_tokens=True,
    ).strip()


def _reduce_transcript(transcript: str, lang_name: str, cfg: dict) -> str:
    context_cfg = cfg.get("context", {})
    max_input = int(context_cfg.get("max_input_tokens", 6000))
    requested_chunk = int(context_cfg.get("map_reduce_chunk_size", 1200))
    chunk_budget = min(requested_chunk, max_input - 500)
    if chunk_budget < 1:
        raise ValueError(
            "context.max_input_tokens must exceed 500 tokens for map-reduce."
        )

    chunks = split_text_by_tokens(transcript, chunk_budget, _llm_tokenizer)
    if len(chunks) <= 1:
        raise ValueError(
            "Transcript exceeds the prompt budget but cannot be split with "
            "the configured map-reduce token budget."
        )

    logger.info("Notes map-reduce: summarizing %d transcript groups", len(chunks))
    summaries = [
        _generate(
            notes_chunk_summary_prompt(chunk, lang_name),
            max_new_tokens=300,
            temperature=0.1,
        )
        for chunk in chunks
    ]

    reduction_budget = max(1, max_input - 1000)
    for _ in range(8):
        combined = "\n\n".join(summaries)
        if _count_tokens(combined) <= reduction_budget:
            return combined
        groups = split_text_by_tokens(combined, chunk_budget, _llm_tokenizer)
        summaries = [
            _generate(
                notes_chunk_summary_prompt(group, lang_name),
                max_new_tokens=300,
                temperature=0.1,
            )
            for group in groups
        ]

    raise RuntimeError(
        "Map-reduce summaries still exceed the configured prompt budget after "
        "8 reduction rounds; increase context.max_input_tokens or reduce the "
        "lecture length."
    )


def make_notes(transcript: str, lang: str = "hi") -> dict[str, Any]:
    """Return structured notes, rendering metadata, and measured latency."""
    if not transcript.strip():
        raise ValueError("Cannot create notes from an empty transcript.")
    if is_mock_mode():
        mock_data = MOCK_NOTES_DATA_EN if lang == "en" else MOCK_NOTES_DATA
        result = dict(mock_data)
        result["key_points"] = list(mock_data["key_points"])
        result["key_terms"] = [dict(term) for term in mock_data["key_terms"]]
        result.update(
            markdown=notes_to_markdown(result),
            latency_s=0.01,
            used_map_reduce=False,
        )
        return result

    cfg = load_config()
    lang_name = cfg.get("supported_languages", {}).get(lang, lang)
    context_cfg = cfg.get("context", {})
    max_input_tokens = int(context_cfg.get("max_input_tokens", 6000))
    _load_llm()

    with timer("Notes generation", logger) as timing:
        if _count_tokens(transcript) > max_input_tokens - 1000:
            reduced_transcript = _reduce_transcript(transcript, lang_name, cfg)
            prompt = notes_combine_prompt(reduced_transcript, lang_name)
            used_map_reduce = True
        else:
            prompt = notes_prompt(transcript, lang_name)
            used_map_reduce = False

        raw_output = _generate(
            prompt,
            max_new_tokens=cfg.get("generation", {}).get(
                "max_new_tokens_notes", 800
            ),
            temperature=min(
                0.3,
                float(cfg.get("generation", {}).get("temperature", 0.2)),
            ),
        )
        try:
            document = NotesDocument.model_validate(safe_parse_json(raw_output))
        except Exception as exc:
            raise ValueError(f"Param-1 notes failed structured validation: {exc}") from exc

    result = document.model_dump()
    result.update(
        markdown=notes_to_markdown(result),
        latency_s=timing.elapsed,
        used_map_reduce=used_map_reduce,
    )
    return result
