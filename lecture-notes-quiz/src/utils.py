"""
src/utils.py
------------
Shared utilities: config loading, logging helpers, JSON repair/validation.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=False)
load_dotenv(_PROJECT_ROOT.parent / ".env", override=False)

# ── Logging setup ─────────────────────────────────────────────────────────────

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger with the shared format."""
    return logging.getLogger(name)


# ── Config loading ────────────────────────────────────────────────────────────

_CONFIG_CACHE: dict[str, Any] | None = None
_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def load_config(path: Path | None = None) -> dict[str, Any]:
    """
    Load config.yaml (cached after first call).
    Environment variables can override selected keys at runtime.
    """
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    cfg_path = path or _CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    # Allow env-var overrides for common settings
    if os.getenv("LLM_QUANTIZATION"):
        cfg.setdefault("memory", {})["llm_quantization"] = os.getenv("LLM_QUANTIZATION")
    if os.getenv("DEFAULT_LANGUAGE"):
        cfg["default_language"] = os.getenv("DEFAULT_LANGUAGE")

    _CONFIG_CACHE = cfg
    return cfg


def is_mock_mode() -> bool:
    """Return True when MOCK_MODELS=1 is set in the environment."""
    return os.getenv("MOCK_MODELS", "0").strip() == "1"


# ── Latency timing ────────────────────────────────────────────────────────────

@contextmanager
def timer(label: str, logger: logging.Logger | None = None):
    """
    Context manager that measures wall-clock time and logs it.

    Usage::
        with timer("ASR") as t:
            result = transcribe(...)
        print(t.elapsed)  # seconds
    """
    log = logger or get_logger("timer")
    start = time.perf_counter()

    class _T:
        elapsed: float = 0.0

    t = _T()
    try:
        yield t
    finally:
        t.elapsed = time.perf_counter() - start
        log.info("⏱  %s completed in %.2f s", label, t.elapsed)


# ── JSON utilities ────────────────────────────────────────────────────────────

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def extract_json_block(text: str) -> str:
    """
    Pull a JSON object/array out of a model response that may contain
    markdown fences or extra prose around the JSON.
    """
    # Try fenced block first
    m = _JSON_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()

    starts = [index for index in (text.find("{"), text.find("[")) if index >= 0]
    if starts:
        start = min(starts)
        pairs = {"{": "}", "[": "]"}
        stack: list[str] = []
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if escaped:
                escaped = False
                continue
            if char == "\\" and in_string:
                escaped = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char in pairs:
                stack.append(pairs[char])
            elif char in pairs.values() and stack and stack[-1] == char:
                stack.pop()
                if not stack:
                    return text[start : index + 1]
    return text.strip()


def safe_parse_json(text: str) -> Any:
    """
    Parse JSON from a model response, tolerating markdown fences.
    Raises json.JSONDecodeError on failure.
    """
    cleaned = extract_json_block(text)
    return json.loads(cleaned)


def repair_json_truncated(text: str) -> Any:
    """
    Best-effort repair for truncated JSON by closing open brackets.
    Returns parsed object or raises ValueError.
    """
    cleaned = extract_json_block(text)
    # Count unclosed brackets
    stack: list[str] = []
    pairs = {"{": "}", "[": "]"}
    closing = set(pairs.values())
    opening = set(pairs.keys())
    in_string = False
    escape_next = False

    for ch in cleaned:
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in opening:
            stack.append(pairs[ch])
        elif ch in closing:
            if stack and stack[-1] == ch:
                stack.pop()

    repaired = cleaned + "".join(reversed(stack))
    try:
        return json.loads(repaired)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON repair failed: {exc}") from exc


# ── Text normalisation (for WER/CER) ─────────────────────────────────────────

import unicodedata


def normalize_text(text: str) -> str:
    """
    Normalise text before WER/CER computation:
    1. Unicode NFC normalisation
    2. Strip punctuation while keeping Unicode letters, digits, whitespace, and
       combining marks used by Indic conjuncts and vowel signs.
    3. Collapse whitespace
    4. Lowercase

    Note: Python's \\w in re.UNICODE matches Unicode letters and digits, but the
    Devanagari halant (virama, U+094D) is a combining mark (category Mn), not a
    letter, so we preserve it explicitly to keep conjuncts intact.
    """
    text = unicodedata.normalize("NFC", text)
    # Preserve combining marks across Indic scripts so normalization does not
    # change Kannada/Tamil words while removing punctuation and symbols.
    text = "".join(
        char
        if char.isspace() or unicodedata.category(char)[0] in {"L", "N", "M"}
        else " "
        for char in text
    )
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text
