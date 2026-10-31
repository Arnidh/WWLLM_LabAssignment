"""
src/translate.py
----------------
Translation wrapper using AI4Bharat IndicTrans2 (distilled 200 M).

Repo     : https://github.com/AI4Bharat/IndicTrans2
Toolkit  : https://github.com/VarunGumma/IndicTransToolkit
HF hubs  :
  indic→en   : ai4bharat/indictrans2-indic-en-dist-200M
  en→indic   : ai4bharat/indictrans2-en-indic-dist-200M
  indic→indic: ai4bharat/indictrans2-indic-indic-dist-320M

Language code format: Flores-200 codes (e.g. "hin_Deva", "kan_Knda", "eng_Latn").
See docs/MODEL_NOTES.md for the full mapping.
"""
from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from typing import Optional

from .utils import get_logger, is_mock_mode, load_config, timer

logger = get_logger(__name__)

# ── Language code mapping (BCP-47 → Flores-200) ───────────────────────────────
# IndicTrans2 uses Flores-200 script codes, not BCP-47.
# Extend this dict as you add more languages.

BCP47_TO_FLORES: dict[str, str] = {
    "hi": "hin_Deva",
    "en": "eng_Latn",
    "kn": "kan_Knda",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "mr": "mar_Deva",
    "bn": "ben_Beng",
    "gu": "guj_Gujr",
    "pa": "pan_Guru",
    "or": "ory_Orya",
    "ml": "mal_Mlym",
}


def _to_flores(lang: str) -> str:
    """Convert a BCP-47 code to Flores-200; raise ValueError if unknown."""
    if lang in BCP47_TO_FLORES:
        return BCP47_TO_FLORES[lang]
    # Maybe already a Flores code (e.g., "hin_Deva")
    if "_" in lang and len(lang) > 5:
        return lang
    raise ValueError(
        f"Unknown language code '{lang}'. "
        f"Add it to BCP47_TO_FLORES in src/translate.py."
    )


# ── MOCK canned output ────────────────────────────────────────────────────────

MOCK_TRANSLATION = (
    "Newton's first law states that an object remains at rest unless an external "
    "force acts on it. Newton's second law states that force equals mass times "
    "acceleration. Newton's third law states that every action has an equal and "
    "opposite reaction. These laws are used in rocket science. They also apply "
    "to vehicle design and sports."
)

# ── Model state ───────────────────────────────────────────────────────────────

_trans_model = None
_trans_tokenizer = None
_trans_ip = None          # IndicProcessor from IndicTransToolkit
_trans_direction: str = "none"   # "indic_en" | "en_indic" | "indic_indic"


def _load_translator(src_lang: str, tgt_lang: str) -> str:
    """
    Lazy-load the appropriate IndicTrans2 checkpoint.
    Returns the direction string.
    """
    global _trans_model, _trans_tokenizer, _trans_ip, _trans_direction

    en_flores = BCP47_TO_FLORES.get("en", "eng_Latn")
    src_flores = _to_flores(src_lang)
    tgt_flores = _to_flores(tgt_lang)

    # Determine direction
    if tgt_flores == en_flores:
        direction = "indic_en"
    elif src_flores == en_flores:
        direction = "en_indic"
    else:
        direction = "indic_indic"

    if _trans_model is not None and _trans_direction == direction:
        return direction  # already loaded correct checkpoint

    # Free old model if direction changed
    if _trans_model is not None:
        _free_translator()

    cfg = load_config()
    direction_key_map = {
        "indic_en":    "translation_to_en",
        "en_indic":    "translation_to_indic",
        "indic_indic": "translation_indic_indic",
    }
    model_id: str = cfg["models"][direction_key_map[direction]]
    hf_token = os.getenv("HF_TOKEN")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading IndicTrans2 '%s' (%s) on %s …", model_id, direction, device)

    # ── Try IndicTransToolkit (recommended) ───────────────────────────────
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from IndicTransToolkit import IndicProcessor  # type: ignore[import]

        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, token=hf_token
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            token=hf_token,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        model.eval()
        ip = IndicProcessor(inference=True)

        _trans_tokenizer = tokenizer
        _trans_model = model
        _trans_ip = ip
        _trans_direction = direction
        logger.info("IndicTrans2 loaded via IndicTransToolkit  ✓")
        return direction

    except ImportError:
        logger.warning(
            "IndicTransToolkit not installed. "
            "Install with: pip install git+https://github.com/VarunGumma/IndicTransToolkit.git"
        )
    except Exception as exc:
        logger.error("IndicTrans2 load failed: %s", exc)
        raise


def _free_translator() -> None:
    global _trans_model, _trans_tokenizer, _trans_ip, _trans_direction
    _trans_model = None
    _trans_tokenizer = None
    _trans_ip = None
    _trans_direction = "none"
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    logger.info("Translator freed.")


# ── Sentence splitting ────────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """
    Simple sentence splitter suitable for Indic + Latin scripts.
    Splits on '.', '।', '?', '!' followed by whitespace.
    """
    import re
    sentences = re.split(r"(?<=[.।?!])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


# ── Public API ────────────────────────────────────────────────────────────────

@dataclass
class TranslationResult:
    translated_text: str
    src_lang: str
    tgt_lang: str
    latency_s: float


def translate(text: str, src_lang: str, tgt_lang: str) -> TranslationResult:
    """
    Translate `text` from `src_lang` to `tgt_lang` using IndicTrans2.

    Parameters
    ----------
    text : str
    Input text, usually a lecture transcript.
    src_lang : str
        BCP-47 source language code (e.g., "hi").
    tgt_lang : str
        BCP-47 target language code (e.g., "en").

    Returns
    -------
    TranslationResult
        Translated text + metadata.
    """
    if is_mock_mode():
        logger.info("MOCK MODE: returning canned translation (→ English)")
        # Identity: if same language, return input unchanged even in MOCK mode
        if src_lang == tgt_lang:
            return TranslationResult(
                translated_text=text,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                latency_s=0.0,
            )
        return TranslationResult(
            translated_text=MOCK_TRANSLATION,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            latency_s=0.01,
        )

    if src_lang == tgt_lang:
        return TranslationResult(
            translated_text=text,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            latency_s=0.0,
        )

    cfg = load_config()
    free_gpu: bool = cfg.get("memory", {}).get("free_gpu_between_stages", True)

    src_flores = _to_flores(src_lang)
    tgt_flores = _to_flores(tgt_lang)

    direction = _load_translator(src_lang, tgt_lang)

    with timer(f"Translation {src_lang}→{tgt_lang}", logger) as t:
        sentences = _split_sentences(text)
        translated_sentences = _translate_batch(sentences, src_flores, tgt_flores)
        translated_text = " ".join(translated_sentences)

    if free_gpu:
        _free_translator()

    return TranslationResult(
        translated_text=translated_text,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        latency_s=t.elapsed,
    )


def _translate_batch(
    sentences: list[str],
    src_flores: str,
    tgt_flores: str,
    batch_size: int = 8,
) -> list[str]:
    """Run IndicTrans2 on batches of sentences."""
    import torch

    all_translations: list[str] = []

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        logger.info("Translating sentences %d–%d …", i + 1, i + len(batch))

        # Pre-process with IndicProcessor
        processed = _trans_ip.preprocess_batch(
            batch,
            src_lang=src_flores,
            tgt_lang=tgt_flores,
        )

        inputs = _trans_tokenizer(
            processed,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            max_length=256,
        )

        device = next(_trans_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = _trans_model.generate(
                **inputs,
                num_beams=5,
                num_return_sequences=1,
                max_length=256,
            )

        decoded = _trans_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        postprocessed = _trans_ip.postprocess_batch(decoded, lang=tgt_flores)
        all_translations.extend(postprocessed)

    return all_translations
