"""
src/asr.py
----------
ASR wrapper for AI4Bharat IndicConformer (600 M multilingual).

Model card  : https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual
Fallback    : bharatgenai/Shrutam-2  (see docs/MODEL_NOTES.md)

Usage
-----
    from src.asr import transcribe
    result = transcribe("lecture.wav", lang="hi")
    text = result["full_text"]

The function is lazy: the model is loaded on first call and optionally freed
after use (config: memory.free_gpu_between_stages).
"""
from __future__ import annotations

import gc
import os
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import AudioChunk, load_and_chunk
from .utils import get_logger, is_mock_mode, load_config, timer

logger = get_logger(__name__)

# ── MOCK canned output ────────────────────────────────────────────────────────

MOCK_TRANSCRIPT = (
    "नमस्ते, आज हम न्यूटन के गति के नियमों के बारे में पढ़ेंगे। "
    "न्यूटन का पहला नियम कहता है कि कोई वस्तु तब तक विरामावस्था में रहती है "
    "जब तक उस पर कोई बाहरी बल न लगे। "
    "दूसरा नियम बताता है कि बल, द्रव्यमान और त्वरण का गुणनफल होता है। "
    "तीसरा नियम: प्रत्येक क्रिया के लिए समान और विपरीत प्रतिक्रिया होती है। "
    "इन नियमों का उपयोग रॉकेट विज्ञान, वाहन डिज़ाइन और खेल-कूद में होता है।"
)

MOCK_CHUNKS: list[dict] = [
    {
        "index": 0,
        "start_s": 0.0,
        "end_s": 30.0,
        "text": MOCK_TRANSCRIPT,
    }
]


# ── Model state ───────────────────────────────────────────────────────────────

_asr_model = None
_asr_tokenizer = None
_asr_backend: str = "none"


def _load_model() -> None:
    """Lazy-load the verified IndicConformer checkpoint or BharatGen fallback."""
    global _asr_model, _asr_tokenizer, _asr_backend

    if _asr_model is not None:
        return

    cfg = load_config()
    hf_token = os.getenv("HF_TOKEN")
    model_id: str = cfg["models"]["asr_primary"]

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading ASR model '%s' on %s …", model_id, device)

    from transformers import AutoModel

    primary_error: Exception | None = None
    try:
        model = AutoModel.from_pretrained(
            model_id, token=hf_token, trust_remote_code=True
        )
        model.eval()
        _asr_model = model
        _asr_backend = "indicconformer"
        if device == "cuda":
            try:
                import onnxruntime
            except ImportError as exc:
                raise RuntimeError(
                    "IndicConformer requires ONNX Runtime. Install "
                    "`onnxruntime-gpu==1.20.1` for CUDA or "
                    "`onnxruntime==1.20.1` for CPU."
                ) from exc
            if "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
                logger.warning(
                    "CUDA is available in PyTorch, but ONNX Runtime has no "
                    "CUDAExecutionProvider; inference will use its CPU provider."
                )
        logger.info("ASR backend: IndicConformer AutoModel  ✓")
        return
    except Exception as exc:
        primary_error = exc
        _asr_model = None
        _asr_tokenizer = None
        _asr_backend = "none"
        logger.warning(
            "IndicConformer load failed (%s). Trying BharatGen fallback …",
            primary_error,
        )

    fallback_id: str = cfg["models"].get("asr_fallback", "")
    try:
        from transformers import AutoTokenizer

        model = AutoModel.from_pretrained(
            fallback_id, token=hf_token, trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(fallback_id, token=hf_token)
        model.to(device)
        model.eval()
        _asr_model = model
        _asr_tokenizer = tokenizer
        _asr_backend = "shrutam2"
        logger.warning("ASR backend: BharatGen fallback '%s'", fallback_id)
    except Exception as fallback_error:
        raise RuntimeError(
            f"Unable to load primary ASR model '{model_id}': {primary_error}. "
            f"Unable to load BharatGen fallback '{fallback_id}': {fallback_error}"
        ) from fallback_error


def free_gpu_memory() -> None:
    """Unload the ASR model and release cached GPU memory."""
    global _asr_model, _asr_tokenizer, _asr_backend
    _asr_model = None
    _asr_tokenizer = None
    _asr_backend = "none"
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    logger.info("ASR model freed from memory.")


# ── Transcription ─────────────────────────────────────────────────────────────

@dataclass
class TranscriptChunk:
    """Transcript of a single audio chunk with timing."""
    index: int
    start_s: float
    end_s: float
    text: str


@dataclass
class TranscriptResult:
    """Full transcript plus per-chunk detail."""
    full_text: str
    chunks: list[TranscriptChunk]
    latency_s: float

    def to_dict(self) -> dict:
        return {
            "full_text": self.full_text,
            "chunks": [
                {
                    "index": chunk.index,
                    "start_s": chunk.start_s,
                    "end_s": chunk.end_s,
                    "text": chunk.text,
                }
                for chunk in self.chunks
            ],
            "latency_s": self.latency_s,
        }


def transcribe(audio_path: str | Path, lang: str = "hi") -> dict:
    """
    Transcribe an audio file using IndicConformer.

    Parameters
    ----------
    audio_path : str or Path
        Path to the audio file.
    lang : str
        BCP-47 language code (default: "hi" for Hindi).

    Returns
    -------
    dict
        Full transcript, per-chunk text and timestamps, and ASR latency.
    """
    if is_mock_mode():
        logger.info("MOCK MODE: returning canned transcript")
        return TranscriptResult(
            full_text=MOCK_TRANSCRIPT,
            chunks=[TranscriptChunk(**chunk) for chunk in MOCK_CHUNKS],
            latency_s=0.01,
        ).to_dict()

    cfg = load_config()
    supported_languages = cfg.get("asr_supported_languages", [])
    if supported_languages and lang not in supported_languages:
        raise ValueError(
            f"IndicConformer does not list language code '{lang}'. "
            f"Choose one of: {', '.join(supported_languages)}."
        )
    free_gpu = cfg.get("memory", {}).get("free_gpu_between_stages", True)
    transcript_chunks: list[TranscriptChunk] = []

    try:
        with timer("ASR stage", logger) as t:
            audio_info = load_and_chunk(audio_path)
            logger.info(
                "Audio loaded: %d chunks, %.1f s total",
                audio_info.n_chunks,
                audio_info.duration_s,
            )
            _load_model()

            for chunk in audio_info.chunks:
                text = _transcribe_chunk(chunk, lang)
                transcript_chunks.append(
                    TranscriptChunk(
                        index=chunk.index,
                        start_s=chunk.start_s,
                        end_s=chunk.end_s,
                        text=text,
                    )
                )
                logger.info(
                    "  Chunk %d [%.1f–%.1f s]: %s…",
                    chunk.index,
                    chunk.start_s,
                    chunk.end_s,
                    text[:80],
                )
    finally:
        if free_gpu:
            free_gpu_memory()

    full_text = " ".join(
        chunk.text.strip() for chunk in transcript_chunks if chunk.text.strip()
    )
    return TranscriptResult(
        full_text=full_text,
        chunks=transcript_chunks,
        latency_s=t.elapsed,
    ).to_dict()


def _transcribe_chunk(chunk: AudioChunk, lang: str) -> str:
    """Transcribe one audio chunk using the loaded backend."""
    if _asr_backend == "indicconformer":
        import torch

        waveform = torch.from_numpy(chunk.audio).unsqueeze(0)
        with torch.inference_mode():
            text = _asr_model(waveform, lang, "ctc")
        if not isinstance(text, str):
            raise TypeError(
                f"IndicConformer returned {type(text).__name__}; expected transcript text."
            )
        return text.strip()

    if _asr_backend == "shrutam2":
        lang_name = load_config().get("supported_languages", {}).get(lang)
        if not lang_name:
            raise ValueError(
                f"Language code '{lang}' has no configured display name for Shrutam-2."
            )
        return _transcribe_shrutam_chunk(chunk, lang_name)

    raise RuntimeError(f"Unknown ASR backend: {_asr_backend}")


def _transcribe_shrutam_chunk(chunk: AudioChunk, lang_name: str) -> str:
    """Run the documented Shrutam-2 file-based API on one timestamped chunk."""
    audio_int16 = np.clip(chunk.audio * 32767, -32768, 32767).astype(np.int16)
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
            temp_path = audio_file.name
        with wave.open(temp_path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(chunk.sample_rate)
            wav_file.writeframes(audio_int16.tobytes())

        transcript = _asr_model.transcribe(
            temp_path,
            prompts=f"Transcribe speech to {lang_name} text.",
            tokenizer=_asr_tokenizer,
        )
        if not isinstance(transcript, str):
            raise TypeError(
                f"Shrutam-2 returned {type(transcript).__name__}; expected transcript text."
            )
        return transcript.strip()
    finally:
        if temp_path is not None:
            Path(temp_path).unlink(missing_ok=True)
