"""
src/audio.py
------------
Audio loading, resampling, and chunking utilities.

Accepts: wav, mp3, m4a, flac (via pydub/ffmpeg).
Output: 16 kHz mono numpy arrays (one per chunk) plus timestamps.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .utils import get_logger, is_mock_mode, load_config

logger = get_logger(__name__)

# ── Canned MOCK output ────────────────────────────────────────────────────────

MOCK_CHUNK = np.zeros(16000, dtype=np.float32)  # 1 second of silence


@dataclass
class AudioChunk:
    """A single audio chunk ready for the ASR model."""
    index: int
    start_s: float          # start time in seconds (in the original file)
    end_s: float            # end time in seconds
    audio: np.ndarray       # float32, 16 kHz, mono
    sample_rate: int = 16000


@dataclass
class AudioInfo:
    """Metadata about the source audio file."""
    path: str
    duration_s: float
    original_sr: int
    n_chunks: int
    chunks: list[AudioChunk] = field(default_factory=list)


# ── Public API ────────────────────────────────────────────────────────────────

def load_and_chunk(audio_path: str | Path) -> AudioInfo:
    """
    Load an audio file, resample to 16 kHz mono, split into chunks.

    Splitting strategy (configurable in config.yaml):
      1. Silence-based splitting via pydub (preferred – avoids cutting mid-word).
      2. Fixed-length chunking with overlap as fallback.

    Parameters
    ----------
    audio_path : str or Path
        Path to the audio file (.wav, .mp3, .m4a, .flac).

    Returns
    -------
    AudioInfo
        Metadata plus a list of AudioChunk objects.
    """
    if is_mock_mode():
        logger.info("MOCK MODE: returning fake audio chunks")
        chunk = AudioChunk(index=0, start_s=0.0, end_s=30.0, audio=MOCK_CHUNK)
        return AudioInfo(
            path=str(audio_path),
            duration_s=30.0,
            original_sr=16000,
            n_chunks=1,
            chunks=[chunk],
        )

    cfg = load_config()
    audio_cfg = cfg.get("audio", {})
    target_sr: int = audio_cfg.get("sample_rate", 16000)
    chunk_len_s: float = float(audio_cfg.get("chunk_length_s", 30))
    overlap_s: float = float(audio_cfg.get("overlap_s", 1))
    use_silence_split: bool = audio_cfg.get("silence_split", True)
    silence_thresh: float = float(audio_cfg.get("silence_thresh_dbfs", -40))

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # ── Load with pydub (handles mp3/m4a/flac transparently via ffmpeg) ───
    try:
        from pydub import AudioSegment
        from pydub.silence import detect_nonsilent
    except ImportError as exc:
        raise RuntimeError(
            "pydub is required for audio loading. Install with: pip install pydub"
        ) from exc

    logger.info("Loading audio: %s", audio_path)
    try:
        seg: AudioSegment = AudioSegment.from_file(str(audio_path))
    except FileNotFoundError as exc:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                f"Cannot decode '{audio_path}': FFmpeg is not installed or is "
                "not on PATH. Install it with `winget install --id "
                "Gyan.FFmpeg -e`, restart the terminal and app, then retry."
            ) from exc
        raise
    original_sr = seg.frame_rate
    duration_s = len(seg) / 1000.0
    logger.info("Duration: %.1f s, original SR: %d Hz", duration_s, original_sr)

    if not len(seg):
        raise ValueError(f"Audio file has no samples: {audio_path}")
    if target_sr <= 0:
        raise ValueError("audio.sample_rate must be greater than zero.")

    # Resample + mono
    seg = seg.set_frame_rate(target_sr).set_channels(1).set_sample_width(2)

    if chunk_len_s <= 0:
        raise ValueError("audio.chunk_length_s must be greater than zero.")
    if overlap_s < 0 or overlap_s >= chunk_len_s:
        raise ValueError(
            "audio.overlap_s must be non-negative and shorter than chunk_length_s."
        )

    chunks: list[AudioChunk] = []

    if use_silence_split:
        ranges = detect_nonsilent(
            seg,
            min_silence_len=500,
            silence_thresh=silence_thresh,
            seek_step=10,
        )
        if ranges:
            chunks = _silence_chunks(
                seg, ranges, chunk_len_s, overlap_s, target_sr
            )
            logger.info("Silence-based split → %d chunks", len(chunks))
        else:
            logger.warning("No speech regions found; using fixed-length chunking")
            chunks = _fixed_chunks(seg, chunk_len_s, overlap_s, target_sr)
    else:
        chunks = _fixed_chunks(seg, chunk_len_s, overlap_s, target_sr)

    return AudioInfo(
        path=str(audio_path),
        duration_s=duration_s,
        original_sr=original_sr,
        n_chunks=len(chunks),
        chunks=chunks,
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _seg_to_numpy(seg) -> np.ndarray:
    """Convert a pydub AudioSegment (mono, 16 kHz) to float32 numpy array."""
    samples = np.frombuffer(seg.raw_data, dtype=np.int16).astype(np.float32)
    return samples / 32768.0  # normalise to [-1, 1]


def _fixed_chunks(
    seg,
    chunk_len_s: float,
    overlap_s: float,
    target_sr: int,
) -> list[AudioChunk]:
    """Split audio into fixed-length chunks with optional overlap."""
    chunk_ms = int(chunk_len_s * 1000)
    overlap_ms = int(overlap_s * 1000)
    return _chunks_for_range(
        seg, 0, len(seg), chunk_ms, overlap_ms, target_sr, 0
    )


def _chunks_for_range(
    seg,
    start_ms: int,
    end_ms: int,
    chunk_ms: int,
    overlap_ms: int,
    target_sr: int,
    first_index: int,
) -> list[AudioChunk]:
    chunks: list[AudioChunk] = []
    step_ms = chunk_ms - overlap_ms
    chunk_start_ms = start_ms

    while chunk_start_ms < end_ms:
        chunk_end_ms = min(chunk_start_ms + chunk_ms, end_ms)
        chunks.append(
            AudioChunk(
                index=first_index + len(chunks),
                start_s=chunk_start_ms / 1000.0,
                end_s=chunk_end_ms / 1000.0,
                audio=_seg_to_numpy(seg[chunk_start_ms:chunk_end_ms]),
                sample_rate=target_sr,
            )
        )
        if chunk_end_ms == end_ms:
            break
        chunk_start_ms += step_ms
    return chunks


def _silence_chunks(
    seg,
    speech_ranges: list[list[int]],
    chunk_len_s: float,
    overlap_s: float,
    target_sr: int,
) -> list[AudioChunk]:
    """Chunk detected speech ranges without losing their original timestamps."""
    chunk_ms = int(chunk_len_s * 1000)
    overlap_ms = int(overlap_s * 1000)
    padding_ms = 200
    padded_ranges = [
        (max(0, start - padding_ms), min(len(seg), end + padding_ms))
        for start, end in speech_ranges
    ]

    groups: list[tuple[int, int]] = []
    current_start, current_end = padded_ranges[0]
    for start_ms, end_ms in padded_ranges[1:]:
        if end_ms - current_start <= chunk_ms:
            current_end = end_ms
        else:
            groups.append((current_start, current_end))
            current_start, current_end = start_ms, end_ms
    groups.append((current_start, current_end))

    chunks: list[AudioChunk] = []
    for start_ms, end_ms in groups:
        chunks.extend(
            _chunks_for_range(
                seg,
                start_ms,
                end_ms,
                chunk_ms,
                overlap_ms,
                target_sr,
                len(chunks),
            )
        )
    return chunks
