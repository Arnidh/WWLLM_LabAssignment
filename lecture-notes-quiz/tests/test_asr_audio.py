from __future__ import annotations

import wave

import numpy as np
import pytest
from pydub import AudioSegment

from src import audio


def _write_wav(
    path,
    sample_rate: int,
    duration_s: int,
    channels: int = 1,
    speech_ranges: tuple[tuple[int, int], ...] = (),
) -> None:
    samples = np.zeros(sample_rate * duration_s, dtype=np.int16)
    for start_s, end_s in speech_ranges:
        samples[start_s * sample_rate : end_s * sample_rate] = 9000
    if channels == 2:
        samples = np.column_stack((samples, samples))

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.tobytes())


def test_fixed_chunks_resample_to_mono_and_keep_timestamps(
    tmp_path, monkeypatch
):
    audio_path = tmp_path / "lecture.wav"
    _write_wav(audio_path, sample_rate=8000, duration_s=5, channels=2)
    monkeypatch.setattr(audio, "is_mock_mode", lambda: False)
    monkeypatch.setattr(
        audio,
        "load_config",
        lambda: {
            "audio": {
                "sample_rate": 16000,
                "chunk_length_s": 2,
                "overlap_s": 0.25,
                "silence_split": False,
            }
        },
    )

    result = audio.load_and_chunk(audio_path)

    assert result.original_sr == 8000
    assert result.n_chunks == 3
    assert [(chunk.start_s, chunk.end_s) for chunk in result.chunks] == [
        (0.0, 2.0),
        (1.75, 3.75),
        (3.5, 5.0),
    ]
    assert all(chunk.sample_rate == 16000 for chunk in result.chunks)
    assert all(chunk.audio.dtype == np.float32 for chunk in result.chunks)
    assert all(chunk.audio.ndim == 1 for chunk in result.chunks)
    assert len(result.chunks[0].audio) == 32000


def test_silence_chunks_keep_source_timestamps(tmp_path, monkeypatch):
    audio_path = tmp_path / "lecture.wav"
    _write_wav(
        audio_path,
        sample_rate=16000,
        duration_s=8,
        speech_ranges=((1, 2), (5, 6)),
    )
    monkeypatch.setattr(audio, "is_mock_mode", lambda: False)
    monkeypatch.setattr(
        audio,
        "load_config",
        lambda: {
            "audio": {
                "sample_rate": 16000,
                "chunk_length_s": 3,
                "overlap_s": 0.5,
                "silence_split": True,
                "silence_thresh_dbfs": -40,
            }
        },
    )

    result = audio.load_and_chunk(audio_path)

    assert result.n_chunks == 2
    assert result.chunks[0].start_s < 1.0
    assert 1.9 <= result.chunks[0].end_s <= 2.3
    assert 4.7 <= result.chunks[1].start_s <= 5.0
    assert 5.9 <= result.chunks[1].end_s <= 6.3


def test_missing_ffmpeg_has_actionable_decode_error(tmp_path, monkeypatch):
    audio_path = tmp_path / "lecture.mp3"
    audio_path.touch()
    monkeypatch.setattr(audio, "is_mock_mode", lambda: False)
    monkeypatch.setattr(audio, "load_config", lambda: {"audio": {}})

    def raise_missing_decoder(*_args, **_kwargs):
        raise FileNotFoundError(2, "not found")

    monkeypatch.setattr(
        AudioSegment,
        "from_file",
        raise_missing_decoder,
    )
    monkeypatch.setattr(audio.shutil, "which", lambda _command: None)

    with pytest.raises(RuntimeError, match="FFmpeg is not installed or is not on PATH"):
        audio.load_and_chunk(audio_path)
