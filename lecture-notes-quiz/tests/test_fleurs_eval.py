from __future__ import annotations

import io
import csv
import tarfile

import pytest

from eval import run_asr_inference
from eval.prepare_fleurs_eval import (
    FleursSample,
    _extract_selected_audio,
    parse_test_tsv,
    select_samples,
)


def test_parse_fleurs_tsv_uses_original_reference_text():
    rows = (
        "1839\tclip.wav\tनमस्ते दुनिया। \"हैलो\"\tनमस्ते दुनिया\tन म स् ते\t16000\tFEMALE\n"
        "1840\tclip2.wav\tआज मौसम अच्छा है।\tआज मौसम अच्छा है\tआज\t12000\tMALE\n"
    )

    samples = parse_test_tsv(rows)

    assert [sample.source_id for sample in samples] == ["1839", "1840"]
    assert samples[0].reference_text == 'नमस्ते दुनिया। "हैलो"'
    assert samples[0].gender == "FEMALE"


def test_parse_fleurs_tsv_rejects_unsafe_audio_paths():
    with pytest.raises(ValueError, match="Unsafe or invalid"):
        parse_test_tsv("1\t..\\\\outside.wav\treference\traw\twords\t16000\tFEMALE\n")


def test_selection_is_seeded_and_rejects_impossible_count():
    samples = [
        FleursSample(str(index), f"{index}.wav", f"text {index}", "FEMALE")
        for index in range(20)
    ]

    assert select_samples(samples, 5, 3235) == select_samples(samples, 5, 3235)
    assert len(select_samples(samples, 5, 3235)) == 5
    with pytest.raises(ValueError, match="only 20 are available"):
        select_samples(samples, 21, 3235)


def test_archive_extraction_writes_only_selected_files_to_output_dir(tmp_path):
    archive_path = tmp_path / "test.tar.gz"
    with tarfile.open(archive_path, mode="w:gz") as archive:
        for name, contents in (
            ("nested/clip.wav", b"audio"),
            ("ignored.wav", b"not selected"),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(contents)
            archive.addfile(info, io.BytesIO(contents))

    output_dir = tmp_path / "audio"
    extracted = _extract_selected_audio(
        archive_path,
        output_dir,
        [FleursSample("1", "clip.wav", "reference", "FEMALE")],
        "hi_in",
    )

    assert extracted == {"clip.wav": "fleurs_hi_in_clip.wav"}
    assert (output_dir / "fleurs_hi_in_clip.wav").read_bytes() == b"audio"
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "fleurs_hi_in_clip.wav"
    ]


def test_real_inference_refuses_mock_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(run_asr_inference, "is_mock_mode", lambda: True)

    with pytest.raises(RuntimeError, match="Refusing to record MOCK output"):
        run_asr_inference.run_asr_inference(
            str(tmp_path / "manifest.csv"),
            str(tmp_path / "transcripts"),
            str(tmp_path / "metadata.json"),
        )


def test_real_inference_rejects_missing_audio_before_model_loading(
    tmp_path, monkeypatch
):
    manifest_path = tmp_path / "test_cases.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest:
        writer = csv.DictWriter(
            manifest,
            fieldnames=["id", "audio_file", "language", "reference_text"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "id": "sample-1",
                "audio_file": str(tmp_path / "missing.wav"),
                "language": "hi",
                "reference_text": "संदर्भ",
            }
        )
    monkeypatch.setattr(run_asr_inference, "is_mock_mode", lambda: False)

    with pytest.raises(FileNotFoundError, match="missing audio for sample-1"):
        run_asr_inference.run_asr_inference(
            str(manifest_path),
            str(tmp_path / "transcripts"),
            str(tmp_path / "metadata.json"),
        )
