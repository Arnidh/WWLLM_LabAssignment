"""Prepare a small, reproducible FLEURS read-speech ASR benchmark."""
from __future__ import annotations

import argparse
import csv
import io
import logging
import random
import shutil
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

DATASET_ID = "google/fleurs"
DATASET_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
DATASET_LICENSE = "CC-BY-4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
LANGUAGE_CONFIGS = {"hi": "hi_in", "kn": "kn_in", "ta": "ta_in"}
MANIFEST_FIELDS = [
    "id",
    "audio_file",
    "language",
    "test_type",
    "reference_text",
    "notes",
    "dataset_id",
    "dataset_config",
    "dataset_split",
    "dataset_revision",
    "dataset_license",
    "source_id",
    "gender",
]
LOGGER = logging.getLogger("prepare_fleurs_eval")


@dataclass(frozen=True)
class FleursSample:
    source_id: str
    filename: str
    reference_text: str
    gender: str


def parse_test_tsv(content: str) -> list[FleursSample]:
    samples: list[FleursSample] = []
    reader = csv.reader(
        io.StringIO(content), delimiter="\t", quoting=csv.QUOTE_NONE
    )
    for line_number, row in enumerate(reader, 1):
        if not row or (row[0].strip().lower() == "id" and line_number == 1):
            continue
        if len(row) < 7:
            raise ValueError(f"Malformed FLEURS TSV row {line_number}: expected 7 columns.")

        filename = row[1].strip()
        if (
            PurePosixPath(filename).name != filename
            or PureWindowsPath(filename).name != filename
            or not filename.lower().endswith(".wav")
        ):
            raise ValueError(f"Unsafe or invalid audio filename on TSV row {line_number}.")

        source_id = row[0].strip()
        reference_text = row[2].strip()
        if not source_id or not reference_text:
            raise ValueError(f"Missing ID or reference transcript on TSV row {line_number}.")
        samples.append(
            FleursSample(
                source_id=source_id,
                filename=filename,
                reference_text=reference_text,
                gender=row[6].strip(),
            )
        )
    if not samples:
        raise ValueError("The FLEURS TSV did not contain any test samples.")
    return samples


def select_samples(
    samples: list[FleursSample], count: int, seed: int
) -> list[FleursSample]:
    if count < 1:
        raise ValueError("The per-language sample count must be at least 1.")
    if count > len(samples):
        raise ValueError(f"Requested {count} samples, but only {len(samples)} are available.")
    selected_indexes = set(random.Random(seed).sample(range(len(samples)), count))
    return [sample for index, sample in enumerate(samples) if index in selected_indexes]


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url, headers={"User-Agent": "lecture-notes-quiz-fleurs-eval/1.0"}
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(request, timeout=120) as response:
        with destination.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)


def _extract_selected_audio(
    archive_path: Path,
    output_dir: Path,
    samples: list[FleursSample],
    language_config: str,
) -> dict[str, str]:
    wanted = {sample.filename: sample for sample in samples}
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted: dict[str, str] = {}
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            member_name = member.name.replace("\\", "/")
            filename = PurePosixPath(member_name).name
            if filename not in wanted:
                continue
            if filename in extracted:
                raise ValueError(f"Duplicate audio filename in dataset archive: {filename}")
            source = archive.extractfile(member)
            if source is None:
                raise OSError(f"Could not read dataset archive member: {member.name}")
            target_name = f"fleurs_{language_config}_{filename}"
            target_path = output_dir / target_name
            with source, target_path.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            extracted[filename] = target_name

    missing = sorted(set(wanted) - set(extracted))
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} selected audio file(s) were missing from the archive: "
            + ", ".join(missing[:5])
        )
    return extracted


def prepare_dataset(root: Path, per_language: int, seed: int) -> int:
    root = root.resolve()
    audio_dir = root / "data" / "audio"
    manifest_path = root / "data" / "test_cases.csv"
    rows: list[dict[str, str]] = []
    base_url = (
        f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{DATASET_REVISION}"
    )

    for language, config in LANGUAGE_CONFIGS.items():
        with tempfile.TemporaryDirectory(prefix=f"fleurs-{config}-") as temporary_dir:
            temporary_path = Path(temporary_dir)
            tsv_path = temporary_path / "test.tsv"
            _download(f"{base_url}/data/{config}/test.tsv", tsv_path)
            samples = select_samples(
                parse_test_tsv(tsv_path.read_text(encoding="utf-8")),
                per_language,
                seed,
            )

            missing_samples = [
                sample
                for sample in samples
                if not (audio_dir / f"fleurs_{config}_{sample.filename}").is_file()
            ]
            if missing_samples:
                archive_path = temporary_path / "test.tar.gz"
                LOGGER.info("Downloading the %s test audio archive.", config)
                _download(
                    f"{base_url}/data/{config}/audio/test.tar.gz", archive_path
                )
                extracted = _extract_selected_audio(
                    archive_path, audio_dir, missing_samples, config
                )
            else:
                extracted = {sample.filename: f"fleurs_{config}_{sample.filename}" for sample in samples}

            for sample in samples:
                rows.append(
                    {
                        "id": f"fleurs_{language}_{sample.source_id}",
                        "audio_file": (
                            Path("data") / "audio" / extracted[sample.filename]
                        ).as_posix(),
                        "language": language,
                        "test_type": "read_speech",
                        "reference_text": sample.reference_text,
                        "notes": "",
                        "dataset_id": DATASET_ID,
                        "dataset_config": config,
                        "dataset_split": "test",
                        "dataset_revision": DATASET_REVISION,
                        "dataset_license": DATASET_LICENSE,
                        "source_id": sample.source_id,
                        "gender": sample.gender,
                    }
                )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = manifest_path.with_suffix(".csv.tmp")
    try:
        with temporary_manifest.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        temporary_manifest.replace(manifest_path)
    finally:
        temporary_manifest.unlink(missing_ok=True)

    LOGGER.info(
        "Prepared %d licensed FLEURS test clips in %s.",
        len(rows),
        manifest_path,
    )
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a deterministic Hindi/Kannada/Tamil FLEURS test subset."
    )
    parser.add_argument(
        "--per-language",
        type=int,
        default=10,
        help="Number of read-speech examples per language (default: 10).",
    )
    parser.add_argument("--seed", type=int, default=3235)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    count = prepare_dataset(args.project_root, args.per_language, args.seed)
    print(
        f"Prepared {count} examples. FLEURS is CC-BY-4.0 read speech; "
        "it is not a lecture/noisy-speech benchmark."
    )


if __name__ == "__main__":
    main()
