from __future__ import annotations

import sys
import tarfile
from pathlib import Path

import httpx
import zstandard as zstd
from tqdm import tqdm

DEFAULT_DATA_URL = "https://s3.jbecker.dev/data.tar.zst"
DEFAULT_ARCHIVE_NAME = "data.tar.zst"
DEFAULT_SENTINEL_NAME = ".download_complete"


def _is_within_directory(base_dir: Path, target_path: Path) -> bool:
    """Return True when target_path resolves underneath base_dir."""
    try:
        target_path.resolve().relative_to(base_dir.resolve())
        return True
    except ValueError:
        return False


def download_data_archive(
    url: str = DEFAULT_DATA_URL,
    data_dir: Path = Path("data"),
    archive_name: str = DEFAULT_ARCHIVE_NAME,
) -> Path:
    """Download the compressed data archive to data_dir."""
    data_dir.mkdir(parents=True, exist_ok=True)

    archive_path = data_dir / archive_name
    temp_path = archive_path.with_name(f"{archive_path.name}.part")

    if archive_path.exists():
        print(f"Using existing archive at {archive_path}")
        return archive_path

    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=None) as response:
            response.raise_for_status()
            total_bytes = response.headers.get("Content-Length")
            total = int(total_bytes) if total_bytes is not None else None

            with (
                temp_path.open("wb") as handle,
                tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc="Downloading data",
                ) as progress,
            ):
                for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                    handle.write(chunk)
                    progress.update(len(chunk))
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    temp_path.replace(archive_path)
    return archive_path


def extract_data_archive(archive_path: Path, destination: Path = Path(".")) -> None:
    """Extract a .tar.zst archive using Python-only tooling."""
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    destination.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive_path}...")

    with archive_path.open("rb") as compressed_stream:
        decompressor = zstd.ZstdDecompressor()
        with decompressor.stream_reader(compressed_stream) as decompressed_stream:
            with tarfile.open(fileobj=decompressed_stream, mode="r|") as archive:
                for member in archive:
                    target_path = destination / member.name
                    if not _is_within_directory(destination, target_path):
                        raise ValueError(f"Refusing to extract unsafe archive member: {member.name}")
                    extract_kwargs = {"path": destination}
                    if sys.version_info >= (3, 12):
                        extract_kwargs["filter"] = "data"
                    archive.extract(member, **extract_kwargs)

    print("Extraction complete.")


def setup_data(
    url: str = DEFAULT_DATA_URL,
    data_dir: Path = Path("data"),
    archive_name: str = DEFAULT_ARCHIVE_NAME,
    sentinel_name: str = DEFAULT_SENTINEL_NAME,
    cleanup_archive: bool = True,
) -> bool:
    """Download and extract the bundled dataset."""
    sentinel_path = data_dir / sentinel_name
    if sentinel_path.exists():
        print("Data already downloaded and extracted, skipping.")
        return True

    archive_path: Path | None = None
    try:
        archive_path = download_data_archive(url=url, data_dir=data_dir, archive_name=archive_name)
        extract_data_archive(archive_path, destination=data_dir.parent)
        if cleanup_archive and archive_path.exists():
            print("Cleaning up downloaded archive...")
            archive_path.unlink()
        sentinel_path.touch()
        print("Data directory ready.")
        return True
    except Exception as exc:
        print(f"Error: {exc}")
        return False


def package_data(data_dir: Path = Path("data"), output_path: Path = Path("data.tar.zst")) -> bool:
    """Package the data directory into a zstd-compressed tar archive."""
    if not data_dir.exists():
        print(f"Error: Data directory '{data_dir}' does not exist.")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f"{output_path.name}.part")
    print(f"Packaging {data_dir} -> {output_path}")

    try:
        with temp_path.open("wb") as raw_stream:
            compressor = zstd.ZstdCompressor(level=3)
            with compressor.stream_writer(raw_stream) as compressed_stream:
                with tarfile.open(fileobj=compressed_stream, mode="w|") as archive:
                    archive.add(data_dir, arcname=data_dir.name)
        temp_path.replace(output_path)
    except Exception as exc:
        temp_path.unlink(missing_ok=True)
        print(f"Error: {exc}")
        return False

    print(f"Successfully created {output_path}")
    return True
