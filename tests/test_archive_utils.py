from __future__ import annotations

from pathlib import Path

from src.common.util.package import extract_data_archive, package_data


def test_package_and_extract_roundtrip(tmp_path: Path):
    data_dir = tmp_path / "data"
    nested_dir = data_dir / "kalshi" / "markets"
    nested_dir.mkdir(parents=True)
    source_file = nested_dir / "markets.parquet"
    source_file.write_text("test-payload")

    archive_path = tmp_path / "data.tar.zst"
    assert package_data(data_dir=data_dir, output_path=archive_path)
    assert archive_path.exists()

    extract_dir = tmp_path / "extracted"
    extract_data_archive(archive_path, destination=extract_dir)

    extracted_file = extract_dir / "data" / "kalshi" / "markets" / "markets.parquet"
    assert extracted_file.read_text() == "test-payload"
