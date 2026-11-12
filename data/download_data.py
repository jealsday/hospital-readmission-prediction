"""Download and extract the UCI "Diabetes 130-US Hospitals" dataset.

Usage:
    python data/download_data.py

The archive is committed as data/data.zip. It is downloaded from UCI only if it is
missing, and its SHA-256 checksum is verified before extraction.
"""

from __future__ import annotations

import hashlib
import urllib.request
import zipfile
from pathlib import Path


URL = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes+130-us+hospitals+for+years+1999-2008.zip"
)
SHA256 = "f82ac129da2ddd2299391ff6fbae3a6a58b3edcf59ac9d7bd480c00fe453112a"
DATA_DIR = Path(__file__).resolve().parent
ARCHIVE = DATA_DIR / "data.zip"
FILES = ["diabetic_data.csv", "IDS_mapping.csv"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if not ARCHIVE.exists():
        print(f"Downloading {URL}")
        urllib.request.urlretrieve(URL, ARCHIVE)

    checksum = sha256(ARCHIVE)
    if checksum != SHA256:
        raise SystemExit(f"Checksum mismatch for {ARCHIVE}: {checksum}")

    with zipfile.ZipFile(ARCHIVE) as archive:
        for name in FILES:
            archive.extract(name, DATA_DIR)
            print(f"Extracted {DATA_DIR / name}")


if __name__ == "__main__":
    main()
