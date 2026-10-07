import argparse
import hashlib
import urllib.request
from pathlib import Path

from ..config import RAW_DATA_DIR
from .sources import RAW_FILES, RawFile


def md5_of(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(raw_file: RawFile, target_dir: Path, force: bool = False) -> Path:
    target = target_dir / raw_file.name
    if target.exists() and not force and md5_of(target) == raw_file.md5:
        return target
    target_dir.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(raw_file.url, target)
    checksum = md5_of(target)
    if checksum != raw_file.md5:
        target.unlink()
        raise ValueError(
            f"{raw_file.name}: expected md5 {raw_file.md5}, got {checksum}"
        )
    return target


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the raw EV charging datasets"
    )
    parser.add_argument("--target-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for raw_file in RAW_FILES:
        path = download(raw_file, args.target_dir, args.force)
        print(f"{raw_file.name}: {path}")


if __name__ == "__main__":
    main()
