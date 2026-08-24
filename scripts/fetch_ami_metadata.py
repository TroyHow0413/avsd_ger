"""Fetch the official AMI corpus-resource metadata needed by AV preparation."""
from __future__ import annotations

import argparse
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


DEFAULT_URL = (
    "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/"
    "ami_public_manual_1.6.2.zip"
)
RESOURCE_NAMES = ("meetings.xml", "participants.xml")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("datasets/ami/corpusResources"))
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if (args.out_dir / "meetings.xml").exists() and not args.overwrite:
        print(f"[skip] {args.out_dir / 'meetings.xml'} already exists")
        return 0

    with tempfile.TemporaryDirectory(prefix="ami_metadata_") as temp_dir:
        archive = Path(temp_dir) / "ami_public_manual_1.6.2.zip"
        print(f"[download] {args.url}")
        urllib.request.urlretrieve(args.url, archive)
        with zipfile.ZipFile(archive) as zf:
            members = {Path(name).name: name for name in zf.namelist()}
            for resource in RESOURCE_NAMES:
                member = members.get(resource)
                if member is None:
                    if resource == "meetings.xml":
                        raise FileNotFoundError(f"{resource} is missing from {args.url}")
                    continue
                destination = args.out_dir / resource
                with zf.open(member) as src, destination.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                print(f"[wrote] {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
