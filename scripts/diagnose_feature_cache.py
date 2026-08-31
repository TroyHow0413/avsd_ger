"""Compare saved feature-cache signatures with the current repository code."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--pair", action="append", nargs=2, metavar=("MANIFEST", "CACHE_DIR"),
        required=True,
    )
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    module = importlib.import_module("scripts.train_stage2_pro6000")
    cfg = module.load_config(args.config)
    print(f"repo={repo}")
    print(f"git_commit={module._git_commit()}")
    print(f"feature_source_fingerprint={module._feature_source_fingerprint()}")
    for manifest_raw, cache_raw in args.pair:
        manifest = Path(manifest_raw)
        cache = Path(cache_raw)
        index = json.loads((cache / "index.json").read_text(encoding="utf-8"))
        current = module._cache_signature(cfg, manifest)
        print(json.dumps({
            "manifest": str(manifest),
            "manifest_sha256": module._sha256_file(manifest),
            "cache": str(cache),
            "index_version": index.get("version"),
            "records": index.get("records"),
            "saved_signature": index.get("signature"),
            "current_signature": current,
            "matches": index.get("signature") == current,
        }, indent=2))


if __name__ == "__main__":
    main()
