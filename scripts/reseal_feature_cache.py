"""Validate frozen feature shards and migrate their cache signature in place.

This does not rebuild or alter any shard.  It is intended for signature-policy
changes where frozen feature extraction inputs and code are known unchanged.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any
from itertools import zip_longest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from avsd_ger.utils import load_config
from scripts import train_stage2_pro6000 as trainer


def _resolved_index_manifest(index: dict[str, Any]) -> Path:
    path = Path(str(index.get("manifest", "")))
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def reseal_one(
    *,
    cfg: dict[str, Any],
    manifest: Path,
    cache_dir: Path,
    reason: str,
    expected_legacy_signature: str,
) -> None:
    manifest = manifest.resolve()
    cache_dir = cache_dir.resolve()
    index_path = cache_dir / "index.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    if not index_path.is_file():
        raise FileNotFoundError(f"Cache index not found: {index_path}")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("version") != trainer.CACHE_VERSION:
        raise RuntimeError(
            f"Cache schema {index.get('version')} is not {trainer.CACHE_VERSION}: {cache_dir}"
        )
    indexed_manifest = _resolved_index_manifest(index)
    if indexed_manifest != manifest:
        raise RuntimeError(
            f"Index belongs to a different manifest: {indexed_manifest} != {manifest}"
        )
    old_signature = str(index.get("signature", ""))
    if old_signature != expected_legacy_signature:
        raise RuntimeError(
            "Saved signature is not the explicitly expected legacy signature; "
            f"saved={old_signature}, expected={expected_legacy_signature}"
        )

    missing = [name for name in index.get("shards", []) if not (cache_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Cache is incomplete; missing shards: {missing[:3]}")

    print(
        f"[validate] {cache_dir}: validating all frozen records against the manifest",
        flush=True,
    )
    identity_fields = (
        "utt_id", "target", "dataset_build_id", "participant_id", "speaker_id",
        "lip_conf_source",
    )
    count = 0
    cached_records = trainer.iter_cached_records(index_path)
    source_records = trainer.base.iter_manifest(manifest)
    for row_number, (cached, source) in enumerate(
        zip_longest(cached_records, source_records), start=1
    ):
        if cached is None or source is None:
            raise RuntimeError(
                f"Manifest/cache length mismatch at row {row_number}: {cache_dir}"
            )
        for field in identity_fields:
            if cached.get(field) != source.get(field):
                raise RuntimeError(
                    f"Manifest/cache mismatch at row {row_number}, field={field}: "
                    f"cached={cached.get(field)!r}, manifest={source.get(field)!r}"
                )
        for field in ("start", "end"):
            if abs(float(cached.get(field)) - float(source.get(field))) > 1e-6:
                raise RuntimeError(
                    f"Manifest/cache mismatch at row {row_number}, field={field}: "
                    f"cached={cached.get(field)!r}, manifest={source.get(field)!r}"
                )
        count += 1
    if count != int(index.get("records", -1)):
        raise RuntimeError(f"Record count mismatch: validated={count}, index={index.get('records')}")

    new_payload = trainer._cache_signature_payload(cfg, manifest)
    new_signature = trainer._cache_signature(cfg, manifest)
    if old_signature == new_signature:
        print(f"[unchanged] {cache_dir}: signature already current ({count} records)")
        return

    history = list(index.get("signature_history", []))
    history.append({
        "signature": old_signature,
        "signature_policy": index.get("signature_policy", "legacy-global-git-commit"),
        "replaced_at": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
    })
    index.update({
        "signature": new_signature,
        "signature_policy": trainer.CACHE_SIGNATURE_POLICY,
        "signature_payload": new_payload,
        "signature_history": history,
        "last_resealed_git_commit": trainer._git_commit(),
        "last_resealed_reason": reason,
    })

    temporary = index_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(index, indent=2), encoding="utf-8")
    os.replace(temporary, index_path)
    print(
        f"[resealed] {cache_dir}: {count} records; "
        f"{old_signature[:12]} -> {new_signature[:12]}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--pair", action="append", nargs=3,
        metavar=("MANIFEST", "CACHE_DIR", "EXPECTED_LEGACY_SIGNATURE"), required=True,
    )
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    for manifest_raw, cache_raw, expected_signature in args.pair:
        reseal_one(
            cfg=cfg,
            manifest=Path(manifest_raw),
            cache_dir=Path(cache_raw),
            reason=args.reason,
            expected_legacy_signature=expected_signature,
        )


if __name__ == "__main__":
    main()
