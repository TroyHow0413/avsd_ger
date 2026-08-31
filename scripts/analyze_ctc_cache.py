"""Report CTC temporal-expansion requirements in cached feature shards."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from avsd_ger.training.ctc_loss import CTCHead, CharVocab


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache_dirs", nargs="+")
    parser.add_argument("--limit", type=int, default=16)
    args = parser.parse_args()
    vocab = CharVocab()
    rows = []
    for raw in args.cache_dirs:
        cache_dir = Path(raw)
        for shard in sorted(cache_dir.glob("shard-*.pt")):
            for record in torch.load(shard, map_location="cpu"):
                ids = vocab.encode(str(record["target"]))
                aligned_tokens = int(record["asr_tok"].shape[0])
                minimum = CTCHead.minimum_ctc_steps(ids)
                required = (minimum + aligned_tokens - 1) // aligned_tokens
                rows.append({
                    "cache": cache_dir.name,
                    "utt_id": record.get("utt_id"),
                    "aligned_tokens": aligned_tokens,
                    "target_chars": len(ids),
                    "minimum_steps": minimum,
                    "required_expansion": required,
                    "target": record["target"],
                    "asr_nbest": record.get("asr_nbest"),
                })
    if not rows:
        raise SystemExit("No cached records found")
    empty = [row for row in rows if row["target_chars"] == 0]
    bad = [row for row in rows if row["required_expansion"] > args.limit]
    print(f"records={len(rows)}")
    print(f"max_required_expansion={max(row['required_expansion'] for row in rows)}")
    print(f"empty_after_normalization={len(empty)}")
    print(f"above_limit_{args.limit}={len(bad)}")
    for row in empty:
        print({"empty_after_normalization": row})
    for row in sorted(bad, key=lambda item: item["required_expansion"], reverse=True):
        print(row)


if __name__ == "__main__":
    main()
