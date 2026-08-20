"""Print an auditable leaf-level diff between two effective AVSD-GER configs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from avsd_ger.utils import load_config  # noqa: E402


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    result: dict[str, Any] = {}
    for key in sorted(value):
        dotted = f"{prefix}.{key}" if prefix else str(key)
        result.update(_flatten(value[key], dotted))
    return result


def config_diff(left: dict[str, Any], right: dict[str, Any]) -> list[dict[str, Any]]:
    a, b = _flatten(left), _flatten(right)
    return [
        {"key": key, "left": a.get(key, "<missing>"),
         "right": b.get(key, "<missing>")}
        for key in sorted(set(a) | set(b)) if a.get(key) != b.get(key)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("left")
    parser.add_argument("right")
    parser.add_argument("--out")
    args = parser.parse_args()
    payload = {
        "left": args.left,
        "right": args.right,
        "comparison_mode": "strictly-controlled only if non-model differences are empty",
        "differences": config_diff(load_config(args.left), load_config(args.right)),
    }
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
