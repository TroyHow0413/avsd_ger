"""Print the canonical AVSD frontend experiment profiles.

This is a documentation helper, not a model runner. The actual third-party
frontends are intentionally kept outside the repo so experiments can swap
pyannote, Sortformer, TalkNet, Light-ASD, or oracle turns without changing the
core AVSD-GER pipeline.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from avsd_ger.frontend import list_frontend_profiles, render_frontend_profiles_markdown


def main() -> int:
    p = argparse.ArgumentParser(description="Show AVSD frontend profile metadata.")
    p.add_argument("--format", choices=["markdown", "json"], default="markdown")
    args = p.parse_args()

    if args.format == "json":
        print(json.dumps([asdict(p) for p in list_frontend_profiles()], indent=2))
    else:
        print(render_frontend_profiles_markdown())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
