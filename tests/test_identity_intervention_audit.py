from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_identity_intervention import analyze


def _turn(turn_id: str, *, mode: str, norm: float, source: str | None, text: str):
    mapping = {"spk1": "spk2", "spk2": "spk1"} if mode == "shuffled" else {}
    return {
        "summary": {
            "turn_id": turn_id,
            "final_text": text,
            "hyp_speaker": "spk1",
            "fallback_applied": False,
        },
        "c1_effective": {
            "top_ids": ["spk1"],
            "identity_conditioning_mode": mode,
            "identity_conditioning_source_id": source,
            "identity_causal_eligible": True,
            "identity_derangement_map": mapping,
            "conditioning_z_id": {"norm": norm},
        },
        "trace": [{
            "raw_ger_text": text,
            "raw_generation": text,
            "final_source": "GER",
            "alignment": {"f_align": {"shape": [1, 2], "norm": norm}},
        }],
    }


def _write(root: Path, ablation: str, turn: dict) -> None:
    path = root / "debug" / ablation / "ES2011a.debug.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"ablation": ablation, "manifest": "ES2011a.json", "turns": [turn]}),
        encoding="utf-8",
    )


def test_analyze_identity_intervention(tmp_path: Path) -> None:
    _write(tmp_path, "identity_normal", _turn("t1", mode="normal", norm=1.0, source="spk1", text="a"))
    _write(tmp_path, "zero_z_id", _turn("t1", mode="zero", norm=0.0, source=None, text="a"))
    _write(tmp_path, "shuffled_z_id", _turn("t1", mode="shuffled", norm=1.0, source="spk2", text="b"))

    report = analyze(tmp_path)

    assert report["integrity_status"] == "pass"
    assert report["comparisons"]["zero_z_id"]["changed_rates"]["final_text"] == 0.0
    assert report["comparisons"]["shuffled_z_id"]["changed_rates"]["raw_ger_text"] == 1.0
    assert report["comparisons"]["shuffled_z_id"]["by_series"]["ES2011"]["n_turns"] == 1
