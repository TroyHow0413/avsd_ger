from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.av_preprocess_common import (
    atomic_jsonl,
    ensure_build_config,
    manifest_path,
    valid_npy,
)
from scripts.prepare_lrs2_full import parse_filelist, read_transcript
from scripts.prepare_voxceleb2_c1 import VoxItem, curate, validation_speakers


def test_lrs2_filelist_tags_and_transcript(tmp_path: Path) -> None:
    source = tmp_path / "main"
    clip = source / "speaker" / "00001"
    clip.parent.mkdir(parents=True)
    clip.with_suffix(".mp4").write_bytes(b"video")
    clip.with_suffix(".txt").write_text("Text: HELLO WORLD\nConf: 6\n", encoding="utf-8")
    filelist = tmp_path / "test.txt"
    filelist.write_text("speaker/00001 NF\n", encoding="utf-8")

    items = parse_filelist(filelist, "test", source)

    assert len(items) == 1
    assert items[0].clip_id == "speaker/00001"
    assert items[0].tags == ("NF",)
    assert read_transcript(Path(items[0].transcript)) == "HELLO WORLD"


def test_voxceleb_curation_limits_each_source_video() -> None:
    items = [
        VoxItem(i, "official-dev", "id0001", relative, f"/{relative}")
        for i, relative in enumerate(
            [
                "id0001/youtube_a/00001.mp4",
                "id0001/youtube_a/00002.mp4",
                "id0001/youtube_b/00001.mp4",
            ]
        )
    ]

    selected = curate(items, max_per_speaker=None, max_per_video=1, min_per_speaker=2)

    assert [item.relative for item in selected] == [
        "id0001/youtube_a/00001.mp4",
        "id0001/youtube_b/00001.mp4",
    ]


def test_voxceleb_validation_split_is_deterministic_and_disjoint() -> None:
    speakers = [f"id{index:04d}" for index in range(100)]
    first = validation_speakers(speakers, 10.0, 1337)
    second = validation_speakers(reversed(speakers), 10.0, 1337)

    assert first == second
    assert len(first) == 10
    assert first < set(speakers)


def test_common_manifest_and_npy_helpers(tmp_path: Path) -> None:
    array_path = tmp_path / "cache" / "roi.npy"
    array_path.parent.mkdir()
    np.save(array_path, np.zeros((2, 1, 96, 96), dtype=np.uint8))
    assert valid_npy(array_path, ndim=4, dtype="uint8", shape_tail=(1, 96, 96))
    assert manifest_path(array_path, tmp_path, absolute=False) == "cache/roi.npy"

    manifest = tmp_path / "manifest.jsonl"
    atomic_jsonl(manifest, [{"speaker_id": "spk1"}, {"speaker_id": "spk2"}])
    assert [json.loads(line) for line in manifest.read_text().splitlines()] == [
        {"speaker_id": "spk1"},
        {"speaker_id": "spk2"},
    ]


def test_build_config_rejects_mixed_resume(tmp_path: Path) -> None:
    path = tmp_path / "build_config.json"
    ensure_build_config(path, {"roi_backend": "dlib"}, overwrite=False)

    try:
        ensure_build_config(path, {"roi_backend": "haar"}, overwrite=False)
    except RuntimeError as exc:
        assert "configuration differs" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("mixed preprocessing should be rejected")
