"""Shared, dependency-light helpers for server-side AV dataset preparation."""
from __future__ import annotations

import json
import os
import subprocess
import wave
from pathlib import Path
from typing import Any, Iterable

import numpy as np


VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".webm", ".avi"}


class AtomicJsonlWriter:
    """Incrementally write a JSONL file, publishing it only on clean exit."""

    def __init__(self, path: Path):
        self.path = path
        self.temporary = path.with_name(f".{path.name}.tmp")
        self._handle: Any = None

    def __enter__(self) -> "AtomicJsonlWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.temporary.open("w", encoding="utf-8", newline="\n")
        return self

    def write(self, row: dict[str, Any]) -> None:
        if self._handle is None:
            raise RuntimeError("AtomicJsonlWriter is not open")
        self._handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
        self._handle.write("\n")

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._handle is not None:
            self._handle.close()
        if exc_type is None:
            os.replace(self.temporary, self.path)
        else:
            self.temporary.unlink(missing_ok=True)


def run_checked(command: list[str]) -> None:
    """Run a command without shell interpolation and surface useful stderr."""
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        stderr = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"command failed ({result.returncode}): {stderr}")


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with AtomicJsonlWriter(path) as writer:
        for row in rows:
            writer.write(row)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def manifest_path(path: Path, root: Path, *, absolute: bool) -> str:
    resolved = path.resolve()
    if absolute:
        return resolved.as_posix()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"output {resolved} is outside --manifest-root {root.resolve()}; "
            "choose a common root or pass --absolute-paths"
        ) from exc


def valid_wav(path: Path) -> bool:
    try:
        if path.stat().st_size <= 44:
            return False
        with wave.open(str(path), "rb") as handle:
            return (
                handle.getnchannels() == 1
                and handle.getframerate() == 16000
                and handle.getnframes() > 0
            )
    except (OSError, EOFError, wave.Error):
        return False


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate())


def valid_npy(
    path: Path,
    *,
    ndim: int | None = None,
    dtype: str | None = None,
    shape_tail: tuple[int, ...] | None = None,
) -> bool:
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        return (
            array.size > 0
            and (ndim is None or array.ndim == ndim)
            and (dtype is None or array.dtype == np.dtype(dtype))
            and (shape_tail is None or array.shape[-len(shape_tail):] == shape_tail)
        )
    except (OSError, ValueError):
        return False


def valid_image(path: Path) -> bool:
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
        return path.stat().st_size > 0
    except (OSError, ValueError):
        return False


def extract_wav(
    source: Path,
    destination: Path,
    *,
    ffmpeg: str = "ffmpeg",
    threads: int = 1,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.tmp.wav")
    try:
        run_checked(
            [
                ffmpeg,
                "-nostdin",
                "-y",
                "-loglevel",
                "error",
                "-threads",
                str(threads),
                "-i",
                str(source),
                "-map",
                "0:a:0",
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(temporary),
            ]
        )
        if not valid_wav(temporary):
            raise RuntimeError("ffmpeg produced an invalid 16 kHz mono WAV")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def extract_face(
    source: Path,
    destination: Path,
    *,
    ffmpeg: str = "ffmpeg",
    size: int = 224,
    threads: int = 1,
) -> None:
    """Extract a representative frame from an already face-tracked video."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.tmp.jpg")
    try:
        run_checked(
            [
                ffmpeg,
                "-nostdin",
                "-y",
                "-loglevel",
                "error",
                "-threads",
                str(threads),
                "-i",
                str(source),
                "-vf",
                f"thumbnail=30,scale={size}:{size}:force_original_aspect_ratio=increase,crop={size}:{size}",
                "-frames:v",
                "1",
                "-q:v",
                "2",
                str(temporary),
            ]
        )
        if not valid_image(temporary):
            raise RuntimeError("ffmpeg produced an invalid face image")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def save_npy_atomic(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def ensure_build_config(path: Path, config: dict[str, Any], *, overwrite: bool) -> None:
    """Prevent a resume from silently mixing incompatible preprocessing."""
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != config and not overwrite:
            raise RuntimeError(
                f"preprocessing configuration differs from {path}; use a new "
                "--output-root or pass --overwrite to rebuild every artifact"
            )
    atomic_json(path, config)
