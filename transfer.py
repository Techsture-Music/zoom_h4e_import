"""
Streaming copy and stitch, each verified by checksum, with a progress line.

Everything is written to NAME.partial and only renamed into place once
verified, so a file at its final name is always a complete, checked one.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from wavfile import (WAVE_FORMAT_PCM, WavInfo, WavWriter, container_for, merge_cues,
                     planned_size, read_wav)

CHUNK_SIZE = 8 * 1024 * 1024

# Chunks a stitched file gets from its own writer rather than from the parts.
REBUILT_CHUNKS = {b"fmt ", b"fact", b"data", b"ds64", b"JUNK", b"junk", b"PAD ", b"FLLR", b"cue "}
# Per-part metadata that the first part's copy supersedes, so dropping later
# parts' copies isn't worth a note.
SUPERSEDED_CHUNKS = {b"bext", b"iXML", b"LIST"}


class TransferError(Exception):
    pass


def partial_path(dest: Path) -> Path:
    return dest.with_name(dest.name + ".partial")


# --- progress ----------------------------------------------------------------

def fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1000:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000
    return f"{n:.2f} TB"


def fmt_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


class Progress:
    """One status line per file: stage, bar, percent, speed, ETA.

    Redraws in place on a terminal; elsewhere (logs, pipes) prints every 10%.
    The total covers both the write pass and the verification read-back.
    """

    def __init__(self, total: int, stage: str, stream=sys.stderr):
        self.total = max(total, 1)
        self.stage_name = stage
        self.stream = stream
        self.tty = stream.isatty()
        self.done = 0
        self.started = time.monotonic()
        self.last_draw = 0.0
        self.next_step = 10

    def stage(self, name: str):
        self.stage_name = name
        self._draw()

    def update(self, n: int):
        self.done += n
        if self.tty:
            if time.monotonic() - self.last_draw >= 0.1:
                self._draw()
        elif self.done * 100 // self.total >= self.next_step:
            self._draw()
            self.next_step = self.done * 100 // self.total // 10 * 10 + 10

    def _draw(self):
        self.last_draw = now = time.monotonic()
        fraction = min(self.done / self.total, 1.0)
        rate = self.done / max(now - self.started, 1e-6)
        eta = fmt_duration((self.total - self.done) / rate) if rate and self.done else "--:--"
        text = f"{self.stage_name:<7} {fraction * 100:5.1f}%  {fmt_bytes(rate)}/s  ETA {eta}"
        if self.tty:
            width = shutil.get_terminal_size((80, 20)).columns
            bar_width = max(10, min(30, width - len(text) - 10))
            filled = int(bar_width * fraction)
            bar = "█" * filled + "░" * (bar_width - filled)
            self.stream.write(f"\r    {self.stage_name:<7} {bar} {text[8:]}\x1b[K")
        else:
            self.stream.write(f"    {text}\n")
        self.stream.flush()

    def finish(self, summary: str):
        elapsed = time.monotonic() - self.started
        line = f"    done    {summary} in {fmt_duration(elapsed)}"
        self.stream.write(("\r" + line + "\x1b[K\n") if self.tty else line + "\n")
        self.stream.flush()

    def abort(self):
        if self.tty:
            self.stream.write("\n")
            self.stream.flush()


# --- helpers -----------------------------------------------------------------

def _uncached(fd: int):
    """Ask macOS to keep this file's pages out of the buffer cache, so a
    verification read-back comes from the disk rather than memory. Best effort."""
    try:
        import fcntl
        flag = getattr(fcntl, "F_NOCACHE", 48 if sys.platform == "darwin" else None)
        if flag is not None:
            fcntl.fcntl(fd, flag, 1)
    except (ImportError, OSError):
        pass


def _discard(path: Path):
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def hash_range(path: Path, offset: int, length: int, algo: str, progress: Progress) -> str:
    digest = hashlib.new(algo)
    with open(path, "rb") as f:
        _uncached(f.fileno())
        f.seek(offset)
        remaining = length
        while remaining:
            block = f.read(min(CHUNK_SIZE, remaining))
            if not block:
                raise TransferError(f"{path.name} is shorter than expected during verification")
            digest.update(block)
            remaining -= len(block)
            progress.update(len(block))
    return digest.hexdigest()


# --- copy --------------------------------------------------------------------

def copy_verified(src: Path, dest: Path, algo: str) -> str:
    """Copy src to dest, then re-read dest and compare checksums.
    Returns the file's checksum. The source is only ever opened read-only."""
    total = src.stat().st_size
    tmp = partial_path(dest)
    progress = Progress(total * 2, "copy")
    digest = hashlib.new(algo)
    try:
        with open(src, "rb") as fin, open(tmp, "wb") as fout:
            _uncached(fout.fileno())
            while True:
                block = fin.read(CHUNK_SIZE)
                if not block:
                    break
                digest.update(block)
                fout.write(block)
                progress.update(len(block))
            fout.flush()
            os.fsync(fout.fileno())
        written = tmp.stat().st_size
        if written != total:
            raise TransferError(f"wrote {written:,} bytes but {src.name} is {total:,}")
        progress.stage("verify")
        if hash_range(tmp, 0, written, algo, progress) != digest.hexdigest():
            raise TransferError(f"checksum mismatch: the copy of {src.name} doesn't match the card")
        st = src.stat()
        os.utime(tmp, (st.st_atime, st.st_mtime))
        os.replace(tmp, dest)
    except BaseException:
        progress.abort()
        _discard(tmp)
        raise
    progress.finish(f"{fmt_bytes(total)} copied and verified")
    return digest.hexdigest()


# --- stitch ------------------------------------------------------------------

@dataclass
class StitchLayout:
    """What a stitched file will contain, worked out before writing anything."""
    parts: List[WavInfo]
    fmt: bytes
    chunks: List[Tuple[bytes, bytes]]
    fact: bool
    frames: int
    audio_bytes: int
    notes: List[str] = field(default_factory=list)

    @property
    def size(self) -> int:
        return planned_size(self.fmt, self.chunks, self.fact, self.audio_bytes)

    def container(self, rf64_mode: str) -> str:
        return container_for(self.size, rf64_mode)


def stitch_layout(parts: Sequence[WavInfo], keep_ixml: bool = True) -> StitchLayout:
    """Header for the stitched file: the first part's fmt and metadata (bext
    included, so its TimeReference still marks the start), plus every part's
    cue points shifted to their position in the joined audio."""
    first = parts[0]
    notes: List[str] = []
    offsets, frames = [], 0
    for part in parts:
        offsets.append(frames)
        frames += part.frames
    cue, dropped = merge_cues([(p.payload(b"cue "), off) for p, off in zip(parts, offsets)])

    chunks: List[Tuple[bytes, bytes]] = []
    for chunk in first.chunks:
        if chunk.id == b"cue " and cue:
            chunks.append((b"cue ", cue))
            cue = None
        if chunk.id in REBUILT_CHUNKS or (chunk.id == b"iXML" and not keep_ixml):
            continue
        if chunk.payload is None:
            notes.append(f"'{chunk.name}' chunk is too large to carry over; dropped")
            continue
        chunks.append((chunk.id, chunk.payload))
    if cue:
        chunks.append((b"cue ", cue))

    if dropped:
        notes.append(f"{dropped} marker(s) beyond 2^32 frames can't be stored and were dropped")
    if any(c.id == b"LIST" and (c.payload or b"")[:4] == b"adtl" for p in parts[1:] for c in p.chunks):
        notes.append("marker labels from parts after the first aren't carried over (the markers themselves are)")
    extra = sorted({c.name for p in parts[1:] for c in p.chunks
                    if c.id not in REBUILT_CHUNKS | SUPERSEDED_CHUNKS})
    if extra:
        notes.append(f"only the first part's {', '.join(extra)} chunk(s) are kept")

    fact = first.find(b"fact") is not None or first.format_tag != WAVE_FORMAT_PCM
    return StitchLayout(list(parts), first.find(b"fmt ").payload, chunks, fact, frames,
                        sum(p.audio_bytes for p in parts), notes)


def stitch_verified(layout: StitchLayout, dest: Path, algo: str, rf64_mode: str) -> Tuple[str, str]:
    """Join the parts' audio frame-for-frame into dest, then re-read the
    result and check its header and an audio checksum against the sources.
    Returns (container, audio checksum)."""
    tmp = partial_path(dest)
    progress = Progress(layout.audio_bytes * 2, "stitch")
    digest = hashlib.new(algo)
    writer: Optional[WavWriter] = None
    try:
        writer = WavWriter(tmp, layout.fmt, layout.chunks, rf64_mode, layout.fact)
        _uncached(writer.f.fileno())
        for part in layout.parts:
            with open(part.path, "rb") as fin:
                fin.seek(part.data.offset)
                remaining = part.audio_bytes
                while remaining:
                    block = fin.read(min(CHUNK_SIZE, remaining))
                    if not block:
                        raise TransferError(f"{part.path.name} ended before its data chunk did")
                    digest.update(block)
                    writer.write(block)
                    remaining -= len(block)
                    progress.update(len(block))
        container = writer.close()
        writer = None

        progress.stage("verify")
        out = read_wav(tmp)
        first = layout.parts[0]
        problems = [f"read back as {name} {got}, expected {want}" for name, got, want in (
            ("container", out.container, container),
            ("format", out.describe(), first.describe()),
            ("frame count", out.frames, layout.frames),
            ("data size", out.data.size, layout.audio_bytes),
        ) if got != want]
        problems += out.warnings
        if out.bext and first.bext and out.bext.time_reference != first.bext.time_reference:
            problems.append("BEXT time reference changed")
        if problems:
            raise TransferError("stitched file failed verification: " + "; ".join(problems))
        if hash_range(tmp, out.data.offset, out.audio_bytes, algo, progress) != digest.hexdigest():
            raise TransferError("audio checksum mismatch: the stitched file doesn't match the parts")
        mtime = layout.parts[-1].path.stat().st_mtime
        os.utime(tmp, (mtime, mtime))
        os.replace(tmp, dest)
    except BaseException:
        progress.abort()
        if writer is not None:
            writer.abort()
        _discard(tmp)
        raise
    progress.finish(f"{fmt_bytes(layout.audio_bytes)} of audio stitched and verified")
    return container, digest.hexdigest()
