"""
RIFF/RF64 WAV container reading and writing.

Audio is treated as opaque frames: nothing here decodes a sample, so any PCM
or float format passes through bit-for-bit. What this module understands is
the container: chunk layout, RF64's 64-bit ds64 sizes, and the fmt, fact,
bext and cue payloads that stitching has to read or rewrite.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field
from datetime import date, time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

U32_MAX = 0xFFFFFFFF

WAVE_FORMAT_PCM = 0x0001
WAVE_FORMAT_IEEE_FLOAT = 0x0003
WAVE_FORMAT_EXTENSIBLE = 0xFFFE
FORMAT_NAMES = {WAVE_FORMAT_PCM: "PCM", WAVE_FORMAT_IEEE_FLOAT: "float"}

DS64_BODY = struct.Struct("<QQQI")  # riffSize, dataSize, sampleCount, tableLength
DS64_TABLE_ENTRY = struct.Struct("<4sQ")
BEXT_HEAD = struct.Struct("<256s32s32s10s8sQH")  # through Version; TimeReference is low+high u32
CUE_POINT = struct.Struct("<II4sIII")  # id, position, fccChunk, chunkStart, blockStart, sampleOffset

# Metadata chunks larger than this are left on disk instead of read into memory.
MAX_METADATA_CHUNK = 16 * 1024 * 1024


class WavError(Exception):
    """A file that can't be read or written as a WAV."""


@dataclass
class Chunk:
    id: bytes
    offset: int  # file offset of the payload, just past the 8-byte header
    size: int  # payload size, with RF64 sizes resolved through ds64
    payload: Optional[bytes] = None  # loaded for metadata chunks, never for data

    @property
    def name(self) -> str:
        return self.id.decode("latin-1").strip()


@dataclass
class Bext:
    description: str
    originator: str
    originator_reference: str
    origination_date: str
    origination_time: str
    time_reference: int  # samples since midnight, at the file's sample rate
    version: int

    @classmethod
    def parse(cls, payload: bytes) -> Optional["Bext"]:
        if len(payload) < BEXT_HEAD.size:
            return None
        desc, orig, ref, day, clock, time_ref, version = BEXT_HEAD.unpack_from(payload)
        return cls(_text(desc), _text(orig), _text(ref), _text(day), _text(clock), time_ref, version)

    def origination(self) -> Tuple[Optional[date], Optional[time]]:
        """OriginationDate/Time as values; EBU allows any separator, so only digits count."""
        d, t = _digits(self.origination_date), _digits(self.origination_time)
        day = clock = None
        if len(d) == 8:
            try:
                day = date(int(d[:4]), int(d[4:6]), int(d[6:]))
            except ValueError:
                pass
        if len(t) == 6:
            try:
                clock = time(int(t[:2]), int(t[2:4]), int(t[4:]))
            except ValueError:
                pass
        return day, clock


@dataclass
class WavInfo:
    path: Path
    container: str  # "RIFF", "RF64" or "BW64"
    file_size: int
    chunks: List[Chunk]
    format_tag: int  # WAVE_FORMAT_EXTENSIBLE is resolved to its sub-format
    channels: int
    sample_rate: int
    bits_per_sample: int
    block_align: int
    data: Chunk
    bext: Optional[Bext]
    warnings: List[str] = field(default_factory=list)

    @property
    def frames(self) -> int:
        return self.data.size // self.block_align

    @property
    def audio_bytes(self) -> int:
        """Data chunk bytes that make up whole frames."""
        return self.frames * self.block_align

    @property
    def duration(self) -> float:
        return self.frames / self.sample_rate

    @property
    def signature(self) -> Tuple[int, int, int, int, int]:
        """Everything that has to match for two files' frames to be concatenated."""
        return (self.format_tag, self.channels, self.sample_rate, self.bits_per_sample, self.block_align)

    def describe(self) -> str:
        kind = FORMAT_NAMES.get(self.format_tag, f"format 0x{self.format_tag:04x}")
        chans = {1: "mono", 2: "stereo"}.get(self.channels, f"{self.channels}ch")
        return f"{self.sample_rate / 1000:g} kHz {self.bits_per_sample}-bit {kind} {chans}"

    def find(self, chunk_id: bytes) -> Optional[Chunk]:
        return next((c for c in self.chunks if c.id == chunk_id), None)

    def payload(self, chunk_id: bytes) -> Optional[bytes]:
        chunk = self.find(chunk_id)
        return chunk.payload if chunk else None


def read_wav(path) -> WavInfo:
    """Parse a WAV's chunk list and format without reading its audio.

    Sizes are taken from the file rather than trusted blindly: a data chunk
    whose header was never finalized (size 0, or larger than the file, as
    after a power loss) is read to the end of the file, with a warning.
    """
    path = Path(path)
    with open(path, "rb") as f:
        file_size = os.fstat(f.fileno()).st_size
        head = f.read(12)
        if len(head) < 12:
            raise WavError(f"{path.name}: too short to be a WAV file")
        riff_id, _riff_size, wave_id = struct.unpack("<4sI4s", head)
        if riff_id not in (b"RIFF", b"RF64", b"BW64") or wave_id != b"WAVE":
            raise WavError(f"{path.name}: not a RIFF/RF64 WAVE file")

        warnings: List[str] = []
        chunks: List[Chunk] = []
        big_sizes = {}
        pos = 12
        while pos + 8 <= file_size:
            f.seek(pos)
            chunk_id, size = struct.unpack("<4sI", f.read(8))
            if not _plausible_id(chunk_id):
                warnings.append(f"unreadable chunk header at byte {pos}; ignored the rest of the file")
                break
            payload_at = pos + 8
            if size == U32_MAX and chunk_id in big_sizes:
                size = big_sizes[chunk_id]
            elif chunk_id == b"data" and size in (0, U32_MAX) and file_size > payload_at:
                f.seek(payload_at)
                if not _plausible_id(f.read(4)):
                    warnings.append("data chunk size was never written (recording not finalized?); "
                                    "using everything to the end of the file")
                    size = file_size - payload_at
            if payload_at + size > file_size:
                if chunk_id != b"data":
                    warnings.append(f"'{chunk_id.decode('latin-1')}' chunk runs past the end of the file; ignored")
                    break
                warnings.append(f"data chunk claims {size} bytes but only {file_size - payload_at} "
                                "are present (file truncated?); using what is there")
                size = file_size - payload_at

            chunk = Chunk(chunk_id, payload_at, size)
            if chunk_id != b"data" and size <= MAX_METADATA_CHUNK:
                f.seek(payload_at)
                chunk.payload = f.read(size)
            if chunk_id == b"ds64" and not chunks:
                big_sizes = _parse_ds64(chunk.payload or b"", path)
            chunks.append(chunk)
            pos = payload_at + size + (size & 1)

    fmt = next((c for c in chunks if c.id == b"fmt "), None)
    data = next((c for c in chunks if c.id == b"data"), None)
    if fmt is None or fmt.payload is None or len(fmt.payload) < 16:
        raise WavError(f"{path.name}: missing or short fmt chunk")
    if data is None:
        raise WavError(f"{path.name}: no data chunk")
    format_tag, channels, rate, _byte_rate, block_align, bits = struct.unpack_from("<HHIIHH", fmt.payload)
    if format_tag == WAVE_FORMAT_EXTENSIBLE and len(fmt.payload) >= 40:
        format_tag = struct.unpack_from("<H", fmt.payload, 24)[0]
    if not (channels and rate and block_align):
        raise WavError(f"{path.name}: fmt chunk has zero channels, rate or block size")
    if data.size % block_align:
        warnings.append(f"data ends with a partial frame ({data.size % block_align} bytes); it is ignored")

    bext_chunk = next((c for c in chunks if c.id == b"bext"), None)
    bext = Bext.parse(bext_chunk.payload) if bext_chunk and bext_chunk.payload else None
    return WavInfo(path, riff_id.decode(), file_size, chunks, format_tag, channels, rate,
                   bits, block_align, data, bext, warnings)


def _parse_ds64(body: bytes, path: Path) -> dict:
    if len(body) < DS64_BODY.size:
        raise WavError(f"{path.name}: ds64 chunk is too short")
    _riff, data_size, _samples, table_len = DS64_BODY.unpack_from(body)
    sizes = {b"data": data_size}
    for i in range(table_len):
        at = DS64_BODY.size + i * DS64_TABLE_ENTRY.size
        if at + DS64_TABLE_ENTRY.size > len(body):
            break
        chunk_id, size = DS64_TABLE_ENTRY.unpack_from(body, at)
        sizes[chunk_id] = size
    return sizes


def _plausible_id(chunk_id: bytes) -> bool:
    return len(chunk_id) == 4 and all(0x20 <= b <= 0x7E for b in chunk_id)


def _text(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("latin-1").strip()


def _digits(s: str) -> str:
    return "".join(c for c in s if c.isdigit())


# --- writing ---------------------------------------------------------------

def _chunk_len(payload_size: int) -> int:
    return 8 + payload_size + (payload_size & 1)


def planned_size(fmt: bytes, chunks: Sequence[Tuple[bytes, bytes]], fact: bool, data_size: int) -> int:
    """Exact size of the file WavWriter produces for these inputs."""
    size = 12 + _chunk_len(DS64_BODY.size) + _chunk_len(len(fmt))
    if fact:
        size += _chunk_len(4)
    size += sum(_chunk_len(len(payload)) for _, payload in chunks)
    return size + _chunk_len(data_size)


def container_for(file_size: int, mode: str) -> str:
    """'RIFF' or 'RF64' for a file of file_size bytes under --rf64 auto/always/never."""
    if mode == "always":
        return "RF64"
    if file_size - 8 > U32_MAX:
        if mode == "never":
            raise WavError(f"{file_size:,} bytes is over the 4 GB WAV limit and --rf64 is 'never'")
        return "RF64"
    return "RIFF"


class WavWriter:
    """Streams audio into a new WAV whose container is settled on close().

    The header reserves a JUNK chunk exactly where RF64 puts its ds64 chunk
    (EBU Tech 3306), so one byte layout serves both containers. close() either
    leaves it as JUNK and fills in 32-bit sizes, or rewrites it as ds64 with
    64-bit sizes. The choice is made from the bytes actually written, so RF64
    output doesn't depend on anyone predicting the size correctly.
    """

    def __init__(self, path, fmt: bytes, chunks: Sequence[Tuple[bytes, bytes]],
                 rf64: str = "auto", fact: bool = False):
        self.path = Path(path)
        self.rf64 = rf64
        self.block_align = struct.unpack_from("<H", fmt, 12)[0]
        self.data_size = 0
        self.fact_at = None
        self.f = open(self.path, "wb")
        try:
            self.f.write(b"RIFF\0\0\0\0WAVE")
            self._chunk(b"JUNK", bytes(DS64_BODY.size))
            self._chunk(b"fmt ", fmt)
            if fact:
                self.fact_at = self.f.tell() + 8
                self._chunk(b"fact", bytes(4))
            for chunk_id, payload in chunks:
                self._chunk(chunk_id, payload)
            self.data_size_at = self.f.tell() + 4
            self.f.write(b"data\0\0\0\0")
        except BaseException:
            self.abort()
            raise

    def _chunk(self, chunk_id: bytes, payload: bytes):
        self.f.write(chunk_id + struct.pack("<I", len(payload)) + payload)
        if len(payload) & 1:
            self.f.write(b"\0")

    def write(self, block: bytes):
        self.f.write(block)
        self.data_size += len(block)

    def close(self) -> str:
        """Finalize sizes, fsync, and return the container used."""
        f = self.f
        if self.data_size & 1:
            f.write(b"\0")
        file_size = f.tell()
        container = container_for(file_size, self.rf64)
        frames = self.data_size // self.block_align
        if container == "RF64":
            f.seek(0)
            f.write(b"RF64" + struct.pack("<I", U32_MAX) + b"WAVE")
            f.write(b"ds64" + struct.pack("<I", DS64_BODY.size)
                    + DS64_BODY.pack(file_size - 8, self.data_size, frames, 0))
            data_field = U32_MAX
        else:
            f.seek(4)
            f.write(struct.pack("<I", file_size - 8))
            data_field = self.data_size
        f.seek(self.data_size_at)
        f.write(struct.pack("<I", data_field))
        if self.fact_at is not None:
            f.seek(self.fact_at)
            f.write(struct.pack("<I", min(frames, U32_MAX)))
        f.flush()
        os.fsync(f.fileno())
        f.close()
        return container

    def abort(self):
        """Close and delete the unfinished file."""
        try:
            self.f.close()
        finally:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass


def merge_cues(parts: Sequence[Tuple[Optional[bytes], int]]) -> Tuple[Optional[bytes], int]:
    """Combine consecutive parts' cue chunks into one for the stitched file.

    parts is (cue payload or None, frame offset of that part in the output).
    The first part's cue points keep their IDs, so LIST/adtl labels copied
    from it stay valid; later parts' points are renumbered after them and
    shifted by their offset. Returns (payload or None, points dropped because
    they would land past 2^32 frames).
    """
    points = []
    dropped = 0
    for index, (payload, offset) in enumerate(parts):
        if not payload or len(payload) < 4:
            continue
        count = min(struct.unpack_from("<I", payload)[0], (len(payload) - 4) // CUE_POINT.size)
        for i in range(count):
            cue_id, position, fcc, chunk_start, block_start, sample_offset = \
                CUE_POINT.unpack_from(payload, 4 + i * CUE_POINT.size)
            if index:
                position += offset
                sample_offset += offset
                if max(position, sample_offset) > U32_MAX:
                    dropped += 1
                    continue
                cue_id = None
            points.append([cue_id, position, fcc, chunk_start, block_start, sample_offset])
    if not points:
        return None, dropped
    next_id = max((p[0] for p in points if p[0] is not None), default=0) + 1
    for point in points:
        if point[0] is None:
            point[0] = next_id
            next_id += 1
    body = struct.pack("<I", len(points)) + b"".join(CUE_POINT.pack(*p) for p in points)
    return body, dropped
