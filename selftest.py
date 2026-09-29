#!/usr/bin/env python3
"""
End-to-end self-test: builds synthetic H4essential cards and imports them.

A real 2 GB split takes ~47 minutes of recording at 96 kHz, so this
fabricates cards that follow the documented layout instead: 32-bit float
WAVs with bext/iXML/cue chunks, split takes (including one across midnight),
Trash/Export folders and macOS AppleDouble files. It then checks the import
sample by sample, and checks that the card is byte-for-byte untouched.

    python3 selftest.py            # small files, a few seconds
    python3 selftest.py --big      # also stitches two 2 GiB parts into a
                                   # 4 GiB RF64 file (needs ~9 GB free)
"""

import argparse
import filecmp
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from array import array
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOL = HERE / "zoom_h4e_import.py"
sys.path.insert(0, str(HERE))
from wavfile import U32_MAX, read_wav  # noqa: E402

GUID_FLOAT = bytes.fromhex("0300000000001000800000aa00389b71")
failures = []


def check(condition, message):
    print(("  ok    " if condition else "  FAIL  ") + message)
    if not condition:
        failures.append(message)


# --- fixture writing ---------------------------------------------------------

def chunk(cid, payload):
    return cid + struct.pack("<I", len(payload)) + payload + (b"\0" if len(payload) & 1 else b"")


def fmt_chunk(rate, channels, extensible):
    align = channels * 4
    if extensible:
        return struct.pack("<HHIIHHHHI16s", 0xFFFE, channels, rate, rate * align, align, 32,
                           22, 32, 3 if channels == 2 else 4, GUID_FLOAT)
    return struct.pack("<HHIIHHH", 3, channels, rate, rate * align, align, 32, 0)


def bext_chunk(when, time_ref):
    body = struct.pack("<256s32s32s10s8sQH", b"test", b"ZOOM H4essential", b"ref",
                       when.strftime("%Y-%m-%d").encode(), when.strftime("%H:%M:%S").encode(),
                       time_ref, 1)
    return body + bytes(602 - len(body)) + b"A=PCM,F=96000,W=32\r\n"


def ramp(first, frames, channels, part_tag=0.0):
    """Frame n (counted across the whole recording) holds n + 0.5*channel."""
    values = array("f", bytes(frames * channels * 4))
    for c in range(channels):
        values[c::channels] = array("f", [first + i + 0.5 * c + part_tag for i in range(frames)])
    return values.tobytes()


def write_wav(path, rate, channels, frames, first, when, time_ref, *, extensible=False,
              cues=(), label=None, data_size_field=None):
    audio = ramp(first, frames, channels)
    head = [chunk(b"bext", bext_chunk(when, time_ref)),
            chunk(b"fmt ", fmt_chunk(rate, channels, extensible)),
            chunk(b"fact", struct.pack("<I", frames)),
            chunk(b"iXML", b"<?xml version=\"1.0\"?><BWFXML><PROJECT>selftest</PROJECT></BWFXML>")]
    size = len(audio) if data_size_field is None else data_size_field
    body = b"".join(head) + b"data" + struct.pack("<I", size) + audio
    if cues:  # after the data chunk, like a recorder that appends markers at stop
        body += chunk(b"cue ", struct.pack("<I", len(cues)) + b"".join(
            struct.pack("<II4sIII", i + 1, pos, b"data", 0, 0, pos) for i, pos in enumerate(cues)))
        if label:
            note = struct.pack("<I", 1) + label + b"\0"
            body += chunk(b"LIST", b"adtl" + chunk(b"labl", note))
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body) + 4) + b"WAVE" + body)
    end = when + timedelta(seconds=frames / rate)
    os.utime(path, (end.timestamp(), end.timestamp()))


def make_take(card, when, tracks, rate, frames, first=0, time_ref=None, date_fmt="%y%m%d", **kw):
    folder = card / f"{when.strftime(date_fmt)}_{when:%H%M%S}"
    folder.mkdir(parents=True)
    if time_ref is None:
        time_ref = (when.hour * 3600 + when.minute * 60 + when.second) * rate
    for name, channels in tracks.items():
        write_wav(folder / f"{folder.name}_{name}.WAV", rate, channels, frames, first, when, time_ref,
                  extensible=(channels == 2), **kw)
    return folder


def split(card, start, tracks, rate, part_frames, date_fmt="%y%m%d", ref_error=0):
    """A recording split into parts whose folder names carry the (truncated)
    continuation time, as the recorder does at 2 GB."""
    t0 = start.hour * 3600 + start.minute * 60 + start.second
    first = 0
    for i, frames in enumerate(part_frames):
        exact = start + timedelta(seconds=first / rate)
        when = exact.replace(microsecond=0)
        ref = ((t0 * rate + first) % (86400 * rate)) + (ref_error if i else 0)
        cues = (1000,) if i == 0 else (500,)
        make_take(card, when, tracks, rate, frames, first, ref, date_fmt,
                  cues=cues, label=b"mark" if i < 2 else None)
        first += frames


STD = {"TrMic": 2, "Tr1": 1, "Tr2": 1}


def build_card_a(card):
    card.mkdir(parents=True)
    (card / "ZOOM.SYS").write_bytes(b"ZOOM SYSTEM FILE H4essential")
    split(card, datetime(2026, 9, 25, 21, 30, 0), STD, 96000, [240000, 240000, 115200])
    make_take(card, datetime(2026, 9, 25, 22, 0, 0), STD, 96000, 240000)
    make_take(card, datetime(2026, 9, 25, 22, 0, 6), STD, 96000, 96000)  # 3.5 s after the last ended
    make_take(card, datetime(2026, 9, 25, 23, 10, 0), dict(STD, TrLR=2), 48000, 48000)
    split(card, datetime(2026, 9, 25, 23, 59, 59), STD, 96000, [240000, 96000])  # over midnight
    folder = make_take(card, datetime(2026, 9, 27, 10, 0, 0), {"TrMic": 2, "Tr1-2": 2}, 96000, 48000)
    (folder / f"{folder.name}_TrX.WAV").write_bytes(b"not a wav at all")
    shutil.copy(folder / f"{folder.name}_TrMic.WAV", folder / f"._{folder.name}_TrMic.WAV")
    (card / "stray.wav").write_bytes(b"RIFF")
    make_take(card / "Trash", datetime(2026, 9, 20, 9, 0, 0), STD, 96000, 1000)
    (card / "Export").mkdir()
    shutil.copy(folder / f"{folder.name}_TrMic.WAV", card / "Export" / "export.WAV")


def snapshot(root):
    state = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            p = Path(dirpath) / f
            st = p.stat()
            state[str(p)] = (st.st_size, st.st_mtime, hashlib.sha256(p.read_bytes()).hexdigest())
    return state


def run(card, dest, *args):
    result = subprocess.run([sys.executable, str(TOOL), str(dest), "--source", str(card), *args],
                            capture_output=True, text=True)
    return result.returncode, result.stdout + result.stderr


def samples(path, first_frame, count):
    info = read_wav(path)
    with open(path, "rb") as f:
        f.seek(info.data.offset + first_frame * info.block_align)
        return array("f", f.read(count * info.block_align)), info


# --- tests -------------------------------------------------------------------

def test_small(work):
    card, out = work / "cardA", work / "out"
    build_card_a(card)
    before = snapshot(card)

    print("dry run")
    code, text = run(card, out, "--dry-run")
    check(code == 0, f"dry run exits 0 (got {code})")
    check(not out.exists(), "dry run creates nothing")
    check("YYMMDD (auto-detected" in text, "date format auto-detected as YYMMDD")
    check("split across 3 folders" in text, "3-part split detected")
    check("BEXT time reference sample-exact" in text, "continuation confirmed by BEXT time reference")
    check("outside the ±2s continuation tolerance" in text, "3.5 s gap reported as a near miss")
    check("Trash/: 3 WAV file(s) ignored" in text and "Export/: 1 WAV" in text, "Trash and Export ignored")
    check("stray.wav: not inside a DATE_TIME take folder" in text, "stray root WAV reported")
    check("._" not in text, "AppleDouble files ignored")
    check("Tr1-2" in text, "unknown stereo-linked suffix detected as a track")

    print("import")
    code, text = run(card, out)
    check(code == 0, f"import exits 0 (got {code})" + ("" if code == 0 else "\n" + text))
    s25, s27 = out / "2026-09-25", out / "2026-09-27"
    expected = [s25 / f"2026-09-25_{t}_{n}.wav" for t in ("213000", "220000", "220006", "231000", "235959")
                for n in STD] + [s25 / "2026-09-25_231000_TrLR.wav", s27 / "2026-09-27_TrMic.wav",
                                 s27 / "2026-09-27_Tr1-2.wav", s27 / "260927_100000_TrX.WAV"]
    missing = [str(p.relative_to(out)) for p in expected if not p.exists()]
    check(not missing, f"all {len(expected)} expected files written" + (f"; missing {missing}" if missing else ""))
    check(not list(out.rglob("*.partial")), "no .partial files left behind")

    stitched = s25 / "2026-09-25_213000_TrMic.wav"
    got, info = samples(stitched, 0, 595200)
    check(info.container == "RIFF" and info.frames == 595200, f"stitched TrMic has 595200 frames ({info.frames})")
    check(got == array("f", ramp(0, 595200, 2)), "stitched TrMic is sample-exact across both joins")
    part1 = read_wav(card / "260925_213000" / "260925_213000_TrMic.WAV")
    check(info.payload(b"bext") == part1.payload(b"bext"), "first part's bext carried over byte-for-byte")
    check(struct.unpack("<I", info.payload(b"fact"))[0] == 595200, "fact chunk rewritten with the new length")
    cue = info.payload(b"cue ")
    points = [struct.unpack_from("<II4sIII", cue, 4 + 24 * i) for i in range(struct.unpack_from("<I", cue)[0])]
    check([(p[0], p[1]) for p in points] == [(1, 1000), (2, 240500), (3, 480500)],
          f"markers merged and shifted {[(p[0], p[1]) for p in points]}")
    check(info.payload(b"iXML") is not None, "iXML carried over")

    mono, _ = samples(s25 / "2026-09-25_213000_Tr1.wav", 0, 595200)
    check(mono == array("f", ramp(0, 595200, 1)), "stitched mono Tr1 is sample-exact")
    night, ninfo = samples(s25 / "2026-09-25_235959_TrMic.wav", 0, 336000)
    check(night == array("f", ramp(0, 336000, 2)), "split across midnight stitched sample-exactly")

    copy_src = card / "260925_231000" / "260925_231000_TrLR.WAV"
    copy_dst = s25 / "2026-09-25_231000_TrLR.wav"
    check(filecmp.cmp(copy_src, copy_dst, shallow=False), "unsplit take copied byte-for-byte")
    check(int(copy_src.stat().st_mtime) == int(copy_dst.stat().st_mtime), "copy keeps the card's mtime")

    manifest = json.loads((s25 / "import_manifest.json").read_text())
    entry = manifest["files"]["2026-09-25_213000_TrMic.wav"]
    check(entry["action"] == "stitch" and len(entry["sources"]) == 3 and entry["frames"] == 595200,
          "manifest records the stitch and its sources")
    check(snapshot(card) == before, "card is byte-for-byte unchanged, mtimes included")

    print("re-run")
    mtimes = {p: p.stat().st_mtime for p in out.rglob("*.wav")}
    code, text = run(card, out)
    check(code == 0 and "Nothing new to import" in text, "second run finds nothing new")
    check(all(p.stat().st_mtime == m for p, m in mtimes.items()), "second run rewrites nothing")

    print("options")
    code, text = run(card, work / "o1", "--skip-lr")
    check(code == 0 and not list((work / "o1").rglob("*TrLR*")), "--skip-lr leaves TrLR out")

    code, text = run(card, work / "o2", "--rf64", "always", "--only-track", "TrMic")
    rf = read_wav(work / "o2" / "2026-09-25" / "2026-09-25_213000_TrMic.wav")
    got, _ = samples(rf.path, 0, 595200)
    check(code == 0 and rf.container == "RF64" and got == array("f", ramp(0, 595200, 2)),
          "--rf64 always writes a readable, sample-exact RF64")
    written = sorted(p.name for p in (work / "o2").rglob("*") if p.suffix.lower() == ".wav")
    check(len(written) == 6 and all("TrMic" in n for n in written), f"--only-track TrMic imports only TrMic {written}")
    third_party(rf.path, 595200, 96000)
    third_party(copy_dst, 48000, 48000)

    code, text = run(card, work / "o3", "--no-stitch", "--only-track", "Tr1")
    parts = sorted(p.name for p in (work / "o3" / "2026-09-25").glob("*213000_Tr1_part*"))
    check(code == 0 and len(parts) == 3, f"--no-stitch copies parts separately {parts}")

    code, text = run(card, work / "o4", "--date-format", "DDMMYY", "--dry-run")
    check(code == 2 and "contradicts the card" in text and "consistent with YYMMDD" in text,
          "wrong --date-format is caught by the BEXT check")
    code, text = run(card, work / "o4", "--date-format", "DDMMYY", "--ignore-date-check", "--dry-run")
    check(code == 0 and "2025-09-26/" in text, "--ignore-date-check proceeds with the given format")

    (work / "o5" / "2026-09-27").mkdir(parents=True)
    (work / "o5" / "2026-09-27" / "2026-09-27_TrMic.wav").write_bytes(b"mine")
    code, text = run(card, work / "o5")
    check(code == 2 and "already exist" in text and not (work / "o5" / "2026-09-25").exists(),
          "existing file is a conflict, caught before anything is written")
    code, text = run(card, work / "o5", "--overwrite")
    check(code == 0 and (work / "o5" / "2026-09-27" / "2026-09-27_TrMic.wav").stat().st_size > 4,
          "--overwrite replaces it")

    code, text = run(card, card / "inside")
    check(code == 2 and "on the card itself" in text and not (card / "inside").exists(),
          "refuses a destination on the card")
    check(snapshot(card) == before, "card still unchanged after every option")


def test_other_card(work):
    """MMDDYY names, a BEXT gap the folder names hide, and an unfinalized header."""
    print("MMDDYY card")
    card = work / "cardB"
    card.mkdir()
    split(card, datetime(2026, 3, 4, 8, 0, 0), STD, 96000, [240000, 96000], date_fmt="%m%d%y")
    # Names say continuous, but BEXT says the second part starts 3 s late.
    split(card, datetime(2026, 3, 4, 9, 0, 0), STD, 96000, [240000, 96000], date_fmt="%m%d%y",
          ref_error=3 * 96000)
    folder = card / "030526_120000"
    folder.mkdir()
    write_wav(folder / "030526_120000_TrMic.WAV", 96000, 2, 48000, 0, datetime(2026, 3, 5, 12), 0,
              data_size_field=0)
    code, text = run(card, work / "outB")
    check(code == 0 and "MMDDYY (auto-detected" in text, "MMDDYY detected")
    check("2026-03-04/" in text and "2026-03-05/" in text, "MMDDYY dates parsed correctly")
    check("BEXT time references show a +3.000s gap" in text, "BEXT gap overrides matching folder names")
    check("never written" in text, "unfinalized data size reported")
    got, info = samples(work / "outB" / "2026-03-04" / "2026-03-04_080000_TrMic.wav", 0, 336000)
    check(got == array("f", ramp(0, 336000, 2)), "MMDDYY split stitched sample-exactly")


def test_second_card(work):
    """A later card with a new recording on a date that's already imported."""
    print("second card, same date")
    out = work / "outD"
    make_take(work / "cardD1", datetime(2026, 10, 1, 8, 0, 0), STD, 96000, 9600)
    make_take(work / "cardD2", datetime(2026, 10, 1, 18, 0, 0), STD, 96000, 9600)
    code, _ = run(work / "cardD1", out)
    check(code == 0 and (out / "2026-10-01" / "2026-10-01_TrMic.wav").exists(),
          "first card's only recording gets the plain {date}_{track} name")
    code, text = run(work / "cardD2", out)
    check(code == 0 and (out / "2026-10-01" / "2026-10-01_180000_TrMic.wav").exists(),
          "second card's recording gets a time-stamped name instead of colliding")
    files = json.loads((out / "2026-10-01" / "import_manifest.json").read_text())["files"]
    check(len(files) == 6, f"manifest lists both imports ({len(files)} files)")


def third_party(path, frames, rate):
    """ffprobe and afinfo (Core Audio) must agree on the length."""
    if shutil.which("ffprobe"):
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=duration_ts,sample_rate",
                              "-of", "json", str(path)], capture_output=True, text=True).stdout
        stream = json.loads(out)["streams"][0]
        check(int(stream["duration_ts"]) == frames, f"ffprobe reads {path.name}: {stream['duration_ts']} frames")
    if shutil.which("afinfo"):
        out = subprocess.run(["afinfo", str(path)], capture_output=True, text=True).stdout
        check(f"{frames} valid frames" in out or f"estimated duration: {frames / rate:.6f}" in out
              or f"{frames / rate:.3f}" in out, f"afinfo reads {path.name}")


def test_big(work):
    """Two parts of exactly 2 GiB of audio: the stitched data is 2^32 bytes,
    one past what a RIFF size field can hold."""
    print("4 GiB RF64 stitch")
    card, out = work / "cardC", work / "outC"
    rate, channels, block = 96000, 2, 1 << 20
    part_frames = (1 << 31) // (channels * 4)
    start = datetime(2026, 9, 25, 20, 0, 0)
    for part in range(2):
        first = part * part_frames
        when = (start + timedelta(seconds=first / rate)).replace(microsecond=0)
        folder = card / f"{when:%y%m%d_%H%M%S}"
        folder.mkdir(parents=True)
        path = folder / f"{folder.name}_TrMic.WAV"
        pattern = ramp(0, block, channels, part_tag=0.125 * part)
        head = (chunk(b"fmt ", fmt_chunk(rate, channels, True))
                + chunk(b"bext", bext_chunk(when, (72000 * rate + first) % (86400 * rate))))
        with open(path, "wb") as f:
            f.write(b"RIFF" + struct.pack("<I", 4 + len(head) + 8 + (1 << 31)) + b"WAVE" + head)
            f.write(b"data" + struct.pack("<I", 1 << 31))
            for _ in range(part_frames // block):
                f.write(pattern)
        end = (when + timedelta(seconds=part_frames / rate)).timestamp()
        os.utime(path, (end, end))
    code, text = run(card, out)
    check(code == 0, f"4 GiB import exits 0 (got {code})" + ("" if code == 0 else "\n" + text[-2000:]))
    path = out / "2026-09-25" / "2026-09-25_TrMic.wav"
    info = read_wav(path)
    check(info.container == "RF64" and info.data.size == 1 << 32 and info.frames == 2 * part_frames,
          f"auto-selected RF64 holding 2^32 bytes of audio ({info.container}, {info.data.size:,})")
    with open(path, "rb") as f:
        riff, riff_field = struct.unpack("<4sI", f.read(8))
        f.seek(12)
        ds64, ds64_size, riff64, data64, samples64, table = struct.unpack("<4sIQQQI", f.read(36))
        f.seek(info.data.offset - 4)
        data_field = struct.unpack("<I", f.read(4))[0]
    check(riff == b"RF64" and riff_field == U32_MAX, "RF64 header with a -1 RIFF size")
    check((ds64, ds64_size, riff64, data64, samples64, table)
          == (b"ds64", 28, path.stat().st_size - 8, 1 << 32, 2 * part_frames, 0),
          "ds64 holds the true RIFF size, data size and sample count")
    check(data_field == U32_MAX, "data chunk's 32-bit size is -1, per RF64")
    edge = part_frames - 1
    got, _ = samples(path, edge, 2)
    want = array("f", [(edge % block) + 0.0, (edge % block) + 0.5, 0.125, 0.625])
    check(got == want, f"frames either side of the join are the parts' last and first ({list(got)})")
    last, _ = samples(path, 2 * part_frames - 1, 1)
    check(last == array("f", [block - 1 + 0.125, block - 1 + 0.625]), "final frame intact past 4 GiB")
    third_party(path, 2 * part_frames, rate)
    if shutil.which("ffmpeg"):
        tail = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{(2 * part_frames - 4) / rate:.6f}", "-i",
                               str(path), "-f", "f32le", "-"], capture_output=True).stdout
        decoded = array("f", tail)
        check(len(decoded) >= 2 and decoded[-2:] == last,
              "ffmpeg decodes the last frame past the 4 GiB mark correctly")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--big", action="store_true", help="also run the 4 GiB RF64 test")
    p.add_argument("--workdir", type=Path, help="where to build test cards (default: a temp dir)")
    p.add_argument("--keep", action="store_true", help="don't delete the test files afterwards")
    args = p.parse_args()
    work = Path(tempfile.mkdtemp(prefix="h4e_selftest_", dir=args.workdir))
    try:
        test_small(work)
        test_other_card(work)
        test_second_card(work)
        if args.big:
            test_big(work)
    finally:
        if args.keep:
            print(f"test files kept in {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)
    print(f"\n{'FAILED: ' + str(len(failures)) if failures else 'all checks passed'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
