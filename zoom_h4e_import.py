#!/usr/bin/env python3
"""
Import recordings from a Zoom H4essential microSD card.

Copies every take off the card, stitches recordings the recorder split at
2 GB back into one file per track, and files them in one dated folder per
session. Every file is verified by checksum. Nothing on the card is ever
modified: it is only opened for reading.

Usage:
    python3 zoom_h4e_import.py ~/Music/Recordings --dry-run
    python3 zoom_h4e_import.py ~/Music/Recordings
    python3 zoom_h4e_import.py ~/Music/Recordings --source /Volumes/ZOOM_H4E --skip-lr
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import string
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from card import (DATE_FORMATS, CardError, Recording, Session, assign_starts, chain_takes,
                  group_sessions, resolve_date_format, scan_card)
from transfer import (StitchLayout, TransferError, copy_verified, fmt_bytes, fmt_duration,
                      stitch_layout, stitch_verified)
from wavfile import WavError, WavInfo

DEFAULT_NAME = "{date}_{track}"
DEFAULT_NAME_MULTI = "{date}_{time}_{track}"
FOLDER_FIELDS = {"date", "time"}
NAME_FIELDS = {"date", "time", "track", "take", "orig"}
EXT = ".wav"
MANIFEST = "import_manifest.json"

PATTERN_HELP = """\
naming patterns:
  {date}   session date, formatted with --date-style (default 2026-09-25)
  {time}   recording start time, HHMMSS
  {track}  track name from the card, e.g. TrMic, Tr1, TrLR
  {take}   recording number within the session: 01, 02, ...
  {orig}   the recording's first folder name on the card, e.g. 260925_213000
  (--folder accepts {date} and {time} only)

A session holding several recordings defaults to {date}_{time}_{track} so
their files don't collide.
"""


@dataclass
class Job:
    kind: str  # "copy", "stitch" or "skip"
    session: Session
    recording: Recording
    label: str  # track name, or the file name for files copied as-is
    sources: List[Path]
    dest: Path
    info: Optional[WavInfo] = None  # first part's header; None for files copied as-is
    layout: Optional[StitchLayout] = None
    size: int = 0
    container: str = ""
    reason: str = ""


# --- setup -------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Import recordings from a Zoom H4essential card: copy, stitch split "
                    "recordings, rename, and verify by checksum. The card is never modified.",
        epilog=PATTERN_HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dest", type=Path, help="folder to import into; one subfolder per session is made inside it")
    p.add_argument("--source", type=Path,
                   help="card root (default: the mounted card whose ZOOM.SYS names the H4essential)")
    p.add_argument("-n", "--dry-run", action="store_true",
                   help="show the planned copies, renames and stitches without writing anything")

    g = p.add_argument_group("dates and sessions")
    g.add_argument("--date-format", choices=("auto",) + DATE_FORMATS, default="auto",
                   help="the recorder's date format setting, as used in folder names "
                        "(default: detect it from BEXT dates and file modification times)")
    g.add_argument("--ignore-date-check", action="store_true",
                   help="import even if the date format disagrees with the files' own dates")
    g.add_argument("--tolerance", type=float, default=2.0, metavar="SECONDS",
                   help="how far a take may start from the previous take's end and still be its "
                        "continuation (default: 2; folder names only have 1-second resolution)")
    g.add_argument("--session-gap", type=float, metavar="MINUTES",
                   help="start a new session after a break this long, instead of one session per day")

    g = p.add_argument_group("naming")
    g.add_argument("--folder", default="{date}", help="session folder pattern (default: {date})")
    g.add_argument("--name", help="file name pattern, without extension (default: {date}_{track})")
    g.add_argument("--date-style", default="%Y-%m-%d", help="strftime format for {date} (default: %%Y-%%m-%%d)")

    g = p.add_argument_group("tracks")
    g.add_argument("--skip-lr", action="store_true", help="don't import TrLR (stereo mix) files")
    g.add_argument("--skip-track", action="append", default=[], metavar="NAME",
                   help="don't import this track, e.g. Tr2 (repeatable)")
    g.add_argument("--only-track", action="append", default=[], metavar="NAME",
                   help="import only these tracks (repeatable)")

    g = p.add_argument_group("output")
    g.add_argument("--rf64", choices=("auto", "always", "never"), default="auto",
                   help="stitched files use RF64 only when over 4 GB (auto), always, or never "
                        "(refuse instead) (default: auto)")
    g.add_argument("--no-stitch", action="store_true",
                   help="copy split parts as separate _part1, _part2... files instead of stitching")
    g.add_argument("--no-ixml", action="store_true",
                   help="leave the first part's iXML chunk out of stitched files")
    g.add_argument("--hash", choices=("blake2b", "sha256", "md5"), default="blake2b",
                   help="checksum used for verification (default: blake2b)")
    g.add_argument("--overwrite", action="store_true",
                   help="replace files already at the destination instead of stopping")
    g.add_argument("--reimport", action="store_true",
                   help="import again even what a session's manifest shows was already imported")

    args = p.parse_args(argv)
    if args.tolerance < 0:
        p.error("--tolerance can't be negative")
    if args.session_gap is not None and args.session_gap <= 0:
        p.error("--session-gap must be positive")
    for flag, pattern, allowed in (("--folder", args.folder, FOLDER_FIELDS),
                                   ("--name", args.name, NAME_FIELDS)):
        if pattern is None:
            continue
        try:
            used = {f for _, f, _, _ in string.Formatter().parse(pattern) if f is not None}
        except ValueError as e:
            p.error(f"{flag} {pattern!r}: {e}")
        if used - allowed:
            p.error(f"{flag} {pattern!r}: unknown placeholder(s) {', '.join(sorted(used - allowed))}; "
                    f"available: {', '.join('{' + f + '}' for f in sorted(allowed))}")
    return args


def find_card() -> Path:
    """The one mounted volume whose ZOOM.SYS says it came from an H4essential."""
    volumes = Path("/Volumes")
    found = []
    for vol in sorted(volumes.iterdir()) if volumes.is_dir() else []:
        try:
            with open(vol / "ZOOM.SYS", "rb") as f:
                if b"H4essential" in f.read(4096):
                    found.append(vol)
        except OSError:
            continue
    if len(found) == 1:
        return found[0]
    if not found:
        raise CardError("no H4essential card found under /Volumes; pass --source")
    raise CardError(f"several H4essential cards are mounted ({', '.join(map(str, found))}); pass --source")


def home(path: Path) -> str:
    text, root = str(path), str(Path.home())
    return "~" + text[len(root):] if text == root or text.startswith(root + os.sep) else text


def render(pattern: str, **fields) -> str:
    name = pattern.format(**fields)
    if not name.strip() or name in (".", "..") or "/" in name or "\0" in name:
        raise CardError(f"pattern {pattern!r} produces an unusable name: {name!r}")
    return name


# --- manifest ----------------------------------------------------------------

def load_manifest(folder: Path) -> dict:
    path = folder / MANIFEST
    if not path.exists():
        return {"tool": "zoom_h4e_import", "files": {}}
    try:
        data = json.loads(path.read_text())
        data.setdefault("files", {})
        return data
    except (OSError, ValueError) as e:
        raise CardError(f"can't read {path}: {e}")


def source_key(card_root: Path, sources: List[Path]) -> list:
    """How a manifest recognizes the same recording on a later import."""
    return [[p.relative_to(card_root).as_posix(), p.stat().st_size] for p in sources]


def record(job: Job, card_root: Path, container: str, digest: str, args, date_format: str):
    manifest = load_manifest(job.dest.parent)
    manifest.update(source=str(card_root), date_format=date_format)
    entry = {
        "action": job.kind,
        "sources": source_key(card_root, job.sources),
        "bytes": job.dest.stat().st_size,
        "container": container,
        "checksum": {"algorithm": args.hash,
                     "of": "audio data (matches the parts' data chunks joined)" if job.kind == "stitch"
                     else "whole file (matches the card)",
                     "value": digest},
        "recording_start": job.recording.start.isoformat(),
        "imported_at": datetime.now().isoformat(timespec="seconds"),
    }
    if job.info:
        frames = job.layout.frames if job.layout else job.info.frames
        entry.update(format=job.info.describe(), sample_rate=job.info.sample_rate,
                     channels=job.info.channels, bits_per_sample=job.info.bits_per_sample,
                     frames=frames, duration_seconds=round(frames / job.info.sample_rate, 6))
        if job.info.bext:
            entry["bext_time_reference"] = job.info.bext.time_reference
    manifest["files"][job.dest.name] = entry
    path = job.dest.parent / MANIFEST
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(tmp, path)


# --- planning ----------------------------------------------------------------

def name_sessions(sessions: List[Session], args):
    used = set()
    for session in sessions:
        start = session.recordings[0].start
        name = render(args.folder, date=session.day.strftime(args.date_style), time=f"{start:%H%M%S}")
        if name in used:  # only possible with --session-gap or a folder pattern without {date}
            name = f"{name}_{start:%H%M%S}"
        used.add(name)
        session.folder = name


def plan(card_root: Path, dest_root: Path, sessions: List[Session], args) -> tuple:
    jobs: List[Job] = []
    notes: List[str] = []
    stitch_notes = defaultdict(list)  # the same note usually applies to every track of a recording
    skip = {n.lower() for n in args.skip_track} | ({"trlr"} if args.skip_lr else set())
    only = {n.lower() for n in args.only_track}
    seen_tracks = set()

    def excluded(track: str) -> str:
        t = track.lower()
        if t in skip:
            return "--skip-lr" if t == "trlr" and args.skip_lr else "--skip-track"
        if only and t not in only:
            return "not an --only-track"
        return ""

    for session in sessions:
        folder = dest_root / session.folder
        # A folder that already holds another recording (from an earlier card)
        # counts too, or the default name would collide with it.
        starts = {rec.start.isoformat() for rec in session.recordings}
        earlier = {e.get("recording_start") for e in load_manifest(folder)["files"].values()} - starts
        multi = len(session.recordings) > 1 or bool(earlier)
        pattern = args.name or (DEFAULT_NAME_MULTI if multi else DEFAULT_NAME)
        for index, rec in enumerate(session.recordings, 1):
            fields = dict(date=session.day.strftime(args.date_style), time=f"{rec.start:%H%M%S}",
                          take=f"{index:02d}", orig=rec.takes[0].name)
            for track in rec.track_names:
                seen_tracks.add(track.lower())
                parts = [take.track(track) for take in rec.takes]
                stem = render(pattern, track=track, **fields)
                if args.no_stitch and len(parts) > 1:
                    groups = [([p], f"{stem}_part{n}") for n, p in enumerate(parts, 1)]
                else:
                    groups = [(parts, stem)]
                for group, name in groups:
                    job = Job("copy", session, rec, track, [p.path for p in group], folder / (name + EXT),
                              info=group[0].info)
                    job.reason = excluded(track)
                    if job.reason:
                        job.kind = "skip"
                    elif len(group) > 1:
                        job.kind = "stitch"
                        job.layout = stitch_layout([p.info for p in group], keep_ixml=not args.no_ixml)
                        job.size = job.layout.size
                        try:
                            job.container = job.layout.container(args.rf64)
                        except WavError as e:
                            raise CardError(f"{job.dest.name}: {e}")
                        for note in job.layout.notes:
                            stitch_notes[note].append(job.dest.name)
                    else:
                        job.size = group[0].info.file_size
                        job.container = group[0].info.container
                    jobs.append(job)
            for take in rec.takes:
                for path, why in take.extras:
                    job = Job("copy", session, rec, path.name, [path], folder / path.name,
                              size=path.stat().st_size)
                    prefix = take.name + "_"
                    if path.stem.startswith(prefix) and len(path.stem) > len(prefix):
                        job.reason = excluded(path.stem[len(prefix):])
                    if job.reason:
                        job.kind = "skip"
                    else:
                        notes.append(f"{path.relative_to(card_root)}: {why}; copied as-is")
                    jobs.append(job)

    notes.extend(f"{note}: {', '.join(names)}" for note, names in stitch_notes.items())
    unmatched = sorted((skip - {"trlr"} if args.skip_lr else skip) | only) if seen_tracks else []
    for name in unmatched:
        if name not in seen_tracks:
            notes.append(f"no track named {name!r} on the card (tracks seen: {', '.join(sorted(seen_tracks))})")

    # Two planned files with one name would silently overwrite each other.
    targets = defaultdict(list)
    for job in jobs:
        if job.kind != "skip":
            targets[job.dest].append(job)
    clashes = [dest for dest, same in targets.items() if len(same) > 1]
    if clashes:
        raise CardError(f"the naming pattern gives several files the name {clashes[0].name}; "
                        "add {time} or {take} to --name")

    # Recordings a manifest already lists are skipped; any other existing file is a conflict.
    conflicts = []
    manifests: Dict[Path, dict] = {}
    for job in jobs:
        if job.kind == "skip":
            continue
        folder = job.dest.parent
        if folder not in manifests:
            manifests[folder] = load_manifest(folder)
        files = manifests[folder]["files"]
        key = source_key(card_root, job.sources)
        prior = next((name for name, entry in files.items() if entry.get("sources") == key), None)
        if prior and not args.reimport:
            existing = folder / prior
            if existing.exists() and existing.stat().st_size == files[prior].get("bytes"):
                job.kind = "skip"
                job.reason = "already imported" + (f" as {prior}" if prior != job.dest.name else "")
                continue
        if job.dest.exists():
            conflicts.append(job.dest)
    if conflicts and not args.overwrite:
        listed = "\n  ".join(home(c) for c in conflicts[:5])
        more = f"\n  ...and {len(conflicts) - 5} more" if len(conflicts) > 5 else ""
        raise CardError(f"{len(conflicts)} destination file(s) already exist and aren't recorded as "
                        f"imports of these recordings:\n  {listed}{more}\n"
                        "Choose a different --name or --folder, or use --overwrite to replace them.")
    return jobs, notes


def card_notes(recordings: List[Recording]) -> List[str]:
    notes = []
    for rec in recordings:
        for take in rec.takes:
            durations = [t.info.duration for t in take.tracks.values()]
            if durations and max(durations) - min(durations) > 0.05:
                notes.append(f"{take.name}: tracks differ in length "
                             f"({min(durations):.2f}s to {max(durations):.2f}s)")
            for track in take.tracks.values():
                notes.extend(f"{track.path.name}: {w}" for w in track.info.warnings)
    return notes


# --- output ------------------------------------------------------------------

def print_plan(source, dest, date_format, n_takes, recordings, sessions, jobs, notes, free):
    print(f"Card     {home(source)}")
    print(f"Import   {home(dest)}")
    print(f"Dates    {date_format.explanation}")
    print(f"Found    {n_takes} take folder(s) -> {len(recordings)} recording(s) "
          f"in {len(sessions)} session(s)")

    by_recording = defaultdict(list)
    for job in jobs:
        by_recording[id(job.recording)].append(job)

    def left(job: Job) -> str:
        if job.kind == "stitch":
            return f"{job.label} x{len(job.sources)} parts"
        return job.sources[0].name

    width = max((len(left(j)) for j in jobs), default=0)
    dest_width = max((len(j.dest.name) for j in jobs if j.kind != "skip"), default=0)
    for session in sessions:
        print(f"\n{session.folder}/")
        for rec in session.recordings:
            where = (rec.takes[0].name if len(rec.takes) == 1 else
                     f"split across {len(rec.takes)} folders: " + ", ".join(t.name for t in rec.takes))
            print(f"  {rec.start:%Y-%m-%d %H:%M:%S}  {fmt_duration(rec.duration)}  {where}")
            for link in rec.links:
                print(f"      {link}")
            for job in by_recording[id(rec)]:
                if job.kind == "skip":
                    print(f"    skip    {left(job):<{width}}  ({job.reason})")
                    continue
                detail = job.info.describe() if job.info else "copied as-is"
                size = fmt_bytes(job.size) + (f" {job.container}" if job.kind == "stitch" else "")
                print(f"    {job.kind:<7} {left(job):<{width}} -> {job.dest.name:<{dest_width}}  "
                      f"{detail}, {size}")

    if notes:
        print("\nNotes")
        for note in notes:
            print(f"  - {note}")

    todo = [j for j in jobs if j.kind != "skip"]
    copies = sum(1 for j in todo if j.kind == "copy")
    stitches = sum(1 for j in todo if j.kind == "stitch")
    skipped = len(jobs) - len(todo)
    print(f"\nPlan: {copies} cop{'y' if copies == 1 else 'ies'}, {stitches} stitch{'es' * (stitches != 1)}, "
          f"{skipped} skipped. {fmt_bytes(sum(j.size for j in todo))} to write, {fmt_bytes(free)} free.")


def execute(jobs: List[Job], card_root: Path, dest_root: Path, args, date_format: str) -> List[str]:
    todo = [j for j in jobs if j.kind != "skip"]
    failures = []
    for i, job in enumerate(todo, 1):
        rel = job.dest.relative_to(dest_root)
        what = (f"{job.label} from {len(job.sources)} parts" if job.kind == "stitch"
                else job.sources[0].name)
        print(f"\n[{i}/{len(todo)}] {job.kind} {what} -> {rel} ({fmt_bytes(job.size)})", flush=True)
        job.dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            if job.kind == "stitch":
                container, digest = stitch_verified(job.layout, job.dest, args.hash, args.rf64)
            else:
                container, digest = job.container, copy_verified(job.sources[0], job.dest, args.hash)
            record(job, card_root, container, digest, args, date_format)
        except (TransferError, WavError, OSError) as e:
            print(f"    FAILED: {e}", flush=True)
            failures.append(f"{rel}: {e}")
    return failures


def existing_ancestor(path: Path) -> Path:
    path = path.resolve()
    while not path.exists():
        path = path.parent
    return path


# --- main --------------------------------------------------------------------

def run(args) -> int:
    source = (args.source or find_card()).expanduser()
    if not source.is_dir():
        raise CardError(f"{source} is not a folder")
    dest = args.dest.expanduser()
    if dest.exists() and not dest.is_dir():
        raise CardError(f"{dest} exists and isn't a folder")
    src_real, dest_real = source.resolve(), dest.resolve()
    if dest_real == src_real or src_real in dest_real.parents:
        raise CardError("the destination is on the card itself; import to a folder on your computer")

    scan = scan_card(source)
    if not scan.takes:
        print(f"No recordings found on {source}.")
        for note in scan.notes:
            print(f"  - {note}")
        return 0

    date_format = resolve_date_format(scan.takes, args.date_format, args.ignore_date_check)
    assign_starts(scan.takes, date_format.fmt)
    recordings, chain_notes = chain_takes(scan.takes, args.tolerance)
    sessions = group_sessions(recordings, args.session_gap)
    name_sessions(sessions, args)
    jobs, plan_notes = plan(source, dest, sessions, args)
    notes = date_format.warnings + chain_notes + card_notes(recordings) + plan_notes + scan.notes

    free = shutil.disk_usage(existing_ancestor(dest)).free
    print_plan(source, dest, date_format, len(scan.takes), recordings, sessions, jobs, notes, free)
    todo = [j for j in jobs if j.kind != "skip"]
    if args.dry_run:
        print("Dry run: nothing was written.")
        return 0
    if not todo:
        print("Nothing new to import.")
        return 0
    needed = sum(j.size for j in todo)
    if needed > free:
        raise CardError(f"this import needs {fmt_bytes(needed)} but only {fmt_bytes(free)} is free")

    failures = execute(jobs, source, dest, args, date_format.fmt)
    done = len(todo) - len(failures)
    print(f"\nImported {done} of {len(todo)} file(s) into {home(dest)}, all verified by {args.hash} checksum."
          if not failures else f"\n{len(failures)} of {len(todo)} file(s) failed:")
    for failure in failures:
        print(f"  - {failure}")
    if failures:
        print("Nothing on the card was changed. Fix the problem and re-run; finished files are skipped.")
    return 1 if failures else 0


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except CardError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted. The unfinished file was removed; finished files are listed in each "
              "session's import_manifest.json, so re-running picks up where this stopped.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
