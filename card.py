"""
Reading a Zoom H4essential card: take folders, their dates, and how they chain.

    card -> Take (one DATE_TIME folder) -> Recording (takes chained by
    continuation) -> Session (recordings sharing one dated output folder)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from wavfile import WavError, WavInfo, read_wav

TAKE_DIR_RE = re.compile(r"^(\d{6})_(\d{6})$")
IGNORED_DIRS = {"trash", "export"}
DATE_FORMATS = ("YYMMDD", "MMDDYY", "DDMMYY")
SECONDS_PER_DAY = 86400
# A take starting this close to the previous one's end, but outside the
# continuation tolerance, is reported so the tolerance can be adjusted.
NEAR_MISS_SECONDS = 60


class CardError(Exception):
    """The card can't be interpreted safely; the message says why."""


@dataclass
class Track:
    name: str  # file name suffix after DATE_TIME_, e.g. "TrMic", "Tr1", "TrLR"
    path: Path
    info: WavInfo
    mtime: float


@dataclass
class Take:
    folder: Path
    date_digits: str
    time_digits: str
    tracks: Dict[str, Track] = field(default_factory=dict)
    # WAVs in the folder that can't be treated as tracks; still copied verbatim.
    extras: List[Tuple[Path, str]] = field(default_factory=list)
    start: Optional[datetime] = None  # set once the date format is known

    @property
    def name(self) -> str:
        return self.folder.name

    @property
    def duration(self) -> float:
        return max((t.info.duration for t in self.tracks.values()), default=0.0)

    @property
    def end(self) -> datetime:
        return self.start + timedelta(seconds=self.duration)

    def track(self, name: str) -> Optional[Track]:
        return next((t for n, t in self.tracks.items() if n.lower() == name.lower()), None)

    def track_key(self) -> frozenset:
        return frozenset(n.lower() for n in self.tracks)

    def bext_date(self) -> Optional[date]:
        for track in self.tracks.values():
            if track.info.bext:
                day, _ = track.info.bext.origination()
                if day:
                    return day
        return None


@dataclass
class Recording:
    takes: List[Take]
    links: List[str] = field(default_factory=list)  # how each continuation was confirmed

    @property
    def start(self) -> datetime:
        return self.takes[0].start

    @property
    def end(self) -> datetime:
        return self.takes[-1].end

    @property
    def duration(self) -> float:
        return sum(t.duration for t in self.takes)

    @property
    def track_names(self) -> List[str]:
        return list(self.takes[0].tracks)


@dataclass
class Session:
    day: date
    recordings: List[Recording]
    folder: str = ""  # output folder name, set by the planner


# --- scanning --------------------------------------------------------------

@dataclass
class Scan:
    takes: List[Take]
    notes: List[str]  # what on the card was left alone, and why


def scan_card(root: Path) -> Scan:
    takes: List[Take] = []
    notes: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        for d in sorted(dirnames):
            if d.lower() in IGNORED_DIRS:
                count = _count_wavs(here / d)
                if count:
                    notes.append(f"{_rel(here / d, root)}/: {count} WAV file(s) ignored")
        dirnames[:] = sorted(d for d in dirnames
                             if not d.startswith(".") and d.lower() not in IGNORED_DIRS)
        wavs = sorted(f for f in filenames if f.lower().endswith(".wav") and not f.startswith("."))

        match = TAKE_DIR_RE.match(here.name)
        if match and _parse_time(match.group(2)) is None:
            notes.append(f"{_rel(here, root)}/: {match.group(2)} isn't a valid time; not treated as a take")
            match = None
        if not match:
            notes.extend(f"{_rel(here / f, root)}: not inside a DATE_TIME take folder; skipped"
                         for f in wavs)
            continue

        take = Take(here, match.group(1), match.group(2))
        prefix = here.name + "_"
        for filename in wavs:
            path = here / filename
            stem = filename[:-4]
            if not stem.startswith(prefix) or len(stem) == len(prefix):
                take.extras.append((path, "name doesn't follow DATE_TIME_TRACK.WAV"))
                continue
            try:
                info = read_wav(path)
            except (WavError, OSError) as e:
                reason = str(e).replace(f"{filename}: ", "", 1)
                take.extras.append((path, f"can't be read as a WAV ({reason})"))
                continue
            take.tracks[stem[len(prefix):]] = Track(stem[len(prefix):], path, info, path.stat().st_mtime)
        if take.tracks or take.extras:
            takes.append(take)
        else:
            notes.append(f"{_rel(here, root)}/: take folder has no WAV files")
    return Scan(takes, notes)


def _count_wavs(folder: Path) -> int:
    return sum(1 for _, _, files in os.walk(folder)
               for f in files if f.lower().endswith(".wav") and not f.startswith("."))


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


# --- dates -----------------------------------------------------------------

def parse_card_date(digits: str, fmt: str) -> Optional[date]:
    a, b, c = int(digits[:2]), int(digits[2:4]), int(digits[4:])
    year, month, day = {"YYMMDD": (a, b, c), "MMDDYY": (c, a, b), "DDMMYY": (c, b, a)}[fmt]
    try:
        return date(2000 + year, month, day)
    except ValueError:
        return None


def _parse_time(digits: str) -> Optional[time]:
    try:
        return time(int(digits[:2]), int(digits[2:4]), int(digits[4:]))
    except ValueError:
        return None


def _date_evidence(take: Take) -> Optional[Tuple[str, date, date]]:
    """(source, earliest, latest) start date the card supports, independent of the name.

    BEXT OriginationDate is the start date itself. File modification times
    fall somewhere between the take's start and end, so they bound the start
    date to a range, widened by a day for clock/time zone slop.
    """
    day = take.bext_date()
    if day:
        return "BEXT date", day, day
    mtimes = [datetime.fromtimestamp(t.mtime) for t in take.tracks.values()]
    mtimes = [m for m in mtimes if m.year >= 2000]  # 1980 = FAT's "no date"
    if not mtimes:
        return None
    first, last = min(mtimes), max(mtimes)
    earliest = (first - timedelta(seconds=take.duration) - timedelta(days=1)).date()
    return "file modification time", earliest, (last + timedelta(days=1)).date()


@dataclass
class FormatCheck:
    fmt: str
    invalid: List[str] = field(default_factory=list)  # take names that aren't a date in this format
    agree: int = 0
    disagree: List[str] = field(default_factory=list)
    unchecked: int = 0
    sources: set = field(default_factory=set)

    def summary(self, total: int) -> str:
        if not self.agree and not self.disagree:
            return "no BEXT dates or usable file times to check it against"
        text = f"{self.agree} of {total} takes match their {' / '.join(sorted(self.sources))}"
        if self.disagree:
            text += f", {len(self.disagree)} don't"
        if self.unchecked:
            text += f", {self.unchecked} couldn't be checked"
        return text


def check_format(takes: List[Take], fmt: str) -> FormatCheck:
    check = FormatCheck(fmt)
    for take in takes:
        day = parse_card_date(take.date_digits, fmt)
        if day is None:
            check.invalid.append(take.name)
            continue
        evidence = _date_evidence(take)
        if evidence is None:
            check.unchecked += 1
            continue
        source, earliest, latest = evidence
        check.sources.add(source)
        if earliest <= day <= latest:
            check.agree += 1
        else:
            says = earliest.isoformat() if earliest == latest else f"{earliest}..{latest}"
            check.disagree.append(f"{take.name} reads as {day} but its {source} says {says}")
    return check


@dataclass
class DateFormat:
    fmt: str
    explanation: str
    warnings: List[str] = field(default_factory=list)


def resolve_date_format(takes: List[Take], requested: str, ignore_check: bool) -> DateFormat:
    """Pick (or check) the recorder's date format against the card's own metadata."""
    total = len(takes)
    checks = {f: check_format(takes, f) for f in DATE_FORMATS}

    def fits() -> str:
        good = [c.fmt for c in checks.values() if not c.invalid and not c.disagree and c.agree]
        return f" The card is consistent with {' or '.join(good)}." if good else ""

    if requested != "auto":
        check = checks[requested]
        if check.invalid:
            raise CardError(f"{len(check.invalid)} take folder(s) aren't valid {requested} dates, "
                            f"e.g. {check.invalid[0]}.{fits()}")
        text = f"{requested} (as configured; {check.summary(total)})"
        if check.disagree:
            detail = f"--date-format {requested} contradicts the card: {check.disagree[0]}"
            if len(check.disagree) > 1:
                detail += f" (and {len(check.disagree) - 1} more)"
            if not ignore_check:
                raise CardError(detail + "." + fits() + " Use --ignore-date-check to import anyway.")
            return DateFormat(requested, text, [detail])
        return DateFormat(requested, text)

    viable = [c for c in checks.values() if not c.invalid]
    if not viable:
        raise CardError("take folder names aren't valid dates in any of " + ", ".join(DATE_FORMATS))
    clean = [c for c in viable if not c.disagree]
    if not clean:
        best = max(viable, key=lambda c: c.agree)
        detail = f"no date format matches every take's metadata; closest is {best.fmt} ({best.disagree[0]})"
        if not ignore_check:
            raise CardError(detail + ". Is the recorder's clock set? Pass --date-format and "
                                     "--ignore-date-check to import anyway.")
        return DateFormat(best.fmt, f"{best.fmt} (best guess; {best.summary(total)})", [detail])

    top = max(c.agree for c in clean)
    winners = [c for c in clean if c.agree == top]
    chosen = winners[0]  # DATE_FORMATS order breaks ties
    warnings = []
    differ = [c.fmt for c in winners[1:]
              if any(parse_card_date(t.date_digits, c.fmt) != parse_card_date(t.date_digits, chosen.fmt)
                     for t in takes)]
    if differ:
        warnings.append(f"couldn't tell {chosen.fmt} from {', '.join(differ)} using the card's metadata; "
                        f"assumed {chosen.fmt}. Pass --date-format if that's wrong.")
    return DateFormat(chosen.fmt, f"{chosen.fmt} (auto-detected; {chosen.summary(total)})", warnings)


def assign_starts(takes: List[Take], fmt: str):
    for take in takes:
        take.start = datetime.combine(parse_card_date(take.date_digits, fmt), _parse_time(take.time_digits))


# --- continuations -----------------------------------------------------------

def chain_takes(takes: List[Take], tolerance: float) -> Tuple[List[Recording], List[str]]:
    """Group takes into recordings, joining each take to the one before it
    when it's a continuation of a 2 GB split."""
    recordings: List[Recording] = []
    notes: List[str] = []
    for take in sorted(takes, key=lambda t: (t.start, t.name)):
        if recordings:
            joined, text = _continuation(recordings[-1].takes[-1], take, tolerance)
            if joined:
                recordings[-1].takes.append(take)
                recordings[-1].links.append(text)
                continue
            if text:
                notes.append(text)
        recordings.append(Recording([take]))
    return recordings, notes


def _continuation(prev: Take, cur: Take, tolerance: float) -> Tuple[bool, Optional[str]]:
    if not prev.tracks or not cur.tracks:
        return False, None
    gap = (cur.start - prev.end).total_seconds()
    if abs(gap) > tolerance:
        if abs(gap) <= NEAR_MISS_SECONDS:
            return False, (f"{cur.name} starts {gap:+.1f}s from the end of {prev.name}, outside the "
                           f"±{tolerance:g}s continuation tolerance; kept as a separate recording")
        return False, None
    if prev.track_key() != cur.track_key():
        return False, (f"{cur.name} starts where {prev.name} ends but has different tracks "
                       f"({', '.join(cur.tracks)} vs {', '.join(prev.tracks)}); not stitched")
    for name, track in prev.tracks.items():
        other = cur.track(name)
        if track.info.signature != other.info.signature:
            return False, (f"{cur.name} starts where {prev.name} ends but {name} changes format "
                           f"({track.info.describe()} -> {other.info.describe()}); not stitched")

    evidence = f"name gap {gap:+.1f}s"
    samples = _time_reference_gap(prev, cur)
    if samples is not None:
        diff, rate = samples
        if abs(diff) / rate > tolerance:
            return False, (f"{cur.name}'s name puts it {gap:+.1f}s from the end of {prev.name}, but their "
                           f"BEXT time references show a {diff / rate:+.3f}s gap; not stitched")
        evidence += ("; BEXT time reference sample-exact" if diff == 0 else
                     f"; BEXT time reference off by {diff:+d} samples ({diff / rate * 1000:+.1f} ms)")
    return True, f"{cur.name} continues {prev.name} ({evidence})"


def _time_reference_gap(prev: Take, cur: Take) -> Optional[Tuple[int, int]]:
    """Samples between prev's end and cur's start by BEXT TimeReference
    (samples since midnight), or None if the files don't carry one."""
    for name, track in prev.tracks.items():
        a, b = track.info.bext, cur.track(name).info.bext
        if a and b and (a.time_reference or b.time_reference):
            rate = track.info.sample_rate
            day = rate * SECONDS_PER_DAY
            expected = (a.time_reference + track.info.frames) % day
            diff = (b.time_reference - expected) % day
            if diff > day // 2:
                diff -= day
            return diff, rate
    return None


# --- sessions ----------------------------------------------------------------

def group_sessions(recordings: List[Recording], gap_minutes: Optional[float]) -> List[Session]:
    """One session per calendar day of recording start, or, with gap_minutes,
    a new session whenever recordings are further apart than that."""
    sessions: List[Session] = []
    for rec in recordings:
        if sessions:
            current = sessions[-1]
            if gap_minutes is None:
                same = rec.start.date() == current.day
            else:
                same = rec.start - current.recordings[-1].end <= timedelta(minutes=gap_minutes)
            if same:
                current.recordings.append(rec)
                continue
        sessions.append(Session(rec.start.date(), [rec]))
    return sessions
