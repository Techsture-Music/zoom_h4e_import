# zoom_h4e_import

Imports recordings from a Zoom H4essential microSD card. It copies every take
off the card, stitches recordings the recorder split at 2 GB back into one
file per track, and files them in one dated folder per session. Every file is
checked against a checksum after it's written.

The card is only ever opened for reading. Nothing on it is modified or
deleted, and the tool refuses to import to a destination on the card itself.

No dependencies beyond Python 3.9's standard library.

## Usage

```
python3 zoom_h4e_import.py ~/Music/Recordings --dry-run    # show the plan
python3 zoom_h4e_import.py ~/Music/Recordings              # do it
```

The card is found automatically (the mounted volume whose `ZOOM.SYS` names
the H4essential). Use `--source /Volumes/NAME` to name it explicitly, or to
import from a folder copy of a card.

The plan lists every copy, rename and stitch, with each file's format read
from its header. It also shows how each continuation was confirmed, and ends
with notes on anything left behind: `Trash`/`Export` folders, stray files,
near-miss continuations. Progress is shown during the run.

```
2026-09-25/
  2026-09-25 21:30:00  1:33:12  split across 2 folders: 260925_213000, 260925_221636
      260925_221636 continues 260925_213000 (name gap -0.2s; BEXT time reference sample-exact)
    stitch  TrMic x2 parts -> 2026-09-25_TrMic.wav  96 kHz 32-bit float stereo, 4.3 GB RF64
    stitch  Tr1 x2 parts   -> 2026-09-25_Tr1.wav    96 kHz 32-bit float mono, 2.1 GB RIFF
```

## What it does

**Sessions.** By default, all recordings that start on the same day go into
one folder named for that date. Use `--session-gap MINUTES` to start a new
session after a break instead. A recording that runs past midnight stays
with the day it started.

**Names.** The default is `{date}_{track}`, e.g. `2026-09-25_TrMic.wav`. If
the session folder holds more than one recording, including recordings from
an earlier import, the default becomes `{date}_{time}_{track}` so files don't
collide. Use `--name` and `--folder` to set your own patterns; `--help` lists
the placeholders. `{date}` is ISO by default so folders sort in order; change
it with `--date-style`.

**Tracks** are whatever suffixes the files on the card have (`TrMic`, `Tr1`,
`TrLR`, or whatever the stereo-linked file turns out to be called), not a
hard-coded list. Use `--skip-lr`, `--skip-track NAME` or `--only-track NAME`
to filter them.

**Dates.** Folder names follow the recorder's date-format setting. By default
the tool works out which format the card uses by comparing each take's name
with its BEXT origination date, or with the file modification times if there's
no BEXT. If you pass `--date-format YYMMDD|MMDDYY|DDMMYY`, it runs the same
check and stops if the card contradicts your setting. Use
`--ignore-date-check` to import anyway.

**Split detection.** A take continues the previous one when:

- it starts within `--tolerance` seconds (default 2) of where the previous
  take ends: the previous take's folder-name time plus its duration,
  measured from its sample count;
- it has the same track names; and
- each track has the same sample rate, channel count and bit depth.

When both takes carry BEXT time references, the tool also checks the gap
between them to the sample. If that gap is larger than the tolerance, the
takes aren't stitched, even when the folder names line up. The plan reports
whether the BEXT timing was sample-exact.

**Stitching** copies each part's audio frames byte-for-byte, so 32-bit float
stays exactly as recorded. Audio is streamed in 8 MB chunks and whole files
are never loaded into memory. The stitched file gets the first part's `fmt`,
`bext` (including the time reference, which still marks the true start),
`iXML`, and other metadata. Its `fact` chunk is rewritten with the new
length, and markers from every part are merged and shifted into position.
Marker labels from parts after the first are dropped, and the plan says so.

**RF64.** Stitched files larger than 4 GB are written as RF64 (EBU Tech
3306). Every stitch is written in the same layout, with space reserved for
RF64's `ds64` chunk. When the file is closed, the header becomes RIFF or
RF64 depending on how many bytes were actually written. So RF64 isn't a
special code path: it's the same writer with a different header at the end.
Use `--rf64 always` for consistent containers, or `--rf64 never` to refuse
anything over 4 GB instead.

**Verification.**

- **Copies:** checksummed on the way off the card, then read back from disk
  and compared.
- **Stitches:** the joined audio from the parts is checksummed and compared
  with the data chunk read back from the new file. The new header is also
  re-parsed and checked for frame count, format and time reference.

Files are written as `NAME.partial` and only renamed once verified, so a file
under its final name is always a complete, checked one.

**Re-running.** Each session folder gets an `import_manifest.json` recording
each file's sources, format, length, BEXT time reference and checksum. On the
next run, recordings already listed there are skipped, so you can import the
same card after every session. An interrupted import resumes where it
stopped. If a file already exists at the destination and isn't recorded as
this recording's import, the run stops before writing anything, unless you
pass `--overwrite`.

**Damaged files.** A WAV in a take folder that can't be parsed is still
copied off the card under its original name, and the plan flags it. A file
whose header was never finalized (after a power loss, say) is read to the end
of the file, with a warning.

## Testing

```
python3 selftest.py          # builds synthetic cards and imports them (~2 s)
python3 selftest.py --big    # also stitches two 2 GiB parts into a 4 GiB RF64 (~9 GB scratch)
```

The self-test builds cards that follow the card layout in the spec. It checks
stitched audio sample by sample, including across a midnight split, and
checks that every card is byte-for-byte unchanged afterwards (mtimes
included). `--big` confirms that `ffprobe`, `afinfo` (Core Audio) and
`ffmpeg` all read the 4 GiB RF64 file correctly, including the audio past
the 4 GiB mark.

## Unconfirmed assumptions about the recorder

The rules below come from the spec. None have been checked against a real
split recording yet. The plan output shows how the tool interprets each
card, so check the first real split with `--dry-run`:

- **Continuation folders.** When a file hits 2 GB, the recorder starts a new
  `DATE_TIME` folder with no gap in the audio, and all tracks split at the
  same moment.
- **Folder location.** Take folders can sit anywhere on the card: the scan is
  recursive and skips `Trash`/`Export` (any capitalization) and hidden files.
- **Stereo-linked name.** The stereo-linked file's suffix is picked up
  whatever it is.

If a real split doesn't stitch, the near-miss notes in the plan show how far
off the timing was. `--tolerance` and `--no-stitch` are there as fallbacks.
