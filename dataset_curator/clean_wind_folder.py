#!/usr/bin/env python3
"""
clean_wind_folder.py
--------------------
Cleans NOISE_MASTER/Nature/Wind ONLY. Nothing else is touched.

THE PROBLEM
-----------
FSD50K's ontology has a class "Wind instrument, woodwind instrument".
The curator matched on the word "wind", so ~83% of the Wind folder is
actually studio recordings of SAXOPHONES, CLARINETS, FLUTES, OBOES and
BASSOONS -- not atmospheric wind at all.

Blacklisting bad keywords does not work here (there are too many).
So this script uses a WHITELIST:

    KEEP a file only if its FSD50K tags contain STRONG evidence of real
    atmospheric wind (wind / gust / gale / storm / howling / rustle /
    leaves / thunderstorm ...) AND contain NO musical-instrument tag and
    NO other-content tag (train, bike, speech, fire, water ...).

    Everything else is DELETED.

Non-FSD50K Wind files (the UAVirBASE ambient channel recordings) are
never touched -- they are genuine field recordings.

Expected: ~2,873 deleted, ~314 FSD50K + 72 UAVirBASE kept.

Usage:
    python clean_wind_folder.py --dry-run     # look first, delete nothing
    python clean_wind_folder.py               # actually delete
"""

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

NOISE_MASTER_DIR = Path(
    r"C:\Users\CARE\Downloads\Acoustic based drone detection"
    r"\Audio files\Dataset\NOISE_MASTER"
)

WIND_DIR = NOISE_MASTER_DIR / "Nature" / "Wind"

FSD50K_META_DIR = Path(
    r"C:\Users\CARE\Downloads\Acoustic based drone detection"
    r"\Audio files\Noise\FSD50K\FSD50K.metadata"
)

FSD50K_DEV_JSON  = FSD50K_META_DIR / "dev_clips_info_FSD50K.json"
FSD50K_EVAL_JSON = FSD50K_META_DIR / "eval_clips_info_FSD50K.json"

REPORT_PATH = NOISE_MASTER_DIR / "wind_deletion_report.csv"
KEEP_PATH   = NOISE_MASTER_DIR / "wind_kept_report.csv"

# ============================================================
# TAG SETS
# ============================================================

# STRONG positive evidence of real atmospheric wind / weather.
# A file must have at least one of these to survive.
STRONG_WIND = {
    "wind", "windy", "breeze", "breezy", "gust", "gusts", "gusty", "gusting",
    "gale", "storm", "stormy", "windstorm", "blizzard", "thunderstorm",
    "hurricane", "tempest", "howling", "howl",
    "wind-noise", "blowing-wind", "whistling-wind",
    "rustle", "rustling", "leaves",
}

# Musical instruments and studio-sampling artifacts.
# This is what contaminated the folder: FSD50K "wind instrument" class.
INSTRUMENT = {
    # saxophones
    "sax", "saxophone", "soprano-sax", "alto-sax", "tenor-sax", "baritone-sax",
    "soprano-saxophone", "alto-saxophone", "tenor-saxophone", "xaphoon",
    # other woodwinds
    "clarinet", "bass-clarinet", "pocket-clarinet",
    "flute", "transverse-flute", "piccolo", "recorder",
    "oboe", "bassoon", "bassoons", "contrabassoon",
    "english-horn", "cor-anglais",
    "woodwind", "woodwinds", "aerophone",
    "double-reed", "single-reed", "reed",
    # folk / misc wind instruments
    "ocarina", "harmonica", "bagpipes", "didgeridoo", "panpipes",
    "whistle", "accordion",
    # brass
    "trumpet", "trombone", "tuba", "horn", "french-horn", "brass",
    "glockenspiel",
    # studio sampling markers (dead giveaway: these are sample libraries)
    "multisample", "single-note", "good-sounds", "neumann-u87",
    "vsco-2", "vst", "sfz", "sampled-instruments", "sampler", "samples",
    "orchestral", "midi",
    # performance articulation markers
    "staccato", "tenuto", "legato", "vibrato", "non-vibrato",
    "multiphonics", "mezzoforte", "fortissimo", "pianissimo",
    # general music vocabulary
    "scale", "melody", "music", "musical", "musical-instruments",
    "instrument", "note", "sustain", "chord", "tune", "tuning",
    "jazz", "classical", "orchestra", "symphony", "concert", "chamber",
    "swing", "klezmer", "improvisation", "synth", "percussion",
    # chimes and bells -- tonal, periodic, wrong for a looped bed
    "chime", "chimes", "windchime", "windchimes", "wind-chimes",
    "windbell", "windbells", "bell", "bells",
}

# Other content that clearly is not wind (trains, bikes, speech, fire...)
OTHER_CONTENT = {
    "train", "trains", "rail", "railway",
    "bicycle", "motorbike", "engine", "traffic", "hoover", "vacuum",
    "speech", "talk", "voice", "child", "children",
    "footsteps", "sloshing",
    "fire", "flames", "burning",
    "stream", "river", "water", "pump",
    "siren", "ship", "gunshot", "explosion",
}

# ============================================================
# HELPERS
# ============================================================

def load_json(path: Path) -> dict:
    if not path.exists():
        print(f"[WARN] JSON not found: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_fsd50k_id(filename: str):
    """Nature_Wind_FSD50K_<ID>.wav  ->  '<ID>'   (None if not FSD50K)"""
    parts = Path(filename).stem.split("_")
    try:
        return parts[parts.index("FSD50K") + 1]
    except (ValueError, IndexError):
        return None


def judge(tags):
    """
    Returns (keep: bool, reason: str)

    KEEP only if: strong wind evidence present
             AND: no instrument tags
             AND: no other-content tags
    """
    s = {t.lower() for t in tags}

    strong = STRONG_WIND    & s
    instr  = INSTRUMENT     & s
    other  = OTHER_CONTENT  & s

    if strong and not instr and not other:
        return True, "wind:" + ",".join(sorted(strong)[:4])

    reasons = []
    if instr:
        reasons.append("INSTRUMENT:" + ",".join(sorted(instr)[:4]))
    if other:
        reasons.append("OTHER:" + ",".join(sorted(other)[:4]))
    if not strong:
        reasons.append("NO-WIND-EVIDENCE")

    return False, " | ".join(reasons)


# ============================================================
# MAIN
# ============================================================

def main(dry_run: bool):

    print("=" * 70)
    print("CLEAN WIND FOLDER  --  NOISE_MASTER/Nature/Wind only")
    if dry_run:
        print("*** DRY RUN -- nothing will be deleted ***")
    print("=" * 70)

    # ---- load FSD50K metadata -------------------------------------------
    print("\nLoading FSD50K metadata...")
    meta = {**load_json(FSD50K_DEV_JSON), **load_json(FSD50K_EVAL_JSON)}
    print(f"  {len(meta):,} clip entries loaded")

    if not meta:
        print("\n[ERROR] No metadata loaded. Check FSD50K_META_DIR path.")
        return

    # ---- scan Wind folder ------------------------------------------------
    if not WIND_DIR.exists():
        print(f"\n[ERROR] Wind folder not found: {WIND_DIR}")
        return

    print(f"\nScanning: {WIND_DIR}")
    wav_files = sorted(WIND_DIR.glob("*.wav"))

    fsd_files   = [f for f in wav_files if "_FSD50K_" in f.name]
    other_files = [f for f in wav_files if "_FSD50K_" not in f.name]

    print(f"  Total Wind files      : {len(wav_files):,}")
    print(f"  FSD50K files (checked): {len(fsd_files):,}")
    print(f"  Non-FSD50K (untouched): {len(other_files):,}   "
          f"[UAVirBASE ambients -- real field recordings]")

    # ---- judge each FSD50K file -----------------------------------------
    print("\nJudging each FSD50K Wind file against its tags...")

    to_delete = []
    to_keep   = []

    for wav in fsd_files:
        cid = extract_fsd50k_id(wav.name)

        if cid is None or cid not in meta:
            # No metadata -> cannot verify -> delete (cannot trust it)
            to_delete.append({
                "filename": wav.name,
                "abs_path": str(wav),
                "clip_id" : cid or "",
                "title"   : "",
                "reason"  : "NO-METADATA",
                "tags"    : "",
            })
            continue

        entry = meta[cid]
        tags  = entry.get("tags", [])
        title = entry.get("title", "")

        keep, reason = judge(tags)

        row = {
            "filename": wav.name,
            "abs_path": str(wav),
            "clip_id" : cid,
            "title"   : title,
            "reason"  : reason,
            "tags"    : ", ".join(tags),
        }

        (to_keep if keep else to_delete).append(row)

    # ---- summary ---------------------------------------------------------
    print()
    print("-" * 70)
    print(f"  KEEP   (real wind)          : {len(to_keep):>6}")
    print(f"  DELETE (contaminated)       : {len(to_delete):>6}")
    print(f"  UAVirBASE ambients (kept)   : {len(other_files):>6}")
    print("-" * 70)
    print(f"  Wind bed pool after cleaning: "
          f"{len(to_keep) + len(other_files):>6} files")
    print("-" * 70)

    print("\n  Deletion reasons:")
    reasons = Counter(r["reason"].split(":")[0].split(" |")[0]
                      for r in to_delete)
    for k, v in reasons.most_common():
        print(f"    {k:<22}: {v:>6}")

    print("\n  Sample of files being DELETED:")
    for r in to_delete[:8]:
        print(f"    {r['filename']}")
        print(f"       reason: {r['reason']}")
        print(f"       tags  : {r['tags'][:90]}")

    print("\n  Sample of files being KEPT:")
    for r in to_keep[:8]:
        print(f"    {r['filename']}")
        print(f"       tags  : {r['tags'][:90]}")

    # ---- write both reports ---------------------------------------------
    fields = ["filename", "abs_path", "clip_id", "title", "reason", "tags"]

    with open(REPORT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(to_delete)

    with open(KEEP_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(to_keep)

    print(f"\n  Deletion report : {REPORT_PATH}")
    print(f"  Kept-files report: {KEEP_PATH}")
    print("  Open BOTH and verify before deleting.")

    if dry_run:
        print()
        print("=" * 70)
        print("DRY RUN COMPLETE -- no files were deleted.")
        print("Re-run without --dry-run to actually delete.")
        print("=" * 70)
        return

    # ---- delete ----------------------------------------------------------
    print("\nDeleting...")
    deleted, errors = 0, []

    for r in to_delete:
        try:
            os.remove(r["abs_path"])
            deleted += 1
        except Exception as e:
            errors.append((r["filename"], str(e)))

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"  Deleted                     : {deleted:,}")
    print(f"  Errors                      : {len(errors)}")
    print(f"  Wind files remaining        : "
          f"{len(to_keep) + len(other_files):,}")

    if errors:
        print("\n  Errors:")
        for name, err in errors[:10]:
            print(f"    {name}: {err}")

    print(f"\n  Full list of deleted files  : {REPORT_PATH}")
    print("\nNext step: re-run rebuild_statistics.py to regenerate")
    print("noise_master_metadata.csv with the cleaned Wind folder.")
    print("=" * 70)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Remove non-wind contamination from NOISE_MASTER/Nature/Wind"
    )
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would be deleted without deleting")
    main(ap.parse_args().dry_run)