#!/usr/bin/env python3
"""
clean_traffic_crowd.py
----------------------
Cleans NOISE_MASTER/Urban/Traffic and NOISE_MASTER/Human/Crowd ONLY.
Nothing else is touched. Same whitelist method as clean_wind_folder.py.

THE PROBLEM
-----------
Urban/Traffic  : 5,371 files, but ~25% are FIREWORKS, MACHINE GUNS and
                 EXPLOSIONS (titles like "Fireworks Boom.wav",
                 "LightMachineGun1.wav"). Looped as a bed, a machine-gun
                 burst repeating every 8 seconds is catastrophic.

Human/Crowd    : 1,503 files, but ~72% is individual LAUGHTER, not crowd
                 murmur ("Baby Laugh.mp3", giggling, evil laughs). A laugh
                 is a transient event, not a continuous bed texture.

WHY WHITELIST, NOT BLACKLIST
----------------------------
Blacklisting misfires. Animals/Dog has 269 files tagged "jazz" -- but the
titles read "Jazz the Dog Howl & Bark". Someone's dog is NAMED Jazz.
"delphidebrain" is an uploader username. Neither is contamination.

So: KEEP a file only if it shows POSITIVE evidence of the texture we want
AND carries no disqualifying tag. Everything else goes.

NOTE ON CROWD: applause / cheering / stadium noise are KEPT -- they are
legitimate continuous crowd ambience. Only individual/comedic laughter,
babies, and music are removed.

Expected:
    Traffic : ~2,409 kept  /  ~2,962 deleted
    Crowd   :   ~333 kept  /  ~1,170 deleted

Usage:
    python clean_traffic_crowd.py --dry-run     # look first, delete nothing
    python clean_traffic_crowd.py               # actually delete
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

FSD50K_META_DIR = Path(
    r"C:\Users\CARE\Downloads\Acoustic based drone detection"
    r"\Audio files\Noise\FSD50K\FSD50K.metadata"
)

FSD50K_DEV_JSON  = FSD50K_META_DIR / "dev_clips_info_FSD50K.json"
FSD50K_EVAL_JSON = FSD50K_META_DIR / "eval_clips_info_FSD50K.json"

REPORT_DELETE = NOISE_MASTER_DIR / "traffic_crowd_deletion_report.csv"
REPORT_KEEP   = NOISE_MASTER_DIR / "traffic_crowd_kept_report.csv"

# ============================================================
# TAG SETS -- TRAFFIC
# ============================================================

TRAFFIC_GOOD = {
    # the texture we actually want: continuous road/vehicle hum
    "traffic", "road", "street", "highway", "motorway", "freeway",
    "roadside", "intersection", "junction", "avenue", "boulevard",
    "rush-hour", "commute", "asphalt",
    "car", "cars", "vehicle", "vehicles", "driving", "passing", "drive-by",
    "truck", "trucks", "bus", "van", "motorcycle", "scooter",
    "engine", "motor", "tyre", "tire",
    "city", "urban", "town",
    "field-recording", "ambience", "ambient", "ambiance",
    "atmosphere", "atmos", "atmo", "soundscape",
}

TRAFFIC_BAD = {
    # weapons / explosives -- the big contaminant
    "explosion", "explosions", "explosive", "blast", "detonation",
    "gun", "guns", "gunshot", "gunshots", "gunfire", "firearm",
    "shot", "shots", "shoot", "shooting",
    "rifle", "pistol", "revolver", "machine-gun", "m16", "ak47",
    "bullet", "bullets", "ammo", "weapon",
    "artillery", "cannon", "bomb", "grenade",
    "war", "battle", "combat", "army",
    "fireworks", "firework", "firecracker",
    # aviation -- belongs in Hard_Negatives, not a road bed
    "aircraft", "airplane", "plane", "jet", "helicopter",
    # discrete alerts
    "siren", "alarm",
    # music / synthetic
    "music", "song", "melody", "beat", "drum", "guitar", "piano",
    "synth", "instrument", "ringtone",
}

# ============================================================
# TAG SETS -- CROWD
# ============================================================

CROWD_GOOD = {
    # continuous crowd murmur / public-space ambience
    "crowd", "crowds", "people", "public", "busy",
    "chatter", "chattering", "murmur", "babble", "hubbub", "walla",
    "voices", "talking", "conversation",
    "audience", "stadium", "arena", "fans", "game", "match", "rally",
    "market", "marketplace", "restaurant", "cafe", "bar", "pub", "mall",
    "station", "airport", "terminal", "lobby", "hall", "concourse",
    "festival", "party", "protest",
    # applause / cheering ARE legitimate continuous crowd texture
    "applause", "cheer", "cheers", "cheering", "clapping",
    "street", "city", "urban",
    "field-recording", "background",
    "ambience", "ambient", "ambiance",
    "atmosphere", "atmos", "atmo", "soundscape",
}

CROWD_BAD = {
    # individual / comedic laughter -- transient, not a bed
    "laugh", "laughs", "laughing", "laughter",
    "giggle", "giggles", "giggling", "chuckle", "snicker",
    "baby", "babies", "infant", "toddler",
    "evil", "funny", "comedy", "cartoon", "joke", "sitcom",
    # isolated vocal bursts
    "scream", "screaming", "shriek",
    # music
    "music", "song", "melody", "beat", "drum", "guitar", "piano",
    "synth", "instrument", "ringtone",
}

# ============================================================
# TARGETS
# ============================================================

TARGETS = [
    {
        "superclass"  : "Urban",
        "target_class": "Traffic",
        "good"        : TRAFFIC_GOOD,
        "bad"         : TRAFFIC_BAD,
    },
    {
        "superclass"  : "Human",
        "target_class": "Crowd",
        "good"        : CROWD_GOOD,
        "bad"         : CROWD_BAD,
    },
]

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
    """Superclass_TargetClass_FSD50K_<ID>.wav -> '<ID>'  (None if not FSD50K)"""
    parts = Path(filename).stem.split("_")
    try:
        return parts[parts.index("FSD50K") + 1]
    except (ValueError, IndexError):
        return None


def judge(tags, good_set, bad_set):
    """
    KEEP only if: positive evidence present AND no disqualifying tag.
    Returns (keep: bool, reason: str)
    """
    s = {t.lower() for t in tags}

    good = good_set & s
    bad  = bad_set  & s

    if good and not bad:
        return True, "good:" + ",".join(sorted(good)[:4])

    reasons = []
    if bad:
        reasons.append("BAD:" + ",".join(sorted(bad)[:4]))
    if not good:
        reasons.append("NO-POSITIVE-EVIDENCE")

    return False, " | ".join(reasons)


# ============================================================
# MAIN
# ============================================================

def main(dry_run: bool):

    print("=" * 72)
    print("CLEAN TRAFFIC + CROWD  --  NOISE_MASTER only")
    if dry_run:
        print("*** DRY RUN -- nothing will be deleted ***")
    print("=" * 72)

    # ---- load metadata ---------------------------------------------------
    print("\nLoading FSD50K metadata...")
    meta = {**load_json(FSD50K_DEV_JSON), **load_json(FSD50K_EVAL_JSON)}
    print(f"  {len(meta):,} clip entries loaded")

    if not meta:
        print("\n[ERROR] No metadata loaded. Check FSD50K_META_DIR path.")
        return

    all_delete = []
    all_keep   = []

    # ---- process each target class ---------------------------------------
    for t in TARGETS:

        sc, tc = t["superclass"], t["target_class"]
        folder = NOISE_MASTER_DIR / sc / tc

        print()
        print("=" * 72)
        print(f"{sc}/{tc}")
        print("=" * 72)

        if not folder.exists():
            print(f"  [ERROR] Folder not found: {folder}")
            continue

        wavs = sorted(folder.glob("*.wav"))
        fsd_wavs   = [w for w in wavs if "_FSD50K_" in w.name]
        other_wavs = [w for w in wavs if "_FSD50K_" not in w.name]

        print(f"  Total files            : {len(wavs):,}")
        print(f"  FSD50K (checked)       : {len(fsd_wavs):,}")
        print(f"  Non-FSD50K (untouched) : {len(other_wavs):,}   "
              f"[US8K / ESC50 -- single-label, not contaminated]")

        keep_rows, del_rows = [], []

        for wav in fsd_wavs:
            cid = extract_fsd50k_id(wav.name)

            if cid is None or cid not in meta:
                del_rows.append({
                    "class": f"{sc}/{tc}", "filename": wav.name,
                    "abs_path": str(wav), "clip_id": cid or "",
                    "title": "", "reason": "NO-METADATA", "tags": "",
                })
                continue

            entry = meta[cid]
            tags  = entry.get("tags", [])
            title = entry.get("title", "")

            keep, reason = judge(tags, t["good"], t["bad"])

            row = {
                "class"   : f"{sc}/{tc}",
                "filename": wav.name,
                "abs_path": str(wav),
                "clip_id" : cid,
                "title"   : title,
                "reason"  : reason,
                "tags"    : ", ".join(tags),
            }

            (keep_rows if keep else del_rows).append(row)

        # ---- per-class summary -------------------------------------------
        remaining = len(keep_rows) + len(other_wavs)
        print()
        print(f"  KEEP   : {len(keep_rows):>6}")
        print(f"  DELETE : {len(del_rows):>6}")
        print(f"  -> {tc} pool after cleaning: {remaining:,} files")

        print("\n  Deletion reasons:")
        reasons = Counter(r["reason"].split(":")[0].split(" |")[0]
                          for r in del_rows)
        for k, v in reasons.most_common():
            print(f"    {k:<24}: {v:>6}")

        print("\n  Sample DELETED:")
        for r in del_rows[:5]:
            print(f"    {r['title'][:52]}")
            print(f"       reason: {r['reason'][:70]}")

        print("\n  Sample KEPT:")
        for r in keep_rows[:5]:
            print(f"    {r['title'][:52]}")
            print(f"       tags  : {r['tags'][:70]}")

        all_keep.extend(keep_rows)
        all_delete.extend(del_rows)

    # ---- write reports ---------------------------------------------------
    fields = ["class", "filename", "abs_path", "clip_id",
              "title", "reason", "tags"]

    for path, rows in [(REPORT_DELETE, all_delete), (REPORT_KEEP, all_keep)]:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    print()
    print("=" * 72)
    print(f"  TOTAL KEEP   : {len(all_keep):>6}")
    print(f"  TOTAL DELETE : {len(all_delete):>6}")
    print("=" * 72)
    print(f"\n  Deletion report : {REPORT_DELETE}")
    print(f"  Kept report     : {REPORT_KEEP}")
    print("  Open BOTH and verify before deleting.")

    if dry_run:
        print()
        print("=" * 72)
        print("DRY RUN COMPLETE -- no files were deleted.")
        print("Re-run without --dry-run to actually delete.")
        print("=" * 72)
        return

    # ---- delete ----------------------------------------------------------
    print("\nDeleting...")
    deleted, errors = 0, []

    for r in all_delete:
        try:
            os.remove(r["abs_path"])
            deleted += 1
        except Exception as e:
            errors.append((r["filename"], str(e)))

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)
    print(f"  Deleted : {deleted:,}")
    print(f"  Errors  : {len(errors)}")

    if errors:
        print("\n  Errors:")
        for name, err in errors[:10]:
            print(f"    {name}: {err}")

    print(f"\n  Full list of deleted files: {REPORT_DELETE}")
    print("\nNext step: re-run rebuild_statistics.py to regenerate")
    print("noise_master_metadata.csv with the cleaned folders.")
    print("=" * 72)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Remove contamination from NOISE_MASTER Traffic + Crowd"
    )
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would be deleted without deleting")
    main(ap.parse_args().dry_run)