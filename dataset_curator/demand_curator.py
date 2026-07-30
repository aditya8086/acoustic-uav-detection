#!/usr/bin/env python3
"""
demand_curator.py
-----------------
Adds DEMAND recordings to NOISE_MASTER as bed material.

WHY DEMAND
----------
The Wind bed pool collapsed to 210 usable clips (>=8 s) after the woodwind
cleaning, which starved the bed builder and dragged Wind down to 14.9% against
a 26% target. DEMAND supplies long-form, genuinely outdoor field recordings:
each environment is a continuous 5-minute capture, versus the ~8.8 s median of
the current Wind pool.

ONE CHANNEL PER ENVIRONMENT -- NOT 16
-------------------------------------
DEMAND records with a 16-mic array spaced 5-21.8 cm apart. At those spacings
the channels are strongly correlated, especially at the low frequencies that
dominate wind. Splitting all 16 would give 16x the FILE COUNT but roughly 1x
the acoustic diversity.

Worse, it would leak. NOISE_MASTER is split train/val/test at the FILE level,
so channel 3 and channel 11 of the same 20-second moment could land on opposite
sides of the split -- the model would train on one view and be tested on a
near-identical one. This is the exact redundancy already engineered out of
DRONE_MASTER (where UAVirBASE's 8 channels were collapsed to 1 random pick).

So: ONE channel per environment, chosen deterministically (seeded), then chopped
into 20 s segments. 5 min / 20 s = 15 segments per environment.

CLASS ROUTING (from listening)
------------------------------
    NFIELD   -> Nature/Wind      open field, wind-dominant
    SPSQUARE -> Nature/Wind      open public square, wind-dominant
    NRIVER   -> Nature/Stream    continuous running water
    STRAFFIC -> Urban/Traffic    continuous road hum
    NPARK    -> Human/Crowd      audible people

OUTPUT NAMING
-------------
    {Superclass}_{TargetClass}_DEMAND_{env}_ch{NN}_seg{NNN}.wav

    e.g. Nature_Wind_DEMAND_nfield_ch05_seg003.wav

The 3rd underscore-token is "DEMAND", which is what rebuild_statistics.py reads
as source_dataset. No change to that script is needed.

INPUT LAYOUT EXPECTED
---------------------
    DEMAND/
        NFIELD/   ch01.wav ... ch16.wav
        NPARK/    ch01.wav ... ch16.wav
        NRIVER/   ...
        SPSQUARE/ ...
        STRAFFIC/ ...

(The script also tolerates one extra nesting level, e.g. NFIELD/NFIELD/ch01.wav,
which is how some DEMAND zips extract.)

Usage:
    python demand_curator.py --dry-run
    python demand_curator.py
"""

import argparse
import csv
import random
import re
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================

DEMAND_DIR = Path(
    r"C:\Users\CARE\Downloads\Acoustic based drone detection"
    r"\Audio files\DEMAND"
)

NOISE_MASTER = Path(
    r"C:\Users\CARE\Downloads\Acoustic based drone detection"
    r"\Audio files\Dataset\NOISE_MASTER"
)

REPORT_CSV = NOISE_MASTER / "demand_curation_report.csv"

# ---- routing --------------------------------------------------------
DEMAND_ROUTING = {
    "NFIELD":   ("Nature", "Wind"),      # open field, wind-dominant
    "SPSQUARE": ("Nature", "Wind"),      # open public square, wind-dominant
    "NRIVER":   ("Nature", "Stream"),    # continuous running water
    "STRAFFIC": ("Urban",  "Traffic"),   # continuous road hum
    "NPARK":    ("Human",  "Crowd"),     # audible people
}

# ---- segmentation ---------------------------------------------------
SEGMENT_SEC = 20.0      # 5 min / 20 s = 15 segments per environment
MIN_SEGMENT_SEC = 15.0  # discard a trailing stub shorter than this

OUTPUT_SUBTYPE = "PCM_16"

SEED = 1337             # deterministic channel choice


# ============================================================
# HELPERS
# ============================================================

def find_channel_files(env_dir: Path):
    """
    Return the sorted list of per-channel WAVs for one environment.

    Handles both layouts:
        DEMAND/NFIELD/ch01.wav
        DEMAND/NFIELD/NFIELD/ch01.wav
    """
    wavs = sorted(env_dir.glob("*.wav"))
    if wavs:
        return wavs

    # one level deeper
    for sub in sorted(env_dir.iterdir()):
        if sub.is_dir():
            wavs = sorted(sub.glob("*.wav"))
            if wavs:
                return wavs
    return []


def channel_number(path: Path) -> str:
    """Extract a zero-padded channel number from a filename like 'ch05.wav'."""
    m = re.search(r"ch(\d+)", path.stem, re.IGNORECASE)
    return f"{int(m.group(1)):02d}" if m else "01"


# ============================================================
# MAIN
# ============================================================

def main(dry_run: bool):

    rng = random.Random(SEED)

    print("=" * 72)
    print("DEMAND CURATOR  ->  NOISE_MASTER")
    if dry_run:
        print("*** DRY RUN -- nothing will be written ***")
    print("=" * 72)

    if not DEMAND_DIR.exists():
        print(f"\n[ERROR] DEMAND folder not found:\n  {DEMAND_DIR}")
        return
    if not NOISE_MASTER.exists():
        print(f"\n[ERROR] NOISE_MASTER not found:\n  {NOISE_MASTER}")
        return

    print(f"\nSource : {DEMAND_DIR}")
    print(f"Target : {NOISE_MASTER}")
    print(f"\nSegment length : {SEGMENT_SEC:.0f} s "
          f"(discard trailing stub < {MIN_SEGMENT_SEC:.0f} s)")
    print(f"Channels used  : 1 per environment (seeded, reproducible)")

    print("\nRouting:")
    for env, (sc, tc) in DEMAND_ROUTING.items():
        print(f"  {env:<10} -> {sc}/{tc}")

    rows, errors = [], []
    total_written = 0

    for env, (superclass, target_class) in DEMAND_ROUTING.items():

        print("\n" + "-" * 72)
        print(f"{env}  ->  {superclass}/{target_class}")
        print("-" * 72)

        env_dir = DEMAND_DIR / env
        if not env_dir.exists():
            print(f"  [SKIP] folder not found: {env_dir}")
            continue

        channels = find_channel_files(env_dir)
        if not channels:
            print(f"  [SKIP] no WAV files found under {env_dir}")
            continue

        # ---- pick ONE channel, deterministically -----------------------
        chosen = channels[rng.randrange(len(channels))]
        ch_tag = channel_number(chosen)

        print(f"  Channels available : {len(channels)}")
        print(f"  Channel chosen     : {chosen.name}  (1 of {len(channels)})")

        try:
            audio, sr = sf.read(str(chosen), always_2d=True)
        except Exception as e:
            errors.append({"env": env, "file": str(chosen), "error": str(e)})
            print(f"  [ERROR] {e}")
            continue

        # force mono (DEMAND per-channel files are already mono)
        audio = audio.mean(axis=1)

        duration = len(audio) / sr
        seg_len  = int(SEGMENT_SEC * sr)
        min_len  = int(MIN_SEGMENT_SEC * sr)

        n_full = len(audio) // seg_len
        tail   = len(audio) - n_full * seg_len
        n_segs = n_full + (1 if tail >= min_len else 0)

        print(f"  Duration           : {duration:.1f} s @ {sr} Hz")
        print(f"  Segments           : {n_segs}"
              f"{'  (+1 partial tail kept)' if tail >= min_len else ''}")

        out_dir = NOISE_MASTER / superclass / target_class

        if not dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)

        written_here = 0

        for i in range(n_segs):
            start = i * seg_len
            end   = min(start + seg_len, len(audio))
            seg   = audio[start:end]

            if len(seg) < min_len:
                continue

            name = (f"{superclass}_{target_class}_DEMAND_"
                    f"{env.lower()}_ch{ch_tag}_seg{i:03d}.wav")
            out_path = out_dir / name

            if not dry_run:
                try:
                    sf.write(str(out_path), seg, sr, subtype=OUTPUT_SUBTYPE)
                except Exception as e:
                    errors.append({"env": env, "file": name, "error": str(e)})
                    continue

            rows.append({
                "output_file"  : name,
                "superclass"   : superclass,
                "target_class" : target_class,
                "environment"  : env,
                "source_file"  : chosen.name,
                "channel"      : ch_tag,
                "segment_index": i,
                "start_sec"    : round(start / sr, 2),
                "duration_sec" : round(len(seg) / sr, 2),
                "sample_rate"  : sr,
            })
            written_here += 1

        total_written += written_here
        print(f"  Written            : {written_here}")

    # ---- report ---------------------------------------------------------
    if rows and not dry_run:
        with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # ---- summary --------------------------------------------------------
    print("\n" + "=" * 72)
    print("DRY RUN COMPLETE" if dry_run else "DONE")
    print("=" * 72)
    print(f"  Segments {'planned' if dry_run else 'written'} : {total_written}")
    print(f"  Errors                : {len(errors)}")

    if rows:
        from collections import Counter
        per_class = Counter((r["superclass"], r["target_class"]) for r in rows)
        print("\n  Per class:")
        for (sc, tc), n in sorted(per_class.items()):
            print(f"    {sc}/{tc:<12}: {n:>4} clips")

        total_min = sum(r["duration_sec"] for r in rows) / 60
        print(f"\n  Total audio added : {total_min:.1f} minutes")

    if errors:
        print("\n  Errors:")
        for e in errors[:10]:
            print(f"    {e['env']}: {e['error']}")

    if dry_run:
        print("\n  Re-run without --dry-run to write the files.")
    else:
        print(f"\n  Report: {REPORT_CSV}")
        print("\n  Next: run windfarm_curator.py, then rebuild_statistics.py")
    print("=" * 72)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Add DEMAND recordings to NOISE_MASTER as bed material"
    )
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would be written without writing")
    main(ap.parse_args().dry_run)