#!/usr/bin/env python3
"""
windfarm_curator.py
-------------------
Adds the Wind Farm Noise Benchmark dataset to NOISE_MASTER/Nature/Wind.

WHY
---
After the woodwind cleaning, the Wind bed pool fell to 210 usable clips (>=8 s).
That starved the bed builder: it repeatedly ran out of Wind material, threw
"Could not fill bed", and the retry drew a different class instead -- pushing
Wind down to 14.9% against a 26% target while Rain and Stream absorbed the
overflow. This dataset supplies ~6,000 additional 10 s outdoor wind recordings,
roughly a 30x increase in raw material.

WHAT THIS DATA IS
-----------------
6,000 x 10 s clips recorded outdoors at two South Australian residences,
980 m (H1) and 1.3 km (H2) from the nearest wind turbine. So it is genuine
outdoor wind, but it also carries a low-frequency amplitude-modulated component
from the turbine blades.

That AM component is mostly BELOW the 120 Hz detection band and is removed by
the highpass in Script 2. The accompanying Rating CSVs score each clip 1-5 for
AM presence (1 = confident absence, 5 = confident presence), and the ratings are
strongly bimodal: about half the clips sit near 1.0, the rest near 4.5-5.0.

MAX_AM_RATING lets you exclude the heavily modulated clips. It defaults to None
(keep everything) because the stationarity gate in build_synthetic_noise.py
already measures in-band energy swing per clip and will reject anything too
modulated on its own merits. Set it to e.g. 2.0 if you would rather filter here.

OUTPUT NAMING
-------------
    Nature_Wind_WINDFARM_{site}_{sample:04d}.wav

    e.g. Nature_Wind_WINDFARM_H1_0042.wav

The 3rd underscore-token is "WINDFARM", which rebuild_statistics.py reads as
source_dataset. No change needed to that script.

INPUT LAYOUT EXPECTED
---------------------
    Wind farm noise benchmark/
        Wind farm 1_H1/         <- 3,000 wav files
        Wind farm 2_H2/         <- 3,000 wav files
        Rating_Wind farm 1_H1.csv
        Rating_Wind farm 2_H2.csv

Usage:
    python windfarm_curator.py --dry-run
    python windfarm_curator.py
    python windfarm_curator.py --max-am 2.0    # only low-modulation clips
"""

import argparse
import csv
import shutil
from pathlib import Path

import pandas as pd
import soundfile as sf
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================

WINDFARM_DIR = Path(
    r"C:\Users\CARE\Downloads\Acoustic based drone detection"
    r"\Audio files\Wind farm noise benchmark"
)

NOISE_MASTER = Path(
    r"C:\Users\CARE\Downloads\Acoustic based drone detection"
    r"\Audio files\Dataset\NOISE_MASTER"
)

OUTPUT_DIR = NOISE_MASTER / "Nature" / "Wind"
REPORT_CSV = NOISE_MASTER / "windfarm_curation_report.csv"

# ---- sites ----------------------------------------------------------
SITES = [
    {
        "tag":        "H1",
        "audio_dir":  "Wind farm 1_H1",
        "rating_csv": "Rating_Wind farm 1_H1.csv",
    },
    {
        "tag":        "H2",
        "audio_dir":  "Wind farm 2_H2",
        "rating_csv": "Rating_Wind farm 2_H2.csv",
    },
]

# Amplitude-modulation filter. None = keep all clips.
# The stationarity gate downstream measures in-band energy swing directly,
# so filtering here is optional. Set to ~2.0 to keep only low-AM clips.
MAX_AM_RATING = None

OUTPUT_SUBTYPE = "PCM_16"


# ============================================================
# HELPERS
# ============================================================

def load_ratings(csv_path: Path) -> dict:
    """
    Read a Rating CSV into {sample_number: rating}.

    The two CSVs are NOT consistently formatted -- H1 has a column named
    'Sample' while H2 has 'Sample ' with a trailing space. Columns are
    stripped before use rather than indexed by name literally.
    """
    if not csv_path.exists():
        print(f"  [WARN] rating CSV not found: {csv_path.name}")
        return {}

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    if "Sample" not in df.columns or "Rating" not in df.columns:
        print(f"  [WARN] unexpected columns in {csv_path.name}: "
              f"{list(df.columns)}")
        return {}

    return dict(zip(df["Sample"].astype(int), df["Rating"].astype(float)))


def sample_number(path: Path):
    """Pull the numeric sample id out of a filename. None if absent."""
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits) if digits else None


# ============================================================
# MAIN
# ============================================================

def main(dry_run: bool, max_am):

    print("=" * 72)
    print("WIND FARM NOISE BENCHMARK CURATOR  ->  NOISE_MASTER/Nature/Wind")
    if dry_run:
        print("*** DRY RUN -- nothing will be written ***")
    print("=" * 72)

    if not WINDFARM_DIR.exists():
        print(f"\n[ERROR] Source folder not found:\n  {WINDFARM_DIR}")
        return
    if not NOISE_MASTER.exists():
        print(f"\n[ERROR] NOISE_MASTER not found:\n  {NOISE_MASTER}")
        return

    print(f"\nSource : {WINDFARM_DIR}")
    print(f"Target : {OUTPUT_DIR}")
    print(f"AM filter : "
          f"{'none (keep all)' if max_am is None else f'keep rating <= {max_am}'}")

    if not dry_run:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows, errors = [], []
    total_written = 0
    total_skipped_am = 0

    for site in SITES:

        tag = site["tag"]
        print("\n" + "-" * 72)
        print(f"SITE {tag}")
        print("-" * 72)

        audio_dir = WINDFARM_DIR / site["audio_dir"]
        if not audio_dir.exists():
            print(f"  [SKIP] audio folder not found: {audio_dir}")
            continue

        ratings = load_ratings(WINDFARM_DIR / site["rating_csv"])
        print(f"  Ratings loaded : {len(ratings):,}")

        wavs = sorted(audio_dir.glob("*.wav"))
        print(f"  WAV files found: {len(wavs):,}")

        if not wavs:
            print("  [SKIP] no WAV files")
            continue

        written_here = 0
        skipped_here = 0

        for wav in tqdm(wavs, desc=f"  {tag}", unit="file"):

            num = sample_number(wav)
            rating = ratings.get(num) if num is not None else None

            # optional AM filter
            if max_am is not None and rating is not None and rating > max_am:
                skipped_here += 1
                continue

            name = (f"Nature_Wind_WINDFARM_{tag}_"
                    f"{num:04d}.wav" if num is not None
                    else f"Nature_Wind_WINDFARM_{tag}_{wav.stem}.wav")
            out_path = OUTPUT_DIR / name

            duration = 0.0
            sr = 0
            try:
                info     = sf.info(str(wav))
                duration = round(info.duration, 3)
                sr       = info.samplerate
            except Exception as e:
                errors.append({"site": tag, "file": wav.name, "error": str(e)})
                continue

            if not dry_run:
                try:
                    shutil.copy2(wav, out_path)
                except Exception as e:
                    errors.append({"site": tag, "file": wav.name,
                                   "error": str(e)})
                    continue

            rows.append({
                "output_file"  : name,
                "site"         : tag,
                "sample_number": num if num is not None else "",
                "am_rating"    : round(rating, 3) if rating is not None else "",
                "source_file"  : wav.name,
                "duration_sec" : duration,
                "sample_rate"  : sr,
            })
            written_here += 1

        total_written += written_here
        total_skipped_am += skipped_here

        print(f"  {'Planned' if dry_run else 'Written'} : {written_here:,}")
        if skipped_here:
            print(f"  Skipped (AM > {max_am}) : {skipped_here:,}")

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
    print(f"  Files {'planned' if dry_run else 'written'} : {total_written:,}")
    if max_am is not None:
        print(f"  Skipped by AM filter  : {total_skipped_am:,}")
    print(f"  Errors                : {len(errors)}")

    if rows:
        durations = [r["duration_sec"] for r in rows if r["duration_sec"]]
        if durations:
            print(f"\n  Clip duration : "
                  f"{min(durations):.1f}-{max(durations):.1f} s "
                  f"(median {sorted(durations)[len(durations)//2]:.1f} s)")
            print(f"  Total audio   : {sum(durations)/60:.1f} minutes "
                  f"({sum(durations)/3600:.2f} hours)")

        rated = [r["am_rating"] for r in rows if r["am_rating"] != ""]
        if rated:
            low  = sum(1 for r in rated if r <= 2.0)
            high = sum(1 for r in rated if r >= 4.0)
            print(f"\n  AM ratings: {low:,} low (<=2.0) | "
                  f"{high:,} high (>=4.0) | {len(rated):,} total")
            print("  (the stationarity gate in build_synthetic_noise.py will")
            print("   independently reject clips whose in-band energy swings)")

    if errors:
        print("\n  Errors:")
        for e in errors[:10]:
            print(f"    {e['site']} {e['file']}: {e['error']}")

    if dry_run:
        print("\n  Re-run without --dry-run to copy the files.")
    else:
        print(f"\n  Report: {REPORT_CSV}")
        print("\n  Next: run rebuild_statistics.py to regenerate")
        print("        noise_master_metadata.csv")
    print("=" * 72)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Add Wind Farm Noise Benchmark clips to NOISE_MASTER"
    )
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would be written without writing")
    ap.add_argument("--max-am", type=float, default=MAX_AM_RATING,
                    help="Only keep clips with AM rating <= this "
                         "(1=no modulation, 5=strong). Default: keep all")
    a = ap.parse_args()
    main(a.dry_run, a.max_am)