import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import soundfile as sf

# ============================================================
# CONFIGURATION
# ============================================================

NOISE_MASTER = r"C:\Users\CARE\Downloads\Acoustic based drone detection\Audio files\Dataset\NOISE_MASTER"

SUPERCLASSES = {
    "Animals",
    "Nature",
    "Human",
    "Urban",
    "Hard_Negatives",
}

# ============================================================
# SCAN
# ============================================================

print("=" * 65)
print("NOISE MASTER — PER FILE METADATA BUILDER")
print("=" * 65)
print(f"\nScanning: {NOISE_MASTER}\n")

records = []
errors  = []

root = Path(NOISE_MASTER)

all_files = [
    p for p in root.rglob("*")
    if p.is_file()
    and p.suffix.lower() in (".wav", ".mp3", ".flac", ".ogg")
]

print(f"Found {len(all_files):,} audio files — reading metadata...\n")

for fpath in tqdm(all_files, unit="file"):

    parts = fpath.parts

    # Find superclass index
    superclass_idx = None
    for i, part in enumerate(parts):
        if part in SUPERCLASSES:
            superclass_idx = i
            break

    if superclass_idx is None or superclass_idx + 1 >= len(parts):
        continue

    superclass   = parts[superclass_idx]
    target_class = parts[superclass_idx + 1]

    # Parse source dataset from filename
    # Format: Superclass_TargetClass_DATASETTAG_OriginalStem.wav
    fname  = fpath.stem   # without extension
    fparts = fname.split("_")

    # Dataset tag is always the 3rd token
    source_dataset = fparts[2] if len(fparts) >= 3 else "UNKNOWN"

    # Read audio metadata
    try:
        info           = sf.info(str(fpath))
        duration_sec   = round(info.duration, 3)
        sample_rate    = info.samplerate
        channels       = info.channels
        subtype        = info.subtype        # e.g. PCM_16, PCM_24
        frames         = info.frames

    except Exception as e:
        errors.append({
            "file"  : str(fpath),
            "error" : str(e),
        })
        duration_sec = 0.0
        sample_rate  = 0
        channels     = 0
        subtype      = "ERROR"
        frames       = 0

    records.append({
        "filename"       : fpath.name,
        "superclass"     : superclass,
        "target_class"   : target_class,
        "source_dataset" : source_dataset,
        "duration_sec"   : duration_sec,
        "duration_min"   : round(duration_sec / 60, 4),
        "sample_rate"    : sample_rate,
        "channels"       : channels,
        "bit_depth"      : subtype,
        "frames"         : frames,
        "abs_path"       : str(fpath),
    })

# ============================================================
# BUILD DATAFRAME
# ============================================================

df = pd.DataFrame(records)

# Sort by superclass → target_class → duration descending
# so longest files float to top within each class
df = df.sort_values(
    ["superclass", "target_class", "duration_sec"],
    ascending=[True, True, False]
).reset_index(drop=True)

# Add a duration category column — useful for synthesis decisions
def duration_bucket(sec):
    if sec < 2:
        return "very_short"     # < 2s
    elif sec < 5:
        return "short"          # 2–5s
    elif sec < 10:
        return "medium"         # 5–10s
    elif sec < 20:
        return "long"           # 10–20s
    else:
        return "very_long"      # 20s+

df["duration_bucket"] = df["duration_sec"].apply(duration_bucket)

# ============================================================
# SAVE
# ============================================================

out_metadata = os.path.join(NOISE_MASTER, "noise_master_metadata.csv")
out_errors   = os.path.join(NOISE_MASTER, "metadata_errors.csv")

df.to_csv(out_metadata, index=False)
pd.DataFrame(errors).to_csv(out_errors, index=False)

# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("DURATION BUCKET SUMMARY")
print("=" * 65)

bucket_order = ["very_short", "short", "medium", "long", "very_long"]
bucket_stats = (
    df.groupby("duration_bucket")
    .agg(files=("filename", "count"), total_sec=("duration_sec", "sum"))
    .reindex(bucket_order)
    .fillna(0)
)

print(f"\n{'Bucket':<15} {'Range':<15} {'Files':>7}  {'Total Min':>10}")
print("-" * 52)
ranges = {
    "very_short" : "< 2s",
    "short"      : "2s – 5s",
    "medium"     : "5s – 10s",
    "long"       : "10s – 20s",
    "very_long"  : "> 20s",
}
for bucket in bucket_order:
    row = bucket_stats.loc[bucket]
    print(
        f"{bucket:<15} "
        f"{ranges[bucket]:<15} "
        f"{int(row['files']):>7,}  "
        f"{row['total_sec']/60:>10.1f}"
    )

print("\n" + "=" * 65)
print("PER CLASS DURATION PROFILE")
print("=" * 65)
print(
    f"{'Superclass':<20} {'Class':<22} "
    f"{'Files':>6}  {'AvgSec':>7}  {'MaxSec':>7}  "
    f"{'LongFiles(>10s)':>15}"
)
print("-" * 82)

for (sc, tc), grp in df.groupby(["superclass", "target_class"]):
    long_files = (grp["duration_sec"] >= 10).sum()
    print(
        f"{sc:<20} {tc:<22} "
        f"{len(grp):>6,}  "
        f"{grp['duration_sec'].mean():>7.1f}  "
        f"{grp['duration_sec'].max():>7.1f}  "
        f"{long_files:>15,}"
    )

print("\n" + "=" * 65)
total_files = len(df)
total_sec   = df["duration_sec"].sum()
print(f"GRAND TOTAL  :  {total_files:,} files")
print(f"             :  {total_sec/60:.1f} minutes")
print(f"             :  {total_sec/3600:.2f} hours")
print("=" * 65)

if errors:
    print(f"\nWarning: {len(errors)} files could not be read.")
    print("See metadata_errors.csv")

print(f"\nSaved: noise_master_metadata.csv")
print(f"       metadata_errors.csv")
print("=" * 65)