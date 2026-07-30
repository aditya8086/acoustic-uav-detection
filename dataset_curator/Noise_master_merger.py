import os
import shutil
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# ============================================================
# CONFIGURATION — update these paths to match your machine
# ============================================================

FSD50K_CURATED = r"C:\Users\CARE\Downloads\Acoustic based drone detection\Audio files\Noise\FSD50K_CURATED"
US8K_CURATED   = r"C:\Users\CARE\Downloads\Acoustic based drone detection\Audio files\Noise\UrbanSound8K_CURATED"
ESC50_CURATED  = r"C:\Users\CARE\Downloads\Acoustic based drone detection\Audio files\Noise\ESC-50-master_CURATED"
NOISE_MASTER   = r"C:\Users\CARE\Downloads\Acoustic based drone detection\Audio files\Noise\NOISE_MASTER_V2"

DATASET_TAG = {
    FSD50K_CURATED : "FSD50K",
    US8K_CURATED   : "US8K",
    ESC50_CURATED  : "ESC50",
}

SUPERCLASSES = {"Animals", "Nature", "Human", "Urban", "Hard_Negatives"}

# ============================================================
# INITIALIZE
# ============================================================

os.makedirs(NOISE_MASTER, exist_ok=True)

seen       = set()
stats      = defaultdict(int)
copied_log = []
skipped_log= []
errors_log = []

# ============================================================
# HELPERS
# ============================================================

def extract_raw_id(stem, superclass, target_class):
    """
    The curators already named files as:
        Superclass_TargetClass_RawID   (FSD50K, ESC50)
        Superclass_TargetClass_OriginalFilename  (US8K)

    We strip the leading "Superclass_TargetClass_" prefix to get
    only the raw original ID, so the merger can add its own clean
    prefix: Superclass_TargetClass_DATASETTAG_RawID

    Example:
        stem      = "Animals_Bird_339932"
        prefix    = "Animals_Bird_"
        raw_id    = "339932"        ← correct
    """
    prefix = f"{superclass}_{target_class}_"
    if stem.startswith(prefix):
        return stem[len(prefix):]
    # If no prefix found (shouldn't happen) return stem as-is
    return stem


def parse_curated_root(root):
    """
    Walks a curated root and yields (superclass, target_class, abs_path).
    Works transparently for both fold-based and flat structures.
    """
    root = Path(root)
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in (".wav", ".mp3", ".flac", ".ogg"):
            continue

        parts = path.parts

        superclass_idx = None
        for i, part in enumerate(parts):
            if part in SUPERCLASSES:
                superclass_idx = i
                break

        if superclass_idx is None:
            continue
        if superclass_idx + 1 >= len(parts):
            continue

        superclass   = parts[superclass_idx]
        target_class = parts[superclass_idx + 1]

        # File must be directly inside TargetClass (depth check)
        if superclass_idx + 2 != len(parts) - 1:
            continue

        yield superclass, target_class, path

# ============================================================
# MERGE
# ============================================================

def merge_dataset(curated_root):

    tag   = DATASET_TAG[curated_root]
    label = f"[{tag}]"

    print(f"\n{label} Scanning: {curated_root}")
    entries = list(parse_curated_root(curated_root))
    print(f"{label} Found {len(entries):,} audio files")

    for superclass, target_class, src_path in tqdm(entries, desc=label, unit="file"):

        try:
            stem = src_path.stem
            ext  = src_path.suffix.lower()

            # Strip curator-added prefix to get the raw original ID
            raw_id = extract_raw_id(stem, superclass, target_class)

            # Build clean filename: Superclass_TargetClass_TAG_RawID.wav
            new_name = f"{superclass}_{target_class}_{tag}_{raw_id}{ext}"

            # Dedup key: same class + same dataset + same raw id
            dedup_key = (superclass, target_class, tag, raw_id)

            if dedup_key in seen:
                skipped_log.append({
                    "source"       : tag,
                    "superclass"   : superclass,
                    "target_class" : target_class,
                    "original_file": src_path.name,
                    "reason"       : "Duplicate within class",
                })
                continue

            seen.add(dedup_key)

            dest_dir = Path(NOISE_MASTER) / superclass / target_class
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / new_name

            shutil.copy2(src_path, dest_path)
            stats[(superclass, target_class)] += 1

            copied_log.append({
                "source"        : tag,
                "superclass"    : superclass,
                "target_class"  : target_class,
                "original_file" : src_path.name,
                "new_file"      : new_name,
                "dest_path"     : str(dest_path),
            })

        except Exception as e:
            errors_log.append({
                "source" : tag,
                "file"   : str(src_path),
                "error"  : str(e),
            })

# ============================================================
# RUN
# ============================================================

print("=" * 65)
print("NOISE MASTER MERGER V2")
print("=" * 65)

for dataset_root in [FSD50K_CURATED, US8K_CURATED, ESC50_CURATED]:
    merge_dataset(dataset_root)

# ============================================================
# SAVE REPORTS
# ============================================================

print("\nSaving reports...")

stats_rows = []
for (sc, tc), count in sorted(stats.items()):
    stats_rows.append({"superclass": sc, "target_class": tc, "files": count})

stats_df = pd.DataFrame(stats_rows)
stats_df.to_csv(os.path.join(NOISE_MASTER, "dataset_statistics.csv"), index=False)

source_breakdown = defaultdict(int)
for row in copied_log:
    source_breakdown[row["source"]] += 1

breakdown_df = pd.DataFrame([
    {"source": k, "files": v} for k, v in sorted(source_breakdown.items())
])
breakdown_df.to_csv(os.path.join(NOISE_MASTER, "source_breakdown.csv"), index=False)

pd.DataFrame(copied_log).to_csv(os.path.join(NOISE_MASTER, "manifest.csv"), index=False)
pd.DataFrame(skipped_log).to_csv(os.path.join(NOISE_MASTER, "skipped_files.csv"), index=False)
pd.DataFrame(errors_log).to_csv(os.path.join(NOISE_MASTER, "errors.csv"), index=False)

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("MERGE COMPLETE")
print("=" * 65)
print(f"\nTotal files copied  : {len(copied_log):,}")
print(f"Duplicates skipped  : {len(skipped_log):,}")
print(f"Errors              : {len(errors_log):,}")

print("\nPer-source breakdown:")
for row in breakdown_df.itertuples():
    print(f"  {row.source:<10} {row.files:>6,} files")

print("\nPer-class breakdown:")
for row in stats_df.itertuples():
    print(f"  {row.superclass:<20} {row.target_class:<25} {row.files:>6,}")

print(f"\nOutput: {NOISE_MASTER}")
print("=" * 65)