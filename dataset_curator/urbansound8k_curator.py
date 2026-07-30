import os
import shutil
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

# ============================================================
# PATHS
# ============================================================

ROOT = r"C:\Users\CARE\Downloads\Acoustic based detection of Drones\Audio files\Noise\UrbanSound8K"

METADATA_CSV = os.path.join(
    ROOT,
    "metadata",
    "UrbanSound8K.csv"
)

OUTPUT_DIR = os.path.join(
    os.path.dirname(ROOT),
    "UrbanSound8K_CURATED"
)

# ============================================================
# SANITY CHECKS
# ============================================================

print("=" * 60)
print("URBANSOUND8K CURATOR")
print("=" * 60)

print("\nChecking paths...\n")

print("ROOT:")
print(ROOT)

print("\nMETADATA:")
print(METADATA_CSV)

if not os.path.exists(ROOT):
    raise FileNotFoundError(
        f"\nROOT folder not found:\n{ROOT}"
    )

if not os.path.exists(METADATA_CSV):
    raise FileNotFoundError(
        f"\nMetadata CSV not found:\n{METADATA_CSV}"
    )

print("\n✓ Paths verified")

# ============================================================
# AUDIT V1 MAPPING
# ============================================================

CLASS_MAPPING = {
    "air_conditioner": ("Hard_Negatives", "AirConditioner"),
    "car_horn": ("Urban", "Traffic"),
    "children_playing": ("Human", "Children"),
    "dog_bark": ("Animals", "Dog"),
    "drilling": ("Hard_Negatives", "Drill"),
    "engine_idling": ("Hard_Negatives", "Engine"),
    "gun_shot": ("Hard_Negatives", "Gunshot"),
    "jackhammer": ("Hard_Negatives", "Jackhammer"),
    "siren": ("Urban", "Siren")
}

REMOVE_CLASSES = {
    "street_music"
}

# ============================================================
# OUTPUT
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

stats = defaultdict(int)

curation_log = []
skipped_files = []
errors = []

# ============================================================
# LOAD METADATA
# ============================================================

df = pd.read_csv(METADATA_CSV)

print(f"\nLoaded metadata for {len(df):,} clips")

# ============================================================
# PROCESS
# ============================================================

for _, row in tqdm(df.iterrows(), total=len(df)):

    try:

        file_name = row["slice_file_name"]

        fold = f"fold{row['fold']}"

        class_name = row["class"]

        if class_name in REMOVE_CLASSES:

            skipped_files.append({
                "file": file_name,
                "reason": "Removed class"
            })

            continue

        if class_name not in CLASS_MAPPING:

            skipped_files.append({
                "file": file_name,
                "reason": "Class not mapped"
            })

            continue

        superclass, target_class = CLASS_MAPPING[class_name]

        src = os.path.join(
            ROOT,
            "audio",
            fold,
            file_name
        )

        if not os.path.exists(src):

            errors.append({
                "file": file_name,
                "error": "Source file missing"
            })

            continue

        out_dir = os.path.join(
            OUTPUT_DIR,
            fold,
            superclass,
            target_class
        )

        os.makedirs(out_dir, exist_ok=True)

        new_name = (
            f"{superclass}_"
            f"{target_class}_"
            f"{file_name}"
        )

        dst = os.path.join(
            out_dir,
            new_name
        )

        shutil.copy2(src, dst)

        stats[f"{superclass}/{target_class}"] += 1

        curation_log.append({
            "original_file": file_name,
            "fold": fold,
            "class": class_name,
            "superclass": superclass,
            "target_class": target_class,
            "new_file": new_name
        })

    except Exception as e:

        errors.append({
            "file": row.get("slice_file_name", "unknown"),
            "error": str(e)
        })

# ============================================================
# SAVE REPORTS
# ============================================================

pd.DataFrame(
    [{"class": k, "count": v}
     for k, v in sorted(stats.items())]
).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "dataset_statistics.csv"
    ),
    index=False
)

pd.DataFrame(curation_log).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "curation_log.csv"
    ),
    index=False
)

pd.DataFrame(skipped_files).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "skipped_files.csv"
    ),
    index=False
)

pd.DataFrame(errors).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "errors.csv"
    ),
    index=False
)

# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("CURATION COMPLETE")
print("=" * 60)

print(f"Total curated files : {len(curation_log):,}")
print(f"Skipped files       : {len(skipped_files):,}")
print(f"Errors              : {len(errors):,}")

print("\nOutput folder:")
print(OUTPUT_DIR)

print("\nGenerated files:")
print("dataset_statistics.csv")
print("curation_log.csv")
print("skipped_files.csv")
print("errors.csv")

print("\n" + "=" * 60)