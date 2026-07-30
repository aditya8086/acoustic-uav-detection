import os
import shutil
from pathlib import Path
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

# ==========================================================
# USER CONFIGURATION
# ==========================================================

ROOT = r"D:\FSD50K"

DEV_AUDIO = os.path.join(ROOT, "FSD50K.dev_audio")
EVAL_AUDIO = os.path.join(ROOT, "FSD50K.eval_audio")

DEV_CSV = os.path.join(ROOT, "FSD50K.ground_truth", "dev.csv")
EVAL_CSV = os.path.join(ROOT, "FSD50K.ground_truth", "eval.csv")

AUDIT_CSV = r"D:\fsd50k_audit_v2.csv"

OUTPUT_DIR = r"D:\FSD50K_CURATED"

# ==========================================================
# CREATE OUTPUT
# ==========================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("FSD50K CURATOR V2")
print("=" * 60)

# ==========================================================
# LOAD AUDIT TABLE
# ==========================================================

print("\nLoading audit file...")

audit = pd.read_csv(AUDIT_CSV)

audit = audit[
    audit["decision"]
    .astype(str)
    .str.upper()
    .eq("KEEP")
]

label_map = {}

for _, row in audit.iterrows():

    class_name = str(row["class_name"]).strip()

    superclass = str(row["target_superclass"]).strip()

    target_class = str(row["target_class"]).strip()

    label_map[class_name] = (
        superclass,
        target_class
    )

print(f"Loaded {len(label_map)} KEEP classes")

# ==========================================================
# LOGGING STRUCTURES
# ==========================================================

stats = defaultdict(int)

curation_log = []

skipped_files = []

errors = []

# ==========================================================
# PROCESS FUNCTION
# ==========================================================

def process_dataset(csv_file, audio_folder, dataset_name):

    print(f"\nProcessing {dataset_name}")

    df = pd.read_csv(csv_file)

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=dataset_name
    ):

        try:

            fname = str(row["fname"]).strip()

            wav_path = os.path.join(
                audio_folder,
                f"{fname}.wav"
            )

            if not os.path.exists(wav_path):

                skipped_files.append({
                    "file_id": fname,
                    "reason": "Audio file missing"
                })

                continue

            raw_labels = str(row["labels"])

            labels = [
                x.strip()
                for x in raw_labels.split(",")
                if x.strip()
            ]

            matched = False

            for label in labels:

                if label not in label_map:
                    continue

                matched = True

                superclass, target_class = label_map[label]

                out_dir = os.path.join(
                    OUTPUT_DIR,
                    superclass,
                    target_class
                )

                os.makedirs(
                    out_dir,
                    exist_ok=True
                )

                new_name = (
                    f"{superclass}_"
                    f"{target_class}_"
                    f"{fname}.wav"
                )

                dst = os.path.join(
                    out_dir,
                    new_name
                )

                if not os.path.exists(dst):

                    shutil.copy2(
                        wav_path,
                        dst
                    )

                stats[
                    (superclass, target_class)
                ] += 1

                curation_log.append({
                    "file_id": fname,
                    "original_label": label,
                    "superclass": superclass,
                    "target_class": target_class,
                    "output_file": dst
                })

            if not matched:

                skipped_files.append({
                    "file_id": fname,
                    "reason": "No KEEP labels"
                })

        except Exception as e:

            errors.append({
                "file_id": row.get(
                    "fname",
                    "UNKNOWN"
                ),
                "error": str(e)
            })

# ==========================================================
# RUN
# ==========================================================

process_dataset(
    DEV_CSV,
    DEV_AUDIO,
    "DEV"
)

process_dataset(
    EVAL_CSV,
    EVAL_AUDIO,
    "EVAL"
)

# ==========================================================
# SAVE STATISTICS
# ==========================================================

print("\nSaving reports...")

stats_rows = []

for (superclass, target_class), count in sorted(
    stats.items()
):

    stats_rows.append({
        "superclass": superclass,
        "target_class": target_class,
        "files": count
    })

stats_df = pd.DataFrame(stats_rows)

stats_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "dataset_statistics.csv"
    ),
    index=False
)

# ==========================================================
# SAVE CURATION LOG
# ==========================================================

pd.DataFrame(
    curation_log
).to_csv(

    os.path.join(
        OUTPUT_DIR,
        "curation_log.csv"
    ),

    index=False
)

# ==========================================================
# SAVE SKIPPED FILES
# ==========================================================

pd.DataFrame(
    skipped_files
).to_csv(

    os.path.join(
        OUTPUT_DIR,
        "skipped_files.csv"
    ),

    index=False
)

# ==========================================================
# SAVE ERRORS
# ==========================================================

pd.DataFrame(
    errors
).to_csv(

    os.path.join(
        OUTPUT_DIR,
        "errors.csv"
    ),

    index=False
)

# ==========================================================
# FINAL SUMMARY
# ==========================================================

print("\n" + "=" * 60)

print("CURATION COMPLETE")

print("=" * 60)

print(
    f"Total curated copies : "
    f"{len(curation_log):,}"
)

print(
    f"Skipped clips : "
    f"{len(skipped_files):,}"
)

print(
    f"Errors : "
    f"{len(errors):,}"
)

print(
    f"\nOutput Folder:\n"
    f"{OUTPUT_DIR}"
)

print("\nGenerated Files:")

print("dataset_statistics.csv")
print("curation_log.csv")
print("skipped_files.csv")
print("errors.csv")

print("=" * 60)