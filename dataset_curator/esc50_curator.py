import os
import shutil
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

# ============================================================
# PATHS
# ============================================================

ROOT = r"C:\Users\CARE\Downloads\Acoustic based detection of Drones\Audio files\Noise\ESC-50-master"

METADATA_CSV = os.path.join(
    ROOT,
    "meta",
    "esc50.csv"
)

OUTPUT_DIR = os.path.join(
    os.path.dirname(ROOT),
    "ESC50_CURATED"
)

# ============================================================
# CLASS MAPPING
# ============================================================

CLASS_MAPPING = {

    # Animals
    "dog": ("Animals", "Dog"),
    "rooster": ("Animals", "Bird"),
    "pig": ("Animals", "DomesticAnimal"),
    "cow": ("Animals", "DomesticAnimal"),
    "frog": ("Animals", "Frog"),
    "cat": ("Animals", "Cat"),
    "hen": ("Animals", "Bird"),
    "insects": ("Animals", "Insects"),
    "sheep": ("Animals", "DomesticAnimal"),
    "crow": ("Animals", "Bird"),

    # Nature
    "rain": ("Nature", "Rain"),
    "sea_waves": ("Nature", "Water"),
    "crackling_fire": ("Nature", "Fire"),
    "crickets": ("Nature", "Insects"),
    "chirping_birds": ("Animals", "Bird"),
    "water_drops": ("Nature", "Water"),
    "wind": ("Nature", "Wind"),
    "pouring_water": ("Nature", "Water"),
    "thunderstorm": ("Nature", "Thunder"),

    # Human
    "crying_baby": ("Human", "Children"),
    "sneezing": ("Human", "HumanNoise"),
    "clapping": ("Human", "HumanNoise"),
    "breathing": ("Human", "HumanNoise"),
    "coughing": ("Human", "HumanNoise"),
    "footsteps": ("Human", "HumanActivity"),
    "laughing": ("Human", "HumanNoise"),
    "snoring": ("Human", "HumanNoise"),

    # Domestic
    "vacuum_cleaner": ("Hard_Negatives", "VacuumCleaner"),

    # Urban / Hard Negatives
    "helicopter": ("Hard_Negatives", "Aircraft"),
    "chainsaw": ("Hard_Negatives", "Chainsaw"),
    "siren": ("Urban", "Siren"),
    "car_horn": ("Urban", "Traffic"),
    "engine": ("Hard_Negatives", "Engine"),
    "train": ("Urban", "RailTransport"),
    "airplane": ("Hard_Negatives", "Aircraft"),
    "fireworks": ("Hard_Negatives", "Explosion"),
    "hand_saw": ("Hard_Negatives", "Saw"),
}

REMOVE_CLASSES = {

    "toilet_flush",
    "brushing_teeth",
    "drinking_sipping",

    "door_wood_knock",
    "mouse_click",
    "keyboard_typing",
    "door_wood_creaks",
    "can_opening",
    "washing_machine",
    "clock_alarm",
    "clock_tick",
    "glass_breaking",

    "church_bells"
}

# ============================================================
# INITIALIZE
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

stats = defaultdict(int)
curation_log = []
skipped_files = []
errors = []

# ============================================================
# LOAD CSV
# ============================================================

print("=" * 60)
print("ESC50 CURATOR")
print("=" * 60)

df = pd.read_csv(METADATA_CSV)

print(f"\nLoaded metadata for {len(df):,} clips")

# ============================================================
# PROCESS
# ============================================================

for _, row in tqdm(df.iterrows(), total=len(df)):

    try:

        filename = row["filename"]
        fold = f"fold{row['fold']}"
        category = row["category"]

        if category in REMOVE_CLASSES:

            skipped_files.append({
                "file": filename,
                "reason": "Removed class"
            })
            continue

        if category not in CLASS_MAPPING:

            skipped_files.append({
                "file": filename,
                "reason": "Class not mapped"
            })
            continue

        superclass, target_class = CLASS_MAPPING[category]

        src = os.path.join(
            ROOT,
            "audio",
            filename
        )

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
            f"{filename}"
        )

        dst = os.path.join(out_dir, new_name)

        shutil.copy2(src, dst)

        stats[f"{superclass}/{target_class}"] += 1

        curation_log.append({
            "original_file": filename,
            "fold": fold,
            "category": category,
            "superclass": superclass,
            "target_class": target_class,
            "new_file": new_name
        })

    except Exception as e:

        errors.append({
            "file": filename,
            "error": str(e)
        })

# ============================================================
# REPORTS
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

print("\n" + "=" * 60)
print("CURATION COMPLETE")
print("=" * 60)

print(f"Total curated files : {len(curation_log):,}")
print(f"Skipped files       : {len(skipped_files):,}")
print(f"Errors              : {len(errors):,}")

print("\nOutput folder:")
print(OUTPUT_DIR)