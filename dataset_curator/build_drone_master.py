#!/usr/bin/env python3
"""
build_drone_master.py
---------------------
Extracts individual channels from all drone recordings across:
    - UaVirBASE (original, 10m and 20m recordings)
    - UaVirBASE_30m through UaVirBASE_400m (simulated distances)
    - NASA uav (62 converted recordings)

and saves them as mono WAV files into a single DRONE_MASTER folder.

Skips ambient recordings (label.json drone.sound_source == "Ambient Noise")
for all UAVirBase variants.

Output naming:
    UAVirBase:  UAVirBASE_{variant}_{folder_name}_ch{n}.wav
                e.g. UAVirBASE_original_20241115_093611_ch1.wav
                     UAVirBASE_100m_20241115_093611_ch3.wav

    NASA:       NASA_{folder_name}_ch{n}.wav
                e.g. NASA_edge_flyover_066_ch1.wav

Output format: PCM_16, original sample rate (no resampling here)

After running, DRONE_MASTER will contain all mono drone audio files
ready to be used by the soundscape synthesizer.

Usage:
    python build_drone_master.py
"""

import json
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================

DATASETS_DIR = Path(
    r"C:\Users\CARE\Downloads\Acoustic based drone detection"
    r"\Audio files\Datasets"
)

DRONE_MASTER = DATASETS_DIR / "DRONE_MASTER"

# All UAVirBase variant folder names
# Original + 13 simulated (30m-325m) + 3 more (350/375/400m)
UAVIRBASE_VARIANTS = [
    "UaVirBASE",
    "UaVirBASE_30m",
    "UaVirBASE_50m",
    "UaVirBASE_75m",
    "UaVirBASE_100m",
    "UaVirBASE_125m",
    "UaVirBASE_150m",
    "UaVirBASE_175m",
    "UaVirBASE_200m",
    "UaVirBASE_225m",
    "UaVirBASE_250m",
    "UaVirBASE_275m",
    "UaVirBASE_300m",
    "UaVirBASE_325m",
    "UaVirBASE_350m",
    "UaVirBASE_375m",
    "UaVirBASE_400m",
]

NASA_DIR = DATASETS_DIR / "NASA Drone dataset" / "nasa uav"

OUTPUT_SUBTYPE = "PCM_16"

# ============================================================
# HELPERS
# ============================================================

def is_ambient(folder: Path) -> bool:
    """Returns True if this UAVirBase folder is an ambient (no-drone) recording."""
    label_path = folder / "label.json"
    if not label_path.exists():
        return False
    with open(label_path, "r", encoding="utf-8") as f:
        label = json.load(f)
    return label.get("drone", {}).get("sound_source", "") == "Ambient Noise"


def extract_channels(wav_path: Path, out_prefix: str, out_dir: Path,
                     copied: list, skipped: list, errors: list):
    """
    Load a multi-channel WAV, split into mono channels,
    save each as {out_prefix}_ch{n}.wav in out_dir.
    """
    try:
        audio, sample_rate = sf.read(str(wav_path), always_2d=True)
        n_samples, n_channels = audio.shape

        for ch_idx in range(n_channels):
            ch_num   = ch_idx + 1
            out_name = f"{out_prefix}_ch{ch_num}.wav"
            out_path = out_dir / out_name

            if out_path.exists():
                skipped.append(out_name)
                continue

            sf.write(
                str(out_path),
                audio[:, ch_idx],
                sample_rate,
                subtype=OUTPUT_SUBTYPE,
            )

            copied.append({
                "file"        : out_name,
                "sample_rate" : sample_rate,
                "duration_sec": round(n_samples / sample_rate, 3),
                "channels"    : n_channels,
            })

    except Exception as e:
        errors.append({
            "source": str(wav_path),
            "error" : str(e),
        })


# ============================================================
# MAIN
# ============================================================

DRONE_MASTER.mkdir(parents=True, exist_ok=True)

copied  = []
skipped = []
errors  = []

print("=" * 65)
print("BUILD DRONE MASTER")
print("=" * 65)
print(f"\nOutput: {DRONE_MASTER}\n")

# ── UAVirBase variants ─────────────────────────────────────────────────────

print("=" * 65)
print("PROCESSING UAVirBase VARIANTS")
print("=" * 65)

for variant in UAVIRBASE_VARIANTS:

    variant_dir = DATASETS_DIR / variant

    if not variant_dir.exists():
        print(f"\n[SKIP] {variant} — folder not found, skipping.")
        continue

    # Variant tag for filename: "original" for UaVirBASE, "30m" for UaVirBASE_30m etc.
    tag = "original" if variant == "UaVirBASE" else variant.replace("UaVirBASE_", "")

    session_folders = sorted([f for f in variant_dir.iterdir() if f.is_dir()])
    drone_folders   = [f for f in session_folders if not is_ambient(f)]
    ambient_count   = len(session_folders) - len(drone_folders)

    print(f"\n[{variant}]  {len(drone_folders)} drone + {ambient_count} ambient (skipped)")

    for folder in tqdm(drone_folders, desc=f"  {variant}", unit="folder"):

        wav_path = folder / "output.wav"

        if not wav_path.exists():
            errors.append({"source": str(folder), "error": "output.wav not found"})
            continue

        out_prefix = f"UAVirBASE_{tag}_{folder.name}"

        extract_channels(wav_path, out_prefix, DRONE_MASTER, copied, skipped, errors)

# ── NASA ───────────────────────────────────────────────────────────────────

print(f"\n{'=' * 65}")
print("PROCESSING NASA UAV")
print("=" * 65)

if not NASA_DIR.exists():
    print(f"\n[SKIP] NASA dir not found: {NASA_DIR}")
else:
    nasa_folders = sorted([f for f in NASA_DIR.iterdir() if f.is_dir()])
    print(f"\nFound {len(nasa_folders)} NASA recording(s)")

    for folder in tqdm(nasa_folders, desc="  NASA", unit="folder"):

        wav_path = folder / "audio.wav"

        if not wav_path.exists():
            errors.append({"source": str(folder), "error": "audio.wav not found"})
            continue

        out_prefix = f"NASA_{folder.name}"

        extract_channels(wav_path, out_prefix, DRONE_MASTER, copied, skipped, errors)

# ── SUMMARY ────────────────────────────────────────────────────────────────

import csv

report_path = DRONE_MASTER / "drone_master_manifest.csv"
with open(report_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["file", "sample_rate", "duration_sec", "channels"])
    writer.writeheader()
    writer.writerows(copied)

print(f"\n{'=' * 65}")
print("DONE")
print("=" * 65)
print(f"  Files written          : {len(copied):,}")
print(f"  Already existed (skip) : {len(skipped):,}")
print(f"  Errors                 : {len(errors):,}")

if errors:
    print(f"\nErrors:")
    for e in errors[:10]:
        print(f"  {e['source']}: {e['error']}")
    if len(errors) > 10:
        print(f"  ... and {len(errors)-10} more")

print(f"\nManifest written to: {report_path}")
print("=" * 65)