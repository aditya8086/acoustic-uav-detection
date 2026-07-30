#!/usr/bin/env python3
"""
add_ambient_to_noise.py
-----------------------
Finds the 4 UAVirBase ambient recordings (no drone, recorded as background
captures) and adds them to NOISE_MASTER under Nature/Wind as individual
per-channel WAV files.

Each ambient folder contains:
    a.wav       — 8-channel, 96kHz, 32-bit PCM audio
    label.json  — metadata; drone.sound_source == "Ambient Noise" marks these

Output naming:
    Nature_Wind_UAVirBASE_{folder_name}_ch{n}.wav
    e.g. Nature_Wind_UAVirBASE_20241115_093128_ch1.wav

After running this script, run rebuild_statistics.py to update
noise_master_metadata.csv to include these 32 new files.

Usage:
    python add_ambient_to_noise.py
"""

import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

# ============================================================
# CONFIGURATION — verify these paths match your machine
# ============================================================

UAVIRBASE_DIR = Path(
    r"C:\Users\CARE\Downloads\Acoustic based drone detection\Audio files\Datasets\UaVirBASE"
)

NOISE_MASTER_DIR = Path(
    r"C:\Users\CARE\Downloads\Acoustic based drone detection"
    r"\Audio files\Noise\NOISE_MASTER"
)

# Output goes into Nature/Wind inside NOISE_MASTER
OUTPUT_DIR = NOISE_MASTER_DIR / "Nature" / "Wind"

# Output format — keep original 96kHz, save as PCM_16
# (resampling to 16kHz happens later in the synthesizer, not here)
OUTPUT_SUBTYPE = "PCM_16"

# ============================================================
# SCAN — find ambient folders
# ============================================================

print("=" * 60)
print("ADD UAVIRBASE AMBIENT RECORDINGS TO NOISE_MASTER")
print("=" * 60)
print(f"\nScanning: {UAVIRBASE_DIR}\n")

ambient_folders = []

for folder in sorted(UAVIRBASE_DIR.iterdir()):
    if not folder.is_dir():
        continue

    label_path = folder / "label.json"
    wav_path   = folder / "output.wav"

    if not label_path.exists() or not wav_path.exists():
        continue

    with open(label_path, "r", encoding="utf-8") as f:
        label = json.load(f)

    sound_source = label.get("drone", {}).get("sound_source", "")

    if sound_source == "Ambient Noise":
        ambient_folders.append(folder)

print(f"Found {len(ambient_folders)} ambient folder(s):")
for f in ambient_folders:
    print(f"  {f.name}")

if len(ambient_folders) == 0:
    print("\nNo ambient folders found. Check UAVIRBASE_DIR path.")
    raise SystemExit(1)

# ============================================================
# EXTRACT CHANNELS AND SAVE
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\nOutput directory: {OUTPUT_DIR}")
print(f"Output format   : {OUTPUT_SUBTYPE}\n")

copied   = []
skipped  = []
errors   = []

for folder in tqdm(ambient_folders, desc="Ambient folders", unit="folder"):

    wav_path = folder / "output.wav"

    try:
        audio, sample_rate = sf.read(str(wav_path), always_2d=True)
        # audio shape: (n_samples, n_channels)
        n_samples, n_channels = audio.shape

        print(f"\n  {folder.name}: {n_channels}ch, {sample_rate}Hz, "
              f"{n_samples/sample_rate:.1f}s")

        for ch_idx in range(n_channels):
            ch_num    = ch_idx + 1
            out_name  = f"Nature_Wind_UAVirBASE_{folder.name}_ch{ch_num}.wav"
            out_path  = OUTPUT_DIR / out_name

            if out_path.exists():
                skipped.append(out_name)
                continue

            channel_audio = audio[:, ch_idx]

            sf.write(
                str(out_path),
                channel_audio,
                sample_rate,
                subtype=OUTPUT_SUBTYPE,
            )

            copied.append({
                "source_folder" : folder.name,
                "channel"       : ch_num,
                "output_file"   : out_name,
                "sample_rate"   : sample_rate,
                "duration_sec"  : round(n_samples / sample_rate, 3),
            })

    except Exception as e:
        errors.append({
            "folder" : folder.name,
            "error"  : str(e),
        })
        print(f"\n  ERROR in {folder.name}: {e}")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
print(f"  Files written  : {len(copied)}")
print(f"  Already existed (skipped) : {len(skipped)}")
print(f"  Errors         : {len(errors)}")

if copied:
    print("\nFiles written:")
    for row in copied:
        print(f"  {row['output_file']}  ({row['duration_sec']}s)")

if errors:
    print("\nErrors:")
    for e in errors:
        print(f"  {e['folder']}: {e['error']}")

print(f"\nNext step: run rebuild_statistics.py to update noise_master_metadata.csv")
print("=" * 60)