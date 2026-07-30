#!/usr/bin/env python3
"""
mix_and_label.py   --   SCRIPT 2 of 2
=====================================
Mixes each drone recording with its paired synthetic noise scene at seven SNR
levels, producing the final labelled training dataset.

INPUT
-----
    SYNTHETIC_NOISE/synthetic_noise_manifest.csv   (12,450 scenes from Script 1)
    SYNTHETIC_NOISE/*.wav                          (the noise-only scenes)
    DRONE_MASTER/*.wav                             (the drone recordings)

OUTPUT
------
    MIXED_DATASET/
        train/  val/  test/
            {scene_stem}__label1_snr{X}dB.wav      7 per scene
            {scene_stem}__label0.wav               1 per scene
        mixed_dataset_manifest.csv

    12,450 scenes x 7 SNRs = 87,150 label=1
    12,450 scenes x 1      = 12,450 label=0
    ------------------------------------------
                             99,600 files


WHY THE NOISE IS SCALED AND THE DRONE IS NOT
--------------------------------------------
The drone recordings are the OUTPUT OF THE PROPAGATION MODEL. UAVirBASE_100m
already has the drone attenuated to what it physically sounds like at 100 m --
that calibration is the scientific contribution of the whole simulation
pipeline.

Scaling the drone to hit a target SNR would throw that away: a 100 m drone
scaled down to reach -20 dB would sound like a 400 m drone, not a 100 m drone
against loud background. The physical meaning is lost.

So the drone stays at its recorded level and the NOISE is scaled. Each mix then
means something concrete: "this is what a 100 m drone actually sounds like, with
the background 10 dB louder than it."

(The earlier preview_mix.py did the opposite -- it scaled the drone. That was
convenient for A/B listening because the background stayed at constant level
across the sweep, but it is wrong for training data.)


WHY SNR IS MEASURED IN-BAND
---------------------------
The detector only sees 120-1000 Hz. Everything below 120 Hz is removed by the
highpass and never reaches the model.

Wind and traffic recordings carry substantial sub-120 Hz energy. Measuring SNR
broadband would count that energy, making the noise look far louder than it
actually is inside the detection band -- so a clip labelled "-20 dB" would
really be much easier than -20 dB. Measured on synthetic signals, the gap
between broadband and in-band SNR reached 21 dB.

Both signals are therefore bandpass-filtered to 120-1000 Hz to COMPUTE the
ratio, but the resulting gain is applied to the FULL-BAND noise. Nothing
outside the band is destroyed prematurely; the 120 Hz highpass after mixing
handles that.


CLIPPING
--------
At low SNR the noise gain is large (at -20 dB the noise must be 20 dB above the
drone), so the sum can exceed full scale. When the peak exceeds 0.99 the ENTIRE
MIX is scaled down by a single factor. That preserves the drone:noise ratio
exactly -- only the absolute level changes, which is irrelevant because feature
extraction normalises anyway. The SNR label stays true.


CLASS BALANCE
-------------
7 positives per negative. This is deliberate:

  - The 12,450 negatives are all DISTINCT scenes (different beds, different
    source clips) -- not duplicates.
  - Each negative is the MATCHED background of its 7 positives, so the model
    sees "this exact background with a drone" vs "this exact background
    without" -- the cleanest possible contrast for learning the drone itself
    rather than the background.
  - Duplicating each negative 7x to force 50/50 would make the model see the
    same 40 s background 7 times per epoch: storage cost, overfitting risk,
    zero new information.

Handle the 7:1 ratio at training time with a weighted loss or a weighted
sampler (weight negatives ~7x). The manifest carries a `label` column to make
this straightforward.

Usage:
    python mix_and_label.py --dry-run
    python mix_and_label.py --limit 20      # first 20 scenes only
    python mix_and_label.py
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, sosfilt, resample_poly
from tqdm import tqdm

# =====================================================================
# CONFIGURATION
# =====================================================================

DATASET_DIR = Path(
    r"C:\Users\CARE\Downloads\Acoustic based drone detection"
    r"\Audio files\Dataset"
)

SYNTHETIC_NOISE = DATASET_DIR / "SYNTHETIC_NOISE"
DRONE_MASTER    = DATASET_DIR / "DRONE_MASTER"
OUTPUT_DIR      = DATASET_DIR / "MIXED_DATASET"

SCENE_MANIFEST  = SYNTHETIC_NOISE / "synthetic_noise_manifest.csv"
OUTPUT_MANIFEST = OUTPUT_DIR / "mixed_dataset_manifest.csv"

# ---- audio ----------------------------------------------------------
TARGET_SR      = 8000
OUTPUT_SUBTYPE = "PCM_16"

# ---- detection band -------------------------------------------------
BAND_LOW_HZ  = 120.0    # highpass cutoff, applied AFTER mixing
BAND_HIGH_HZ = 1000.0   # upper band edge, used for the SNR measurement
FILTER_ORDER = 4

# ---- SNR sweep ------------------------------------------------------
# -20 dB is the hard target agreed with the professor -- the drone should still
# be detectable there. +10 dB is the easy end.
SNR_LEVELS_DB = [-20, -15, -10, -5, 0, 5, 10]

PEAK_CEILING = 0.99


# =====================================================================
# DSP
# =====================================================================

def load_mono_8k(path: Path) -> np.ndarray:
    """Load a WAV, force mono, resample to TARGET_SR, return float32."""
    audio, sr = sf.read(str(path), always_2d=True, dtype="float64")
    audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        g = np.gcd(int(sr), TARGET_SR)
        audio = resample_poly(audio, TARGET_SR // g, sr // g)
    return audio.astype(np.float32)


def bandpass(x: np.ndarray) -> np.ndarray:
    """Bandpass to the detection band. Used ONLY to measure energy."""
    nyq  = TARGET_SR / 2.0
    high = min(BAND_HIGH_HZ, nyq * 0.99)
    sos  = butter(FILTER_ORDER, [BAND_LOW_HZ / nyq, high / nyq],
                  btype="bandpass", output="sos")
    return sosfilt(sos, x).astype(np.float32)


def highpass(x: np.ndarray) -> np.ndarray:
    """The real 120 Hz highpass, applied to the MIXED signal after mixing."""
    nyq = TARGET_SR / 2.0
    sos = butter(FILTER_ORDER, BAND_LOW_HZ / nyq,
                 btype="highpass", output="sos")
    return sosfilt(sos, x).astype(np.float32)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))


def mix_at_snr(drone: np.ndarray, noise: np.ndarray, snr_db: float):
    """
    Keep the drone at its recorded level; scale the NOISE to hit snr_db.

    SNR is measured in-band (120-1000 Hz); the gain is applied to the
    full-band noise.

        want   20*log10( d_rms / (g * n_rms) ) = snr_db
        so     g = d_rms / (n_rms * 10^(snr_db/20))

    Returns (mixed, noise_gain).
    """
    d_rms = rms(bandpass(drone))
    n_rms = rms(bandpass(noise))

    if n_rms < 1e-9:
        return drone.copy(), 0.0

    gain  = d_rms / (n_rms * (10.0 ** (snr_db / 20.0)))
    mixed = drone + noise * gain
    return mixed, gain


def peak_guard(x: np.ndarray):
    """
    Scale the WHOLE signal down if it would clip.

    Applied to the mix as a single factor, so the drone:noise ratio -- and
    therefore the SNR label -- is preserved exactly.

    Returns (x, was_scaled).
    """
    peak = float(np.max(np.abs(x)))
    if peak > PEAK_CEILING:
        return x * (PEAK_CEILING / peak), True
    return x, False


def snr_tag(snr_db: int) -> str:
    """-20 -> 'm20', +10 -> 'p10'  (Windows-safe filename token)."""
    return f"{'m' if snr_db < 0 else 'p'}{abs(snr_db):02d}"


# =====================================================================
# MAIN
# =====================================================================

def main(dry_run: bool, limit: int):

    print("=" * 74)
    print("MIX AND LABEL  --  Script 2 of 2")
    if dry_run:
        print("*** DRY RUN -- planning only, no audio written ***")
    print("=" * 74)

    # ---- checks --------------------------------------------------------
    for p in (SCENE_MANIFEST, SYNTHETIC_NOISE, DRONE_MASTER):
        if not p.exists():
            print(f"\n[ERROR] Not found: {p}")
            return

    scenes = pd.read_csv(SCENE_MANIFEST)
    print(f"\nScenes in manifest : {len(scenes):,}")
    print(f"Drone source       : {DRONE_MASTER}")
    print(f"Output             : {OUTPUT_DIR}")

    print(f"\nSNR levels  : {', '.join(f'{s:+d}' for s in SNR_LEVELS_DB)} dB")
    print(f"Band        : {BAND_LOW_HZ:.0f}-{BAND_HIGH_HZ:.0f} Hz "
          f"(SNR measured in-band)")
    print(f"Highpass    : {BAND_LOW_HZ:.0f} Hz, applied AFTER mixing")
    print(f"Scaling     : NOISE scaled, drone kept at recorded level")

    if limit:
        scenes = scenes.head(limit)
        print(f"\n[LIMIT] first {limit} scenes only")

    n_pos = len(scenes) * len(SNR_LEVELS_DB)
    n_neg = len(scenes)
    print(f"\nWill produce:")
    print(f"  label=1 : {n_pos:,}  ({len(scenes):,} scenes x "
          f"{len(SNR_LEVELS_DB)} SNRs)")
    print(f"  label=0 : {n_neg:,}  (1 per scene)")
    print(f"  TOTAL   : {n_pos + n_neg:,} files")
    print(f"  ratio   : {len(SNR_LEVELS_DB)}:1 positive:negative "
          f"-> use weighted loss at training time")

    print("\n  By split:")
    for sp, g in scenes.groupby("split"):
        print(f"    {sp:<6}: {len(g):>6,} scenes -> "
              f"{len(g)*len(SNR_LEVELS_DB):>7,} pos + {len(g):>6,} neg")

    if dry_run:
        print("\n" + "=" * 74)
        print("DRY RUN COMPLETE -- nothing written.")
        print("=" * 74)
        return

    # ---- build ---------------------------------------------------------
    for sp in ("train", "val", "test"):
        (OUTPUT_DIR / sp).mkdir(parents=True, exist_ok=True)

    rows, errors = [], []
    n_clipped = 0
    snr_used  = Counter()

    for _, sc in tqdm(scenes.iterrows(), total=len(scenes),
                      desc="Scenes", unit="scene"):

        scene_path = SYNTHETIC_NOISE / sc.scene_file
        drone_path = DRONE_MASTER / sc.drone_file
        split_dir  = OUTPUT_DIR / sc.split
        stem       = Path(sc.scene_file).stem

        try:
            if not scene_path.exists():
                raise FileNotFoundError(f"scene missing: {sc.scene_file}")
            if not drone_path.exists():
                raise FileNotFoundError(f"drone missing: {sc.drone_file}")

            noise = load_mono_8k(scene_path)
            drone = load_mono_8k(drone_path)

            # Script 1 truncated every scene to the drone's exact sample count.
            # Verify rather than assume -- silently trimming would hide an
            # upstream change.
            if len(drone) != len(noise):
                k = min(len(drone), len(noise))
                errors.append({
                    "scene": sc.scene_file,
                    "error": f"length mismatch drone={len(drone)} "
                             f"noise={len(noise)} -> trimmed to {k}",
                })
                drone, noise = drone[:k], noise[:k]

            duration = round(len(noise) / TARGET_SR, 3)

            # ---- shared metadata ----------------------------------------
            base = {
                "scene_file"  : sc.scene_file,
                "drone_file"  : sc.drone_file,
                "split"       : sc.split,
                "mode"        : sc["mode"],
                "source"      : sc.source,
                "session"     : sc.session,
                "distance"    : sc.distance,
                "channel"     : sc.channel,
                "bed_class"   : sc.bed_class,
                "duration_sec": duration,
                "sample_rate" : TARGET_SR,
            }

            # ---- label = 0 : noise only, highpassed ---------------------
            neg, _   = peak_guard(highpass(noise))
            neg_name = f"{stem}__label0.wav"
            sf.write(str(split_dir / neg_name), neg,
                     TARGET_SR, subtype=OUTPUT_SUBTYPE)

            rows.append({**base,
                         "output_file": neg_name,
                         "label"      : 0,
                         "snr_db"     : "",
                         "noise_gain" : "",
                         "clipped"    : "no"})

            # ---- label = 1 : one file per SNR ---------------------------
            for snr in SNR_LEVELS_DB:
                mixed, gain  = mix_at_snr(drone, noise, snr)
                mixed        = highpass(mixed)
                mixed, clip  = peak_guard(mixed)

                name = f"{stem}__label1_snr{snr_tag(snr)}dB.wav"
                sf.write(str(split_dir / name), mixed,
                         TARGET_SR, subtype=OUTPUT_SUBTYPE)

                if clip:
                    n_clipped += 1
                snr_used[snr] += 1

                rows.append({**base,
                             "output_file": name,
                             "label"      : 1,
                             "snr_db"     : snr,
                             "noise_gain" : round(float(gain), 6),
                             "clipped"    : "yes" if clip else "no"})

        except Exception as e:
            errors.append({"scene": sc.scene_file, "error": str(e)})

    # ---- manifest ------------------------------------------------------
    if rows:
        with open(OUTPUT_MANIFEST, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # ---- summary -------------------------------------------------------
    print("\n" + "=" * 74)
    print("DONE")
    print("=" * 74)

    df   = pd.DataFrame(rows) if rows else pd.DataFrame()
    npos = int((df.label == 1).sum()) if len(df) else 0
    nneg = int((df.label == 0).sum()) if len(df) else 0

    print(f"  Files written : {len(rows):,}")
    print(f"    label=1     : {npos:,}")
    print(f"    label=0     : {nneg:,}")
    print(f"  Errors        : {len(errors):,}")

    if len(df):
        print("\n  By split:")
        for sp, g in df.groupby("split"):
            p = int((g.label == 1).sum())
            n = int((g.label == 0).sum())
            print(f"    {sp:<6}: {p:>7,} pos | {n:>6,} neg | {len(g):>7,} total")

        print("\n  By SNR (label=1):")
        for snr in SNR_LEVELS_DB:
            print(f"    {snr:>+4d} dB : {snr_used[snr]:>7,}")

        pos = df[df.label == 1]
        print("\n  Noise gain applied (median by SNR):")
        for snr in SNR_LEVELS_DB:
            g = pos[pos.snr_db == snr].noise_gain
            if len(g):
                print(f"    {snr:>+4d} dB : {g.median():>10.4f}x")

        print(f"\n  Peak-limited  : {n_clipped:,} of {npos:,} "
              f"({100*n_clipped/max(npos,1):.1f}%)")
        print("    (whole mix scaled by one factor -- SNR ratio preserved)")

        print("\n  Class balance:")
        print(f"    {npos/max(nneg,1):.1f} : 1  positive : negative")
        print(f"    -> weight negatives ~{npos/max(nneg,1):.0f}x in the loss")

        print("\n  Breakdown available in the manifest for:")
        print("    distance | bed_class | mode (static/flyby) | snr_db | split")

    if errors:
        print(f"\n  First errors:")
        for e in errors[:8]:
            print(f"    {e['scene']}: {e['error']}")

    print(f"\n  Manifest : {OUTPUT_MANIFEST}")
    print(f"  Output   : {OUTPUT_DIR}")
    print("=" * 74)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Mix drone + synthetic noise at multiple SNRs "
                    "to build the labelled dataset"
    )
    ap.add_argument("--dry-run", action="store_true",
                    help="Plan only; write no audio")
    ap.add_argument("--limit", type=int, default=0,
                    help="Only process the first N scenes (for testing)")
    a = ap.parse_args()
    main(a.dry_run, a.limit)