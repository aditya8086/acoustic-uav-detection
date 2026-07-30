#!/usr/bin/env python3
"""
build_synthetic_noise.py   --   SCRIPT 1 of 2
=============================================
Builds synthetic acoustic SCENES (noise only, NO drone) from NOISE_MASTER.
Each scene is later mixed with a drone (Script 2) to make label=1, and used
as-is for label=0.

WHAT CHANGED FROM THE FIRST VERSION (important)
-----------------------------------------------
1. GLOBAL WEIGHTED BEDS (not one-per-superclass).
   Previously every drone got 5 scenes, one bed per superclass. Because
   Human and Hard_Negatives each have only ONE bed class (Crowd, Engine),
   those two ended up as 40% of all beds -- while Wind, the dominant real
   deployment condition, was stuck at ~7%. Beds are now drawn from ONE global
   weighted list so the distribution reflects deployment reality (Wind 22%,
   Rain 15%, ... Engine 8%, Crowd 5%), regardless of taxonomy.

2. STATIC vs FLYBY handling.
   UAVirBASE = static drones at 17 simulated distances.
   NASA      = real recordings, filename tags flight mode:
                 "flyover" -> FLYBY  (drone moving past the array)
                 "hover"   -> STATIC (drone hovering)
   Flyby is a distinct, harder deployment condition and was previously only
   2.8% of data. To boost it WITHOUT inflating storage:
       - UAVirBASE          : 1 random channel, 5 scenes   (static)
       - NASA hover         : 1 random channel, 5 scenes   (static, == UAVirBASE)
       - NASA flyover       : ALL 4 channels,  10 scenes   (flyby, oversampled)
   NASA flyover channels are DECORRELATED (drone moves during recording, so
   each mic sees a different Doppler/amplitude trajectory), so using all 4 adds
   real variety rather than duplicates. Result: flyby ~11.6% of scenes.
   Every scene -- static or flyby -- draws from the SAME global bed/event
   distribution, so flyby drones see the full variety of backgrounds.

3. GLOBAL WEIGHTED EVENT POOL (unchanged from the interim fix): any event
   class can drop into any scene; drone-confusable hard negatives up-weighted.

TWO-LAYER SCENE STRUCTURE
-------------------------
Layer 1 -- BED (stationary, continuous):
    One bed class from the GLOBAL weighted list. Stitch SEVERAL different
    clips of that class with 500 ms equal-power crossfades until the drone's
    duration is filled. Different clips (not one clip looped) -> no periodicity.

Layer 2 -- EVENTS (sparse, transient):
    2-5 short clips from the GLOBAL weighted event pool, dropped at random
    positions ON TOP of the bed, mixed -6 to +12 dB relative to the bed so
    they punch through. Hard-clipped at the scene boundary.

LEAKAGE GUARD
-------------
Split at the SESSION level, BEFORE building. All variants/channels of a
session go into ONE split. Noise files partitioned disjointly too.

AUDIO
-----
8 kHz mono PCM_16. Band of interest is 120-1000 Hz; 8 kHz (Nyquist 4 kHz) is
deliberate headroom in case the band later widens. NO highpass here -- the
120 Hz highpass belongs in Script 2, AFTER mixing.

OUTPUT
------
    SYNTHETIC_NOISE/
        {drone_stem}__bed-{BedClass}__ev-{Ev1}-{Ev2}.wav
        synthetic_noise_manifest.csv

Usage:
    python build_synthetic_noise.py --dry-run
    python build_synthetic_noise.py --limit 20
    python build_synthetic_noise.py
"""

import argparse
import csv
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
from tqdm import tqdm

# =====================================================================
# CONFIGURATION
# =====================================================================

DATASET_DIR = Path(
    r"C:\Users\CARE\Downloads\Acoustic based drone detection"
    r"\Audio files\Dataset"
)

NOISE_MASTER   = DATASET_DIR / "NOISE_MASTER"
DRONE_MASTER   = DATASET_DIR / "DRONE_MASTER"
OUTPUT_DIR     = DATASET_DIR / "SYNTHETIC_NOISE"

NOISE_METADATA   = NOISE_MASTER / "noise_master_metadata.csv"
DRONE_MANIFEST   = DRONE_MASTER / "drone_master_manifest.csv"
OUTPUT_MANIFEST  = OUTPUT_DIR   / "synthetic_noise_manifest.csv"

# ---- audio ----------------------------------------------------------
TARGET_SR       = 8000      # Hz. Band 120-1000 Hz; Nyquist 4 kHz = headroom.
OUTPUT_SUBTYPE  = "PCM_16"
CROSSFADE_SEC   = 0.5       # 500 ms equal-power crossfade between bed clips

# ---- bed layer ------------------------------------------------------
BED_MIN_DURATION = 8.0      # seconds (beds loop; short clips sound periodic)

# ---- STATIONARITY GATE ----------------------------------------------
# A bed must have roughly CONSTANT energy over time. If it is bursty, the local
# SNR swings and the clip's single SNR label becomes a fiction.
#
# Measured by framing the clip, computing in-band (120-1000 Hz) RMS per frame,
# and taking the spread between the 95th and 5th percentile in dB.
#
# Calibration on known signals:
#     steady wind ................  0.8 dB   -> excellent bed
#     bird chirps OVER wind ......  3.3 dB   -> fine, the wind fills the gaps
#     bare bird chirps ........... 34.8 dB   -> rejected, silence between chirps
#
# This is why Bird stays as a bed class: a bare bird recording fails, but a
# bird recording with a continuous wind/ambience floor passes. The gate decides
# per FILE, not per class.
# STATIONARITY THRESHOLDS (per-class for clips, global for scenes)
#
# Wind's energy naturally varies with gusts and lulls -- that is physics,
# not bursty content. A global 12 dB clip threshold rejected ~73% of all
# Wind clips and starved the Wind bed pool, causing Rain/Stream to over-
# represent in the full run (Wind 18.3% vs target 26.0%).
#
# Fix: Wind clips get a 16 dB clip-level threshold (accepts natural gusts).
# The scene-level threshold stays at 12 dB for ALL classes -- that is the
# strict final gate your professor asked for, and it applies uniformly.
#
# Dropped classes are excluded by class list, not by threshold, so raising
# Wind's threshold does not accidentally re-admit Bird/Cricket/Livestock.
# Insect (median 15.7 dB) is also explicitly excluded via the class list.
STATIONARITY_CLIP_THRESHOLD_DB = {
    "Wind": 16.0,    # natural gust variability -- stricter scene gate still applies
    "default": 12.0, # all other classes
}
STATIONARITY_SCENE_THRESHOLD_DB   = 12.0  # final gate, ALL classes, strict
STATIONARITY_FRAME_SEC            = 0.25
BAND_LOW_HZ                       = 120.0
BAND_HIGH_HZ                      = 1000.0

# ---- event layer ----------------------------------------------------
# EVENTS DISABLED.
# Events were dropped on the professor's instruction. A loud transient (a
# chainsaw at +12 dB over the bed) makes the LOCAL SNR swing wildly, so a clip
# labelled "-10 dB" is really -25 dB during the event and +5 dB after it. The
# single SNR figure stops describing the clip, and every SNR-vs-recall curve
# built on it becomes blurred. Set ENABLE_EVENTS = True to restore them.
ENABLE_EVENTS = False
EVENTS_MIN = 0
EVENTS_MAX = 0
EVENT_MAX_DURATION = 10.0   # events are transient; no lower bound
EVENT_GAIN_DB = (-6.0, +12.0)   # relative to bed: events punch through

# ---- scenes per signal ---------------------------------------------
SCENES_STATIC = 5           # UAVirBASE + NASA hover
SCENES_FLYBY  = 10          # NASA flyover (oversampled)

# ---- splits ---------------------------------------------------------
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}

SEED = 1337

# =====================================================================
# GLOBAL BED WEIGHTS
#
# Share of ALL beds, set by DEPLOYMENT REALITY, not taxonomy. A border/field
# drone detector mostly fights wind, then rain, with crickets/insects at
# night. Engine and Crowd are useful but must NOT dominate (they did before,
# at 20% each, purely because they were the sole bed in their superclass).
#
# These are relative weights; they are normalised to a probability internally,
# so they need not sum to exactly 100 -- but they are written to.
# =====================================================================

GLOBAL_BED_WEIGHTS = {
    # --- Primary: dominant real deployment conditions (39%) -----------
    "Wind":           26,
    "Rain":           20,
    # --- Common ambient (26%) -----------------------------------------
    "Traffic":        12,
    "Engine":         12,   # steady motor -- machine hard negative
    # --- Secondary plausible environments (35%) -----------------------
    "Ocean":           8,
    "Stream":          8,
    "Crowd":           5,
    "RailTransport":   5,
    "Boat":            4,
}

# CLASSES DROPPED FROM BED (data-driven, not a judgment):
#   Bird      -- median scene DR 16.8 dB, 83% of scenes fail the gate.
#                Individual bird clips are bursty (chirp-gap-chirp).
#   Cricket   -- median 19.4 dB, 57% fail. Same reason.
#   Insect    -- median 15.7 dB, 75% fail. Same reason.
#   Livestock -- median 21.9 dB, 67% fail. Individual moos with silence.
# All four were tested in the limit-20 run and their scene-level dynamic
# ranges show they cannot reliably produce stationary beds. The stationarity
# gate rejects most of their clips individually; the few that pass stitch into
# scenes that still fail the scene-level check. Removing them is cleaner than
# letting the script waste attempts rebuilding them.
#
# Engine stays: its WATCH flag comes from a few individual clip outliers (e.g.
# a single "engine starting" transient) rather than the class being inherently
# bursty. The scene-level gate handles those.
#
# AirConditioner, Drill, Jackhammer, VacuumCleaner: zero clips >=8s.
# Cannot form a bed regardless.

GLOBAL_BED_CLASSES = list(GLOBAL_BED_WEIGHTS.keys())

# =====================================================================
# GLOBAL EVENT WEIGHTS
#
# Events are transient clips dropped on top of the bed, drawn from a GLOBAL
# pool: any class can appear in any scene. This is realistic and it kills the
# Siren monoculture that arose when events were restricted per-superclass.
#
# HIGH (3.0): drone-confusable hard negatives -- buzzy broadband machines that
#             teach the model "this is NOT a drone". We want these often.
# MED  (1.5): everyday transients (animals, speech).
# LOW  (1.0): default for everything else.
# =====================================================================

EVENT_WEIGHTS = {
    "Chainsaw": 3.0, "Drill": 3.0, "Aircraft": 3.0, "Engine": 3.0,
    "Jackhammer": 3.0, "Saw": 3.0, "Siren": 3.0, "Gunshot": 3.0,
    "Dog": 1.5, "Cat": 1.5, "WildAnimal": 1.5, "DomesticAnimal": 1.5,
    "Speech": 1.5, "Conversation": 1.5, "Frog": 1.5,
}
EVENT_EXCLUDE = {"Water"}   # dropped from the corpus usage entirely

# =====================================================================
# HELPERS
# =====================================================================

def weighted_choice(options, weights_map, rng):
    """Pick one option using weights_map (missing keys default to 1.0)."""
    weights = [weights_map.get(o, 1.0) for o in options]
    return rng.choices(options, weights=weights, k=1)[0]


def load_mono_8k(path: Path) -> np.ndarray:
    """Load any WAV, force mono, resample to TARGET_SR, return float32."""
    audio, sr = sf.read(str(path), always_2d=True, dtype="float64")
    audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        g = np.gcd(int(sr), TARGET_SR)
        audio = resample_poly(audio, TARGET_SR // g, sr // g)
    return audio.astype(np.float32)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))


def _bandpass_for_measure(x):
    """Bandpass to the detection band. Used ONLY to measure energy, never to
    alter audio that gets written out."""
    from scipy.signal import butter, sosfilt
    nyq  = TARGET_SR / 2.0
    high = min(BAND_HIGH_HZ, nyq * 0.99)
    sos  = butter(4, [BAND_LOW_HZ / nyq, high / nyq],
                  btype="bandpass", output="sos")
    return sosfilt(sos, x)


def normalise_rms(x: np.ndarray, target_rms: float = 0.05) -> np.ndarray:
    """
    Loudness-normalise a clip so the bed doesn't lurch in level at joins.

    CRITICAL: normalises on IN-BAND (120-1000 Hz) RMS, not broadband.

    Normalising on broadband RMS was a real defect. Wind-farm recordings carry
    heavy sub-120 Hz turbine rumble, and the AMOUNT of rumble varies clip to
    clip. Two clips normalised to identical BROADBAND level can end up ~15 dB
    apart IN-BAND -- measured directly: a heavy-rumble clip landed at in-band
    RMS 0.0036 while a light-rumble clip landed at 0.0193 after both were
    normalised to broadband 0.05.

    Stitch those two and the scene has a 15 dB step in exactly the band the
    stationarity gate measures, so the scene gets rejected -- even though both
    clips passed the clip-level gate individually. That is what starved the
    Wind bed pool (Wind 14.2% against a 26% target) despite 6,000 wind-farm
    clips being available.

    Matching in-band level makes the stitch seamless in the band that matters.
    The scale factor is applied to the FULL-BAND signal, so out-of-band content
    is preserved -- the 120 Hz highpass in Script 2 removes it later.
    """
    r = rms(_bandpass_for_measure(x))
    if r < 1e-9:
        return x
    return x * (target_rms / r)


def dynamic_range_db(x):
    """
    How much a clip's in-band energy swings over time, in dB.

    Frames the signal, takes per-frame RMS in the 120-1000 Hz band, and returns
    the 95th-percentile minus 5th-percentile level. A steady texture returns a
    small number; a bursty one (chirps with silence between) returns a large one.

    Returns None if the clip is too short to frame.
    """
    xb = _bandpass_for_measure(x)
    n  = int(STATIONARITY_FRAME_SEC * TARGET_SR)
    nf = len(xb) // n
    if nf < 4:
        return None
    frames = xb[: nf * n].reshape(nf, n)
    e      = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    edb    = 20.0 * np.log10(e + 1e-12)
    return float(np.percentile(edb, 95) - np.percentile(edb, 5))


def clip_threshold(bed_class):
    """Return the clip-level stationarity threshold for a given bed class."""
    return STATIONARITY_CLIP_THRESHOLD_DB.get(
        bed_class, STATIONARITY_CLIP_THRESHOLD_DB["default"])


def is_stationary_clip(x, bed_class):
    """True if an individual clip is steady enough to use in a bed."""
    dr = dynamic_range_db(x)
    if dr is None:
        return False
    return dr <= clip_threshold(bed_class)


def is_stationary_scene(x):
    """True if the FINISHED stitched scene meets the strict scene threshold."""
    dr = dynamic_range_db(x)
    if dr is None:
        return True   # too short to measure -- accept
    return dr <= STATIONARITY_SCENE_THRESHOLD_DB


def equal_power_crossfade(a, b, n_fade):
    """Concatenate a and b with an equal-power (sqrt) crossfade."""
    n_fade = int(min(n_fade, len(a), len(b)))
    if n_fade <= 0:
        return np.concatenate([a, b])
    t        = np.linspace(0.0, 1.0, n_fade, dtype=np.float32)
    fade_out = np.sqrt(1.0 - t)
    fade_in  = np.sqrt(t)
    head    = a[:-n_fade]
    overlap = a[-n_fade:] * fade_out + b[:n_fade] * fade_in
    tail    = b[n_fade:]
    return np.concatenate([head, overlap, tail])


# =====================================================================
# LAYER 1 -- BED
# =====================================================================

def build_bed(bed_class, n_samples, pool, rng):
    """
    Stitch several DIFFERENT clips of bed_class into a continuous bed of
    exactly n_samples, using 500 ms equal-power crossfades. Hard-truncated to
    n_samples so no noise-only overhang leaks into the drone-positive class.
    """
    candidates = pool[pool.target_class == bed_class]
    if len(candidates) == 0:
        raise RuntimeError(f"No bed clips for '{bed_class}'")

    n_fade   = int(CROSSFADE_SEC * TARGET_SR)
    bed      = np.zeros(0, dtype=np.float32)
    sources  = []
    rejected = []
    used     = set()
    attempts = 0

    while len(bed) < n_samples + n_fade and attempts < 400:
        attempts += 1
        unused = candidates[~candidates.abs_path.isin(used)]
        picks  = unused if len(unused) > 0 else candidates
        row    = picks.iloc[rng.randrange(len(picks))]

        try:
            clip = load_mono_8k(Path(row.abs_path))
        except Exception:
            continue
        if len(clip) < n_fade * 2:
            continue

        # CLIP-LEVEL STATIONARITY GATE.
        # Wind uses a 16 dB threshold (natural gust variability);
        # all other classes use 12 dB.
        if not is_stationary_clip(clip, bed_class):
            rejected.append(row.filename)
            continue

        clip = normalise_rms(clip)
        bed  = clip.copy() if len(bed) == 0 else equal_power_crossfade(bed, clip, n_fade)
        used.add(row.abs_path)
        sources.append(row.filename)

    if len(bed) < n_samples:
        raise RuntimeError(
            f"Could not fill bed for '{bed_class}' "
            f"({len(rejected)} clips rejected by stationarity gate)")

    return bed[:n_samples], sources, rejected


# =====================================================================
# LAYER 2 -- EVENTS  (global weighted pool, no >=8s floor)
# =====================================================================

def add_events(bed, bed_class, pool, rng):
    """
    Events are DISABLED (ENABLE_EVENTS = False).

    Returns the bed unchanged. Kept in place so events can be restored later by
    flipping ENABLE_EVENTS to True without restructuring the script.

    Why disabled: a transient dropped at up to +12 dB over the bed makes the
    LOCAL SNR swing violently. A clip labelled "-10 dB" is really -25 dB while
    a chainsaw is sounding and +5 dB once it stops, so the single SNR number
    stops describing the clip. It also does not reflect deployment -- a real
    border has wind and occasional rain, not a chainsaw, a jackhammer and a
    siren inside 40 seconds.
    """
    if not ENABLE_EVENTS:
        return bed.copy(), [], []

    n_samples = len(bed)
    scene     = bed.copy()

    candidates = pool[
        (pool.target_class != bed_class)
        & (~pool.target_class.isin(EVENT_EXCLUDE))
        & (pool.duration_sec <= EVENT_MAX_DURATION)
    ]
    if len(candidates) == 0:
        return scene, [], []

    ev_class_list = sorted(candidates.target_class.unique())
    n_events      = rng.randint(EVENTS_MIN, EVENTS_MAX)
    bed_rms       = rms(bed)

    ev_classes, ev_details = [], []

    for _ in range(n_events):
        cls = weighted_choice(ev_class_list, EVENT_WEIGHTS, rng)
        sub = candidates[candidates.target_class == cls]
        row = sub.iloc[rng.randrange(len(sub))]

        try:
            ev = load_mono_8k(Path(row.abs_path))
        except Exception:
            continue
        if len(ev) < TARGET_SR * 0.2:
            continue

        ev      = normalise_rms(ev)
        gain_db = rng.uniform(*EVENT_GAIN_DB)
        ev      = ev * (bed_rms / (rms(ev) + 1e-12)) * (10.0 ** (gain_db / 20.0))

        if n_samples <= len(ev):
            start_i, seg = 0, ev[:n_samples]
        else:
            start_i = rng.randrange(0, n_samples - len(ev))
            seg     = ev

        end_i = min(start_i + len(seg), n_samples)
        scene[start_i:end_i] += seg[: end_i - start_i]

        ev_classes.append(cls)
        ev_details.append({
            "class": cls, "file": row.filename,
            "onset_sec": round(start_i / TARGET_SR, 3),
            "duration_sec": round(len(seg) / TARGET_SR, 3),
            "gain_db": round(gain_db, 1),
        })

    return scene, ev_classes, ev_details

# =====================================================================
# DRONE UNIT SELECTION  (static vs flyby)
# =====================================================================

def build_drone_units(drone_df, rng):
    """
    Build the list of drone signals to synthesise scenes for.

    UAVirBASE           : 1 random channel per (session x distance), static, 5 scenes
    NASA hover  (static): 1 random channel per session,              static, 5 scenes
    NASA flyover (flyby): ALL 4 channels per session,                flyby, 10 scenes

    NASA flight mode comes straight from the filename:
        "flyover" -> flyby ,  "hover" -> static
    """
    units = []

    # ---- UAVirBASE : 1 channel per (session x distance) --------------
    uav = drone_df[drone_df.file.str.startswith("UAVirBASE")].copy()
    uav["variant"] = uav.file.str.extract(r"^UAVirBASE_([^_]+)_")
    uav["session"] = uav.file.str.extract(r"^UAVirBASE_[^_]+_(\d{8}_\d{6})")
    uav["ch"]      = uav.file.str.extract(r"_ch(\d+)\.wav$").astype(int)

    for (session, variant), g in uav.groupby(["session", "variant"]):
        row = g.iloc[rng.randrange(len(g))]
        units.append({
            "drone_file": row.file, "drone_stem": Path(row.file).stem,
            "source": "UAVirBASE", "mode": "static",
            "session": f"UAV_{session}", "distance": variant,
            "channel": int(row.ch),
            "duration_sec": float(row.duration_sec),
            "n_scenes": SCENES_STATIC,
        })

    # ---- NASA : split flyover vs hover by filename -------------------
    nasa = drone_df[drone_df.file.str.startswith("NASA")].copy()
    nasa["session"] = (nasa.file
                       .str.replace(r"^NASA_", "", regex=True)
                       .str.replace(r"_ch\d+\.wav$", "", regex=True))
    nasa["ch"]   = nasa.file.str.extract(r"_ch(\d+)\.wav$").astype(int)
    nasa["mode"] = np.where(nasa.file.str.contains("flyover"), "flyby", "static")

    # NASA hover (static): 1 random channel, 5 scenes -- treated like UAVirBASE
    hover = nasa[nasa["mode"] == "static"]
    for session, g in hover.groupby("session"):
        row = g.iloc[rng.randrange(len(g))]
        units.append({
            "drone_file": row.file, "drone_stem": Path(row.file).stem,
            "source": "NASA", "mode": "static",
            "session": f"NASA_{session}", "distance": "hover",
            "channel": int(row.ch),
            "duration_sec": float(row.duration_sec),
            "n_scenes": SCENES_STATIC,
        })

    # NASA flyover (flyby): ALL channels, 10 scenes each
    flyby = nasa[nasa["mode"] == "flyby"]
    for session, g in flyby.groupby("session"):
        for _, row in g.iterrows():
            units.append({
                "drone_file": row.file, "drone_stem": Path(row.file).stem,
                "source": "NASA", "mode": "flyby",
                "session": f"NASA_{session}", "distance": "flyby",
                "channel": int(row.ch),
                "duration_sec": float(row.duration_sec),
                "n_scenes": SCENES_FLYBY,
            })

    return pd.DataFrame(units)


# =====================================================================
# LEAKAGE GUARD -- session-level split
# =====================================================================

def split_sessions(units, rng):
    """
    Split sessions 70/15/15, STRATIFIED BY MODE.

    Flyby sessions and static sessions are split SEPARATELY, then combined.
    Without this, the random split could hand flyby an unlucky ratio (e.g. only
    3 of 36 flyby sessions to val). Stratifying guarantees flyby lands ~70/15/15
    in its own right, so every split has proportional flyby representation --
    important because flyby is the harder, rarer deployment condition and we
    need it present in val (for early stopping) and test (for reporting).

    Still a SESSION-level split: all channels/distances of a session stay
    together, so the leakage guard holds.
    """
    mapping = {}

    for mode in ("static", "flyby"):
        sess = sorted(units[units["mode"] == mode].session.unique())
        rng.shuffle(sess)
        n       = len(sess)
        n_train = int(n * SPLIT_FRACTIONS["train"])
        n_val   = int(n * SPLIT_FRACTIONS["val"])
        for i, sname in enumerate(sess):
            mapping[sname] = ("train" if i < n_train
                              else "val" if i < n_train + n_val
                              else "test")

    units = units.copy()
    units["split"] = units.session.map(mapping)
    return units, mapping


def split_noise_pool(noise, rng):
    noise = noise.copy()
    noise["split"] = ""
    for cls, g in noise.groupby("target_class"):
        idx = list(g.index)
        rng.shuffle(idx)
        n       = len(idx)
        n_train = int(n * SPLIT_FRACTIONS["train"])
        n_val   = int(n * SPLIT_FRACTIONS["val"])
        for i, j in enumerate(idx):
            noise.at[j, "split"] = ("train" if i < n_train
                                    else "val" if i < n_train + n_val
                                    else "test")
    return noise


# =====================================================================
# MAIN
# =====================================================================

def main(dry_run, limit):
    rng = random.Random(SEED)
    np.random.seed(SEED)

    print("=" * 74)
    print("BUILD SYNTHETIC NOISE  --  Script 1 of 2")
    if dry_run:
        print("*** DRY RUN -- planning only, no audio written ***")
    print("=" * 74)

    print("\nLoading metadata...")
    for p in (NOISE_METADATA, DRONE_MANIFEST):
        if not p.exists():
            print(f"[ERROR] Not found: {p}")
            return
    noise = pd.read_csv(NOISE_METADATA)
    drone = pd.read_csv(DRONE_MANIFEST)
    print(f"  NOISE_MASTER : {len(noise):,} files")
    print(f"  DRONE_MASTER : {len(drone):,} files")

    # ---- drone units ---------------------------------------------------
    print("\nSelecting drone signals (static vs flyby)...")
    units = build_drone_units(drone, rng)

    static = units[units["mode"] == "static"]
    flyby  = units[units["mode"] == "flyby"]
    n_static_scenes = int(static.n_scenes.sum())
    n_flyby_scenes  = int(flyby.n_scenes.sum())
    total_scenes    = n_static_scenes + n_flyby_scenes

    print(f"  Static signals: {len(static):,}  -> {n_static_scenes:,} scenes")
    print(f"    UAVirBASE    : {(static.source=='UAVirBASE').sum():,}")
    print(f"    NASA hover   : {(static.source=='NASA').sum():,}")
    print(f"  Flyby signals : {len(flyby):,}  -> {n_flyby_scenes:,} scenes")
    print(f"    NASA flyover : {len(flyby):,}  (all 4 channels)")
    print(f"  TOTAL scenes  : {total_scenes:,}  "
          f"(static {100*n_static_scenes/total_scenes:.1f}% / "
          f"flyby {100*n_flyby_scenes/total_scenes:.1f}%)")

    # channel uniformity for UAVirBASE
    uav_ch = Counter(static[static.source == "UAVirBASE"].channel)
    tot    = sum(uav_ch.values())
    print("\n  UAVirBASE channel spread (should be ~uniform ~12.5%):")
    print("   ", "  ".join(f"ch{k}:{100*v/tot:.1f}%" for k, v in sorted(uav_ch.items())))

    # ---- splits --------------------------------------------------------
    print("\nSession-level split (leakage guard)...")
    units, session_map = split_sessions(units, rng)
    noise = split_noise_pool(noise, rng)
    for sp in ("train", "val", "test"):
        u = units[units.split == sp]
        print(f"  {sp:<6}: {u.session.nunique():>4} sessions | "
              f"{int(u.n_scenes.sum()):>6,} scenes | "
              f"{(noise.split==sp).sum():>6,} noise files")

    # disjointness assertion
    for a in ("train", "val", "test"):
        for b in ("train", "val", "test"):
            if a < b:
                assert not (set(units[units.split==a].session)
                            & set(units[units.split==b].session)), \
                    f"LEAK between {a}/{b}"
    print("  [OK] splits are session-disjoint")

    # ---- bed pool check ------------------------------------------------
    bed_pool = noise[(noise.target_class.isin(GLOBAL_BED_CLASSES))
                     & (noise.duration_sec >= BED_MIN_DURATION)]
    print(f"\nGlobal bed pool (>= {BED_MIN_DURATION:.0f}s): {len(bed_pool):,} files")
    for sp in ("train", "val", "test"):
        for c in GLOBAL_BED_CLASSES:
            if len(bed_pool[(bed_pool.target_class==c)&(bed_pool.split==sp)]) == 0:
                print(f"  [ERROR] bed '{c}' empty in split '{sp}'")
                return
    print(f"  Bed classes: {len(GLOBAL_BED_CLASSES)} "
          f"(Bird, Cricket, Insect, Livestock removed - too bursty)")
    print("  Events     : DISABLED")
    print(f"  Stationarity gate:")
    print(f"    clips  : Wind > {STATIONARITY_CLIP_THRESHOLD_DB['Wind']:.0f} dB, "
          f"others > {STATIONARITY_CLIP_THRESHOLD_DB['default']:.0f} dB  -> reject")
    print(f"    scenes : all classes > "
          f"{STATIONARITY_SCENE_THRESHOLD_DB:.0f} dB  -> reject and retry")
    print("\n  Weighted distribution (target):")
    wsum = sum(GLOBAL_BED_WEIGHTS.values())
    for c, w in sorted(GLOBAL_BED_WEIGHTS.items(), key=lambda x: -x[1]):
        print(f"    {c:<16}: {100*w/wsum:4.1f}%")

    if limit:
        units = units.head(limit)
        print(f"\n[LIMIT] first {limit} drone signals only")

    planned = int(units.n_scenes.sum())
    print(f"\nWill generate {planned:,} scenes from {len(units):,} drone signals")

    if dry_run:
        print("\n" + "=" * 74)
        print("DRY RUN COMPLETE -- nothing written.")
        print("=" * 74)
        return

    # ---- build ---------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput: {OUTPUT_DIR}\n")

    rows, errors = [], []
    bed_used, ev_used, mode_used = Counter(), Counter(), Counter()
    n_rejected_total = 0
    n_scene_rejected = 0
    scene_drs        = []

    for _, u in tqdm(units.iterrows(), total=len(units),
                     desc="Drones", unit="drone"):
        n_samples = int(round(u.duration_sec * TARGET_SR))
        pool      = noise[noise.split == u.split]

        for scene_i in range(int(u.n_scenes)):
            # retry loop: if the finished scene fails the stationarity gate,
            # draw a new bed class and try again (up to MAX_SCENE_ATTEMPTS).
            # 20 attempts, not 8. At 8 the full run lost 4 scenes of 12,450
            # (0.03%) to exhausted retries -- a scene whose bed class draws kept
            # producing stitched results above the 12 dB scene gate. Raising the
            # ceiling costs runtime only on the rare hard cases and makes the
            # loss effectively zero.
            MAX_SCENE_ATTEMPTS = 20
            for _attempt in range(MAX_SCENE_ATTEMPTS):
              try:
                bed_class = weighted_choice(GLOBAL_BED_CLASSES,
                                            GLOBAL_BED_WEIGHTS, rng)
                bed_candidates = pool[(pool.target_class == bed_class)
                                      & (pool.duration_sec >= BED_MIN_DURATION)]
                if len(bed_candidates) == 0:
                    raise RuntimeError(f"empty bed pool {bed_class}/{u.split}")

                bed, bed_srcs, bed_rej = build_bed(
                    bed_class, n_samples, bed_candidates, rng)
                n_rejected_total += len(bed_rej)

                scene, ev_classes, ev_details = add_events(
                    bed, bed_class, pool, rng)

                # Verify the finished scene is stationary. The gate filters
                # individual clips; this confirms the stitched result holds up.
                scene_dr = dynamic_range_db(scene)

                assert len(scene) == n_samples

                # SCENE-LEVEL STATIONARITY GATE
                # The clip-level gate rejects bursty individual clips, but two
                # borderline clips can stitch into a scene that is itself bursty
                # if their quiet moments align. This checks the finished scene.
                # If it fails, skip this scene entirely -- a different bed class
                # will be drawn on the next attempt (the outer retry loop).
                scene_dr = dynamic_range_db(scene)
                if not is_stationary_scene(scene):
                    n_scene_rejected += 1
                    continue   # skip writing, try again with a new bed draw

                peak = float(np.max(np.abs(scene)))
                if peak > 0.99:
                    scene = scene * (0.99 / peak)

                ev_tag = "-".join(dict.fromkeys(ev_classes)) or "none"
                name = f"{u.drone_stem}__s{scene_i}__bed-{bed_class}__ev-{ev_tag}.wav"
                name = re.sub(r"[^\w\.\-]", "_", name)
                sf.write(str(OUTPUT_DIR / name), scene, TARGET_SR, subtype=OUTPUT_SUBTYPE)

                bed_used[bed_class] += 1
                mode_used[u["mode"]] += 1
                if scene_dr is not None:
                    scene_drs.append(scene_dr)
                for c in ev_classes:
                    ev_used[c] += 1

                rows.append({
                    "scene_file": name, "drone_file": u.drone_file,
                    "split": u.split, "mode": u["mode"], "source": u.source,
                    "session": u.session, "distance": u.distance,
                    "channel": u.channel, "bed_class": bed_class,
                    "bed_sources": " | ".join(bed_srcs),
                    "n_bed_clips": len(bed_srcs),
                    "event_classes": ", ".join(ev_classes),
                    "n_events": len(ev_classes),
                    "event_detail": " | ".join(
                        f"{d['class']}@{d['onset_sec']}s"
                        f"({d['duration_sec']}s,{d['gain_db']}dB)"
                        for d in ev_details),
                    "duration_sec": round(n_samples / TARGET_SR, 3),
                    "sample_rate": TARGET_SR,
                    "dynamic_range_db": (round(scene_dr, 2)
                                         if scene_dr is not None else ""),
                })
                break   # scene accepted -- exit the retry loop
              except Exception as e:
                errors.append({"drone_file": u.drone_file, "error": str(e)})
                break   # don't retry on genuine errors

    if rows:
        with open(OUTPUT_MANIFEST, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # ---- summary -------------------------------------------------------
    print("\n" + "=" * 74)
    print("DONE")
    print("=" * 74)
    print(f"  Scenes written : {len(rows):,}")
    print(f"  Errors         : {len(errors):,}")

    if rows:
        df = pd.DataFrame(rows)
        print("\n  By mode:")
        for m, n in df["mode"].value_counts().items():
            print(f"    {m:<8}: {n:>6,} ({100*n/len(df):.1f}%)")
        print("\n  By split:")
        for sp, n in df.split.value_counts().items():
            print(f"    {sp:<8}: {n:>6,}")
        print("\n  Bed class distribution (ACTUAL vs target):")
        wsum = sum(GLOBAL_BED_WEIGHTS.values())
        for c in GLOBAL_BED_CLASSES:
            got = 100*bed_used[c]/len(df)
            tgt = 100*GLOBAL_BED_WEIGHTS[c]/wsum
            print(f"    {c:<16}: {got:4.1f}%  (target {tgt:4.1f}%)")
        print("\n  Top event classes:")
        for c, n in sorted(ev_used.items(), key=lambda x: -x[1])[:12]:
            print(f"    {c:<16}: {n:>6,}")
        print(f"\n  Bed clips/scene  : {df.n_bed_clips.mean():.2f}")
        print(f"  Events           : DISABLED (pure bed scenes)")

        # --- stationarity report ---------------------------------------
        print("\n  STATIONARITY (in-band 120-1000 Hz energy swing):")
        print(f"    Clips rejected (individual) : {n_rejected_total:,}")
        print(f"    Scenes rejected (stitched)  : {n_scene_rejected:,}")
        if scene_drs:
            arr = np.array(scene_drs)
            print(f"    Scene dynamic range    : "
                  f"median {np.median(arr):.1f} dB | "
                  f"p90 {np.percentile(arr,90):.1f} dB | "
                  f"max {arr.max():.1f} dB")
            print(f"    Scenes over {STATIONARITY_SCENE_THRESHOLD_DB:.0f} dB : "
                  f"{(arr > STATIONARITY_SCENE_THRESHOLD_DB).sum():,} "
                  f"({100*(arr > STATIONARITY_SCENE_THRESHOLD_DB).mean():.1f}%)")
            print("    (lower = steadier = SNR label is meaningful)")
        # flyby bed coverage check
        fly = df[df["mode"] == "flyby"]
        if len(fly):
            print(f"\n  Flyby scenes: {len(fly):,} | distinct beds on flyby: "
                  f"{fly.bed_class.nunique()}/{len(GLOBAL_BED_CLASSES)}")

    if errors:
        print("\n  First errors:")
        for e in errors[:8]:
            print(f"    {e['drone_file']}: {e['error']}")

    print(f"\n  Manifest: {OUTPUT_MANIFEST}")
    print("=" * 74)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    main(a.dry_run, a.limit)