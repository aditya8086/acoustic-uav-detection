# Acoustic UAV Detection

Signal processing and AI algorithms for **acoustics-based detection of drones /
unmanned aerial vehicles**. Developed as part of an internship at
**CARE, IIT Delhi**.

The goal is a system that detects the presence of a UAV from its acoustic
signature alone — distinguishing drone from no-drone under a wide range of
real-world ambient conditions — and does so robustly enough to run in real time
on an embedded processor. This repository holds the signal-processing,
dataset-construction, and (in progress) modelling code for that system.

---

## Approach

A UAV produces a characteristic acoustic signature. The difficulty is detecting
it reliably against everything else the environment throws at the microphone —
wind, rain, traffic, machinery, crowds — and at ranges where the drone is faint.

Rather than relying on the limited real drone-in-the-wild recordings available,
this project builds a large, physically-grounded training corpus:

1. **Distance simulation.** Real drone recordings are propagated to a range of
   greater distances using an acoustic attenuation model (geometric spreading +
   atmospheric absorption, after Sinha et al. 2021), so the corpus spans many
   effective ranges from a limited set of source recordings.

2. **Noise curation & synthesis.** A large environmental-noise corpus is
   assembled from open datasets, aggressively cleaned, and composed into
   stationary background "soundscapes" representative of real deployment
   conditions.

3. **Mixing & labelling.** Drones are mixed into those backgrounds at a sweep of
   controlled signal-to-noise ratios, producing a labelled dataset that lets the
   detector be trained and — crucially — evaluated as a function of how faint the
   drone is.

The system is designed for rigorous evaluation: not just window-level accuracy,
but temporal and event-level performance (detection rate, onset latency,
false-alarm rate) that reflects how a real detector would be judged.

---

## Repository layout

```
.
├── attenuation_simulator_uavirbase/   # Distance-simulation model (Sinha et al. 2021)
│   ├── geometric.py                   # Spherical spreading loss
│   ├── atmospheric.py                 # Atmospheric absorption (Eq. 7-11)
│   ├── attenuation.py                 # Combined transfer function -> audio
│   ├── simulator.py                   # Single-recording propagation simulator
│   ├── generate_dataset.py            # Batch-generate distance variants
│   └── uavirbase_inspector.py         # Inspect raw recordings
│
├── dataset_curator/                   # Noise corpus cleaning & sourcing
│   ├── fsd50k_curator.py              # FSD50K       -> NOISE_MASTER
│   ├── esc50_curator.py              # ESC-50       -> NOISE_MASTER
│   ├── urbansound8k_curator.py       # UrbanSound8K -> NOISE_MASTER
│   ├── demand_curator.py             # DEMAND field recordings -> NOISE_MASTER
│   ├── windfarm_curator.py           # Wind Farm Noise Benchmark -> NOISE_MASTER
│   ├── clean_wind_folder.py          # Remove woodwind-instrument contamination
│   ├── clean_traffic_crowd.py        # Remove explosions / gunshots / laughter
│   ├── add_ambient_to_noise.py       # Fold ambient recordings into the corpus
│   ├── Noise_master_merger.py        # Merge curated sources into NOISE_MASTER
│   └── rebuild_statistics.py         # Regenerate noise_master_metadata.csv
│
├── build_synthetic_noise/             # Dataset construction
│   ├── build_drone_master.py          # Assemble DRONE_MASTER + manifest
│   ├── build_synthetic_noise.py       # Compose stationary noise soundscapes
│   └── mix_and_label.py               # Mix drone + noise across an SNR sweep
│
├── requirements.txt
├── .gitignore
└── README.md
```

The audio corpora themselves (drone recordings, noise corpus, synthesised
soundscapes, mixed dataset) are large and are not tracked in git.

---

## Pipeline components

### Distance simulation — `attenuation_simulator_uavirbase/`

Propagates real recordings to further horizontal distances using an incremental
physical model: geometric spreading (amplitude proportional to 1/r) combined
with frequency-dependent atmospheric absorption driven by temperature, humidity
and pressure, following Sinha et al. (2021). This turns a limited set of
close-range recordings into a corpus spanning many effective ranges.

### Noise curation — `dataset_curator/`

Cleans a raw multi-dataset noise corpus (FSD50K, UrbanSound8K, ESC-50, DEMAND,
Wind Farm Noise Benchmark) into a coherent `NOISE_MASTER` using a **whitelist**
strategy — clips are kept only when their metadata positively confirms the
intended sound, which is far more robust than blacklisting known contaminants.

### Dataset construction — `build_synthetic_noise/`

Assembles the drone corpus, composes stationary background soundscapes matched
to each drone recording, and mixes the two across a range of signal-to-noise
ratios to produce the final labelled dataset. Care is taken throughout to keep
the train / validation / test split leakage-free (splitting at the recording-
session level, before any mixing) and to keep the physical calibration of the
propagation model intact (the drone is held at its recorded level; the
background is scaled to set the SNR).

---

## Reference

Sinha et al., *Applied Acoustics* 182 (2021) — the acoustic propagation and
attenuation model implemented in `attenuation_simulator_uavirbase/`.

---

*Internship project — CARE, IIT Delhi.*
