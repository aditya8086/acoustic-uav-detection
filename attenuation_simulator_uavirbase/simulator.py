"""
simulator.py

UaVirBASE single-recording propagation simulator.

Folder layout (confirmed):
    <recording_folder>/
        output.wav      <- 8-channel, 96 kHz, 32-bit PCM, directly here
        label.json      <- metadata/labels, directly here (NOT in a subfolder)

Applies incremental geometric spreading + atmospheric absorption to
simulate the recording at a larger horizontal distance, keeping height
the same.  Ground effects omitted (Version 1).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import soundfile as sf

from attenuation import AttenuationModel


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PA_TO_ATM = 1.0 / 101_325.0   # multiply Pa by this to get atmospheres


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_float(value) -> float:
    """
    Extract the first numeric value from a string such as:
        '4.6 C'      -> 4.6
        '90 % RH'    -> 90.0
        '100810 Pa'  -> 100810.0
        1.3          -> 1.3   (already numeric, passed through)
    """
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r'[-+]?\d+(?:\.\d+)?', str(value))
    if match:
        return float(match.group())
    raise ValueError(f"Cannot extract a number from weather value: {value!r}")


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class UaVirBASESimulator:
    """
    Load one UaVirBASE recording folder and simulate it at a new distance.

    Parameters
    ----------
    recording_folder : str or Path
        Path to a single recording folder, e.g. ``UaVirBASE/20241115_093611``.
        Must contain output.wav and label.json directly inside it.
    """

    def __init__(self, recording_folder):
        self.folder    = Path(recording_folder)
        self.wav_path  = self.folder / "output.wav"
        self.json_path = self.folder / "label.json"

        if not self.wav_path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {self.wav_path}"
            )
        if not self.json_path.exists():
            raise FileNotFoundError(
                f"Label JSON not found: {self.json_path}"
            )

        # --- Load metadata -----------------------------------------------
        with open(self.json_path, "r", encoding="utf-8") as fh:
            self.meta = json.load(fh)

        # --- Parse weather (all values are strings like "4.6 C") ---------
        wx = self.meta["weather_data"]["measurements"]   # lowercase key

        self.temperature_c  = _parse_float(wx["air temperature"])
        self.humidity_pct   = _parse_float(wx["air humidity"])
        pressure_pa         = _parse_float(wx["barometric pressure"])
        self.pressure_atm   = pressure_pa * _PA_TO_ATM   # convert Pa -> atm

        # --- Parse drone metadata ----------------------------------------
        drone        = self.meta.get("drone", {})
        sound_source = (drone.get("sound_source") or "").strip().lower()
        self.is_ambient = (sound_source == "ambient noise")

        if self.is_ambient:
            self.horizontal_distance = None
            self.height              = None
            self.original_slant      = None
        else:
            # distance and height are stored as strings ("20") in the JSON
            self.horizontal_distance = float(drone["distance"])
            self.height              = float(drone["height"])
            # 3-D slant range: the true acoustic propagation path length
            self.original_slant = math.sqrt(
                self.horizontal_distance ** 2 + self.height ** 2
            )

        # --- Load audio --------------------------------------------------
        # sf.read normalises PCM_32 to float64 in [-1, 1]
        self.audio, self.fs = sf.read(self.wav_path, always_2d=True)
        # Shape: (N_samples, 8) for UaVirBASE recordings

    # ------------------------------------------------------------------
    # Core simulation
    # ------------------------------------------------------------------

    def _target_slant(self, target_horizontal_distance: float) -> float:
        """
        Slant range at the target horizontal distance,
        keeping the original drone height constant.
        """
        return math.sqrt(target_horizontal_distance ** 2 + self.height ** 2)

    def simulate(self, target_distance: float) -> np.ndarray:
        """
        Apply incremental attenuation and return the simulated audio array.

        Parameters
        ----------
        target_distance : float
            Desired horizontal distance in metres (e.g. 30, 50, 100).
            Must be larger than the original horizontal distance.

        Returns
        -------
        np.ndarray  shape (N_samples, 8)
        """
        if self.is_ambient:
            return self.audio.copy()

        target_slant = self._target_slant(target_distance)

        if target_slant <= self.original_slant:
            raise ValueError(
                f"Target slant ({target_slant:.2f} m) must be > "
                f"original slant ({self.original_slant:.2f} m). "
                f"(horizontal: {target_distance} m vs {self.horizontal_distance} m, "
                f"height: {self.height} m)"
            )

        model = AttenuationModel(
            temperature_c    = self.temperature_c,
            humidity_percent = self.humidity_pct,
            pressure_atm     = self.pressure_atm,
        )

        return model.apply_waveform(
            waveform         = self.audio,
            sample_rate      = self.fs,
            slant_d_original = self.original_slant,
            slant_d_target   = target_slant,
        )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def updated_metadata(self, target_distance: float) -> dict:
        """Return a deep-copied metadata dict updated for the new distance."""
        meta = json.loads(json.dumps(self.meta))   # deep copy

        if not self.is_ambient:
            meta["drone"]["distance"] = float(target_distance)
            # height stays unchanged

        meta["simulation"] = {
            "generated":                      True,
            "version":                        "v1",
            "ground_effect":                  False,
            "noise_added":                    False,
            "method":                         "Geometric + Atmospheric (Approach 1 / incremental)",
            "original_horizontal_distance_m": self.horizontal_distance,
            "original_height_m":              self.height,
            "original_slant_range_m":
                round(self.original_slant, 4) if self.original_slant else None,
            "target_horizontal_distance_m":   float(target_distance),
            "target_slant_range_m":
                round(self._target_slant(target_distance), 4)
                if not self.is_ambient else None,
        }

        return meta

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, output_folder, target_distance: float) -> Path:
        """
        Simulate and write output.wav + label.json into output_folder.

        The output folder mirrors the original flat structure:
            <output_folder>/
                output.wav
                label.json

        Parameters
        ----------
        output_folder   : str or Path
        target_distance : float  horizontal distance in metres

        Returns
        -------
        Path to the output folder.
        """
        out = Path(output_folder)
        out.mkdir(parents=True, exist_ok=True)

        # Audio
        audio = self.simulate(target_distance)
        sf.write(
            file       = out / "output.wav",
            data       = audio,
            samplerate = self.fs,
            subtype    = "PCM_32",
        )

        # Metadata
        with open(out / "label.json", "w", encoding="utf-8") as fh:
            json.dump(self.updated_metadata(target_distance), fh, indent=4)

        return out

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        if self.is_ambient:
            return f"UaVirBASESimulator({self.folder.name}, AMBIENT)"
        return (
            f"UaVirBASESimulator({self.folder.name}, "
            f"d={self.horizontal_distance}m, h={self.height}m, "
            f"slant={self.original_slant:.2f}m)"
        )


# ---------------------------------------------------------------------------
# Quick single-folder test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ROOT = Path(
        r"C:\Users\CARE\Downloads\Acoustic based drone detection"
        r"\Audio files\Datasets\UaVirBASE"
    )

    # Use the second folder to skip the first ambient recording
    folders = sorted(ROOT.iterdir())
    test_folder = folders[1]

    print(f"Testing: {test_folder.name}")
    sim = UaVirBASESimulator(test_folder)
    print(sim)

    out_path = test_folder.parent / (test_folder.name + "_100m_test")
    sim.save(out_path, target_distance=100)
    print(f"Saved to: {out_path}")