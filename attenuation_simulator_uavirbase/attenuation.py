"""
attenuation.py

Combines geometric spreading and atmospheric absorption into a single
frequency-domain transfer function, then applies it to audio waveforms.

Version 1
---------
  ✓  Geometric spreading   (Eq. 5 of Sinha et al.)
  ✓  Atmospheric absorption (Eq. 7–11 of Sinha et al.)
  ✗  Ground effects          (intentionally omitted)

Approach
--------
  Approach 1 (incremental):
      We have a real recording at slant_range_original.
      We apply ONLY the additional attenuation from
      slant_range_original → slant_range_target.

  H(f) = (d_orig / d_target)            ← geometric
        × 10^( -Δα(f)·Δd / 20 )        ← atmospheric (frequency-dependent)

  where  Δd = slant_range_target − slant_range_original
         Δα(f) is the absorption coefficient in dB/m at frequency f.
"""

from __future__ import annotations

import numpy as np

from geometric   import GeometricSpreading
from atmospheric import AtmosphericModel


class AttenuationModel:
    """
    Combined attenuation model (geometric + atmospheric).

    Parameters
    ----------
    temperature_c : float
        Air temperature in °C.
    humidity_percent : float
        Relative humidity in % (0–100).
    pressure_atm : float
        Atmospheric pressure in **atm** (not Pa).
        Convert: pressure_atm = pressure_pa / 101325.
    """

    def __init__(self,
                 temperature_c:   float,
                 humidity_percent: float,
                 pressure_atm:    float = 1.0):

        self._geo = GeometricSpreading()
        self._atm = AtmosphericModel(
            temperature_c   = temperature_c,
            humidity_percent = humidity_percent,
            pressure_atm    = pressure_atm,
        )

    # ------------------------------------------------------------------
    # Transfer function
    # ------------------------------------------------------------------

    def transfer_function(self,
                          freqs:            np.ndarray,
                          slant_d_original: float,
                          slant_d_target:   float) -> np.ndarray:
        """
        Full H(f) for Approach 1 incremental propagation.

        Parameters
        ----------
        freqs : np.ndarray
            Frequency array in Hz (from np.fft.rfftfreq).
        slant_d_original : float
            3-D slant range at which the recording was made (metres).
        slant_d_target : float
            Desired 3-D slant range to simulate (metres).

        Returns
        -------
        H : np.ndarray
            Real-valued amplitude multiplier, same shape as *freqs*.
        """
        if slant_d_target <= slant_d_original:
            raise ValueError(
                f"target slant range ({slant_d_target:.2f} m) must be greater than "
                f"original slant range ({slant_d_original:.2f} m).  "
                "Simulating closer distances is not supported."
            )

        # --- Geometric ---------------------------------------------------
        geo_gain = self._geo.incremental_gain(slant_d_original, slant_d_target)

        # --- Atmospheric -------------------------------------------------
        # Additional path length the signal must travel beyond the original.
        delta_d  = slant_d_target - slant_d_original
        atm_gain = self._atm.transfer_function(freqs, delta_d)

        return geo_gain * atm_gain  # shape matches freqs

    # ------------------------------------------------------------------
    # Apply to waveform (full FFT approach, frequency-dependent)
    # ------------------------------------------------------------------

    def apply_to_channel(self,
                         samples:          np.ndarray,
                         sample_rate:      int,
                         slant_d_original: float,
                         slant_d_target:   float) -> np.ndarray:
        """
        Apply the propagation transfer function to a mono (1-D) array.

        Parameters
        ----------
        samples : np.ndarray  shape (N,)
            Single audio channel.
        sample_rate : int
            Sampling rate in Hz.
        slant_d_original, slant_d_target : float
            Slant ranges in metres.

        Returns
        -------
        out : np.ndarray  shape (N,)
        """
        N    = len(samples)
        spec = np.fft.rfft(samples)

        # rfftfreq returns N//2+1 frequencies matching the rfft output length
        freqs = np.fft.rfftfreq(N, d=1.0 / sample_rate)

        H    = self.transfer_function(freqs, slant_d_original, slant_d_target)
        return np.fft.irfft(spec * H, n=N)

    def apply_waveform(self,
                       waveform:         np.ndarray,
                       sample_rate:      int,
                       slant_d_original: float,
                       slant_d_target:   float) -> np.ndarray:
        """
        Apply the propagation transfer function to a waveform array.

        Parameters
        ----------
        waveform : np.ndarray
            Shape (N,) for mono or (N, C) for multi-channel.
            UaVirBASE recordings are (N, 8).
        sample_rate : int
        slant_d_original, slant_d_target : float
            Slant ranges in metres.

        Returns
        -------
        out : np.ndarray  same shape as *waveform*, dtype float64.
        """
        x = np.asarray(waveform, dtype=np.float64)

        if x.ndim == 1:
            return self.apply_to_channel(x, sample_rate,
                                         slant_d_original, slant_d_target)

        if x.ndim == 2:
            out = np.empty_like(x)
            for ch in range(x.shape[1]):
                out[:, ch] = self.apply_to_channel(
                    x[:, ch], sample_rate, slant_d_original, slant_d_target
                )
            return out

        raise ValueError(f"waveform must be 1-D or 2-D, got shape {x.shape}")


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import math

    fs    = 96_000
    t     = np.arange(fs, dtype=float) / fs
    sig   = np.sin(2 * np.pi * 1000 * t)   # 1 kHz tone

    # Replicate a recording made at d=10m, h=10m
    slant_orig   = math.sqrt(10**2 + 10**2)   # 14.14 m
    slant_target = math.sqrt(100**2 + 10**2)  # 100.50 m

    model = AttenuationModel(temperature_c=4.6, humidity_percent=90.0, pressure_atm=1.0)
    out   = model.apply_waveform(sig, fs, slant_orig, slant_target)

    rms_in  = np.sqrt(np.mean(sig**2))
    rms_out = np.sqrt(np.mean(out**2))
    print(f"Slant orig  : {slant_orig:.2f} m")
    print(f"Slant target: {slant_target:.2f} m")
    print(f"Input  RMS  : {rms_in:.6f}")
    print(f"Output RMS  : {rms_out:.6f}")
    print(f"Ratio (dB)  : {20*math.log10(rms_out/rms_in):.3f} dB")