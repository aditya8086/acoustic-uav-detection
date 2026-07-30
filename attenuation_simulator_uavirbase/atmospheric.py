"""
atmospheric.py

Atmospheric attenuation model for UaVirBASE dataset generation.

Implements the atmospheric absorption equations (Eq. 7–11) from:
    Sinha et al., Applied Acoustics 182 (2021).

Ground effects intentionally omitted (Version 1).

Units
-----
  temperature : degrees Celsius  (converted internally to Kelvin)
  humidity    : percent (0–100)  (converted internally to fraction)
  pressure    : atmospheres      (caller must convert Pa → atm)
  frequency   : Hz
  distance    : metres
  alpha       : nepers / metre
"""

from __future__ import annotations
import numpy as np

# Reference constants from Sinha et al. / ISO 9613-1
_T0  = 293.15   # K  – reference temperature
_T01 = 273.16   # K  – used in psat formula
_P0  = 1.0      # atm – reference pressure


class AtmosphericModel:
    """
    Frequency-dependent atmospheric absorption.

    Parameters
    ----------
    temperature_c : float
        Ambient air temperature in °C.
    humidity_percent : float
        Relative humidity in percent (e.g. 90 for 90 %).
    pressure_atm : float
        Atmospheric pressure in **atmospheres** (default 1.0).
        Convert from Pascals: pressure_atm = pressure_pa / 101325.
    """

    def __init__(self,
                 temperature_c: float,
                 humidity_percent: float,
                 pressure_atm: float = 1.0):
        self.T  = float(temperature_c) + 273.15   # Kelvin
        self.hr = float(humidity_percent) / 100.0  # fraction
        self.ps = float(pressure_atm)              # atm

    # ------------------------------------------------------------------
    # Internal helpers (Eq. 10, 11, 8, 9 of Sinha et al.)
    # ------------------------------------------------------------------

    def _saturated_vapour_pressure(self) -> float:
        """psat in atm  (Eq. 11)."""
        return _P0 * 10.0 ** (
            -6.8346 * (_T01 / self.T) ** 1.261 + 4.6151
        )

    def _water_vapour_molar_conc(self) -> float:
        """Molar concentration of water vapour h (Eq. 10), dimensionless."""
        psat = self._saturated_vapour_pressure()
        return (_P0 * self.hr / self.ps) * (psat / _P0)

    def _fr_oxygen(self) -> float:
        """Relaxation frequency of O2 in Hz/atm (Eq. 9)."""
        h = self._water_vapour_molar_conc()
        return (1.0 / _P0) * (
            24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h)
        )

    def _fr_nitrogen(self) -> float:
        """Relaxation frequency of N2 in Hz/atm (Eq. 8)."""
        h = self._water_vapour_molar_conc()
        return (1.0 / _P0) * (_T0 / self.T) ** 0.5 * (
            9.0 + 280.0 * h * np.exp(-4.17 * ((_T0 / self.T) ** (1.0 / 3.0) - 1.0))
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def attenuation_coefficient(self, frequency: np.ndarray) -> np.ndarray:
        """
        Atmospheric attenuation coefficient α (nepers / metre).

        Implements Eq. 7 of Sinha et al.

        Parameters
        ----------
        frequency : array-like
            Frequencies in Hz.  Scalar or array, must be ≥ 0.
        """
        f   = np.asarray(frequency, dtype=float)
        FrO = self._fr_oxygen()
        FrN = self._fr_nitrogen()

        # Pressure-scaled frequency (Hz/atm)
        F = f / self.ps

        # Eq. 7 gives α/ps; we multiply by ps at the end.
        # Handle f=0 (DC): F=0 → α=0, which is correct.
        with np.errstate(divide='ignore', invalid='ignore'):
            term_O = np.where(
                FrO > 0,
                0.01278 * np.exp(-2239.1 / self.T) / (FrO + F ** 2 / FrO),
                0.0,
            )
            term_N = np.where(
                FrN > 0,
                0.1068  * np.exp(-3352.0  / self.T) / (FrN + F ** 2 / FrN),
                0.0,
            )

        alpha_over_ps = (
            (F ** 2 / _P0)
            * (
                1.84e-11 * (self.T / _T0) ** (-0.5)
                + (self.T / _T0) ** (-2.5) * (term_O + term_N)
            )
        )

        return alpha_over_ps * self.ps   # nepers / metre

    def attenuation_db(self, frequency: np.ndarray, distance: float) -> np.ndarray:
        """
        Total atmospheric attenuation over *distance* metres, in dB.

        Parameters
        ----------
        frequency : array-like
            Frequencies in Hz.
        distance : float
            Path length in metres.
        """
        # α [nepers/m] × d [m] × (20 log10 e)  →  dB
        return self.attenuation_coefficient(frequency) * float(distance) * 8.685889638

    def transfer_function(self, freqs: np.ndarray, distance: float) -> np.ndarray:
        """
        Linear amplitude multiplier (0–1) for atmospheric absorption.

        transfer_function = 10^( -attenuation_db / 20 )
        """
        att_db = self.attenuation_db(freqs, distance)
        return 10.0 ** (-att_db / 20.0)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = AtmosphericModel(temperature_c=4.6, humidity_percent=90.0, pressure_atm=1.0)
    freqs = np.array([100, 250, 500, 1000, 2000, 4000, 8000, 16000])
    print(f"{'Freq (Hz)':>12}  {'Atten @10m (dB)':>18}  {'Atten @100m (dB)':>18}")
    print("-" * 54)
    for f in freqs:
        a10  = model.attenuation_db(f, 10)
        a100 = model.attenuation_db(f, 100)
        print(f"{f:12.0f}  {a10:18.4f}  {a100:18.4f}")