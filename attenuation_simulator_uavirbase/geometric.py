"""
geometric.py

Geometric (spherical) spreading model for UaVirBASE dataset generation.

Physics
-------
For a point source in free field, amplitude falls off as 1/r.
In dB: spreading loss = 20 log10(d / d_ref).

For **incremental** propagation (Approach 1):
    We already have the signal at distance d_orig.
    The extra gain to reach d_target is:
        gain = d_orig / d_target          (< 1 when d_target > d_orig)
"""

from __future__ import annotations
import numpy as np


class GeometricSpreading:
    """Free-field spherical spreading model."""

    @staticmethod
    def incremental_gain(d_original: float, d_target: float) -> float:
        """
        Amplitude scale factor to go from d_original → d_target.

        gain = d_original / d_target

        For d_target > d_original this is < 1 (attenuation).
        """
        if d_original <= 0 or d_target <= 0:
            raise ValueError(
                f"Both distances must be > 0; got d_original={d_original}, d_target={d_target}"
            )
        return d_original / d_target

    @staticmethod
    def incremental_loss_db(d_original: float, d_target: float) -> float:
        """
        Additional spreading loss in dB going from d_original → d_target.

        loss_dB = 20 log10(d_target / d_original)   [positive = loss]
        """
        if d_original <= 0 or d_target <= 0:
            raise ValueError(
                f"Both distances must be > 0; got d_original={d_original}, d_target={d_target}"
            )
        return 20.0 * np.log10(d_target / d_original)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = GeometricSpreading()
    pairs = [(10, 30), (10, 50), (10, 100), (20, 100)]
    print(f"{'d_orig':>8}  {'d_target':>8}  {'Gain':>8}  {'Loss (dB)':>10}")
    print("-" * 42)
    for d_o, d_t in pairs:
        g  = model.incremental_gain(d_o, d_t)
        db = model.incremental_loss_db(d_o, d_t)
        print(f"{d_o:8.0f}  {d_t:8.0f}  {g:8.4f}  {db:10.3f}")