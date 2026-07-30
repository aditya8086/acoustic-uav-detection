"""
generate_dataset.py

Generate new UaVirBASE dataset(s) at one or more target distances.

Edit the USER CONFIG section, then run:
    python generate_dataset.py

Output folder structure (mirrors the original flat layout):
    UaVirBASE_100m/
        20241115_093611/
            output.wav
            label.json
        ...
"""

from __future__ import annotations

import shutil
from pathlib import Path

from simulator import UaVirBASESimulator


# =============================================================================
# USER CONFIG  <-- edit this section before running
# =============================================================================

INPUT_DATASET = Path(
    r"C:\Users\CARE\Downloads\Acoustic based drone detection"
    r"\Audio files\Datasets\UaVirBASE"
)

# All horizontal target distances (metres) to generate.
# One output folder (e.g. UaVirBASE_100m) is created per entry.
TARGET_DISTANCES = [325.0, 350.0, 375.0, 400.0]

# Parent folder for all generated datasets (defaults to same parent as input).
OUTPUT_PARENT = INPUT_DATASET.parent

# Whether to copy ambient recordings (no drone) into the output datasets.
COPY_AMBIENT = True

# =============================================================================


def _is_recording_folder(folder: Path) -> bool:
    """
    A valid UaVirBASE recording folder contains BOTH of these
    directly inside it (flat, no subfolders):
        output.wav
        label.json
    """
    if not folder.is_dir():
        return False
    return (folder / "output.wav").exists() and (folder / "label.json").exists()


def _process_target(folders:         list[Path],
                    target_distance: float,
                    output_parent:   Path,
                    copy_ambient:    bool) -> None:

    out_root = output_parent / f"UaVirBASE_{int(target_distance)}m"
    out_root.mkdir(parents=True, exist_ok=True)

    total         = len(folders)
    done_drone    = 0
    skip_drone    = 0
    done_ambient  = 0
    skip_ambient  = 0
    errors        = []

    print(f"\n{'=' * 70}")
    print(f"  Target : {target_distance} m")
    print(f"  Output : {out_root}")
    print(f"{'=' * 70}")

    for idx, folder in enumerate(folders, start=1):
        tag = f"[{idx:3d}/{total}]  {folder.name}"

        # --- Load --------------------------------------------------------
        try:
            sim = UaVirBASESimulator(folder)
        except Exception as exc:
            msg = f"{tag}  LOAD ERROR: {exc}"
            print(msg)
            errors.append(msg)
            continue

        out_folder = out_root / folder.name

        # --- Ambient recordings ------------------------------------------
        if sim.is_ambient:
            if copy_ambient:
                if out_folder.exists():
                    shutil.rmtree(out_folder)
                shutil.copytree(folder, out_folder)
                print(f"{tag}  [ambient -> copied]")
                done_ambient += 1
            else:
                print(f"{tag}  [ambient -> skipped]")
                skip_ambient += 1
            continue

        # --- Drone recordings --------------------------------------------
        # Skip if the target is not actually farther than the original
        if target_distance <= sim.horizontal_distance:
            print(
                f"{tag}  [SKIPPED: target {target_distance}m "
                f"<= original {sim.horizontal_distance}m]"
            )
            skip_drone += 1
            continue

        try:
            sim.save(out_folder, target_distance)
            print(
                f"{tag}  "
                f"d={sim.horizontal_distance:.0f}m "
                f"h={sim.height:.0f}m "
                f"slant={sim.original_slant:.1f}m "
                f"-> {target_distance:.0f}m "
                f"(slant={sim._target_slant(target_distance):.1f}m)"
            )
            done_drone += 1
        except Exception as exc:
            msg = f"{tag}  SIMULATION ERROR: {exc}"
            print(msg)
            errors.append(msg)

    # --- Summary ---------------------------------------------------------
    print(f"\n  Summary for target = {target_distance} m")
    print(f"    Drone OK       : {done_drone}")
    print(f"    Drone skipped  : {skip_drone}  (target <= original)")
    print(f"    Ambient copied : {done_ambient}")
    print(f"    Ambient skipped: {skip_ambient}")
    if errors:
        print(f"    Errors         : {len(errors)}")
        for e in errors:
            print(f"      {e}")


def main() -> None:
    if not INPUT_DATASET.exists():
        raise FileNotFoundError(
            f"Input dataset not found:\n  {INPUT_DATASET}\n"
            "Please update INPUT_DATASET in the USER CONFIG section."
        )

    folders = sorted(
        f for f in INPUT_DATASET.iterdir() if _is_recording_folder(f)
    )

    if not folders:
        raise RuntimeError(
            f"No valid recording folders found in:\n  {INPUT_DATASET}\n"
            "Each folder must contain output.wav and label.json directly inside it."
        )

    print(f"UaVirBASE DATASET GENERATOR")
    print(f"Input       : {INPUT_DATASET}")
    print(f"Folders     : {len(folders)}")
    print(f"Targets (m) : {TARGET_DISTANCES}")

    for target in TARGET_DISTANCES:
        _process_target(
            folders         = folders,
            target_distance = target,
            output_parent   = OUTPUT_PARENT,
            copy_ambient    = COPY_AMBIENT,
        )

    print(f"\n{'=' * 70}")
    print("All targets complete.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()