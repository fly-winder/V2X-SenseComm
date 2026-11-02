"""Generate reproducible near-field UPA codebooks for SenseComm.

The default command reproduces the released fine-codebook configuration:

    python codebook/generate_codebook.py

The output keys and tensor layout are compatible with
``communication.codebook_loader.load_codebook``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np


SPEED_OF_LIGHT = 3.0e8
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent


def build_array_positions(
    frequency: float,
    num_rows: int,
    num_cols: int,
    vertical_spacing: float = 0.5,
    horizontal_spacing: float = 0.5,
) -> np.ndarray:
    """Return Sionna planar-array element positions with shape ``[N, 3]``.

    Sionna expresses both spacings in wavelengths. The array lies in the
    local Y-Z plane and faces the local +X direction.
    """
    from sionna.rt import PlanarArray

    array = PlanarArray(
        num_rows=num_rows,
        num_cols=num_cols,
        vertical_spacing=vertical_spacing,
        horizontal_spacing=horizontal_spacing,
        pattern="iso",
        polarization="V",
    )
    positions = array.positions(SPEED_OF_LIGHT / frequency)
    positions = positions.numpy() if hasattr(positions, "numpy") else np.asarray(positions)
    return np.asarray(positions, dtype=np.float64).T


def generate_near_field_vector(
    distance: float,
    elevation_deg: float,
    azimuth_deg: float,
    antenna_positions: np.ndarray,
    frequency: float = 7.0e9,
) -> np.ndarray:
    """Generate a constant-modulus spherical-wave beamforming vector.

    Angles are defined in the local array coordinate system. Azimuth is the
    angle of the X-Y projection from +X, and elevation is measured from the
    X-Y plane. The planar array is located in the Y-Z plane and faces +X.
    """
    elevation = np.deg2rad(elevation_deg)
    azimuth = np.deg2rad(azimuth_deg)
    target_position = np.array(
        [
            distance * np.cos(elevation) * np.cos(azimuth),
            distance * np.cos(elevation) * np.sin(azimuth),
            distance * np.sin(elevation),
        ],
        dtype=np.float64,
    )

    element_distances = np.linalg.norm(antenna_positions - target_position, axis=1)
    wave_number = 2.0 * np.pi * frequency / SPEED_OF_LIGHT
    num_antennas = antenna_positions.shape[0]
    weights = np.exp(1j * wave_number * element_distances) / np.sqrt(num_antennas)
    return weights.astype(np.complex64)


def generate_broadened_beam(
    distance: float,
    elevation_deg: float,
    azimuth_deg: float,
    antenna_positions: np.ndarray,
    frequency: float,
    num_rows: int,
    num_cols: int,
    alpha: float = 2.0,
    beta: float = 2.0,
) -> np.ndarray:
    """Apply the quadratic-phase broadening used by the original notebook."""
    fine_weights = generate_near_field_vector(
        distance,
        elevation_deg,
        azimuth_deg,
        antenna_positions,
        frequency,
    )

    row_indices = np.arange(num_rows) - (num_rows - 1) / 2.0
    col_indices = np.arange(num_cols) - (num_cols - 1) / 2.0
    col_grid, row_grid = np.meshgrid(col_indices, row_indices, indexing="ij")
    phase = np.pi * (
        alpha * (col_grid / num_cols) ** 2
        + beta * (row_grid / num_rows) ** 2
    )
    broadened = fine_weights * np.exp(1j * phase.ravel())
    return (broadened / np.linalg.norm(broadened)).astype(np.complex64)


def build_sampling_axes(
    distance_params: tuple[int, float, float],
    azimuth_params: tuple[int, float, float],
    elevation_params: tuple[int, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build inverse-range and sine-angle uniformly sampled axes."""
    num_distance, min_distance, max_distance = distance_params
    num_azimuth, min_azimuth, max_azimuth = azimuth_params
    num_elevation, min_elevation, max_elevation = elevation_params

    if min_distance <= 0 or max_distance <= min_distance:
        raise ValueError("Distance bounds must satisfy 0 < min < max")
    if num_distance <= 0 or num_azimuth <= 0 or num_elevation <= 0:
        raise ValueError("All axis sizes must be positive")
    if not (-90.0 <= min_azimuth < max_azimuth <= 90.0):
        raise ValueError("Azimuth bounds must lie within [-90, 90] degrees")
    if not (-90.0 <= min_elevation < max_elevation <= 90.0):
        raise ValueError("Elevation bounds must lie within [-90, 90] degrees")

    distance_axis = 1.0 / np.linspace(
        1.0 / min_distance,
        1.0 / max_distance,
        num_distance,
    )
    azimuth_axis = np.rad2deg(
        np.arcsin(
            np.linspace(
                np.sin(np.deg2rad(min_azimuth)),
                np.sin(np.deg2rad(max_azimuth)),
                num_azimuth,
            )
        )
    )
    elevation_axis = np.rad2deg(
        np.arcsin(
            np.linspace(
                np.sin(np.deg2rad(min_elevation)),
                np.sin(np.deg2rad(max_elevation)),
                num_elevation,
            )
        )
    )
    return distance_axis, azimuth_axis, elevation_axis


def generate_4d_codebook(
    distance_params: tuple[int, float, float],
    azimuth_params: tuple[int, float, float],
    elevation_params: tuple[int, float, float],
    frequency: float = 7.0e9,
    num_rows: int = 16,
    num_cols: int = 64,
    vertical_spacing: float = 0.5,
    horizontal_spacing: float = 0.5,
    broaden: bool = False,
    alpha: float = 2.0,
    beta: float = 2.0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Generate ``[range, azimuth, elevation, antenna]`` codebook weights."""
    distance_axis, azimuth_axis, elevation_axis = build_sampling_axes(
        distance_params,
        azimuth_params,
        elevation_params,
    )
    antenna_positions = build_array_positions(
        frequency,
        num_rows,
        num_cols,
        vertical_spacing,
        horizontal_spacing,
    )

    shape = (
        len(distance_axis),
        len(azimuth_axis),
        len(elevation_axis),
        num_rows * num_cols,
    )
    codebook = np.empty(shape, dtype=np.complex64)
    total_beams = np.prod(shape[:3])
    print(f"Generating codebook {shape[:3]}: {total_beams:,} beams, {shape[3]} antennas")
    start_time = time.time()

    for distance_index, distance in enumerate(distance_axis):
        for azimuth_index, azimuth in enumerate(azimuth_axis):
            for elevation_index, elevation in enumerate(elevation_axis):
                if broaden:
                    weights = generate_broadened_beam(
                        distance,
                        elevation,
                        azimuth,
                        antenna_positions,
                        frequency,
                        num_rows,
                        num_cols,
                        alpha,
                        beta,
                    )
                else:
                    weights = generate_near_field_vector(
                        distance,
                        elevation,
                        azimuth,
                        antenna_positions,
                        frequency,
                    )
                codebook[distance_index, azimuth_index, elevation_index] = weights

        elapsed = time.time() - start_time
        print(
            f"Distance layer {distance_index + 1}/{len(distance_axis)} complete "
            f"({elapsed:.1f} s elapsed)"
        )

    axes = {
        "r_axis": distance_axis,
        "az_axis": azimuth_axis,
        "el_axis": elevation_axis,
    }
    return codebook, axes


def save_codebook(
    output_path: Path,
    codebook: np.ndarray,
    axes: dict[str, np.ndarray],
    overwrite: bool = False,
) -> None:
    """Save the codebook using the field names expected by SenseComm."""
    output_path = output_path.resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite to replace it."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        codebook=codebook,
        r_axis=axes["r_axis"],
        az_axis=axes["az_axis"],
        el_axis=axes["el_axis"],
    )

    with np.load(output_path) as data:
        expected_keys = {"codebook", "r_axis", "az_axis", "el_axis"}
        if set(data.files) != expected_keys or data["codebook"].shape != codebook.shape:
            raise RuntimeError("Saved codebook verification failed")
    print(f"Saved and verified: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fine", "coarse"), default="fine")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--frequency", type=float, default=7.0e9)
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--cols", type=int, default=64)
    parser.add_argument("--vertical-spacing", type=float, default=0.5)
    parser.add_argument("--horizontal-spacing", type=float, default=0.5)
    parser.add_argument("--range-count", type=int, default=None)
    parser.add_argument("--azimuth-count", type=int, default=None)
    parser.add_argument("--elevation-count", type=int, default=None)
    parser.add_argument("--range-min", type=float, default=20.0)
    parser.add_argument("--range-max", type=float, default=200.0)
    parser.add_argument("--azimuth-min", type=float, default=None)
    parser.add_argument("--azimuth-max", type=float, default=None)
    parser.add_argument("--elevation-min", type=float, default=None)
    parser.add_argument("--elevation-max", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    is_fine = args.mode == "fine"

    range_count = args.range_count if args.range_count is not None else (16 if is_fine else 8)
    azimuth_count = args.azimuth_count if args.azimuth_count is not None else (128 if is_fine else 32)
    elevation_count = args.elevation_count if args.elevation_count is not None else (32 if is_fine else 8)
    azimuth_min = args.azimuth_min if args.azimuth_min is not None else (-60.0 if is_fine else -58.0)
    azimuth_max = args.azimuth_max if args.azimuth_max is not None else (60.0 if is_fine else 58.0)
    elevation_min = args.elevation_min if args.elevation_min is not None else (-20.0 if is_fine else -18.0)
    elevation_max = args.elevation_max if args.elevation_max is not None else (20.0 if is_fine else 18.0)
    output = args.output or DEFAULT_OUTPUT_DIR / f"{args.mode}_codebook.npz"

    codebook, axes = generate_4d_codebook(
        distance_params=(range_count, args.range_min, args.range_max),
        azimuth_params=(azimuth_count, azimuth_min, azimuth_max),
        elevation_params=(elevation_count, elevation_min, elevation_max),
        frequency=args.frequency,
        num_rows=args.rows,
        num_cols=args.cols,
        vertical_spacing=args.vertical_spacing,
        horizontal_spacing=args.horizontal_spacing,
        broaden=not is_fine,
        alpha=args.alpha,
        beta=args.beta,
    )
    save_codebook(output, codebook, axes, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
