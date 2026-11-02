# SenseComm Near-Field Codebook

This directory contains the reproducible codebook generator used by the
SenseComm data-generation pipeline. The generated NPZ file is loaded by
`communication/codebook_loader.py` and searched by the beam-labeling module.

## Released fine-codebook configuration

| Parameter | Value |
|---|---:|
| Carrier frequency | 7 GHz |
| BS array | 16 x 64 UPA |
| Array plane / boresight | Y-Z plane / local +X |
| Element spacing | 0.5 wavelength horizontally and vertically |
| Range grid | 16 points, 20-200 m, uniformly sampled in 1/r |
| Azimuth grid | 128 points, -60 to 60 degrees, uniformly sampled in sin(azimuth) |
| Elevation grid | 32 points, -20 to 20 degrees, uniformly sampled in sin(elevation) |
| Number of beams | 16 x 128 x 32 = 65,536 |
| Antennas per beam | 1,024 |
| Tensor layout | range x azimuth x elevation x antenna |
| Weight type | complex64, constant modulus |

Angles are defined in the array-local coordinate system. The 10-degree BS
downtilt is applied by the transmitter orientation when the communication
scene is built; it is therefore not embedded in the codebook weights.

## Generate the fine codebook

From the project root, run:

```bash
python codebook/generate_codebook.py
```

The default output is:

```text
codebook/fine_codebook.npz
```

The generator does not overwrite an existing file unless `--overwrite` is
specified:

```bash
python codebook/generate_codebook.py --overwrite
```

The full tensor contains 67,108,864 complex64 values (512 MiB before NPZ
compression). Keep at least 1.5 GB of free memory available while generating
and compressing it.

## Output format

The NPZ archive contains exactly four arrays:

```text
codebook  complex64  [16, 128, 32, 1024]
r_axis    float64    [16]
az_axis   float64    [128]
el_axis   float64    [32]
```

These names match the defaults in `communication/codebook_loader.py`.

## Custom configuration

All important physical and grid parameters can be overridden. For example:

```bash
python codebook/generate_codebook.py \
  --frequency 7e9 \
  --rows 16 \
  --cols 64 \
  --range-count 16 \
  --range-min 20 \
  --range-max 200 \
  --azimuth-count 128 \
  --azimuth-min -60 \
  --azimuth-max 60 \
  --elevation-count 32 \
  --elevation-min -20 \
  --elevation-max 20 \
  --output codebook/fine_codebook.npz
```

The communication array dimensions and carrier frequency used during dataset
generation must match the generated codebook.

## Optional coarse codebook

The current SenseComm data-generation pipeline searches only the fine
codebook. A broadened coarse codebook can nevertheless be reproduced with:

```bash
python codebook/generate_codebook.py --mode coarse
```

This creates `codebook/coarse_codebook.npz` with the original quadratic-phase
broadening rule. It is not required for standard dataset generation.

## Windows LLVM setup

Sionna RT uses Mitsuba and Dr.Jit. If Windows reports that `LLVM-C.dll` cannot
be found, set `DRJIT_LIBLLVM_PATH` before running the generator. For a Conda
environment, the DLL is commonly located at:

```powershell
$env:DRJIT_LIBLLVM_PATH = "$env:CONDA_PREFIX\Library\bin\LLVM-C.dll"
python codebook\generate_codebook.py
```

Use the actual DLL location from your environment if it differs.
