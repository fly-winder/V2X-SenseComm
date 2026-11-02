# V2X-SenseComm

V2X-SenseComm is a configurable platform for generating synchronized V2X
sensing and communication data with Sionna RT. It combines camera, LiDAR,
radar, noisy position measurements, per-element near-field channels, and
near-field beam labels in dynamic urban traffic scenes.

- Project website: <https://fly-winder.github.io/SenseComm/>
- Dataset and generation assets: <https://huggingface.co/datasets/njl123/V2X-SenseComm>

![V2X-SenseComm generation workflow](https://fly-winder.github.io/SenseComm/assets/figures/tutorials/v2x-sensecomm-generation-workflow.png)

## Features

- Synchronized camera, LiDAR, radar, position, channel, and beam-label data.
- Independent sensing and communication scene configurations.
- Per-element ray tracing for XL-MIMO near-field channels.
- Configurable arrays, carrier frequency, subcarriers, sensors, trajectories,
  traffic, weather, and output directories.
- Basic and Advanced Web UI modes with generated-data previews.
- YAML-based command-line generation for server and cluster execution.
- Reproducible 4D near-field codebook generation.

## Repository structure

```text
app/             Streamlit Web UI and example configuration
communication/   Channel simulation, codebook loading, and beam labeling
configs/         Reference sensing and communication configurations
codebook/        Reproducible near-field codebook generator
maps/            Sionna RT XML scenes and PLY meshes
pipeline/        Synchronized dataset and communication pipelines
run/             Web-UI and command-line generation entry points
scene_builder/   Sensing, communication, and traffic scene construction
sensors/         Camera, LiDAR, radar, and position simulators
utils/           Project path and trajectory utilities
```

## Installation

The released code was tested with Python 3.12, Sionna 1.2.0, TensorFlow 2.20.0,
Mitsuba 3.7.1, and Dr.Jit 1.2.0.

```bash
git clone https://github.com/fly-winder/V2X-SenseComm.git
cd V2X-SenseComm

conda create -n sensecomm python=3.12 -y
conda activate sensecomm
python -m pip install -r requirements.txt
```

A CUDA-capable Linux environment is recommended for large-scale generation.
CPU execution can be substantially slower.

### Windows LLVM setup

If Dr.Jit reports that `LLVM-C.dll` cannot be found, set the library path after
activating the Conda environment:

```powershell
$env:DRJIT_LIBLLVM_PATH = "$env:CONDA_PREFIX\Library\bin\LLVM-C.dll"
```

Use the actual location of `LLVM-C.dll` if it differs in your environment.

## Required generation assets

Large binary assets are hosted on Hugging Face rather than GitHub:

- [Fine near-field codebook](https://huggingface.co/datasets/njl123/V2X-SenseComm/resolve/main/generation_assets/codebook/fine_codebook.npz?download=true)
- [Sunny environment map](https://huggingface.co/datasets/njl123/V2X-SenseComm/resolve/main/generation_assets/textures/sunny.exr?download=true)
- [Cloudy environment map](https://huggingface.co/datasets/njl123/V2X-SenseComm/resolve/main/generation_assets/textures/cloudy.exr?download=true)
- [Night environment map](https://huggingface.co/datasets/njl123/V2X-SenseComm/resolve/main/generation_assets/textures/night.exr?download=true)

Download and place them at the following exact paths:

```text
V2X-SenseComm/
|-- codebook/
|   `-- fine_codebook.npz
`-- textures/
    |-- sunny.exr
    |-- cloudy.exr
    `-- night.exr
```

The EXR files are needed only when `sunny`, `cloudy`, or `night` rendering is
selected. The default `clear` mode does not use an environment map. The fine
codebook is required for beam-label generation.

Alternatively, reproduce the fine codebook locally:

```bash
python codebook/generate_codebook.py
```

See [codebook/README.md](codebook/README.md) for the physical definition,
default grid, output format, and custom generation options.

## Web interface

Start the Streamlit application from the project root:

```bash
python -m streamlit run app/generator_app.py
```

Open <http://localhost:8501> in a browser. The interface provides:

- **Basic Mode** for scene, speed, weather, trajectory, and output settings.
- **Advanced Mode** for traffic geometry, individual sensor parameters,
  communication arrays, carrier frequency, subcarriers, and codebook settings.
- A synchronized preview of generated camera, LiDAR, radar, channel, position,
  and beam-label samples.

Click **Run Generation** to save the active configuration and start generation.
The UI writes its runtime configuration to:

```text
app/generated/generation_config.yaml
```

## Command-line generation

Copy or edit the included configuration template:

```text
app/example_generation_config.yaml
```

Then run:

```bash
python run/run_from_config.py --config app/example_generation_config.yaml
```

This is the recommended entry point for servers and computing clusters. The
hard-coded batch example in `run/dataset_generation.py` is retained for users
who prefer configuring experiments directly in Python.

## Main configuration groups

| Group | Examples |
|---|---|
| `run` | scenes, speed list, distance, frame interval, weather, output root |
| `traffic` | road center, road extent, occluding vehicles |
| `sensors.camera` | field of view, resolution, render samples |
| `sensors.lidar` | channels, horizontal resolution, maximum range |
| `sensors.radar` | array size, carrier, bandwidth, chirps, samples, noise SNR |
| `sensors.gps` | three-dimensional position-noise standard deviation |
| `communication` | carrier, BS array size, subcarriers, codebook, top-k labels |

Scene XML files are discovered automatically from `maps/scene*/`. When adding a
new scene, keep its XML and referenced `meshes/*.ply` files in the same scene
directory.

## Output organization

Generated data are organized by scene, run, and synchronized frame:

```text
dataset/
|-- index.csv
`-- scene_01/
    |-- index.csv
    `-- speed_+25/
        |-- meta.json
        |-- index.csv
        |-- camera/frame_000000.png
        |-- lidar/frame_000000.npz
        |-- radar/frame_000000.npz
        |-- gps/frame_000000.npz
        |-- channel/frame_000000.npz
        `-- beam_label/frame_000000.json
```

Each run-level `meta.json` records the scene, trajectory, frame interval,
velocity, weather, and traffic settings. The CSV indices link all synchronized
modalities and labels.

## Released dataset

The generated SenseComm datasets, scene archives, indices, and generation
assets are available at:

<https://huggingface.co/datasets/njl123/V2X-SenseComm>

The dataset repository includes the standard release and the narrowband 100 Hz
release. Dataset files are not stored in this source-code repository.

## Citation

If you use this project, please cite:

> SenseComm: A Multimodal V2X Dataset for XL-MIMO Near-Field Sensing and Communications.

The full BibTeX entry will be added after publication.

## License

The source code is released under the Apache License 2.0. Scene maps and other
third-party assets remain subject to their original licenses and attribution
requirements.
