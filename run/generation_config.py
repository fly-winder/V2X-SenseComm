from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from configs.scene_map import SCENE_MAP
from utils.project_paths import resolve_project_path


DEFAULT_CONFIG: dict[str, Any] = {
    "run": {
        "scene_ids": ["scene_12"],
        "speed_list": [25.0],
        "total_distance": 100.0,
        "frame_interval_s": 0.1,
        "weather_type": "clear",
        "output_root": "dataset",
        "motion_axis": "y",
        "enable_occluder_cars": False,
    },
    "traffic": {
        "road_center_x": 30.0,
        "y_extent": 50.0,
        "road_center_y": 45.0,
        "x_min": 20.0,
        "x_max": 120.0,
    },
    "sensors": {
        "camera": {
            "fov": 120.0,
            "resolution": [512, 512],
            "num_samples": 128,
        },
        "lidar": {
            "num_channels": 128,
            "h_resolution": 256,
            "max_range": 150.0,
        },
        "radar": {
            "array_rows": 4,
            "array_cols": 8,
            "carrier_frequency": 77e9,
            "bandwidth": 200e6,
            "num_chirps": 8,
            "num_samples": 256,
            "noise_snr_db": None,
        },
        "gps": {
            "noise_std": [0.5, 0.5, 0.5],
        },
    },
    "communication": {
        "carrier_frequency": 7e9,
        "bs_tx_num_rows": 16,
        "bs_tx_num_cols": 64,
        "num_subcarriers": 128,
        "codebook_file": "codebook/fine_codebook.npz",
        "top_k": 5,
    },
}

WEATHER_TYPES = ("clear", "sunny", "cloudy", "night")


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_generation_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    with open(path, "r", encoding="utf-8") as f:
        user_config = yaml.safe_load(f) or {}
    config = deep_merge(DEFAULT_CONFIG, user_config)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    run_cfg = config["run"]
    if not run_cfg["scene_ids"]:
        raise ValueError("run.scene_ids must contain at least one scene id")
    if not run_cfg["speed_list"]:
        raise ValueError("run.speed_list must contain at least one speed")
    if any(float(speed) == 0.0 for speed in run_cfg["speed_list"]):
        raise ValueError("run.speed_list cannot contain zero speed")
    if float(run_cfg["total_distance"]) <= 0:
        raise ValueError("run.total_distance must be positive")
    if float(run_cfg["frame_interval_s"]) <= 0:
        raise ValueError("run.frame_interval_s must be positive")
    if run_cfg["weather_type"] not in WEATHER_TYPES:
        raise ValueError(f"run.weather_type must be one of {WEATHER_TYPES}")
    if run_cfg["motion_axis"] not in ("x", "y"):
        raise ValueError("run.motion_axis must be 'x' or 'y'")

    radar_cfg = config["sensors"]["radar"]
    if int(radar_cfg["array_rows"]) <= 0 or int(radar_cfg["array_cols"]) <= 0:
        raise ValueError("radar.array_rows and radar.array_cols must be positive")
    if radar_cfg.get("noise_snr_db") is not None:
        float(radar_cfg["noise_snr_db"])

    comm_cfg = config["communication"]
    if int(comm_cfg["bs_tx_num_rows"]) <= 0 or int(comm_cfg["bs_tx_num_cols"]) <= 0:
        raise ValueError("communication BS array dimensions must be positive")
    if int(comm_cfg["num_subcarriers"]) <= 0:
        raise ValueError("communication.num_subcarriers must be positive")


def discover_scene_map() -> dict[str, Path]:
    scene_map: dict[str, Path] = {
        scene_id: Path(scene_source)
        for scene_id, scene_source in SCENE_MAP.items()
    }

    maps_root = resolve_project_path("maps")
    for scene_dir in sorted(maps_root.glob("scene*")):
        if not scene_dir.is_dir():
            continue
        digits = "".join(ch for ch in scene_dir.name if ch.isdigit())
        if not digits:
            continue
        scene_id = f"scene_{int(digits):02d}"
        xml_files = sorted(scene_dir.glob("*.xml"))
        if xml_files and scene_id not in scene_map:
            scene_map[scene_id] = xml_files[0]

    return scene_map
