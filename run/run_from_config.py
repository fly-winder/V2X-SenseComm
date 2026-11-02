from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from communication.beam_labeler import BeamLabelConfig
from communication.channel_simulator import ChannelConfig
from communication.codebook_loader import CodebookConfig
from configs import base_config
from pipeline.comm_pipeline import CommunicationPipeline, CommunicationPipelineConfig
from pipeline.dataset_pipeline import DatasetPipeline, DatasetPipelineConfig
from scene_builder.comm_scene_builder import (
    CarObjectConfig as CommCarObjectConfig,
    CommArrayConfig,
    CommNodeConfig,
    CommSceneConfig,
)
from scene_builder.sensing_scene_builder import CarObjectConfig, SensingArrayConfig, SensingNodeConfig, SensingSceneConfig
from sensors.camera_simulator import CameraConfig, CameraSimulator
from sensors.gps_simulator import GPSConfig, GPSSimulator
from sensors.lidar_simulator import LidarConfig, LidarSimulator
from sensors.radar_simulator import RadarConfig, RadarSimulator
from run.generation_config import discover_scene_map, load_generation_config
from utils.project_paths import resolve_project_path
from utils.trajectory import get_two_way_three_lane_init, get_x_axis_six_lane_init


START_Z = 0.1
CHANNEL_BANDWIDTH_HZ = 18.36e6


def get_start_position_and_velocity(speed: float, config: dict[str, Any]):
    run_cfg = config["run"]
    traffic_cfg = config["traffic"]
    motion_axis = run_cfg["motion_axis"]

    if motion_axis == "y":
        return get_two_way_three_lane_init(
            speed=speed,
            road_center_x=float(traffic_cfg["road_center_x"]),
            y_extent=float(traffic_cfg["y_extent"]),
            z=START_Z,
        )

    return get_x_axis_six_lane_init(
        speed=speed,
        road_center_y=float(traffic_cfg["road_center_y"]),
        x_min=float(traffic_cfg["x_min"]),
        x_max=float(traffic_cfg["x_max"]),
        z=START_Z,
    )


def get_car_orientation(motion_axis: str):
    if motion_axis == "x":
        return (0.0, 0.0, 0.0)
    if motion_axis == "y":
        return (np.pi / 2, 0.0, 0.0)
    raise ValueError(f"Unsupported motion_axis={motion_axis}")


def build_occluder_cfg(config: dict[str, Any]):
    run_cfg = config["run"]
    traffic_cfg = config["traffic"]
    if not bool(run_cfg["enable_occluder_cars"]):
        return None

    if run_cfg["motion_axis"] == "y":
        return base_config.build_occluder_traffic_cfg(
            motion_axis="y",
            road_center_x=float(traffic_cfg["road_center_x"]),
            y_min=-float(traffic_cfg["y_extent"]),
            y_max=float(traffic_cfg["y_extent"]),
        )

    return base_config.build_occluder_traffic_cfg(
        motion_axis="x",
        road_center_y=float(traffic_cfg["road_center_y"]),
        x_min=float(traffic_cfg["x_min"]),
        x_max=float(traffic_cfg["x_max"]),
    )


def build_sensing_configs(config: dict[str, Any]):
    radar_cfg = config["sensors"]["radar"]

    scene_params = copy.deepcopy(base_config.SENSING_SCENE_PARAMS)
    array_params = copy.deepcopy(base_config.SENSING_ARRAY_PARAMS)
    node_params = copy.deepcopy(base_config.SENSING_NODE_PARAMS)
    car_params = copy.deepcopy(base_config.SENSING_CAR_OBJECT_PARAMS)

    array_params["bs_rx_num_rows"] = int(radar_cfg["array_rows"])
    array_params["bs_rx_num_cols"] = int(radar_cfg["array_cols"])

    car_params.pop("car_mesh_source", None)
    return (
        SensingSceneConfig(**scene_params),
        SensingArrayConfig(**array_params),
        SensingNodeConfig(**node_params),
        CarObjectConfig(**car_params),
    )


def build_comm_configs(config: dict[str, Any]):
    comm_cfg = config["communication"]

    scene_params = copy.deepcopy(base_config.COMM_SCENE_PARAMS)
    array_params = copy.deepcopy(base_config.COMM_ARRAY_PARAMS)
    node_params = copy.deepcopy(base_config.COMM_NODE_PARAMS)
    car_params = copy.deepcopy(base_config.COMM_CAR_OBJECT_PARAMS)

    scene_params["frequency_hz"] = float(comm_cfg["carrier_frequency"])
    array_params["bs_tx_num_rows"] = int(comm_cfg["bs_tx_num_rows"])
    array_params["bs_tx_num_cols"] = int(comm_cfg["bs_tx_num_cols"])

    car_params.pop("car_mesh_source", None)
    return (
        CommSceneConfig(**scene_params),
        CommArrayConfig(**array_params),
        CommNodeConfig(**node_params),
        CommCarObjectConfig(**car_params),
    )


def build_sensors(config: dict[str, Any]):
    sensor_cfg = config["sensors"]

    camera_params = copy.deepcopy(base_config.CAMERA_PARAMS)
    camera_params.update(sensor_cfg["camera"])
    camera_params["resolution"] = tuple(camera_params["resolution"])

    lidar_params = copy.deepcopy(base_config.LIDAR_PARAMS)
    lidar_params.update(sensor_cfg["lidar"])

    radar_params = copy.deepcopy(base_config.RADAR_PARAMS)
    radar_params.update({
        key: value
        for key, value in sensor_cfg["radar"].items()
        if key not in ("array_rows", "array_cols")
    })

    gps_params = copy.deepcopy(base_config.GPS_PARAMS)
    gps_params.update(sensor_cfg["gps"])
    gps_params["noise_std"] = tuple(gps_params["noise_std"])

    return (
        CameraSimulator(CameraConfig(**camera_params)),
        LidarSimulator(LidarConfig(**lidar_params)),
        RadarSimulator(RadarConfig(**radar_params)),
        GPSSimulator(GPSConfig(**gps_params)),
    )


def build_comm_pipeline(config: dict[str, Any]) -> CommunicationPipeline:
    comm_cfg = config["communication"]

    channel_params = copy.deepcopy(base_config.CHANNEL_PARAMS)
    channel_params["carrier_frequency"] = float(comm_cfg["carrier_frequency"])
    half_bw = CHANNEL_BANDWIDTH_HZ / 2.0
    channel_params["frequencies"] = tuple(
        np.linspace(-half_bw, half_bw, int(comm_cfg["num_subcarriers"]))
    )

    codebook_params = copy.deepcopy(base_config.CODEBOOK_PARAMS)
    codebook_params["file"] = resolve_project_path(str(comm_cfg["codebook_file"]))

    beam_label_params = copy.deepcopy(base_config.BEAM_LABEL_PARAMS)
    beam_label_params["top_k"] = int(comm_cfg["top_k"])

    pipeline_cfg = CommunicationPipelineConfig(
        channel_cfg=ChannelConfig(**channel_params),
        codebook_cfg=CodebookConfig(**codebook_params),
        beam_label_cfg=BeamLabelConfig(**beam_label_params),
        label_output_dir=base_config.COMM_PIPELINE_PARAMS["label_output_dir"],
    )
    return CommunicationPipeline(pipeline_cfg)


def run_generation(config: dict[str, Any]) -> None:
    scene_map = discover_scene_map()
    run_cfg = config["run"]

    comm_pipeline = build_comm_pipeline(config)

    for scene_id in run_cfg["scene_ids"]:
        if scene_id not in scene_map:
            raise ValueError(f"Unknown scene_id={scene_id}. Available: {sorted(scene_map)}")

        scene_source_path = resolve_project_path(scene_map[scene_id])
        if not scene_source_path.exists():
            print(f"[Skip] scene file not found: {scene_source_path}")
            continue

        print(f"\n[Scene] {scene_id} -> {scene_source_path}")

        for speed in run_cfg["speed_list"]:
            speed = float(speed)
            start_position, velocity_vec = get_start_position_and_velocity(speed, config)
            car_orientation = get_car_orientation(run_cfg["motion_axis"])

            sensing_scene_cfg, sensing_array_cfg, sensing_node_cfg, sensing_car_cfg = build_sensing_configs(config)
            comm_scene_cfg, comm_array_cfg, comm_node_cfg, comm_car_cfg = build_comm_configs(config)

            sensing_node_cfg.car_orientation = car_orientation
            comm_node_cfg.car_orientation = car_orientation

            dataset_cfg = DatasetPipelineConfig(
                scene_id=scene_id,
                scene_source=scene_source_path,
                run_id=f"speed_{speed:+.0f}",
                num_frames=None,
                total_distance=float(run_cfg["total_distance"]),
                frame_interval_s=float(run_cfg["frame_interval_s"]),
                start_position=start_position,
                velocity_vec=velocity_vec,
                weather_type=str(run_cfg["weather_type"]),
                output_root=str(run_cfg["output_root"]),
                enable_occluder_cars=bool(run_cfg["enable_occluder_cars"]),
                occluder_traffic_cfg=build_occluder_cfg(config),
            )

            camera, lidar, radar, gps = build_sensors(config)
            dataset_pipeline = DatasetPipeline(
                cfg=dataset_cfg,
                sensing_scene_cfg=sensing_scene_cfg,
                sensing_array_cfg=sensing_array_cfg,
                sensing_node_cfg=sensing_node_cfg,
                sensing_car_cfg=sensing_car_cfg,
                comm_scene_cfg=comm_scene_cfg,
                comm_array_cfg=comm_array_cfg,
                comm_node_cfg=comm_node_cfg,
                comm_car_cfg=comm_car_cfg,
                camera_simulator=camera,
                lidar_simulator=lidar,
                radar_simulator=radar,
                gps_simulator=gps,
                communication_pipeline=comm_pipeline,
            )

            print(
                f"[Run] scene={scene_id}, speed={speed:+.0f}, "
                f"motion_axis={run_cfg['motion_axis']}, "
                f"weather={run_cfg['weather_type']}, "
                f"occluder={run_cfg['enable_occluder_cars']}"
            )
            dataset_pipeline.run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate V2X-SenseComm data from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to generation_config.yaml")
    args = parser.parse_args()

    config = load_generation_config(args.config)
    print("=" * 80)
    print("Start config-driven dataset generation")
    print(f"Python executable: {sys.executable}")
    print(f"Working directory: {Path.cwd()}")
    print(f"Config path: {Path(args.config).resolve()}")
    print(f"CONDA_DEFAULT_ENV: {os.environ.get('CONDA_DEFAULT_ENV')}")
    print(f"Scene IDs: {config['run']['scene_ids']}")
    print(f"Speed list: {config['run']['speed_list']}")
    print(f"Motion axis: {config['run']['motion_axis']}")
    print(f"Occluder cars: {config['run']['enable_occluder_cars']}")
    print("=" * 80)
    run_generation(config)
    print("\n" + "=" * 80)
    print("Dataset generation finished")
    print("=" * 80)


if __name__ == "__main__":
    main()
