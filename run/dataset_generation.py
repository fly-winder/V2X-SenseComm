from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.dataset_pipeline import DatasetPipeline, DatasetPipelineConfig
from utils.trajectory import (
    get_two_way_three_lane_init,
    get_x_axis_six_lane_init,
)
from utils.project_paths import resolve_project_path

from configs.base_config import (
    build_sensing_scene_configs,
    build_comm_scene_configs,
    build_sensors,
    build_comm_pipeline,
    build_occluder_traffic_cfg,
)
from configs.scene_map import SCENE_MAP


# ============================================================
# 实验参数
# ============================================================

SPEED_LIST = [5.0, 15.0, 25.0, -5.0, -15.0, -25.0]

TOTAL_DISTANCE = 4.0
FRAME_INTERVAL_S = 0.1
WEATHER_TYPE = "clear"
OUTPUT_ROOT = resolve_project_path("dataset")


MOTION_AXIS = "x" # y直路；x:拐角
ENABLE_OCCLUDER_CARS = True # 是否要遮挡车辆


# ============================================================
# 直路道路参数
# ============================================================

ROAD_CENTER_X = 30.0
Y_EXTENT = 50.0


# ============================================================
# 拐角/横向道路参数
# ============================================================

ROAD_CENTER_Y = 45.0
X_MIN = 20.0
X_MAX = 120.0


# ============================================================
# 车辆高度
# ============================================================

START_Z = 0.1


# ============================================================
# 全局对象：只构建一次
# ============================================================

sensing_scene_cfg, sensing_array_cfg, sensing_node_cfg, sensing_car_cfg = build_sensing_scene_configs()
comm_scene_cfg, comm_array_cfg, comm_node_cfg, comm_car_cfg = build_comm_scene_configs()

# 通信 pipeline 只构建一次，避免重复加载 codebook
comm_pipeline = build_comm_pipeline()


def get_start_position_and_velocity(speed: float):
    """
    根据当前 MOTION_AXIS 生成目标车初始位置和速度。
    """
    if MOTION_AXIS == "y":
        return get_two_way_three_lane_init(
            speed=speed,
            road_center_x=ROAD_CENTER_X,
            y_extent=Y_EXTENT,
            z=START_Z,
        )

    elif MOTION_AXIS == "x":
        return get_x_axis_six_lane_init(
            speed=speed,
            road_center_y=ROAD_CENTER_Y,
            x_min=X_MIN,
            x_max=X_MAX,
            z=START_Z,
        )

    else:
        raise ValueError(f"Unsupported MOTION_AXIS={MOTION_AXIS}")


def get_car_orientation():
    """
    根据当前 MOTION_AXIS 设置目标车辆朝向。

    - x: 拐角/横向道路，车辆沿 x 方向运动
    - y: 直路场景，车辆沿 y 方向运动
    """
    if MOTION_AXIS == "x":
        return (0.0, 0.0, 0.0)

    if MOTION_AXIS == "y":
        return (np.pi / 2, 0.0, 0.0)

    raise ValueError(f"Unsupported MOTION_AXIS={MOTION_AXIS}")


def build_current_occluder_cfg():
    """
    根据当前 MOTION_AXIS 构建动态遮挡车配置。
    """
    if not ENABLE_OCCLUDER_CARS:
        return None

    if MOTION_AXIS == "y":
        return build_occluder_traffic_cfg(
            motion_axis="y",
            road_center_x=ROAD_CENTER_X,
            y_min=-Y_EXTENT,
            y_max=Y_EXTENT,
        )

    elif MOTION_AXIS == "x":
        return build_occluder_traffic_cfg(
            motion_axis="x",
            road_center_y=ROAD_CENTER_Y,
            x_min=X_MIN,
            x_max=X_MAX,
        )

    else:
        raise ValueError(f"Unsupported MOTION_AXIS={MOTION_AXIS}")


def run_single_experiment(
    scene_id: str,
    scene_source: str | Path,
    speed: float,
):
    """
    执行单个场景 + 单个速度的完整数据生成流程。
    """
    start_position, velocity_vec = get_start_position_and_velocity(speed)
    car_orientation = get_car_orientation()

    occluder_traffic_cfg = build_current_occluder_cfg()

    # car orientation depends on the road motion axis, so keep it in the
    # generation script instead of hard-coding a single global value in base_config.
    sensing_node_cfg.car_orientation = car_orientation
    comm_node_cfg.car_orientation = car_orientation

    dataset_cfg = DatasetPipelineConfig(
        scene_id=scene_id,
        scene_source=scene_source,
        run_id=f"speed_{speed:+.0f}",
        num_frames=None,
        total_distance=TOTAL_DISTANCE,
        frame_interval_s=FRAME_INTERVAL_S,
        start_position=start_position,
        velocity_vec=velocity_vec,
        weather_type=WEATHER_TYPE,
        output_root=OUTPUT_ROOT,

        enable_occluder_cars=ENABLE_OCCLUDER_CARS,
        occluder_traffic_cfg=occluder_traffic_cfg,
    )

    # 每个 run 单独创建传感器，避免状态残留
    camera, lidar, radar, gps = build_sensors()

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

    dataset_pipeline.run()


def main():
    print("=" * 80)
    print("Start dataset generation")
    print("=" * 80)

    for scene_id, scene_source in SCENE_MAP.items():
        scene_source_path = resolve_project_path(scene_source)

        if not scene_source_path.exists():
            print(f"[Skip] scene file not found: {scene_source_path}")
            continue

        print(f"\n[Scene] {scene_id} -> {scene_source_path}")

        for speed in SPEED_LIST:
            print(
                f"[Run] scene={scene_id}, speed={speed:+.0f}, "
                f"motion_axis={MOTION_AXIS}, occluder={ENABLE_OCCLUDER_CARS}"
            )

            run_single_experiment(
                scene_id=scene_id,
                scene_source=scene_source_path,
                speed=speed,
            )

    print("\n" + "=" * 80)
    print("Dataset generation finished")
    print("=" * 80)


if __name__ == "__main__":
    main()
