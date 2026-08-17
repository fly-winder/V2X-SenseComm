import numpy as np
from scene_builder.sensing_scene_builder import (
    SensingSceneConfig,
    SensingArrayConfig,
    SensingNodeConfig,
    CarObjectConfig,
)
from scene_builder.comm_scene_builder import (
    CommSceneConfig,
    CommArrayConfig,
    CommNodeConfig,
    CarObjectConfig as CommCarObjectConfig,
)
from sensors.camera_simulator import CameraConfig, CameraSimulator
from sensors.lidar_simulator import LidarConfig, LidarSimulator
from sensors.radar_simulator import RadarConfig, RadarSimulator
from sensors.gps_simulator import GPSConfig, GPSSimulator
from pipeline.comm_pipeline import CommunicationPipeline, CommunicationPipelineConfig
from communication.channel_simulator import ChannelConfig
from communication.codebook_loader import CodebookConfig
from communication.beam_labeler import BeamLabelConfig
from utils.project_paths import resolve_project_path
from scene_builder.traffic_builder import OccluderTrafficConfig


# ============================================================
# 场景构建参数（Sensing / Communication）
# ============================================================

SENSING_SCENE_PARAMS = dict(
    merge_shapes=False
    # 这里不写地图，地图由 DatasetPipelineConfig.scene_source 传入
)

SENSING_ARRAY_PARAMS = dict(
    veh_tx_num_rows=1,
    veh_tx_num_cols=1,
    veh_tx_pattern="iso",
    veh_tx_polarization="V",

    bs_rx_num_rows=4,
    bs_rx_num_cols=8,
    bs_rx_vertical_spacing=0.5,
    bs_rx_horizontal_spacing=0.5,
    bs_rx_pattern="iso",
    bs_rx_polarization="V",
)

SENSING_NODE_PARAMS = dict(
    car_name="target_car",
    car_position=(20.0, -20.0, 0.1),
    car_orientation=(np.pi/2, 0.0, 0.0),

    veh_sens_tx_name="veh_sens_tx",
    veh_sens_tx_position=(20.0, -20.0, 1.5),
    veh_sens_tx_power_dbm=1.0,
    veh_sens_tx_display_radius=1.0,

    bs_sens_rx_name="bs_sens_rx",
    bs_sens_rx_position=(0.0, 0.0, 9.0),
    bs_sens_rx_orientation=(0.0, np.pi / 18, 0.0),
    bs_sens_rx_display_radius=1.0,
)

SENSING_CAR_OBJECT_PARAMS = dict(
    car_material_name="car-material",
    car_material_type="metal",
    car_material_thickness=0.01,
    car_material_color=(0.8, 0.1, 0.1),
)


COMM_SCENE_PARAMS = dict(
    merge_shapes=False,
    frequency_hz=7e9,
    # 这里不写地图，地图由 DatasetPipelineConfig.scene_source 传入
)

COMM_ARRAY_PARAMS = dict(
    bs_tx_num_rows=16,
    bs_tx_num_cols=64,
    bs_tx_vertical_spacing=0.5,
    bs_tx_horizontal_spacing=0.5,
    bs_tx_pattern="iso",
    bs_tx_polarization="V",

    veh_rx_num_rows=1,
    veh_rx_num_cols=1,
    veh_rx_pattern="iso",
    veh_rx_polarization="V",
)

COMM_NODE_PARAMS = dict(
    car_name="target_car",
    car_position=(20.0, -20.0, 0.1),
    car_orientation=(np.pi / 2, 0.0, 0.0),

    bs_comm_tx_name="bs_comm_tx",
    bs_comm_tx_position=(0.0, 0.0, 10.0),
    bs_comm_tx_orientation=(0.0, np.pi / 18, 0.0),
    bs_comm_tx_power_dbm=44.0,
    bs_comm_tx_display_radius=2.0,

    veh_comm_rx_name="veh_comm_rx",
    veh_comm_rx_position=(20.0, -20.0, 1.5),
    veh_comm_rx_display_radius=0.5,
)

COMM_CAR_OBJECT_PARAMS = dict(
    car_material_name="car-material",
    car_material_type="metal",
    car_material_thickness=0.01,
    car_material_color=(0.8, 0.1, 0.1),
)


# ============================================================
# 传感器参数
# ============================================================

CAMERA_PARAMS = dict(
    sensor_id="camera_0",
    position=[0, 0, 10],
    look_at=[100, 0, 0],
    fov=120.0,
    resolution=(512, 512),
    num_samples=128,
    output_dir="outputs/camera",
)

LIDAR_PARAMS = dict(
    sensor_id="lidar_0",
    position=(0.0, 0.0, 10.0),
    num_channels=128,
    h_resolution=256,
    v_fov=(-30.0, 10.0),
    max_range=150.0,
    output_dir="outputs/lidar",
)

RADAR_PARAMS = dict(
    sensor_id="radar_0",
    carrier_frequency=77e9,
    bandwidth=200e6,
    num_samples=256,
    num_chirps=8,
    t_pri=40e-6,
    interval_s=0.1,
    seed=41,
    noise_snr_db=10.0,
    output_dir="outputs/radar",
)

GPS_PARAMS = dict(
    sensor_id="gps_0",
    target_name="target_car",
    noise_std=(0.5, 0.5, 0.5),
    seed=42,
    output_dir="outputs/gps",
)


# ============================================================
# 通信参数
# ============================================================

COMM_PIPELINE_PARAMS = dict(
    label_output_dir="outputs/beam_label",
)

CHANNEL_PARAMS = dict(
    sensor_id="comm_channel_0",
    carrier_frequency=7e9,
    max_depth=3,
    samples_per_src=10000,
    max_num_paths_per_src=8000,
    los=True,
    specular_reflection=True,
    diffraction=False,
    diffuse_reflection=False,
    refraction=True,
    synthetic_array=False,
    seed=41,
    frequencies=tuple(np.linspace(-9.18e6, 9.18e6, 128)), 
    output_dir="outputs/channel",
)

CODEBOOK_PARAMS = dict(
    file=resolve_project_path("codebook/fine_codebook.npz"),
    codebook_key="codebook",
    r_axis_key="r_axis",
    az_axis_key="az_axis",
    el_axis_key="el_axis",
)

BEAM_LABEL_PARAMS = dict(
    top_k=5,

    angle_peak_rel_db_threshold=3.0,
    angle_peak_nms_radius=(5, 5),
    max_num_angle_peaks=None,
)



# ============================================================
# 动态遮挡车参数
# ============================================================

OCCLUDER_TRAFFIC_PARAMS = dict(
    prefix="occluder_car",

    # 运动方向:
    #   "y": 直路场景，车辆沿 y 轴运动，车道沿 x 轴展开
    #   "x": 拐角场景，车辆沿 x 轴运动，车道沿 y 轴展开
    motion_axis="y",

    # ===== 直路场景参数 =====
    # 沿 y 轴运动，车道中心沿 x 轴展开
    road_center_x=30.0,
    y_min=-50.0,
    y_max=50.0,

    # ===== 拐角/横向道路场景参数 =====
    # 沿 x 轴运动，车道中心沿 y 轴展开
    road_center_y=45.0,
    x_min=20.0,
    x_max=120.0,

    # 车辆高度
    z=0.1,

    # 每个非目标车道放置的遮挡车数量
    cars_per_lane=1,

    # 遮挡车尺寸随机范围
    scaling_min=0.7,
    scaling_max=2.0,

    # 遮挡车速度扰动范围
    speed_jitter_min=-2.0,
    speed_jitter_max=2.0,

    # None 表示每次运行随机不同；整数表示可复现
    seed=None,

    # 材料设置
    material_name="occluder-car-material",
    material_type="metal",
    material_thickness=0.01, 
    material_color=(0.1, 0.1, 0.8),
)


# ============================================================
# Scene config builders
# ============================================================

def build_sensing_scene_configs():
    sensing_scene_cfg = SensingSceneConfig(**SENSING_SCENE_PARAMS)
    sensing_array_cfg = SensingArrayConfig(**SENSING_ARRAY_PARAMS)
    sensing_node_cfg = SensingNodeConfig(**SENSING_NODE_PARAMS)

    # car_mesh_source 若传 None，会覆盖默认值，所以这里特殊处理
    car_params = dict(SENSING_CAR_OBJECT_PARAMS)
    if car_params.get("car_mesh_source", None) is None:
        car_params.pop("car_mesh_source", None)
    sensing_car_cfg = CarObjectConfig(**car_params)

    return (
        sensing_scene_cfg,
        sensing_array_cfg,
        sensing_node_cfg,
        sensing_car_cfg,
    )


def build_comm_scene_configs():
    comm_scene_cfg = CommSceneConfig(**COMM_SCENE_PARAMS)
    comm_array_cfg = CommArrayConfig(**COMM_ARRAY_PARAMS)
    comm_node_cfg = CommNodeConfig(**COMM_NODE_PARAMS)

    car_params = dict(COMM_CAR_OBJECT_PARAMS)
    if car_params.get("car_mesh_source", None) is None:
        car_params.pop("car_mesh_source", None)
    comm_car_cfg = CommCarObjectConfig(**car_params)

    return (
        comm_scene_cfg,
        comm_array_cfg,
        comm_node_cfg,
        comm_car_cfg,
    )


# ============================================================
# Sensor builders
# ============================================================

def build_camera():
    return CameraSimulator(CameraConfig(**CAMERA_PARAMS))


def build_lidar():
    return LidarSimulator(LidarConfig(**LIDAR_PARAMS))


def build_radar():
    return RadarSimulator(RadarConfig(**RADAR_PARAMS))


def build_gps():
    return GPSSimulator(GPSConfig(**GPS_PARAMS))


def build_sensors():
    camera = build_camera()
    lidar = build_lidar()
    radar = build_radar()
    gps = build_gps()
    return camera, lidar, radar, gps


# ============================================================
# Communication pipeline builder
# ============================================================

def build_comm_pipeline():
    channel_cfg = ChannelConfig(**CHANNEL_PARAMS)
    codebook_cfg = CodebookConfig(**CODEBOOK_PARAMS)
    beam_label_cfg = BeamLabelConfig(**BEAM_LABEL_PARAMS)

    comm_pipeline_cfg = CommunicationPipelineConfig(
        channel_cfg=channel_cfg,
        codebook_cfg=codebook_cfg,
        beam_label_cfg=beam_label_cfg,
        label_output_dir=COMM_PIPELINE_PARAMS["label_output_dir"],
    )

    return CommunicationPipeline(comm_pipeline_cfg)


# ============================================================
# Occluder traffic builder
# ============================================================

def build_occluder_traffic_cfg(
    motion_axis: str = "y",
    seed: int | None = None,
    **overrides,
):
    """
    构建动态遮挡车配置。

    Parameters
    ----------
    motion_axis : str
        "y": 直路场景，车辆沿 y 轴运动。
        "x": 拐角/横向道路场景，车辆沿 x 轴运动。
    seed : int or None
        None 表示每次随机不同；整数表示固定随机种子。
    **overrides :
        可选覆盖参数，例如：
        build_occluder_traffic_cfg(
            motion_axis="x",
            road_center_y=45.0,
            x_min=20.0,
            x_max=120.0,
        )

    Returns
    -------
    OccluderTrafficConfig
    """
    if motion_axis not in ("x", "y"):
        raise ValueError(
            f"Unsupported motion_axis={motion_axis}. Expected 'x' or 'y'."
        )

    params = dict(OCCLUDER_TRAFFIC_PARAMS)
    params["motion_axis"] = motion_axis

    # 如果显式传 seed，则覆盖；否则使用 OCCLUDER_TRAFFIC_PARAMS 中的 seed
    if seed is not None:
        params["seed"] = seed

    # 允许在调用时覆盖任意参数
    params.update(overrides)

    return OccluderTrafficConfig(**params)