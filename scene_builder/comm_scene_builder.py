from dataclasses import dataclass
from pathlib import Path
import numpy as np
import sionna.rt
from sionna.rt import (
    load_scene,
    Transmitter,
    Receiver,
    PlanarArray,
    ITURadioMaterial,
    SceneObject,
)


@dataclass
class CommSceneConfig:
    merge_shapes: bool = False
    frequency_hz: float = 7e9
    base_scene_source: object = sionna.rt.scene.simple_street_canyon


@dataclass
class CommArrayConfig:
    bs_tx_num_rows: int = 16
    bs_tx_num_cols: int = 64
    bs_tx_vertical_spacing: float = 0.5
    bs_tx_horizontal_spacing: float = 0.5
    bs_tx_pattern: str = "tr38901"
    bs_tx_polarization: str = "V"

    veh_rx_num_rows: int = 1
    veh_rx_num_cols: int = 1
    veh_rx_pattern: str = "iso"
    veh_rx_polarization: str = "V"


@dataclass
class CommNodeConfig:
    car_name: str = "target_car"
    car_position: tuple = (20.0, -20.0, 0.1)
    car_orientation: tuple = (np.pi/2, 0.0, 0.0)

    bs_comm_tx_name: str = "bs_comm_tx"
    bs_comm_tx_position: tuple = (0.0, 0.0, 10.0)
    bs_comm_tx_orientation: tuple = (0.0, np.pi / 18, 0.0)
    bs_comm_tx_power_dbm: float = 44.0
    bs_comm_tx_display_radius: float = 2.0

    veh_comm_rx_name: str = "veh_comm_rx"
    veh_comm_rx_position: tuple = (20.0, -20.0, 1.5)
    veh_comm_rx_display_radius: float = 2.0


@dataclass
class CarObjectConfig:
    car_mesh_source: object = sionna.rt.scene.low_poly_car
    car_material_name: str = "car-material"
    car_material_type: str = "metal"
    car_material_thickness: float = 0.01
    car_material_color: tuple = (0.8, 0.1, 0.1)


def build_comm_scene(
    scene_cfg: CommSceneConfig,
    array_cfg: CommArrayConfig,
    node_cfg: CommNodeConfig,
    car_cfg: CarObjectConfig,
    scene_source=None,
):
    """
    构建通信侧场景：
    1. 加载无车基础场景
    2. 手动加入目标车
    3. 配置基站 16x64 通信发射阵列
    4. 配置车载 1x1 通信接收阵列
    5. 添加基站通信发射机和车载通信接收机
    """
    actual_scene_source = scene_source if scene_source is not None else scene_cfg.base_scene_source

    if isinstance(actual_scene_source, Path):
        actual_scene_source = str(actual_scene_source)

    scene = load_scene(
        actual_scene_source,
        merge_shapes=scene_cfg.merge_shapes,
    )
    

    # 删除默认设备（如果存在）
    if "tx" in scene.transmitters:
        scene.remove("tx")
    if "rx" in scene.receivers:
        scene.remove("rx")

    # 设置通信频率
    scene.frequency = scene_cfg.frequency_hz

    # 1) 加入车辆对象
    car_material = ITURadioMaterial(
        car_cfg.car_material_name,
        car_cfg.car_material_type,
        thickness=car_cfg.car_material_thickness,
        color=car_cfg.car_material_color,
    )

    car = SceneObject(
        fname=car_cfg.car_mesh_source,
        name=node_cfg.car_name,
        radio_material=car_material,
    )

    scene.edit(add=[car])
    scene.get(node_cfg.car_name).position = list(node_cfg.car_position)
    scene.get(node_cfg.car_name).orientation  = node_cfg.car_orientation 

    # 2) 基站通信发射阵列：16x64
    scene.tx_array = PlanarArray(
        num_rows=array_cfg.bs_tx_num_rows,
        num_cols=array_cfg.bs_tx_num_cols,
        vertical_spacing=array_cfg.bs_tx_vertical_spacing,
        horizontal_spacing=array_cfg.bs_tx_horizontal_spacing,
        pattern=array_cfg.bs_tx_pattern,
        polarization=array_cfg.bs_tx_polarization,
    )

    # 3) 车载通信接收阵列：1x1
    scene.rx_array = PlanarArray(
        num_rows=array_cfg.veh_rx_num_rows,
        num_cols=array_cfg.veh_rx_num_cols,
        pattern=array_cfg.veh_rx_pattern,
        polarization=array_cfg.veh_rx_polarization,
    )

    # 4) 添加基站通信发射机
    bs_comm_tx = Transmitter(
        name=node_cfg.bs_comm_tx_name,
        position=list(node_cfg.bs_comm_tx_position),
        orientation=list(node_cfg.bs_comm_tx_orientation),
        power_dbm=node_cfg.bs_comm_tx_power_dbm,
        display_radius=node_cfg.bs_comm_tx_display_radius,
    )
    scene.add(bs_comm_tx)

    # 5) 添加车载通信接收机
    veh_comm_rx = Receiver(
        name=node_cfg.veh_comm_rx_name,
        position=list(node_cfg.veh_comm_rx_position),
        display_radius=node_cfg.veh_comm_rx_display_radius,
    )
    scene.add(veh_comm_rx)

    return scene