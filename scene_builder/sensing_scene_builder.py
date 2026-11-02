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
class SensingSceneConfig:
    merge_shapes: bool = False
    base_scene_source: object = sionna.rt.scene.simple_street_canyon


@dataclass
class SensingArrayConfig:
    veh_tx_num_rows: int = 1
    veh_tx_num_cols: int = 1
    veh_tx_pattern: str = "iso"
    veh_tx_polarization: str = "V"

    bs_rx_num_rows: int = 8
    bs_rx_num_cols: int = 16
    bs_rx_vertical_spacing: float = 0.5
    bs_rx_horizontal_spacing: float = 0.5
    bs_rx_pattern: str = "tr38901"
    bs_rx_polarization: str = "V"


@dataclass
class SensingNodeConfig:
    car_name: str = "target_car"
    car_position: tuple = (20.0, -20.0, 0.1)
    car_orientation: tuple = (np.pi/2, 0.0, 0.0)

    veh_sens_tx_name: str = "veh_sens_tx"
    veh_sens_tx_position: tuple = (20.0, -20.0, 0.0)
    veh_sens_tx_power_dbm: float = 1.0
    veh_sens_tx_display_radius: float = 1.0

    bs_sens_rx_name: str = "bs_sens_rx"
    bs_sens_rx_position: tuple = (0.0, 0.0, 10.0)
    bs_sens_rx_orientation: tuple = (0.0, np.pi / 18, 0.0)
    bs_sens_rx_display_radius: float = 1.0


@dataclass
class CarObjectConfig:
    car_mesh_source: object = sionna.rt.scene.low_poly_car
    car_material_name: str = "car-material"
    car_material_type: str = "metal"
    car_material_thickness: float = 0.01
    car_material_color: tuple = (0.8, 0.1, 0.1)


def build_sensing_scene(
    scene_cfg: SensingSceneConfig,
    array_cfg: SensingArrayConfig,
    node_cfg: SensingNodeConfig,
    car_cfg: CarObjectConfig,
    scene_source=None,
):
    """
    构建感知侧场景：
    1. 加载无车基础场景
    2. 手动加入目标车
    3. 配置车载单天线感知发射阵列
    4. 配置基站 8x16 感知接收阵列
    5. 添加车载感知发射机和基站感知接收机
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

    # 2) 车载感知发射阵列：单天线
    scene.tx_array = PlanarArray(
        num_rows=array_cfg.veh_tx_num_rows,
        num_cols=array_cfg.veh_tx_num_cols,
        pattern=array_cfg.veh_tx_pattern,
        polarization=array_cfg.veh_tx_polarization,
    )

    # 3) 基站感知接收阵列：8x16
    scene.rx_array = PlanarArray(
        num_rows=array_cfg.bs_rx_num_rows,
        num_cols=array_cfg.bs_rx_num_cols,
        vertical_spacing=array_cfg.bs_rx_vertical_spacing,
        horizontal_spacing=array_cfg.bs_rx_horizontal_spacing,
        pattern=array_cfg.bs_rx_pattern,
        polarization=array_cfg.bs_rx_polarization,
    )

    # 4) 添加车载感知发射机
    veh_sens_tx = Transmitter(
        name=node_cfg.veh_sens_tx_name,
        position=list(node_cfg.veh_sens_tx_position),
        power_dbm=node_cfg.veh_sens_tx_power_dbm,
        display_radius=node_cfg.veh_sens_tx_display_radius,
    )
    scene.add(veh_sens_tx)

    # 5) 添加基站感知接收机
    bs_sens_rx = Receiver(
        name=node_cfg.bs_sens_rx_name,
        position=list(node_cfg.bs_sens_rx_position),
        orientation=list(node_cfg.bs_sens_rx_orientation),
        display_radius=node_cfg.bs_sens_rx_display_radius,
    )
    scene.add(bs_sens_rx)

    return scene