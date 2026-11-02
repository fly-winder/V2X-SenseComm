from dataclasses import dataclass
import numpy as np
import sionna.rt
from sionna.rt import ITURadioMaterial, SceneObject
from utils.trajectory import get_lane_key_from_speed


@dataclass
class OccluderTrafficConfig:
    """
    双向六车道遮挡车配置。

    motion_axis:
        - "y": 直路场景，车辆沿 y 轴运动，车道沿 x 轴展开
        - "x": 拐角/横向道路场景，车辆沿 x 轴运动，车道沿 y 轴展开
    """
    prefix: str = "occluder_car"

    # 运动方向
    motion_axis: str = "y"

    # ===== 直路场景参数：沿 y 轴运动，车道中心沿 x 轴展开 =====
    road_center_x: float = 30.0
    y_min: float = -50.0
    y_max: float = 50.0

    # ===== 拐角场景参数：沿 x 轴运动，车道中心沿 y 轴展开 =====
    road_center_y: float = 45.0
    x_min: float = 20.0
    x_max: float = 120.0

    # 车辆高度
    z: float = 0.1

    # 每个非目标车道的遮挡车辆数
    cars_per_lane: int = 1

    # 车辆尺寸随机范围
    scaling_min: float = 1.2
    scaling_max: float = 4.0

    # 速度扰动范围
    speed_jitter_min: float = -2.0
    speed_jitter_max: float = 2.0

    # None 表示每次随机不同；整数表示可复现
    seed: int | None = None

    # 材料
    material_name: str = "occluder-car-material"
    material_type: str = "metal"
    material_thickness: float = 0.01
    material_color: tuple = (0.2, 0.3, 0.8)


def get_lane_offset_map():
    """
    六车道相对道路中心线的偏移。

    正方向:
        +25: 靠中心快车道
        +15: 中间车道
        +5 : 外侧慢车道

    反方向:
        -25: 靠中心快车道
        -15: 中间车道
        -5 : 外侧慢车道
    """
    return {
        25.0: 1.75,
        15.0: 5.25,
        5.0: 8.75,

        -25.0: -1.75,
        -15.0: -5.25,
        -5.0: -8.75,
    }


def get_lane_coord_map(cfg: OccluderTrafficConfig):
    """
    根据 motion_axis 返回每个基准速度对应的车道中心坐标。

    Returns
    -------
    dict
        motion_axis == "y":
            {lane_speed: lane_x}
        motion_axis == "x":
            {lane_speed: lane_y}
    """
    lane_offset_map = get_lane_offset_map()

    if cfg.motion_axis == "y":
        center = cfg.road_center_x
    elif cfg.motion_axis == "x":
        center = cfg.road_center_y
    else:
        raise ValueError(
            f"Unsupported motion_axis={cfg.motion_axis}. "
            f"Expected 'x' or 'y'."
        )

    return {
        lane_speed: center + offset
        for lane_speed, offset in lane_offset_map.items()
    }


def _make_vehicle_orientation_for_axis(motion_axis: str):
    """
    车辆模型朝向。

    这里保留你当前工程里的 orientation 约定：
        orientation = [pi/2, 0, 0]

    如果后续你发现沿 x 轴和沿 y 轴的车头方向不一致，
    可以在这里针对 motion_axis 分别设置。
    """
    if motion_axis == "y":
        return np.array([np.pi / 2, 0.0, 0.0], dtype=float)
    elif motion_axis == "x":
        return np.array([0.0, 0.0, 0.0], dtype=float)
    else:
        raise ValueError(
            f"Unsupported motion_axis={motion_axis}. "
            f"Expected 'x' or 'y'."
        )


def generate_occluder_specs(
    target_speed: float,
    cfg: OccluderTrafficConfig,
):
    """
    生成遮挡车初始状态。

    规则：
    - 根据 target_speed 判断目标车所在车道
    - 跳过目标车所在车道，避免同车道碰撞
    - 在其余 5 条车道生成遮挡车
    - 每条车道 cars_per_lane 辆
    - motion_axis == "y":
        车辆沿 y 轴运动，车道坐标为 x，初始 y 在 [y_min, y_max] 内随机
    - motion_axis == "x":
        车辆沿 x 轴运动，车道坐标为 y，初始 x 在 [x_min, x_max] 内随机
    - 速度为车道基准速度 + 随机扰动
    - 尺寸 scaling 在 [scaling_min, scaling_max] 内随机

    Returns
    -------
    specs : list[dict]
        每辆遮挡车的初始配置
    """
    rng = np.random.default_rng(cfg.seed)

    lane_coord_map = get_lane_coord_map(cfg)
    target_speed = float(target_speed)
    target_lane_key = get_lane_key_from_speed(target_speed)

    if target_lane_key not in lane_coord_map:
        raise ValueError(
            f"target_lane_key={target_lane_key} is not in lane definition. "
            f"Supported lane keys: {list(lane_coord_map.keys())}"
        )

    specs = []
    car_idx = 0

    for lane_speed, lane_coord in lane_coord_map.items():

        # 跳过目标车所在车道，避免同车道碰撞
        if lane_speed == target_lane_key:
            continue

        for _ in range(cfg.cars_per_lane):
            scaling = rng.uniform(cfg.scaling_min, cfg.scaling_max)

            jitter = rng.uniform(cfg.speed_jitter_min, cfg.speed_jitter_max)
            v = lane_speed + jitter

            # 防止低速车加扰动后方向反转或变成 0
            if lane_speed > 0:
                v = max(1.0, v)
            else:
                v = min(-1.0, v)

            if cfg.motion_axis == "y":
                # 直路：车辆沿 y 轴运动，车道坐标为 x
                x0 = lane_coord
                y0 = rng.uniform(cfg.y_min, cfg.y_max)

                position = np.array([x0, y0, cfg.z], dtype=float)
                velocity = np.array([0.0, v, 0.0], dtype=float)

            elif cfg.motion_axis == "x":
                # 拐角/横向道路：车辆沿 x 轴运动，车道坐标为 y
                x0 = rng.uniform(cfg.x_min, cfg.x_max)
                y0 = lane_coord

                position = np.array([x0, y0, cfg.z], dtype=float)
                velocity = np.array([v, 0.0, 0.0], dtype=float)

            else:
                raise ValueError(
                    f"Unsupported motion_axis={cfg.motion_axis}. "
                    f"Expected 'x' or 'y'."
                )

            orientation = _make_vehicle_orientation_for_axis(cfg.motion_axis)

            specs.append(
                {
                    "name": f"{cfg.prefix}_{car_idx}",
                    "lane_base_speed": float(lane_speed),
                    "position0": position,
                    "velocity": velocity,
                    "orientation": orientation,
                    "scaling": float(scaling),
                }
            )

            car_idx += 1

    return specs


def add_occluder_cars(
    scene,
    occluder_specs,
    cfg: OccluderTrafficConfig,
):
    """
    向 scene 中添加遮挡车辆。

    注意：
    - sensing_scene 和 comm_scene 都应该调用一次
    - 两边使用相同的 occluder_specs
    """
    car_material = ITURadioMaterial(
        cfg.material_name,
        cfg.material_type,
        thickness=cfg.material_thickness,
        color=cfg.material_color,
    )

    cars = []
    for spec in occluder_specs:
        car = SceneObject(
            fname=sionna.rt.scene.low_poly_car,
            name=spec["name"],
            radio_material=car_material,
        )
        cars.append(car)

    scene.edit(add=cars)

    for spec in occluder_specs:
        obj = scene.get(spec["name"])
        obj.position = spec["position0"].tolist()
        obj.orientation = spec["orientation"].tolist()
        obj.scaling = spec["scaling"]
        obj.velocity = spec["velocity"].tolist()

    return scene


def update_occluder_cars(
    scene,
    occluder_specs,
    frame_idx: int,
    frame_interval_s: float,
    cfg: OccluderTrafficConfig,
    wrap: bool = True,
    wrap_y: bool | None = None,
):
    """
    更新遮挡车位置。

    Parameters
    ----------
    scene :
        Sionna scene
    occluder_specs : list[dict]
        generate_occluder_specs() 生成的遮挡车状态
    frame_idx : int
        当前帧编号
    frame_interval_s : float
        帧间隔
    cfg : OccluderTrafficConfig
        遮挡车配置
    wrap : bool
        是否在运动范围内循环，保持交通密度
    wrap_y : bool or None
        兼容旧接口。如果传入 wrap_y，则使用其值覆盖 wrap。
    """
    if wrap_y is not None:
        wrap = wrap_y

    t = frame_idx * frame_interval_s

    for spec in occluder_specs:
        obj = scene.get(spec["name"])
        if obj is None:
            continue

        pos = spec["position0"] + spec["velocity"] * t

        if wrap:
            if cfg.motion_axis == "y":
                y_range = cfg.y_max - cfg.y_min
                if y_range <= 0:
                    raise ValueError(
                        f"Invalid y range: y_min={cfg.y_min}, y_max={cfg.y_max}"
                    )

                if pos[1] > cfg.y_max:
                    pos[1] = cfg.y_min + (pos[1] - cfg.y_min) % y_range
                elif pos[1] < cfg.y_min:
                    pos[1] = cfg.y_max - (cfg.y_max - pos[1]) % y_range

            elif cfg.motion_axis == "x":
                x_range = cfg.x_max - cfg.x_min
                if x_range <= 0:
                    raise ValueError(
                        f"Invalid x range: x_min={cfg.x_min}, x_max={cfg.x_max}"
                    )

                if pos[0] > cfg.x_max:
                    pos[0] = cfg.x_min + (pos[0] - cfg.x_min) % x_range
                elif pos[0] < cfg.x_min:
                    pos[0] = cfg.x_max - (cfg.x_max - pos[0]) % x_range

            else:
                raise ValueError(
                    f"Unsupported motion_axis={cfg.motion_axis}. "
                    f"Expected 'x' or 'y'."
                )

        obj.position = pos.tolist()
        obj.velocity = spec["velocity"].tolist()
        obj.orientation = spec["orientation"].tolist()
        obj.scaling = spec["scaling"]
