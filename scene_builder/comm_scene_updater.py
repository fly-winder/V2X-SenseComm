import numpy as np

def update_comm_vehicle_pose(
    scene,
    vehicle_position,
    vehicle_velocity,
    rx_offset=(0.0, 0.0, 1.5),
    car_name: str = "target_car",
    veh_rx_name: str = "veh_comm_rx",
):
    """
    同步更新：
    1. 车辆位置与速度
    2. 车载通信接收机位置与速度（带 offset）
    """

    pos = np.asarray(vehicle_position, dtype=float).reshape(3,)
    vel = np.asarray(vehicle_velocity, dtype=float).reshape(3,)
    offset = np.asarray(rx_offset, dtype=float).reshape(3,)

    car = scene.get(car_name)
    rx = scene.get(veh_rx_name)

    car.position = pos.tolist()
    car.velocity = vel.tolist()

    rx.position = (pos + offset).tolist()
    rx.velocity = vel.tolist()