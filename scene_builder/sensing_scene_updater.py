import numpy as np

def update_sensing_vehicle_pose(
    scene,
    vehicle_position,
    vehicle_velocity,
    tx_offset=(0.0, 0.0, 1.5),
    car_name: str = "target_car",
    veh_tx_name: str = "veh_sens_tx",
):
    """
    同步更新：
    1. 车辆位置与速度
    2. 车载感知发射机位置与速度
    """
    pos = np.asarray(vehicle_position, dtype=float).reshape(3,)
    vel = np.asarray(vehicle_velocity, dtype=float).reshape(3,)
    offset = np.asarray(tx_offset, dtype=float).reshape(3,)

    car = scene.get(car_name)
    tx = scene.get(veh_tx_name)

    car.position = pos.tolist()
    car.velocity = vel.tolist()

    tx.position = (pos + offset).tolist()
    tx.velocity = vel.tolist()