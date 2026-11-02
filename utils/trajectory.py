def get_lane_band_offset(abs_speed: float) -> float:
    """
    Map the absolute speed to a lane offset magnitude.

    Rules
    -----
    - |v| < 10        -> outer slow lane   -> 8.75
    - 10 <= |v| < 20  -> middle lane       -> 5.25
    - |v| >= 20       -> inner fast lane   -> 1.75
    """
    abs_speed = float(abs_speed)
    if abs_speed < 10.0:
        return 8.75
    if abs_speed < 20.0:
        return 5.25
    return 1.75


def get_signed_lane_offset(speed: float) -> float:
    """Return the signed lane offset based on the signed vehicle speed."""
    speed = float(speed)
    if speed == 0.0:
        raise ValueError("Speed cannot be zero for lane assignment.")
    offset = get_lane_band_offset(abs(speed))
    return offset if speed > 0 else -offset


def get_lane_key_from_speed(speed: float) -> float:
    """
    Convert an arbitrary speed to a canonical lane key.

    Returns one of:
    - +5, +15, +25
    - -5, -15, -25

    The lane key is used by the traffic builder so that arbitrary speeds such
    as +/-35 still map to the predefined slow/middle/fast lanes.
    """
    speed = float(speed)
    if speed == 0.0:
        raise ValueError("Speed cannot be zero for lane assignment.")

    abs_speed = abs(speed)
    if abs_speed < 10.0:
        lane_key = 5.0
    elif abs_speed < 20.0:
        lane_key = 15.0
    else:
        lane_key = 25.0

    return lane_key if speed > 0 else -lane_key


def get_two_way_three_lane_init(
    speed,
    road_center_x=30.0,
    y_extent=50.0,
    z=0.1,
):
    """
    Straight-road scene: the vehicle moves along the y axis.

    The lane is selected from the speed magnitude:
    - |v| < 10        -> outer slow lane
    - 10 <= |v| < 20  -> middle lane
    - |v| >= 20       -> inner fast lane
    """
    speed = float(speed)
    x = road_center_x + get_signed_lane_offset(speed)

    if speed > 0:
        start_position = (x, -y_extent, z)
    else:
        start_position = (x, y_extent, z)

    velocity_vec = (0.0, speed, 0.0)
    return start_position, velocity_vec


def get_x_axis_six_lane_init(
    speed,
    road_center_y=45.0,
    x_min=20.0,
    x_max=120.0,
    z=0.1,
):
    """
    Corner / cross-road scene: the vehicle moves along the x axis.

    The lane is selected from the speed magnitude:
    - |v| < 10        -> outer slow lane
    - 10 <= |v| < 20  -> middle lane
    - |v| >= 20       -> inner fast lane
    """
    speed = float(speed)
    y = road_center_y + get_signed_lane_offset(speed)

    if speed > 0:
        start_position = (x_min, y, z)
    else:
        start_position = (x_max, y, z)

    velocity_vec = (speed, 0.0, 0.0)
    return start_position, velocity_vec
