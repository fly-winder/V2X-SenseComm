from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class GPSConfig:
    sensor_id: str = "gps_0"
    target_name: str = "target_car"  # 默认采集 scene 中车辆位置
    noise_std: tuple = (0.5, 0.5, 0.5)  # x, y, z 三个方向的高斯噪声标准差（米）
    seed: int = 42
    output_dir: str = "outputs/gps"


@dataclass
class GPSFrame:
    sensor_id: str
    frame_idx: int
    true_position: np.ndarray       # shape: (3,)
    measured_position: np.ndarray   # shape: (3,)


class GPSSimulator:
    """
    简单 GPS 仿真器：
    - 从场景中读取目标位置
    - 加入高斯噪声，模拟测量误差
    """

    def __init__(self, cfg: GPSConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        self.output_dir = Path(cfg.output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def simulate(self, scene, frame_idx: int) -> GPSFrame:
        """
        采集当前帧的 GPS 位置。
        """
        target = scene.get(self.cfg.target_name)

        if target is None:
            raise ValueError(
                f"[GPS] target '{self.cfg.target_name}' not found in scene"
            )

        true_position = np.asarray(target.position, dtype=float).reshape(3,)

        noise = self.rng.normal(
            loc=0.0,
            scale=np.asarray(self.cfg.noise_std, dtype=float),
            size=3,
        )

        measured_position = true_position + noise

        return GPSFrame(
            sensor_id=self.cfg.sensor_id,
            frame_idx=frame_idx,
            true_position=true_position,
            measured_position=measured_position,
        )

    def save_frame(self, frame: GPSFrame, output_dir=None) -> Path:
        """
        保存单帧 GPS 数据

        Parameters
        ----------
        frame : GPSFrame
        output_dir : str or Path or None
            - None：使用默认目录
            - 指定：保存到外部给定目录（推荐 dataset_pipeline 使用）

        Returns
        -------
        Path
        """
        save_dir = Path(output_dir).resolve() if output_dir is not None else self.output_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        save_path = save_dir / f"frame_{frame.frame_idx:06d}.npz"

        np.savez_compressed(
            save_path,
            true_position=frame.true_position,         
            measured_position=frame.measured_position,
        )

        return save_path