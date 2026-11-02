from dataclasses import dataclass
from pathlib import Path
import numpy as np


# ===== Camera 配置 =====
@dataclass
class CameraConfig:
    sensor_id: str = "camera_0"
    position: list = None
    look_at: list = None
    fov: float = 120.0
    resolution: tuple = (512, 512)
    num_samples: int = 128
    output_dir: str = "outputs/camera"


# ===== Camera Simulator =====
class CameraSimulator:
    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg

        # 默认输出目录
        self.output_dir = Path(cfg.output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化 Sionna Camera
        from sionna.rt import Camera
        self.camera = Camera(
            position=cfg.position,
            look_at=cfg.look_at
        )

    def _get_lighting(self, weather_type):
        if weather_type == "sunny":
            return 2.0, "textures/sunny.exr"
        elif weather_type == "cloudy":
            return 0.8, "textures/cloudy.exr"
        elif weather_type == "night":
            return 0.5, "textures/night.exr"
        else:
            return 1.0, None

    def simulate(self, scene, frame_idx, weather_type="", output_dir=None):
        """
        渲染并保存当前帧图像

        Parameters
        ----------
        scene : Sionna scene
        frame_idx : int
            当前帧编号
        weather_type : str
            天气/光照类型
        output_dir : str or Path or None
            - None：使用默认目录
            - 指定：保存到外部给定目录（推荐 dataset_pipeline 使用）

        Returns
        -------
        str
            当前帧图像保存路径
        """
        # 光照设置
        lighting_scale, envmap = self._get_lighting(weather_type)

        # 输出目录
        save_dir = Path(output_dir).resolve() if output_dir is not None else self.output_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        # 输出路径
        filename = save_dir / f"frame_{frame_idx:06d}.png"

        # 渲染
        scene.render_to_file(
            camera=self.camera,
            show_devices=False,
            filename=str(filename),
            fov=self.cfg.fov,
            lighting_scale=lighting_scale,
            envmap=envmap,
            num_samples=self.cfg.num_samples,
            resolution=self.cfg.resolution
        )

        return str(filename)