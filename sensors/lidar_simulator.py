from dataclasses import dataclass
from pathlib import Path
import numpy as np
import mitsuba as mi


@dataclass
class LidarConfig:
    sensor_id: str = "lidar_0"
    position: tuple = (0.0, 0.0, 10.0)
    num_channels: int = 128
    h_resolution: int = 256
    v_fov: tuple = (-30.0, 10.0)   # (min_deg, max_deg)
    max_range: float = 150.0
    output_dir: str = "outputs/lidar"


@dataclass
class LidarFrame:
    sensor_id: str
    frame_idx: int
    xyz: np.ndarray         # (N, 3)
    distance: np.ndarray    # (N,)


class LidarSimulator:
    """
    基于 Sionna + Mitsuba 的旋转式机械 LiDAR 仿真器。
    默认执行一帧完整 360° 扫描，并返回有效点云。
    """

    def __init__(self, cfg: LidarConfig, mi_scene=None):
        self.cfg = cfg
        self.mi_scene = mi_scene

        self.position = np.asarray(cfg.position, dtype=np.float32)
        self.output_dir = Path(cfg.output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._dirs = self._build_directions()  # (N_rays, 3)
        self._origins = np.tile(self.position, (len(self._dirs), 1)).astype(np.float32)

        total = cfg.num_channels * cfg.h_resolution
        # print(
        #     f"[Lidar] 初始化完成 | "
        #     f"线束: {cfg.num_channels} | 水平分辨率: {cfg.h_resolution} | "
        #     f"每帧射线数: {total:,} | 最大距离: {cfg.max_range} m"
        # )

    def set_scene(self, mi_scene) -> None:
        """更新 Mitsuba 场景。动态物体移动后可重新设置。"""
        self.mi_scene = mi_scene

    def _build_directions(self) -> np.ndarray:
        """生成完整 360° 扫描的射线方向。"""
        v_rad = np.linspace(
            np.radians(self.cfg.v_fov[0]),
            np.radians(self.cfg.v_fov[1]),
            self.cfg.num_channels,
        )
        h_rad = np.linspace(
            0.0,
            2.0 * np.pi,
            self.cfg.h_resolution,
            endpoint=False,
        )

        V, H = np.meshgrid(v_rad, h_rad, indexing="ij")
        V = V.flatten()
        H = H.flatten()

        dx = np.cos(V) * np.cos(H)
        dy = np.cos(V) * np.sin(H)
        dz = np.sin(V)

        dirs = np.stack([dx, dy, dz], axis=-1).astype(np.float32)
        return dirs

    def scan(self) -> dict:
        """
        执行一次完整 360° 扫描。

        Returns
        -------
        dict:
            'xyz'      : np.ndarray (N, 3)
            'distance' : np.ndarray (N,)
        """
        if self.mi_scene is None:
            raise ValueError("mi_scene is None. 请先通过 set_scene(scene.mi_scene) 设置场景。")

        ray = mi.Ray3f(
            mi.Point3f(
                self._origins[:, 0],
                self._origins[:, 1],
                self._origins[:, 2],
            ),
            mi.Vector3f(
                self._dirs[:, 0],
                self._dirs[:, 1],
                self._dirs[:, 2],
            ),
        )
        ray.maxt = self.cfg.max_range

        si = self.mi_scene.ray_intersect(ray)

        valid = np.array(si.is_valid(), dtype=bool)
        t_all = np.array(si.t)
        p_all = np.array(si.p).T

        xyz = p_all[valid]
        distance = t_all[valid]

        return {
            "xyz": xyz,
            "distance": distance,
        }

    def simulate(self, scene, frame_idx: int) -> LidarFrame:
        """
        对外统一接口。
        每一帧调用一次即可；如果场景中车辆已移动，请确保 scene.mi_scene 已是最新状态。
        """
        self.set_scene(scene.mi_scene)
        result = self.scan()

        return LidarFrame(
            sensor_id=self.cfg.sensor_id,
            frame_idx=frame_idx,
            xyz=result["xyz"],
            distance=result["distance"],
        )

    def save_frame(self, frame: LidarFrame, output_dir=None) -> Path:
        """
        将单帧点云保存为 npz。

        Parameters
        ----------
        frame : LidarFrame
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
            xyz=frame.xyz,
            distance=frame.distance,
        )
        return save_path

    def visualize_frame(self, frame: LidarFrame) -> None:
        """可选：使用 Open3D 可视化单帧点云。"""
        try:
            import open3d as o3d
        except ImportError:
            print("未安装 open3d，无法可视化。可执行: pip install open3d")
            return

        if len(frame.xyz) == 0:
            print("当前帧没有有效点，跳过可视化。")
            return

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(frame.xyz)

        dist = frame.distance
        norm_dist = (dist - dist.min()) / (dist.max() - dist.min() + 1e-6)
        colors = np.zeros((len(dist), 3))
        colors[:, 0] = norm_dist
        colors[:, 2] = 1.0 - norm_dist
        pcd.colors = o3d.utility.Vector3dVector(colors)

        o3d.visualization.draw_geometries(
            [pcd],
            window_name=f"Lidar Frame {frame.frame_idx}",
            point_show_normal=False,
        )
