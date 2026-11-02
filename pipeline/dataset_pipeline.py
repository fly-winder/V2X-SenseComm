from dataclasses import dataclass
from pathlib import Path
import json
import csv
import numpy as np

from scene_builder.sensing_scene_builder import build_sensing_scene
from scene_builder.comm_scene_builder import build_comm_scene
from scene_builder.sensing_scene_updater import update_sensing_vehicle_pose
from scene_builder.comm_scene_updater import update_comm_vehicle_pose

from scene_builder.traffic_builder import (
    OccluderTrafficConfig,
    generate_occluder_specs,
    add_occluder_cars,
    update_occluder_cars,
)

from pipeline.comm_pipeline import CommunicationPipeline
from utils.project_paths import resolve_project_path


@dataclass
class DatasetPipelineConfig:
    scene_id: str = "scene_00"
    scene_source: str | Path | object = "scenes/scene_00.xml"
    run_id: str = "default_run"

    num_frames: int | None = None
    total_distance: float | None = None
    frame_interval_s: float = 0.1

    start_position: tuple = (20.0, 0.0, 0.0)
    velocity_vec: tuple = (20.0, 0.0, 0.0)

    sensing_tx_offset: tuple = (0.0, 0.0, 1.5)
    comm_rx_offset: tuple = (0.0, 0.5, 1.5)

    weather_type: str = "clear"

    output_root: str = "dataset"

    # ===== 动态遮挡车配置 =====
    enable_occluder_cars: bool = False
    occluder_traffic_cfg: OccluderTrafficConfig | None = None


class DatasetPipeline:
    """
    总数据生成流程：
    1. 构建 sensing_scene 和 comm_scene
    2. 可选：添加动态遮挡车
    3. 每帧更新目标车位置
    4. 每帧更新遮挡车位置
    5. 采集 sensing 数据
    6. 生成 channel + beam label
    7. 按模态目录保存
    8. 自动生成：
       - run级 index.csv
       - scene级 index.csv
       - 全局 dataset/index.csv
    """

    def __init__(
        self,
        cfg: DatasetPipelineConfig,

        # scene builder configs
        sensing_scene_cfg,
        sensing_array_cfg,
        sensing_node_cfg,
        sensing_car_cfg,

        comm_scene_cfg,
        comm_array_cfg,
        comm_node_cfg,
        comm_car_cfg,

        # sensor simulators
        camera_simulator,
        lidar_simulator,
        radar_simulator,
        gps_simulator,

        # communication pipeline
        communication_pipeline: CommunicationPipeline,
    ):
        self.cfg = cfg

        self.sensing_scene_cfg = sensing_scene_cfg
        self.sensing_array_cfg = sensing_array_cfg
        self.sensing_node_cfg = sensing_node_cfg
        self.sensing_car_cfg = sensing_car_cfg

        self.comm_scene_cfg = comm_scene_cfg
        self.comm_array_cfg = comm_array_cfg
        self.comm_node_cfg = comm_node_cfg
        self.comm_car_cfg = comm_car_cfg

        self.camera = camera_simulator
        self.lidar = lidar_simulator
        self.radar = radar_simulator
        self.gps = gps_simulator

        self.communication_pipeline = communication_pipeline

        self.output_root = resolve_project_path(cfg.output_root)
        self.scene_root_dir = self.output_root / cfg.scene_id
        self.run_output_dir = self.scene_root_dir / cfg.run_id
        self.run_output_dir.mkdir(parents=True, exist_ok=True)

        # ===== 自动计算帧数 =====
        if self.cfg.num_frames is None:
            if self.cfg.total_distance is None:
                raise ValueError("Either num_frames or total_distance must be set")

            speed = np.linalg.norm(self.cfg.velocity_vec)
            if speed == 0:
                raise ValueError("Velocity cannot be zero when using total_distance")

            self.cfg.num_frames = max(
                1,
                int(self.cfg.total_distance / (speed * self.cfg.frame_interval_s))
            )

        self.scene_dirs = self._make_scene_dirs()

        # 每个 run 的逐帧记录（绝对路径版）
        self.index_records_abs = []

        # ===== 构建 sensing scene =====
        self.sensing_scene = build_sensing_scene(
            scene_cfg=self.sensing_scene_cfg,
            array_cfg=self.sensing_array_cfg,
            node_cfg=self.sensing_node_cfg,
            car_cfg=self.sensing_car_cfg,
            scene_source=self.cfg.scene_source,
        )

        # ===== 构建 comm scene =====
        self.comm_scene = build_comm_scene(
            scene_cfg=self.comm_scene_cfg,
            array_cfg=self.comm_array_cfg,
            node_cfg=self.comm_node_cfg,
            car_cfg=self.comm_car_cfg,
            scene_source=self.cfg.scene_source,
        )

        # ===== 初始化动态遮挡车 =====
        self.occluder_specs = None
        self.occluder_traffic_cfg = None

        if self.cfg.enable_occluder_cars:
            self.occluder_traffic_cfg = (
                self.cfg.occluder_traffic_cfg
                if self.cfg.occluder_traffic_cfg is not None
                else OccluderTrafficConfig()
            )

            target_speed = self._get_target_speed_for_occluders()

            self.occluder_specs = generate_occluder_specs(
                target_speed=target_speed,
                cfg=self.occluder_traffic_cfg,
            )

            add_occluder_cars(
                scene=self.sensing_scene,
                occluder_specs=self.occluder_specs,
                cfg=self.occluder_traffic_cfg,
            )

            add_occluder_cars(
                scene=self.comm_scene,
                occluder_specs=self.occluder_specs,
                cfg=self.occluder_traffic_cfg,
            )

        # 保存本次采集元信息
        self._write_meta()

    def _get_target_speed_for_occluders(self) -> float:
        """
        根据 occluder_traffic_cfg.motion_axis 选择目标车所在车道对应的速度。

        motion_axis == "y":
            直路场景，目标车沿 y 轴运动，使用 vy 判断所在车道。

        motion_axis == "x":
            拐角/横向道路场景，目标车沿 x 轴运动，使用 vx 判断所在车道。
        """
        if self.occluder_traffic_cfg is None:
            raise ValueError("occluder_traffic_cfg is None")

        velocity = np.asarray(self.cfg.velocity_vec, dtype=float).reshape(3,)

        if self.occluder_traffic_cfg.motion_axis == "y":
            target_speed = float(velocity[1])
        elif self.occluder_traffic_cfg.motion_axis == "x":
            target_speed = float(velocity[0])
        else:
            raise ValueError(
                f"Unsupported motion_axis={self.occluder_traffic_cfg.motion_axis}. "
                f"Expected 'x' or 'y'."
            )

        if target_speed == 0:
            raise ValueError(
                f"Target speed along motion_axis={self.occluder_traffic_cfg.motion_axis} "
                f"is zero. velocity_vec={self.cfg.velocity_vec}"
            )

        return target_speed

    def _make_scene_dirs(self):
        dirs = {
            "run": self.run_output_dir,
            "camera": self.run_output_dir / "camera",
            "lidar": self.run_output_dir / "lidar",
            "radar": self.run_output_dir / "radar",
            "gps": self.run_output_dir / "gps",
            "channel": self.run_output_dir / "channel",
            "beam_label": self.run_output_dir / "beam_label",
        }

        for path in dirs.values():
            path.mkdir(parents=True, exist_ok=True)

        return dirs

    def _write_meta(self):
        occluder_meta = {
            "enable_occluder_cars": bool(self.cfg.enable_occluder_cars),
            "num_occluder_cars": 0 if self.occluder_specs is None else len(self.occluder_specs),
        }

        # if self.occluder_traffic_cfg is not None:
        #     occluder_meta.update(
        #         {
        #             "occluder_prefix": self.occluder_traffic_cfg.prefix,
        #             "occluder_motion_axis": self.occluder_traffic_cfg.motion_axis,
        #             "occluder_cars_per_lane": int(self.occluder_traffic_cfg.cars_per_lane),

        #             "occluder_road_center_x": float(self.occluder_traffic_cfg.road_center_x),
        #             "occluder_y_min": float(self.occluder_traffic_cfg.y_min),
        #             "occluder_y_max": float(self.occluder_traffic_cfg.y_max),

        #             "occluder_road_center_y": float(self.occluder_traffic_cfg.road_center_y),
        #             "occluder_x_min": float(self.occluder_traffic_cfg.x_min),
        #             "occluder_x_max": float(self.occluder_traffic_cfg.x_max),

        #             "occluder_z": float(self.occluder_traffic_cfg.z),
        #             "occluder_scaling_min": float(self.occluder_traffic_cfg.scaling_min),
        #             "occluder_scaling_max": float(self.occluder_traffic_cfg.scaling_max),
        #             "occluder_speed_jitter_min": float(self.occluder_traffic_cfg.speed_jitter_min),
        #             "occluder_speed_jitter_max": float(self.occluder_traffic_cfg.speed_jitter_max),
        #             "occluder_seed": (
        #                 None if self.occluder_traffic_cfg.seed is None
        #                 else int(self.occluder_traffic_cfg.seed)
        #             ),
        #         }
        #     )

        meta = {
            "scene_id": self.cfg.scene_id,
            "scene_source": str(self.cfg.scene_source),
            "run_id": self.cfg.run_id,
            "num_frames": int(self.cfg.num_frames),
            "total_distance": None if self.cfg.total_distance is None else float(self.cfg.total_distance),
            "frame_interval_s": float(self.cfg.frame_interval_s),
            "start_position": list(self.cfg.start_position),
            "velocity_vec": list(self.cfg.velocity_vec),
            "sensing_tx_offset": list(self.cfg.sensing_tx_offset),
            "comm_rx_offset": list(self.cfg.comm_rx_offset),
            "weather_type": self.cfg.weather_type,
            **occluder_meta,
        }

        meta_path = self.run_output_dir / "meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def _get_vehicle_position(self, frame_idx: int) -> np.ndarray:
        start_pos = np.asarray(self.cfg.start_position, dtype=float).reshape(3,)
        velocity = np.asarray(self.cfg.velocity_vec, dtype=float).reshape(3,)
        return start_pos + frame_idx * self.cfg.frame_interval_s * velocity

    def _run_sensing_frame(self, frame_idx: int):
        camera_path = self.camera.simulate(
            scene=self.sensing_scene,
            frame_idx=frame_idx,
            weather_type=self.cfg.weather_type,
            output_dir=self.scene_dirs["camera"],
        )

        lidar_frame = self.lidar.simulate(
            scene=self.sensing_scene,
            frame_idx=frame_idx,
        )
        lidar_path = self.lidar.save_frame(
            lidar_frame,
            output_dir=self.scene_dirs["lidar"],
        )

        gps_frame = self.gps.simulate(
            scene=self.sensing_scene,
            frame_idx=frame_idx,
        )
        gps_path = self.gps.save_frame(
            gps_frame,
            output_dir=self.scene_dirs["gps"],
        )

        radar_frame = self.radar.simulate(
            scene=self.sensing_scene,
            frame_idx=frame_idx,
        )
        radar_path = self.radar.save_frame(
            radar_frame,
            output_dir=self.scene_dirs["radar"],
        )

        return {
            "camera_path": Path(camera_path).resolve(),
            "lidar_path": Path(lidar_path).resolve(),
            "gps_path": Path(gps_path).resolve(),
            "radar_path": Path(radar_path).resolve(),
        }

    def _run_communication_frame(self, frame_idx: int):
        label, label_path, channel_path = self.communication_pipeline.run_single_frame(
            scene=self.comm_scene,
            frame_idx=frame_idx,
            channel_output_dir=self.scene_dirs["channel"],
            label_output_dir=self.scene_dirs["beam_label"],
        )

        return {
            "label": label,
            "label_path": Path(label_path).resolve(),
            "channel_path": Path(channel_path).resolve(),
        }

    def _make_abs_record(
        self,
        frame_idx: int,
        current_pos: np.ndarray,
        sensing_outputs: dict,
        comm_outputs: dict,
    ):
        return {
            "frame_idx": int(frame_idx),
            "scene_id": self.cfg.scene_id,
            "run_id": self.cfg.run_id,

            "camera_path": sensing_outputs["camera_path"],
            "lidar_path": sensing_outputs["lidar_path"],
            "radar_path": sensing_outputs["radar_path"],
            "gps_path": sensing_outputs["gps_path"],
            "channel_path": comm_outputs["channel_path"],
            "beam_label_path": comm_outputs["label_path"],

            "pos_x": float(current_pos[0]),
            "pos_y": float(current_pos[1]),
            "pos_z": float(current_pos[2]),
            "speed": float(np.linalg.norm(self.cfg.velocity_vec)),
            "weather_type": self.cfg.weather_type,
            "enable_occluder_cars": bool(self.cfg.enable_occluder_cars),
        }

    def _record_for_base(self, record_abs: dict, base_dir: Path) -> dict:
        """
        将绝对路径记录转换成相对于 base_dir 的路径记录。
        """
        return {
            "frame_idx": int(record_abs["frame_idx"]),
            "scene_id": record_abs["scene_id"],
            "run_id": record_abs["run_id"],

            "camera_path": Path(record_abs["camera_path"]).relative_to(base_dir).as_posix(),
            "lidar_path": Path(record_abs["lidar_path"]).relative_to(base_dir).as_posix(),
            "radar_path": Path(record_abs["radar_path"]).relative_to(base_dir).as_posix(),
            "gps_path": Path(record_abs["gps_path"]).relative_to(base_dir).as_posix(),
            "channel_path": Path(record_abs["channel_path"]).relative_to(base_dir).as_posix(),
            "beam_label_path": Path(record_abs["beam_label_path"]).relative_to(base_dir).as_posix(),

            "weather_type": record_abs["weather_type"],
            "enable_occluder_cars": record_abs["enable_occluder_cars"],
        }

    def _upsert_csv(self, csv_path: Path, new_records: list[dict]):
        """
        以 (scene_id, run_id, frame_idx) 为键，对 CSV 做追加/更新。
        """
        if len(new_records) == 0:
            return

        fieldnames = list(new_records[0].keys())
        merged = {}

        if csv_path.exists():
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = (row["scene_id"], row["run_id"], int(row["frame_idx"]))
                    merged[key] = row

        for rec in new_records:
            key = (rec["scene_id"], rec["run_id"], int(rec["frame_idx"]))
            merged[key] = rec

        rows = list(merged.values())
        rows.sort(key=lambda x: (x["scene_id"], x["run_id"], int(x["frame_idx"])))

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _write_run_index_csv(self):
        """
        生成当前 run 的 index.csv，路径相对于 run 根目录。
        """
        run_records = [
            self._record_for_base(rec, self.run_output_dir)
            for rec in self.index_records_abs
        ]
        run_index_path = self.run_output_dir / "index.csv"
        self._upsert_csv(run_index_path, run_records)
        print(f"[DatasetPipeline] run index saved to: {run_index_path}")

    def _update_scene_index_csv(self):
        """
        追加/更新 scene 级 index.csv，路径相对于 scene 根目录。
        """
        scene_records = [
            self._record_for_base(rec, self.scene_root_dir)
            for rec in self.index_records_abs
        ]
        scene_index_path = self.scene_root_dir / "index.csv"
        self._upsert_csv(scene_index_path, scene_records)
        print(f"[DatasetPipeline] scene index updated: {scene_index_path}")

    def _update_global_index_csv(self):
        """
        追加/更新全局 dataset/index.csv，路径相对于 dataset 根目录。
        """
        global_records = [
            self._record_for_base(rec, self.output_root)
            for rec in self.index_records_abs
        ]
        global_index_path = self.output_root / "index.csv"
        self._upsert_csv(global_index_path, global_records)
        print(f"[DatasetPipeline] global index updated: {global_index_path}")

    def run(self):
        for frame_idx in range(self.cfg.num_frames):
            current_pos = self._get_vehicle_position(frame_idx)

            # 1. 更新目标车：sensing scene
            update_sensing_vehicle_pose(
                scene=self.sensing_scene,
                vehicle_position=current_pos,
                vehicle_velocity=self.cfg.velocity_vec,
                tx_offset=self.cfg.sensing_tx_offset,
                car_name="target_car",
                veh_tx_name="veh_sens_tx",
            )

            # 2. 更新目标车：comm scene
            update_comm_vehicle_pose(
                scene=self.comm_scene,
                vehicle_position=current_pos,
                vehicle_velocity=self.cfg.velocity_vec,
                rx_offset=self.cfg.comm_rx_offset,
                car_name="target_car",
                veh_rx_name="veh_comm_rx",
            )

            # 3. 更新动态遮挡车
            if self.occluder_specs is not None:
                update_occluder_cars(
                    scene=self.sensing_scene,
                    occluder_specs=self.occluder_specs,
                    frame_idx=frame_idx,
                    frame_interval_s=self.cfg.frame_interval_s,
                    cfg=self.occluder_traffic_cfg,
                    wrap=True,
                )

                update_occluder_cars(
                    scene=self.comm_scene,
                    occluder_specs=self.occluder_specs,
                    frame_idx=frame_idx,
                    frame_interval_s=self.cfg.frame_interval_s,
                    cfg=self.occluder_traffic_cfg,
                    wrap=True,
                )

            # 4. 感知数据采集
            sensing_outputs = self._run_sensing_frame(frame_idx=frame_idx)

            # 5. 通信信道与 beam label
            comm_outputs = self._run_communication_frame(frame_idx=frame_idx)

            # 6. 记录索引
            record_abs = self._make_abs_record(
                frame_idx=frame_idx,
                current_pos=current_pos,
                sensing_outputs=sensing_outputs,
                comm_outputs=comm_outputs,
            )
            self.index_records_abs.append(record_abs)

            # 可选调试输出
            # print(
            #     f"[DatasetPipeline] scene={self.cfg.scene_id} | "
            #     f"run={self.cfg.run_id} | "
            #     f"frame={frame_idx:06d} | "
            #     f"vehicle_pos={current_pos.tolist()} | "
            #     f"best_beam={comm_outputs['label'].best_beam_index_3d} | "
            #     f"occluders={0 if self.occluder_specs is None else len(self.occluder_specs)}"
            # )

        # 写三级索引
        self._write_run_index_csv()
        self._update_scene_index_csv()
        self._update_global_index_csv()