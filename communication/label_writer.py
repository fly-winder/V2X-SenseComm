from pathlib import Path
import json
import numpy as np


class LabelWriter:
    """
    负责保存 beam label 到 JSON 文件
    """

    def __init__(self, output_dir: str = "outputs/beam_label"):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _to_list(self, value):
        """
        将 numpy array / list / None 转成可 JSON 序列化对象
        """
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    def save_label(self, label, output_dir=None) -> Path:
        """
        保存单帧 label

        Parameters
        ----------
        label : BeamLabel
            由 BeamLabeler 生成的标签对象
        output_dir : str or Path or None
            可选的输出目录。
            - 若为 None，则使用初始化时的默认 output_dir
            - 若指定，则优先保存到该目录

        Returns
        -------
        Path
            保存路径
        """
        save_dir = Path(output_dir).resolve() if output_dir is not None else self.output_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        save_path = save_dir / f"frame_{label.frame_idx:06d}.json"

        label_dict = {
            # ===== single best beam =====
            "best_beam_index_3d": [int(x) for x in label.best_beam_index_3d],
            "best_power": float(label.best_power),
            "best_distance": float(label.best_distance),
            "best_azimuth": float(label.best_azimuth),
            "best_elevation": float(label.best_elevation),
            "los_exists": bool(label.los_exists),

            # ===== traditional top-k beams =====
            "topk_indices": self._to_list(label.topk_indices),
            "topk_powers": self._to_list(label.topk_powers),

            # ===== angle peaks on the best distance slice =====
            # shape: (P, 3), each row = [r_idx, az_idx, el_idx]
            "angle_peak_indices": self._to_list(label.angle_peak_indices),
            "angle_peak_powers": self._to_list(label.angle_peak_powers),
            "angle_peak_rel_db": self._to_list(label.angle_peak_rel_db),
            "angle_peak_azimuths": self._to_list(label.angle_peak_azimuths),
            "angle_peak_elevations": self._to_list(label.angle_peak_elevations),
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(label_dict, f, indent=2)

        return save_path