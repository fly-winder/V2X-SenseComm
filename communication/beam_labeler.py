from dataclasses import dataclass
import numpy as np

from communication.codebook_loader import LoadedCodebook
from communication.channel_simulator import ChannelFrame


@dataclass
class BeamLabelConfig:
    top_k: int = 5
    power_unit: str = "linear"   # 当前 best_power 存线性平均功率

    # ===== 多角度峰值检测参数 =====
    # 在最优距离切片 r_idx 上，保存相对最强角度峰值多少 dB 内的局部峰
    angle_peak_rel_db_threshold: float = 3.0

    # 角度平面上的 NMS 半径: (az_radius, el_radius)
    # 如果一个峰附近被拆成多个峰，可以适当增大
    angle_peak_nms_radius: tuple[int, int] = (4, 2)

    # 最多保存多少个角度峰；None 表示不限制
    max_num_angle_peaks: int | None = None


@dataclass
class BeamLabel:
    frame_idx: int
    best_beam_index_3d: tuple[int, int, int]
    best_power: float
    best_distance: float
    best_azimuth: float
    best_elevation: float
    los_exists: bool

    topk_indices: np.ndarray | None = None   # shape: (K, 3)
    topk_powers: np.ndarray | None = None    # shape: (K,)

    # ===== 新增：固定最优距离切片后的多个角度峰 =====
    angle_peak_indices: np.ndarray | None = None      # shape: (P, 3), [r_idx, az_idx, el_idx]
    angle_peak_powers: np.ndarray | None = None       # shape: (P,)
    angle_peak_rel_db: np.ndarray | None = None       # shape: (P,)
    angle_peak_azimuths: np.ndarray | None = None     # shape: (P,)
    angle_peak_elevations: np.ndarray | None = None   # shape: (P,)


class BeamLabeler:
    """
    根据信道 H 和精细码本，计算：
    - 最优码字三维索引
    - 最优平均功率
    - 对应的距离/方位角/俯仰角
    - top-k beam
    - 固定最优距离切片后的多个角度峰值
    """

    def __init__(self, codebook: LoadedCodebook, cfg: BeamLabelConfig | None = None):
        self.codebook = codebook
        self.cfg = cfg if cfg is not None else BeamLabelConfig()

        # 4D codebook: (Nr, Naz, Nel, Nt)
        self.cb_shape = self.codebook.weights.shape
        self.num_beams = self.cb_shape[0] * self.cb_shape[1] * self.cb_shape[2]

        # 展平成 (N_beams, Nt)
        self.cb_flat = self.codebook.weights.reshape(self.num_beams, -1)

    def compute_power_map(self, H: np.ndarray) -> np.ndarray:
        """
        计算宽带全码本 3D 平均接收功率图。

        Parameters
        ----------
        H : np.ndarray
            宽带信道矩阵，shape = (Nt, Nf)

        Returns
        -------
        np.ndarray
            功率图，shape = (Nr, Naz, Nel)
        """
        if H.ndim != 2:
            raise ValueError(f"H should be 2D with shape (Nt, Nf), but got {H.shape}")

        num_tx_ant, num_subcarriers = H.shape

        if self.cb_flat.shape[1] != num_tx_ant:
            raise ValueError(
                f"Codebook antenna dimension {self.cb_flat.shape[1]} "
                f"does not match H dimension {num_tx_ant}"
            )

        # 响应矩阵:
        # cb_flat: (N_beams, Nt)
        # H:       (Nt, Nf)
        # responses: (N_beams, Nf)
        responses = self.cb_flat @ H

        # 每个 beam 在每个子载波上的功率
        power_per_subcarrier = np.abs(responses) ** 2

        # 对子载波求平均
        avg_power_flat = np.mean(power_per_subcarrier, axis=1)

        power_grid = avg_power_flat.reshape(
            self.cb_shape[0],
            self.cb_shape[1],
            self.cb_shape[2],
        )

        return power_grid

    def find_angle_peaks_within_rel_db(
        self,
        angle_power_map: np.ndarray,
        rel_db_threshold: float = 3.0,
        nms_radius: tuple[int, int] = (4, 2),
        max_num_peaks: int | None = None,
    ):
        """
        在固定距离切片上的二维角度功率图中提取多个局部峰值。

        Parameters
        ----------
        angle_power_map : np.ndarray
            shape = (Naz, Nel)
        rel_db_threshold : float
            保存最强峰值 rel_db_threshold dB 内的角度峰
        nms_radius : tuple[int, int]
            (az_radius, el_radius)
        max_num_peaks : int or None
            最多保存多少个角度峰

        Returns
        -------
        peak_indices_2d : np.ndarray
            shape = (P, 2), 每行 [az_idx, el_idx]
        peak_powers : np.ndarray
            shape = (P,)
        peak_rel_db : np.ndarray
            shape = (P,)
        """
        if angle_power_map.size == 0:
            return (
                np.empty((0, 2), dtype=int),
                np.empty((0,), dtype=float),
                np.empty((0,), dtype=float),
            )

        max_power = float(np.max(angle_power_map))

        if max_power <= 0:
            return (
                np.empty((0, 2), dtype=int),
                np.empty((0,), dtype=float),
                np.empty((0,), dtype=float),
            )

        threshold_power = max_power * (10.0 ** (-rel_db_threshold / 10.0))

        work = angle_power_map.astype(float).copy()
        peak_indices = []
        peak_powers = []

        az_rad, el_rad = nms_radius

        while True:
            flat_idx = int(np.argmax(work))
            peak_power = float(work.reshape(-1)[flat_idx])

            if peak_power < threshold_power:
                break

            az_peak_idx, el_peak_idx = np.unravel_index(flat_idx, work.shape)

            peak_indices.append((az_peak_idx, el_peak_idx))
            peak_powers.append(peak_power)

            if max_num_peaks is not None and len(peak_indices) >= max_num_peaks:
                break

            az0 = max(0, az_peak_idx - az_rad)
            az1 = min(work.shape[0], az_peak_idx + az_rad + 1)

            el0 = max(0, el_peak_idx - el_rad)
            el1 = min(work.shape[1], el_peak_idx + el_rad + 1)

            work[az0:az1, el0:el1] = -np.inf

        if len(peak_indices) == 0:
            return (
                np.empty((0, 2), dtype=int),
                np.empty((0,), dtype=float),
                np.empty((0,), dtype=float),
            )

        peak_indices = np.asarray(peak_indices, dtype=int)
        peak_powers = np.asarray(peak_powers, dtype=float)
        peak_rel_db = 10.0 * np.log10(peak_powers / max_power + 1e-30)

        return peak_indices, peak_powers, peak_rel_db

    def _compute_topk(self, power_grid: np.ndarray):
        """
        计算传统 top-k beam。
        """
        topk_indices = None
        topk_powers = None

        if self.cfg.top_k is not None and self.cfg.top_k > 0:
            flat_power = power_grid.reshape(-1)
            k = min(self.cfg.top_k, flat_power.size)

            topk_flat_idx = np.argpartition(flat_power, -k)[-k:]
            topk_flat_idx = topk_flat_idx[np.argsort(flat_power[topk_flat_idx])[::-1]]

            topk_indices = np.array(
                np.unravel_index(topk_flat_idx, power_grid.shape)
            ).T.astype(int)

            topk_powers = flat_power[topk_flat_idx].astype(float)

        return topk_indices, topk_powers

    def label(self, channel_frame: ChannelFrame) -> BeamLabel:
        """
        根据 ChannelFrame 生成 beam label。
        """
        H = channel_frame.H
        power_grid = self.compute_power_map(H)

        # ===== best beam =====
        best_idx_tuple = np.unravel_index(np.argmax(power_grid), power_grid.shape)
        best_power = float(power_grid[best_idx_tuple])

        r_idx, az_idx, el_idx = best_idx_tuple

        best_distance = float(self.codebook.r_axis[r_idx])
        best_azimuth = float(self.codebook.az_axis[az_idx])
        best_elevation = float(self.codebook.el_axis[el_idx])

        # ===== top-k beam =====
        topk_indices, topk_powers = self._compute_topk(power_grid)

        # ===== angle peaks on the best distance slice =====
        # 固定最优距离切片，只在角度平面上找多个峰
        angle_power_map = power_grid[r_idx, :, :]  # shape = (Naz, Nel)

        angle_peak_indices_2d, angle_peak_powers, angle_peak_rel_db = (
            self.find_angle_peaks_within_rel_db(
                angle_power_map=angle_power_map,
                rel_db_threshold=self.cfg.angle_peak_rel_db_threshold,
                nms_radius=self.cfg.angle_peak_nms_radius,
                max_num_peaks=self.cfg.max_num_angle_peaks,
            )
        )

        if angle_peak_indices_2d.shape[0] > 0:
            fixed_r = np.full(
                (angle_peak_indices_2d.shape[0], 1),
                int(r_idx),
                dtype=int,
            )

            # shape: (P, 3), [r_idx, az_idx, el_idx]
            angle_peak_indices = np.concatenate(
                [fixed_r, angle_peak_indices_2d],
                axis=1,
            ).astype(int)

            angle_peak_azimuths = self.codebook.az_axis[
                angle_peak_indices[:, 1]
            ].astype(float)

            angle_peak_elevations = self.codebook.el_axis[
                angle_peak_indices[:, 2]
            ].astype(float)
        else:
            angle_peak_indices = np.empty((0, 3), dtype=int)
            angle_peak_azimuths = np.empty((0,), dtype=float)
            angle_peak_elevations = np.empty((0,), dtype=float)

        return BeamLabel(
            frame_idx=channel_frame.frame_idx,
            best_beam_index_3d=(int(r_idx), int(az_idx), int(el_idx)),
            best_power=best_power,
            best_distance=best_distance,
            best_azimuth=best_azimuth,
            best_elevation=best_elevation,
            los_exists=channel_frame.los_exists,

            topk_indices=topk_indices,
            topk_powers=topk_powers,

            angle_peak_indices=angle_peak_indices,
            angle_peak_powers=angle_peak_powers,
            angle_peak_rel_db=angle_peak_rel_db,
            angle_peak_azimuths=angle_peak_azimuths,
            angle_peak_elevations=angle_peak_elevations,
        )