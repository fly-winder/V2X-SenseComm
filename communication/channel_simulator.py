from dataclasses import dataclass
from pathlib import Path
import numpy as np
from sionna.rt import PathSolver

from communication.path_utils import get_los_label


@dataclass
class ChannelConfig:
    sensor_id: str = "comm_channel_0"
    carrier_frequency: float = 7e9

    # PathSolver 参数
    max_depth: int = 3
    samples_per_src: int = 50000
    max_num_paths_per_src: int = 10000
    los: bool = True
    specular_reflection: bool = True
    diffraction: bool = False
    diffuse_reflection: bool = False
    refraction: bool = True
    synthetic_array: bool = False # 近场/远场
    seed: int = 41

    # 多载波
    frequencies: tuple = tuple(np.linspace(-10e6, 10e6, 16))

    output_dir: str = "outputs/channel"


@dataclass
class ChannelFrame:
    sensor_id: str
    frame_idx: int
    H: np.ndarray           # shape: (num_tx_ant, num_subcarriers)
    los_exists: bool


class ChannelSimulator:
    """
    通信侧单帧信道仿真器

    输入:
        当前时刻的 comm_scene

    输出:
        ChannelFrame:
            - H : shape = (num_tx_ant, num_subcarriers)
            - los_exists : 当前帧是否存在 LoS 路径
    """

    def __init__(self, cfg: ChannelConfig):
        self.cfg = cfg
        self.path_solver = PathSolver()

        self.output_dir = Path(cfg.output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def simulate(self, scene, frame_idx: int) -> ChannelFrame:
        """
        生成当前帧的信道 H
        """
        # 1) 设置通信频率
        scene.frequency = self.cfg.carrier_frequency

        # 2) 路径求解
        paths = self.path_solver(
            scene=scene,
            max_depth=self.cfg.max_depth,
            samples_per_src=self.cfg.samples_per_src,
            max_num_paths_per_src=self.cfg.max_num_paths_per_src,
            los=self.cfg.los,
            specular_reflection=self.cfg.specular_reflection,
            diffraction=self.cfg.diffraction,
            diffuse_reflection=self.cfg.diffuse_reflection,
            refraction=self.cfg.refraction,
            synthetic_array=self.cfg.synthetic_array,
            seed=self.cfg.seed,
        )

        # 3) 获取频域信道
        h_freq = paths.cfr(
            frequencies=list(self.cfg.frequencies),
            normalize=False,
            out_type="numpy",
        )

        # 4) 转为 numpy 并压缩维度
        h_freq = np.asarray(h_freq)
        H = np.squeeze(h_freq)

        num_tx_ant = scene.tx_array.array_size
        num_subcarriers = len(self.cfg.frequencies)

        # 统一成 (num_tx_ant, num_subcarriers)
        H = H.reshape(num_tx_ant, num_subcarriers).astype(np.complex64)

        # if H.ndim == 0:
        #     # 极端情况：1x1 链路
        #     H = H.reshape(1, 1)

        # elif H.ndim == 1:
        #     # 常见情况：单接收天线，shape 可能是 (num_tx_ant,) 或 (num_rx_ant,)
        #     if H.size == num_tx_ant:
        #         H = H.reshape(num_tx_ant, 1)
        #     elif H.size == num_rx_ant:
        #         H = H.reshape(1, num_rx_ant)
        #     else:
        #         raise ValueError(
        #             f"Unexpected 1D H shape: {H.shape}, "
        #             f"num_tx_ant={num_tx_ant}, num_rx_ant={num_rx_ant}"
        #         )

        # elif H.ndim == 2:
        #     # 可能是 (num_rx_ant, num_tx_ant) 或 (num_tx_ant, num_rx_ant)
        #     if H.shape == (num_rx_ant, num_tx_ant):
        #         H = H.T
        #     elif H.shape == (num_tx_ant, num_rx_ant):
        #         pass
        #     else:
        #         raise ValueError(
        #             f"Unexpected 2D H shape: {H.shape}, "
        #             f"expected ({num_tx_ant}, {num_rx_ant}) or ({num_rx_ant}, {num_tx_ant})"
        #         )

        # else:
        #     raise ValueError(
        #         f"Unexpected H ndim: {H.ndim}, raw h_freq shape: {h_freq.shape}"
        #     )

        # 6) 判断是否存在 LoS 路径
        los_exists = get_los_label(paths)

        return ChannelFrame(
            sensor_id=self.cfg.sensor_id,
            frame_idx=frame_idx,
            H=H,
            los_exists=los_exists,
        )

    def save_frame(self, frame: ChannelFrame, output_dir=None) -> Path:
        """
        保存单帧信道数据
        这里只保存 H，LoS 放到后续 label 里

        Parameters
        ----------
        frame : ChannelFrame
            当前帧信道数据
        output_dir : str or Path or None
            可选输出目录：
            - 若为 None，则使用初始化时的默认 output_dir
            - 若指定，则优先保存到该目录

        Returns
        -------
        Path
            保存路径
        """
        save_dir = Path(output_dir).resolve() if output_dir is not None else self.output_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        save_path = save_dir / f"frame_{frame.frame_idx:06d}.npz"

        np.savez_compressed(
            save_path,
            H=frame.H,
        )

        return save_path