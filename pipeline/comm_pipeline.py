from dataclasses import dataclass

from communication.channel_simulator import ChannelSimulator, ChannelConfig
from communication.codebook_loader import load_codebook, CodebookConfig
from communication.beam_labeler import BeamLabeler, BeamLabelConfig
from communication.label_writer import LabelWriter


@dataclass
class CommunicationPipelineConfig:
    # ===== Channel =====
    channel_cfg: ChannelConfig

    # ===== Codebook =====
    codebook_cfg: CodebookConfig

    # ===== Beam Label =====
    beam_label_cfg: BeamLabelConfig

    # ===== Output =====
    label_output_dir: str = "outputs/beam_label"


class CommunicationPipeline:
    """
    通信侧完整流程：
    scene → H → beam label → 保存
    """

    def __init__(self, cfg: CommunicationPipelineConfig):
        self.cfg = cfg

        # 1️⃣ Channel simulator
        self.channel_sim = ChannelSimulator(cfg.channel_cfg)

        # 2️⃣ Load codebook（只加载一次）
        self.codebook = load_codebook(cfg.codebook_cfg)

        # 3️⃣ Beam labeler
        self.beam_labeler = BeamLabeler(
            codebook=self.codebook,
            cfg=cfg.beam_label_cfg,
        )

        # 4️⃣ Label writer
        self.label_writer = LabelWriter(cfg.label_output_dir)

    def run_single_frame(
        self,
        scene,
        frame_idx: int,
        channel_output_dir=None,
        label_output_dir=None,
    ):
        """
        处理单帧：
        - 生成 H
        - 保存 H
        - 生成 label
        - 保存 label

        Parameters
        ----------
        scene :
            当前通信场景
        frame_idx : int
            当前帧编号
        channel_output_dir : str or Path or None
            若指定，则把 channel 保存到该目录
        label_output_dir : str or Path or None
            若指定，则把 beam label 保存到该目录

        Returns
        -------
        label : BeamLabel
            当前帧生成的标签
        label_path : Path
            label 的保存路径
        channel_path : Path
            channel 的保存路径
        """

        # 1️⃣ 生成信道
        channel_frame = self.channel_sim.simulate(scene, frame_idx)

        # 2️⃣ 保存 H
        channel_path = self.channel_sim.save_frame(
            channel_frame,
            output_dir=channel_output_dir,
        )

        # 3️⃣ 生成 label
        label = self.beam_labeler.label(channel_frame)

        # 4️⃣ 保存 label
        label_path = self.label_writer.save_label(
            label,
            output_dir=label_output_dir,
        )

        return label, label_path, channel_path