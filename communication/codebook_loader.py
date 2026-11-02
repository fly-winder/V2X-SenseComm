from dataclasses import dataclass
from pathlib import Path
import numpy as np


# ===== 配置 =====
@dataclass
class CodebookConfig:
    file: str
    codebook_key: str = "codebook"
    r_axis_key: str = "r_axis"
    az_axis_key: str = "az_axis"
    el_axis_key: str = "el_axis"


# ===== 加载后的数据结构 =====
@dataclass
class LoadedCodebook:
    weights: np.ndarray   # shape: (Nr, Naz, Nel, Nt)
    r_axis: np.ndarray
    az_axis: np.ndarray
    el_axis: np.ndarray


# ===== 主加载函数 =====
def load_codebook(cfg: CodebookConfig) -> LoadedCodebook:
    """
    从 npz 文件加载 4D 波束码本

    Returns
    -------
    LoadedCodebook
        weights: shape = (Nr, Naz, Nel, Nt)
        r_axis, az_axis, el_axis: 坐标轴
    """
    path = Path(cfg.file)

    if not path.exists():
        raise FileNotFoundError(f"❌ Codebook file not found: {path}")

    print(f"Loading codebook from: {path}")

    with np.load(path) as data:
        weights = data[cfg.codebook_key]
        r_axis = data[cfg.r_axis_key]
        az_axis = data[cfg.az_axis_key]
        el_axis = data[cfg.el_axis_key]

    print(f"Codebook loaded!")
    # print(f"   shape: {weights.shape}")
    # print(f"   dtype: {weights.dtype}")

    return LoadedCodebook(
        weights=weights,
        r_axis=r_axis,
        az_axis=az_axis,
        el_axis=el_axis,
    )
