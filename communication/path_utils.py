import numpy as np


def get_los_label(paths) -> int:
    """
    严格 LoS 判定（逐天线）：

    对每根天线：
    - 只要存在至少一条路径在所有交互层上都为 0，
      就认为该天线存在 LoS；

    对整帧：
    - 只有当所有天线都存在至少一条 LoS 路径时，
      才判定该帧为 LoS。

    Returns
    -------
    int
        1: 所有天线都存在 LoS
        0: 至少有一根天线不存在 LoS
    """
    inter_tensor = paths.interactions.numpy()

    # 第一步：沿交互层判断，每个“天线-路径”组合是否为 LoS
    # 例如 shape: (1, 1, 1, 1024, 12)
    is_path_los = np.all(inter_tensor == 0, axis=0)

    # 第二步：对每根天线，只要存在任意一条 LoS 路径，就记该天线有 LoS
    # 例如 shape: (1, 1, 1, 1024)
    ant_has_los = np.any(is_path_los, axis=-1)

    # 第三步：只有所有天线都有 LoS，才判定整帧为 LoS
    has_los = np.all(ant_has_los)

    return int(has_los)


# def get_los_label(paths) -> int:
#     """
#     判断当前 paths 中是否存在 LoS 路径,只要有一根天线有LoS就算有LoS。

#     Returns
#     -------
#     int
#         1: 存在视距路径
#         0: 不存在视距路径
#     """
#     inter_tensor = paths.interactions.numpy()
#     is_path_los = np.all(inter_tensor == 0, axis=0)
#     has_los = np.any(is_path_los)
#     return int(has_los)

# import numpy as np
