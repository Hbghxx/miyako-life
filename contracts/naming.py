"""naming —— 跨工序命名单一来源（约束 A：工序间共享只经 contracts）。

V1 dataset 主 list 文件名曾由 voice（落盘）与 pack（就绪检查）各自硬编码
"miyako.list"（2026-09-18 收敛）——profile 改名/换角色时会静默漂移出双重口径。
"""


def dataset_list_name(profile: dict) -> str:
    """V1 dataset 规范 list 文件名：{package_id}.list（V1 写 / pack 查，同源）。"""
    return f"{profile['character']['package_id']}.list"
