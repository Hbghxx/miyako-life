"""pet 同口径 token 估算（aligned-copy）。

权威源 = miyako `server/tokens.py`（§0.4 定版公式：CJK 字符 1 token/字、
其余字符 len//4）。本模块为 life 侧唯一镜像——工序一律从这里 import，
禁止再写第二份实现（L7 同源纪律；漂移经 interface.yaml 对账提醒）。
"""

from __future__ import annotations

_CJK_RANGES = (
    (0x3000, 0x303F),   # CJK 符号与标点
    (0x3040, 0x309F),   # 平假名
    (0x30A0, 0x30FF),   # 片假名
    (0x3400, 0x4DBF),   # 汉字扩展 A
    (0x4E00, 0x9FFF),   # CJK 统一汉字
    (0xAC00, 0xD7AF),   # 谚文
    (0xF900, 0xFAFF),   # CJK 兼容表意
    (0xFF00, 0xFFEF),   # 全角形式
)


def _is_cjk_char(ch: str) -> bool:
    o = ord(ch)
    return any(lo <= o <= hi for lo, hi in _CJK_RANGES)


def estimate_tokens(text: str) -> int:
    """CJK 1 token/字 + 其余字符 len//4（pet 公式字面实现）。"""
    if not text:
        return 0
    cjk = sum(1 for ch in text if _is_cjk_char(ch))
    return cjk + (len(text) - cjk) // 4
