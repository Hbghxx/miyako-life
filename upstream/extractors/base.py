"""extractors/base.py —— 台本提取公共层（L1.1；语义源 = extract_rules.yaml + 雏形实证）。

树递归扫描与 character 状态归因是 kirikiri/PSB 台本的通用语义，沉淀于此；
engine 特有判定（voice id 正则、字段位、文件集）由具体 extractor 提供。
约束 A：本包只依赖 stdlib + contracts/，不 import 其他工序目录与 build.py。

扫描顺序纪律：walker 先判当前节点、命中即产出，再按子元素顺序递归——
与手工雏形逐字节一致（E1 产物需复现冻结件 sha256，顺序错=hash 断）。
"""

from __future__ import annotations

from pathlib import Path


class UpstreamFailFast(RuntimeError):
    """上游工序失败——禁止继续（plan §4.1 失败矩阵：上游错污染全链）。"""


def iter_scn_files(raw_root: Path, pattern: str,
                   exclude_substr: str = ".resx.") -> list[Path]:
    """原料区台本文件集：profile engine.scn_pattern 匹配 + 排除汉化空壳/人工副本语义。"""
    root = Path(raw_root)
    files = sorted(p for p in root.glob(pattern) if exclude_substr not in p.name)
    if not files:
        raise UpstreamFailFast(
            f"原料区无匹配台本：raw={root} pattern={pattern}"
            f"（先跑 E0 输入验证确认 --raw 指向解包原料区）")
    return files


def scan_tree(node, match_fn, emit) -> None:
    """深度优先扫描 JSON 树：match_fn(list_node) 命中 → emit(rows)；随后照常递归。

    顺序 = 雏形 walk()：先当前节点后子元素；dict 按 key 插入序（json.load 保序）。
    """
    if isinstance(node, list):
        rows = match_fn(node)
        if rows:
            emit(rows)
        for child in node:
            scan_tree(child, match_fn, emit)
    elif isinstance(node, dict):
        for value in node.values():
            scan_tree(value, match_fn, emit)


def collect_character_states(data, tag: str = "character") -> list[dict]:
    """在任意子树中收集 [名, tag, {...redraw.imageFile.options}, ...] 立绘状态条目。

    状态块键序（char/file/dress/face/pose/add/pos）与雏形一致——进入冻结产物。
    """
    out: list[dict] = []

    def _walk(node) -> None:
        if isinstance(node, list):
            if (len(node) >= 3 and node[1] == tag
                    and isinstance(node[0], str) and isinstance(node[2], dict)):
                redraw = node[2].get("redraw") or {}
                imgf = redraw.get("imageFile") or {}
                opts = imgf.get("options") or {}
                out.append({
                    "char": node[0],
                    "file": imgf.get("file"),
                    "dress": opts.get("dress"),
                    "face": opts.get("face"),
                    "pose": opts.get("pose"),
                    "add": opts.get("add"),
                    "pos": node[2].get("posName"),
                })
            for item in node:
                _walk(item)
        elif isinstance(node, dict):
            for value in node.values():
                _walk(value)

    _walk(data)
    return out
