"""logging_util —— 全链路运行日志（contracts 共享件，约束 A 合规）。

设计（2026-09-19 用户需求"涵盖全链路的 logs 日志记录"）：
- 单文件追加：work/logs/pipeline.log（utf-8）。work/ 已 gitignore——运行时痕迹
  不入 git、不进包；L6 幂等只约束产物字节，不含日志。
- 每个 CLI 入口（build / pack / report / 各工序直跑）先 setup_logging(work_dir)：
  **幂等**（重复调用不叠加 handler）；未 setup 的环境（单测）无 handler——
  INFO 静默、WARNING+ 走 lastResort，测试零污染。
- logger 命名：miyako.<stage>（build / e0 / e15 / e1 / e2 / e3 / e4 / e6 /
  v1 / v2 / report / pack）。
- 纪律：日志只记计量与路径（计数 / hash 前缀 / 耗时 / 失败原因），**产物正文
  与游戏文本不进日志**（L8 外发面最小化——日志可能随排障外发）。
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

ROOT_NAME = "miyako"
LOG_DIR_NAME = "logs"
LOG_FILE_NAME = "pipeline.log"
_FMT = "%(asctime)s | %(levelname)-7s | %(name)-10s | %(message)s"


def log_path_of(work_dir: Path) -> Path:
    return Path(work_dir) / LOG_DIR_NAME / LOG_FILE_NAME


def _remove_miyako_handlers() -> None:
    """摘除全部 miyako 文件 handler（关闭并移除；只动自己挂的，不碰外来 handler）。"""
    root = logging.getLogger(ROOT_NAME)
    for h in list(root.handlers):
        if getattr(h, "_miyako_path", None) is not None:
            h.close()
            root.removeHandler(h)


def setup_logging(work_dir: Path, level: int = logging.INFO) -> Path:
    """CLI 入口调用：给 miyako 根 logger 挂 work/logs/pipeline.log 文件 handler。

    语义：**同一进程始终恰好一个 miyako 文件 handler**——同路径重复 setup 幂等
    返回；换 work 目录（不同日志位）则替换旧 handler，不累积。
    返回日志文件路径。
    """
    log_path = log_path_of(work_dir)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger(ROOT_NAME)
    root.setLevel(level)
    root.propagate = False
    for h in root.handlers:
        if getattr(h, "_miyako_path", None) == str(log_path):
            return log_path
    _remove_miyako_handlers()
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh._miyako_path = str(log_path)
    fh.setFormatter(logging.Formatter(_FMT))
    root.addHandler(fh)
    return log_path


def close_logging() -> None:
    """收尾（测试 teardown / 长驻进程）：关闭并摘除**本模块挂的** miyako 文件
    handler（带 _miyako_path 标记者）——与 _remove_miyako_handlers 同谓词，
    不碰外来 handler（pytest caplog 等挂到 miyako logger 的捕获 handler）。
    2026-09-19 复查修正：原实现无差别摘除全部 handler，与"只动自己挂的"
    设计声明相悖，会误杀测试框架的日志捕获。"""
    root = logging.getLogger(ROOT_NAME)
    for h in list(root.handlers):
        if getattr(h, "_miyako_path", None) is None:
            continue
        h.close()
        root.removeHandler(h)


def run_separator(action: str, detail: str = "") -> None:
    """run 起止分隔行：单文件追加式日志里切分每次 CLI 运行（logger=miyako.run）。"""
    get_logger("run").info("=" * 20 + " %s%s " + "=" * 20,
                           action, f"　{detail}" if detail else "")


def get_logger(stage: str) -> logging.Logger:
    return logging.getLogger(f"{ROOT_NAME}.{stage}")
