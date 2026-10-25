"""LLM 成本计量（R3）—— 三方共享（contracts，约束 A 允许的唯一共享层）。

build 与各工序（E4 induce / E6 judge / E1.5 translate）统一经此记账：
按槽位/模型/样本组粒度累计，超 profile cost.quota_warning 警示不中断（§4.1）。
落盘无时间戳字段——重跑字节可复现（L6 幂等纪律）。
"""

from __future__ import annotations

import json
from pathlib import Path


class CostMeter:
    """LLM 成本计量框架：record() 记账 → save() 落盘 → 报告页消费。"""

    def __init__(self, quota_warning: float, path: Path):
        self.quota_warning = quota_warning
        self.path = Path(path)
        # 构造即载入历史账（2026-09-19 修复）：任何工序 save() 都在历史之上累加。
        # 事故实证——e15 空批次（embedded 基线在、无待译）无 merge_from 直接
        # save()，把 45.39 元累计账本覆写成空账；E4/E6 随后 merge 只剩零。
        self.records: list[dict] = []
        if self.path.is_file():
            try:
                self.records = list(json.loads(
                    self.path.read_text(encoding="utf-8")).get("records", []))
            except (json.JSONDecodeError, OSError):
                self.records = []   # 账本损坏：宁重开不炸工序（计量非关键路径）

    def record(self, slot: str, model: str, prompt_tokens: int,
               completion_tokens: int, cost_yuan: float, group: str | None = None) -> None:
        self.records.append({
            "slot": slot, "model": model,
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "cost_yuan": round(cost_yuan, 4), "group": group,
        })

    @property
    def total_yuan(self) -> float:
        return round(sum(r["cost_yuan"] for r in self.records), 4)

    def by_slot(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for r in self.records:
            out[r["slot"]] = round(out.get(r["slot"], 0.0) + r["cost_yuan"], 4)
        return out

    @property
    def over_quota(self) -> bool:
        return self.total_yuan > self.quota_warning

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "total_yuan": self.total_yuan,
            "by_slot": self.by_slot(),
            "quota_warning": self.quota_warning,
            "over_quota": self.over_quota,
            "records": self.records,
        }
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return self.path

    def merge_from(self, other_path: Path) -> None:
        """把已落盘的成本记录并入当前账本（工序独立运行后由 build 汇总）。"""
        data = json.loads(Path(other_path).read_text(encoding="utf-8"))
        self.records.extend(data.get("records", []))
