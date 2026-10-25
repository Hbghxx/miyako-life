"""build.py —— 总编排 DAG + LLM 成本计量（plan L0.4；DAG 语义源 = 方案契约 §四）。

四个 CLI 入口（§0.1）：build（本文件）→ report → pack → accept。
本骨架交付：工序注册表 + 依赖拓扑序 + dry-run 工序计划打印 + 成本计量框架；
各工序实现随里程碑落地（L1~L5），未实现工序被调用时 fail-fast（NotImplementedError，
不静默跳过——上游错污染全链，plan §4.1）。

入参：--profile（游戏 profile）/ --raw（原料区根，只读）/ --resume（断点续跑，样本组
粒度，L2.3 生效）/ --dry-run（只打印工序计划与依赖序）。

profile 纪律：bilingual_source 非 auto 即锁定态，build 只读不写（L-0.2 定案，
E0 负责一致性校验）；本文件对 profile 永无写路径。

成本纪律（R3 + §0.4）：所有 LLM 调用经 CostMeter 计量（按槽位/模型/样本组），
超过 profile cost.quota_warning 报告页警示不中断；计量数据落 work/llm_cost.json
（无时间戳字段，重跑字节可复现——L6 幂等纪律）。

本文件可被任何工序反向引用吗？不——约束 A：四工序目录禁止 import build.py；
build → 工序单向编排。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

from contracts.cost import CostMeter  # 三方共享计量（约束 A：工序不得 import build，反之经 contracts 共享）
from contracts.logging_util import get_logger, run_separator, setup_logging

REPO_ROOT = Path(__file__).resolve().parent

# ── 工序注册表（依赖序由 topo_sort 计算；impl=False = 里程碑未落地）────────
# DAG 语义源=契约 §四；v3_acceptance 属 accept 入口（人工训练节点后），不在 build DAG。
STAGES: list[dict] = [
    {"name": "e0_input_check", "deps": [], "impl": True,
     "summary": "输入验证：三探测 → work/input_profile.json（bilingual_source 锁定校验）"},
    {"name": "e15_translate", "deps": ["e0_input_check"], "impl": True,
     "summary": "E1.5 翻译：上下文窗口粗译 → 双语 voice_map（origin 标记；translate 路径；需 GLM key）"},
    {"name": "e1_semantic", "deps": ["e0_input_check", "e15_translate"], "impl": True,
     "summary": "E1 语义提取：台本 → voice_map.json + sha256 冻结链（embedded 时 e15 空转）"},
    {"name": "e2_sprites", "deps": ["e0_input_check"], "impl": True,
     "summary": "E2 立绘索引：三轴 manifest + 覆盖率 + fallback 模拟"},
    {"name": "e3_dna_stats", "deps": ["e1_semantic"], "impl": True,
     "summary": "E3 表达 DNA 统计（零 LLM）：中日成对 + 词表候选（Y4①②）"},
    {"name": "e4_persona", "deps": ["e3_dna_stats"], "impl": True,
     "summary": "E4 五层提炼（induce 槽）：样本组断点 + refine≤2 + 全量扫描兜底"},
    {"name": "e6_validate", "deps": ["e4_persona"], "impl": True,
     "summary": "E6 自动验证（judge 槽，异型号 Y8）：20 例已知问题 + 证据回查 + 覆盖面"},
    {"name": "v1_audio_prep", "deps": ["e1_semantic"], "impl": True,
     "summary": "V1 音频工程：切片 + 三份 .list + 九桶预录"},
    {"name": "v2_train_prep", "deps": ["v1_audio_prep"], "impl": True,
     "summary": "V2 参考初筛：rank=abs(duration-7.0)，per_emotion 3"},
    {"name": "report_builder", "deps": ["e6_validate", "v2_train_prep"], "impl": True,
     "summary": "L4.1 体检报告页 + L4.2 决策账本状态（人工决策收敛单页）"},
    {"name": "pack", "deps": ["report_builder", "e2_sprites"], "impl": True,
     "summary": "pack：voice_tier 伪码 + write_scopes 执法 + 出厂检查单（fail-closed）"},
]

# e0 → upstream.e0_input_check.run；其余 runners 见 run_stage（当前唯一注册用途：
# IMPLEMENTED 表保留给需要 importlib 动态装载的工序，未来工序接入时填这里）
# 注：pack 于 2026-09-18 接线（L4.3 已实现但注册表曾滞留 stub 态——全量 DAG 会
# 在烧完 LLM 成本后于末序崩出 NotImplementedError，且顶层账本 save 不执行）。
IMPLEMENTED = {
    "e0_input_check": ("upstream.e0_input_check", "run"),
}


def topo_sort(stages: list[dict]) -> list[dict]:
    """Kahn 拓扑排序；依赖缺失/成环即 fail-fast（编排错误属上游错误）。"""
    by_name = {s["name"]: s for s in stages}
    indeg = {s["name"]: 0 for s in stages}
    children: dict[str, list[str]] = {s["name"]: [] for s in stages}
    for s in stages:
        for dep in s["deps"]:
            if dep not in by_name:
                raise SystemExit(f"[build][fail-fast] 工序 {s['name']} 依赖未知工序 {dep}")
            indeg[s["name"]] += 1
            children[dep].append(s["name"])
    ready = [s["name"] for s in stages if indeg[s["name"]] == 0]
    ordered: list[str] = []
    while ready:
        name = ready.pop(0)
        ordered.append(name)
        for child in children[name]:
            indeg[child] -= 1
            if indeg[child] == 0:
                ready.append(child)
    if len(ordered) != len(stages):
        raise SystemExit("[build][fail-fast] 工序依赖成环，注册表非法")
    return [by_name[n] for n in ordered]


def _run_e0(ctx):
    from upstream.e0_input_check import run
    return run(ctx["raw"], ctx["profile_path"], ctx["work"] / "input_profile.json")


def _run_e15(ctx):
    from upstream.e15_translate import run
    return run(ctx["raw"], ctx["profile_path"], ctx["work"])


def _run_e1(ctx):
    from upstream.e1_semantic import run
    return run(ctx["raw"], ctx["profile_path"], ctx["work"])


def _run_e2(ctx):
    from upstream.e2_sprites import run
    return run(ctx["raw"], ctx["profile_path"], ctx["work"])


def _run_e3(ctx):
    from persona.e3_dna_stats import run
    return run(ctx["raw"], ctx["profile_path"], ctx["work"])


def _run_e4(ctx):
    from persona.e4_persona import run
    return run(ctx["raw"], ctx["profile_path"], ctx["work"])


def _run_e6(ctx):
    from persona.e6_validate import run
    return run(ctx["raw"], ctx["profile_path"], ctx["work"])


def _run_v1(ctx):
    from voice.v1_audio_prep import run
    return run(ctx["raw"], ctx["profile_path"], ctx["work"])


def _run_v2(ctx):
    from voice.v2_train_prep import run
    return run(ctx["raw"], ctx["profile_path"], ctx["work"])


def _run_report(ctx):
    from pack.report_builder import run
    return run(ctx["raw"], ctx["profile_path"], ctx["work"])


def _run_pack(ctx):
    from pack.packager import run
    return run(ctx["raw"], ctx["profile_path"], ctx["work"])


def run_stage(stage: dict, ctx: dict, meter: CostMeter) -> None:
    runners = {"e0_input_check": _run_e0, "e15_translate": _run_e15,
               "e1_semantic": _run_e1, "e2_sprites": _run_e2,
               "e3_dna_stats": _run_e3, "e4_persona": _run_e4,
               "e6_validate": _run_e6, "v1_audio_prep": _run_v1,
               "v2_train_prep": _run_v2, "report_builder": _run_report,
               "pack": _run_pack}
    if stage["name"] not in runners:
        raise NotImplementedError(
            f"工序 {stage['name']} 尚未实现（后续里程碑落地）：{stage['summary']}")
    print(f"[build] ▶ {stage['name']}")
    runners[stage["name"]](ctx)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="miyako_life build：总编排 DAG")
    parser.add_argument("--profile", default="config/game_9nine_ep1.yaml")
    parser.add_argument("--raw", help="原料区根目录（只读）")
    parser.add_argument("--resume", action="store_true",
                        help="断点续跑：样本组缓存保留，只补缺口（L2.3 生效）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印工序计划与依赖序，不执行")
    args = parser.parse_args(argv)

    work_dir = REPO_ROOT / "work"
    setup_logging(work_dir)
    log = get_logger("build")
    profile_path = REPO_ROOT / args.profile
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    ordered = topo_sort(STAGES)
    meter = CostMeter(profile["cost"]["quota_warning"], work_dir / "llm_cost.json")

    run_separator("BUILD START", f"profile={args.profile}")
    log.info("run begin profile=%s game=%s bilingual_source=%s（锁定态） raw=%s "
             "resume=%s stages=%d", args.profile, profile["game_id"],
             profile["input"]["bilingual_source"], args.raw, args.resume,
             len(ordered))

    print(f"[build] profile={args.profile} game={profile['game_id']} "
          f"bilingual_source={profile['input']['bilingual_source']}（锁定态，只读）")
    print(f"[build] resume={args.resume} quota_warning={meter.quota_warning}元 "
          f"工作区=work/（gitignore）")
    print(f"[build] 工序计划（依赖拓扑序，共 {len(ordered)} 序）：")
    for i, stage in enumerate(ordered, 1):
        mark = "impl" if stage["impl"] else "stub"
        print(f"  {i:2d}. {stage['name']:<16} [{mark}] deps={stage['deps'] or '—'}"
              f"  {stage['summary']}")
    if args.dry_run:
        print(f"[build] dry-run 完毕：{sum(s['impl'] for s in ordered)}/{len(ordered)} 工序已实现")
        log.info("dry-run 完毕 %d/%d 工序已实现",
                 sum(s["impl"] for s in ordered), len(ordered))
        run_separator("BUILD DRY-RUN END")
        return 0

    if not args.raw:
        parser.error("--raw 必填（原料区根目录）；仅 --dry-run 可省略")
    ctx = {"raw": Path(args.raw), "profile_path": profile_path,
           "profile": profile, "work": work_dir}
    for stage in ordered:
        t0 = time.perf_counter()
        log.info("▶ %s start", stage["name"])
        try:
            run_stage(stage, ctx, meter)
        except SystemExit as exc:
            log.error("✗ %s exited（code=%s）——fail-closed 闸门或 CLI 退出",
                      stage["name"], exc.code)
            raise
        except Exception:
            log.exception("✗ %s failed（进行到 %.1fs 处）",
                          stage["name"], time.perf_counter() - t0)
            raise
        log.info("✓ %s done %.1fs", stage["name"], time.perf_counter() - t0)
    # 工序各自 merge+落盘账本；顶层以落盘文件为准回读后再 save——
    # 否则空账覆盖 work/llm_cost.json（审评 P1-3，2026-09-12）
    if meter.path.is_file():
        meter.records = json.loads(
            meter.path.read_text(encoding="utf-8")).get("records", [])
    path = meter.save()
    warn = " ⚠️ 超阈值（报告页警示，不中断）" if meter.over_quota else ""
    print(f"[build] 全部工序完成；LLM 成本 {meter.total_yuan} 元 → {path}{warn}")
    log.info("run end 全部工序完成 LLM成本=%.4f元 账本=%s", meter.total_yuan, path)
    run_separator("BUILD END")
    return 0


if __name__ == "__main__":
    sys.exit(main())
