"""E0 输入验证 —— 第 0 号前置工序（plan L-0.2 定案；语义权威 = 方案契约 v1.0.3 §五 E0）。

三探测项：
  ① scn 文本语言判定（日文假名占比 / 中文汉字占比）；
  ② 伴生汉化脚本存在性（汉化补丁常见文件名模式扫描 + resx 有效条目计数）；
  ③ 若双语存在：逐 voice_id 对齐率（translate 路径下 N/A → null）。

产出：work/input_profile.json
  {bilingual_source, source: locked|detected, alignment_rate, probes{...}}
  → 报告页顶部常驻显示。

锁定语义（L-0.2 验收项：profile 锁定不可被 build 覆写）：
  profile input.bilingual_source 非 auto 即锁定态——本工序只校验探测结论与锁定值
  一致，不一致 fail-fast（原料变更或锁定失效），绝不回填、绝不改写 profile；
  仅 auto 态才探测后回填实际值并锁定（契约 §3.2）。
  2026-09-11 用户定案：EP1 锁定 embedded——Steam 官方三语内嵌 scn
  （每句 [日文, 英文, 繁中]，证据 = steam 台本目录的多语言节点），
  原 translate 锁定（resx 空壳时代）降级为历史锚。

fail-fast（上游错污染全链，plan §4.1）：
  * 原料区缺失 / 无 scn 文件 → 终止；
  * 锁定值 != 探测结论 → 终止；
  * embedded 模式且 alignment_rate < 0.90 → 终止（错位对齐会污染 verbal_tics 成对统计）。

对齐率探测③说明：translate 锁定路径下恒为 null（不参与统计）。embedded/patch
路径的逐 voice_id 对齐在 extract_rules.yaml（L0.4）落盘前用文件名存在性启发式
预实现，L1.4 收尾补全并挂单测（plan L1.4）。

用法：
  python upstream/e0_input_check.py --raw <原料区根> \
      [--profile config/game_9nine_ep1.yaml] [--out work/input_profile.json]

本模块只依赖 stdlib + PyYAML，不 import 任何其他工序目录（约束 A）。
"""

from __future__ import annotations

# 直跑脚本兜底（python <subdir>/<file>.py）：sys.path[0] 是本目录而非仓库根，
# contracts 导入先于 __main__ 里的 sys.path.insert 执行——必须在导入前自举。
# （包内导入/-m 方式 __package__ 非空，自动跳过；_sys/_Path 避免遮蔽下方正式导入）
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

from contracts.logging_util import get_logger

LOG = get_logger("e0")

# ── 探测阈值与采样上限（确定性常量；调整即属实现变更，需过 golden 对账）──────
KANA_RATIO_JA = 0.10        # 假名/(假名+汉字) ≥ 此值 → 日文 scn（日文文本假名占比通常 0.3+）
KANA_RATIO_ZH = 0.02        # < 此值且含汉字 → 中文主导 scn（embedded 候选）
ALIGNMENT_MIN = 0.90        # 契约 §五 E0：低于此值禁止 embedded 模式继续
MAX_SCN_FILES = 50          # 语言判定采样文件数上限
PER_FILE_SAMPLE_BYTES = 400_000   # 语言判定单文件采样字节上限（EP1 实测单章 ≈2.5MB）
MAX_SAMPLE_BYTES = 10_000_000     # 语言判定采样总字节上限（保证多章覆盖，不赌单章）
MAX_RESX_FILES = 200        # 汉化资源扫描文件数上限

# 汉化补丁常见文件名模式（探测②；命中只报存在性，有效性由内容计数判定）
PATCH_NAME_PATTERNS = ["汉化", "chinese", "zh_CN", "zh_TW", "zh-Hans", "zh-Hant"]

# Steam 官方三语内嵌（embedded · 2026-09-11）：中文不在 resx，而在 .scn 的
# 多语言节点里。证据 = steam 台本目录存在含 ≥3 语言子表的台词节点
#（probe_multilang，经反编译 json 判定）。

# resx 家族：.resx（XML 资源）与 .resx.json（PSB 工具导出的 json 化 resx，
# EP1 实测形态：156 字节空壳头 PsbVersion/PsbType/Context{}/Resources{}）
_RESX_META_KEYS = {"PsbVersion", "PsbType", "Platform", "CryptKey",
                   "ExternalTextures", "Context", "Resources"}

_RESX_DATA_RE = re.compile(r"<data[\s>][\s\S]*?</data>")
_RESX_VALUE_RE = re.compile(r"<value>\s*\S[\s\S]*?</value>")

# ── steam 台本目录约定 ────────────────────────────────────────────────────
def steam_json_dir(raw_root: Path, profile: dict) -> Path:
    """多语言台本目录：profile input.steam_json_dir 覆盖，缺省=原料区同级的材料库。"""
    override = (profile.get("input", {}) or {}).get("steam_json_dir")
    if override:
        return Path(override)
    return raw_root.parent / "miyako_life材料库" / "分类整理" / "01_日语台本"


def probe_multilang(steam_dir: Path) -> dict:
    """探测 Steam 三语台本：反编译 json 中存在 ≥3 语言子表的台词节点 → embedded 证据。"""
    d = Path(steam_dir)
    if not d.is_dir():
        return {"files": 0, "multilang_files": 0, "evidence": False}
    files = multilang_files = 0
    for f in sorted(d.glob("a*.steam.json")):
        files += 1
        try:
            data = json.loads(f.read_text(encoding="utf-8-sig"))
        except (json.JSONDecodeError, OSError):
            continue
        found = False
        for scene in data.get("scenes", []):
            for t in scene.get("texts", []):
                if (isinstance(t, list) and len(t) > 2
                        and isinstance(t[2], list) and len(t[2]) >= 3):
                    found = True
                    break
            if found:
                break
        if found:
            multilang_files += 1
    return {"files": files, "multilang_files": multilang_files,
            "evidence": multilang_files > 0}


class E0FailFast(RuntimeError):
    """E0 探测失败/校验不一致——上游 fail-fast，禁止 build 继续。"""


# ── 探测①：scn 语言判定 ────────────────────────────────────────────────────
def _cjk_ratios(text: str) -> tuple[int, int]:
    """返回 (假名字符数, 汉字字符数)。只统计 CJK 两类，结构与引擎无关。"""
    kana = sum(1 for ch in text if "\u3040" <= ch <= "\u30ff")
    han = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return kana, han


def probe_language(raw_root: Path, scn_files: list[Path]) -> dict:
    kana = han = 0
    scanned = 0
    budget = MAX_SAMPLE_BYTES
    for f in scn_files[:MAX_SCN_FILES]:
        if budget <= 0:
            break
        text = f.read_text(encoding="utf-8", errors="replace")[:PER_FILE_SAMPLE_BYTES]
        budget -= PER_FILE_SAMPLE_BYTES
        k, h = _cjk_ratios(text)
        kana += k
        han += h
        scanned += 1
    total = kana + han
    if total == 0:
        raise E0FailFast(
            f"scn 采样（{scanned} 个文件）中未发现任何 CJK 文本——原料区可能不是台本目录"
        )
    kana_ratio = kana / total
    han_ratio = han / total
    if kana_ratio >= KANA_RATIO_JA:
        detected_lang = "ja"
    elif kana_ratio < KANA_RATIO_ZH:
        detected_lang = "zh"
    else:
        detected_lang = "mixed"
    return {
        "scn_files_scanned": scanned,
        "kana_ratio": round(kana_ratio, 4),
        "han_ratio": round(han_ratio, 4),
        "detected_lang": detected_lang,
    }


# ── 探测②：伴生汉化脚本存在性 ─────────────────────────────────────────────
def _resx_json_effective_entries(path: Path) -> int:
    """.resx.json（PSB 导出）有效条目 = 元数据键之外的非空叶子字符串数（空壳头 → 0）。"""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return 0

    def _count(node: object) -> int:
        if isinstance(node, str):
            return 1 if node.strip() else 0
        if isinstance(node, dict):
            return sum(_count(v) for k, v in node.items() if k not in _RESX_META_KEYS)
        if isinstance(node, list):
            return sum(_count(v) for v in node)
        return 0

    return _count(doc)


def probe_patch(raw_root: Path) -> dict:
    resx_files = sorted(
        list(raw_root.rglob("*.resx")) + list(raw_root.rglob("*.resx.json"))
    )[:MAX_RESX_FILES]
    effective_entries = 0
    for f in resx_files:
        if f.suffix == ".json":  # .resx.json
            effective_entries += _resx_json_effective_entries(f)
            continue
        text = f.read_text(encoding="utf-8", errors="replace")
        for chunk in _RESX_DATA_RE.findall(text):
            if _RESX_VALUE_RE.search(chunk):
                effective_entries += 1
    name_hits = sorted(
        {
            str(p.relative_to(raw_root))
            for p in raw_root.rglob("*")
            if p.is_file()
            and any(pat.lower() in p.name.lower() for pat in PATCH_NAME_PATTERNS)
        }
    )
    return {
        "resx_files_found": len(resx_files),
        "resx_effective_entries": effective_entries,
        "patch_name_hits": name_hits,
        "patch_effective": effective_entries > 0,
    }


# ── 探测③：逐 voice_id 对齐率（embedded/patch 路径；translate 恒 null）────
def probe_alignment(raw_root: Path, profile: dict, scn_files: list[Path]) -> float:
    """对齐率 = 台本 voice_id 引用命中 voice/ 磁盘文件的比例（extract_rules 语义）。

    embedded/patch 路径要求中日文本逐 voice_id 可对齐——本探测是文件存在性层，
    文本级对齐由 E1（extractors）承担；L1.4 起与 E1 共用同一提取器，口径一致。
    """
    from upstream.extractors.kirikiri_9nine import Kirikiri9nineExtractor

    voice_dir = (raw_root / profile["engine"]["voice_dir"]).resolve()
    extractor = Kirikiri9nineExtractor()
    referenced = {e["voice"] for e in extractor.extract_files(scn_files)}
    known = extractor.voice_stems(voice_dir)
    if not known or not referenced:
        return 0.0
    hit = len(referenced & known)
    return round(hit / len(referenced), 4)


# ── 主流程 ────────────────────────────────────────────────────────────────
def decide_source(lang_probe: dict, patch_probe: dict, multilang_probe: dict) -> str:
    """探测结论 → bilingual_source（契约 §五 E0：auto 态回填用；锁定态仅用于比对）。"""
    if multilang_probe.get("evidence"):
        return "embedded"  # Steam 官方三语内嵌（日/英/繁同节点）
    if lang_probe["detected_lang"] in ("zh", "mixed"):
        return "embedded"  # scn 自身中文主导 → 内嵌双语/汉化台本
    if patch_probe["patch_effective"]:
        return "patch"
    return "translate"  # 日文 scn + 无有效汉化 → 翻译路径


def run(raw_root: Path, profile_path: Path, out_path: Path) -> dict:
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    pattern = profile["engine"]["scn_pattern"]
    scn_files = sorted(raw_root.glob(pattern)) if raw_root.is_dir() else []
    if not scn_files:
        raise E0FailFast(
            f"原料区缺失或无匹配台本：raw={raw_root} pattern={pattern}"
            f"（E0 上游 fail-fast：先确认 --raw 指向解包原料区）"
        )

    lang_probe = probe_language(raw_root, scn_files)
    patch_probe = probe_patch(raw_root)
    sdir = steam_json_dir(raw_root, profile)
    multilang_probe = probe_multilang(sdir)
    detected = decide_source(lang_probe, patch_probe, multilang_probe)
    LOG.info("探测完成 scn_files=%d lang=%s patch_effective=%s multilang=%s "
             "→ detected=%s", len(scn_files), lang_probe["detected_lang"],
             patch_probe["patch_effective"], multilang_probe["evidence"], detected)

    locked = profile["input"]["bilingual_source"]
    if locked != "auto":
        # 锁定态：只校验不回填（profile 锁定不可被 build/E0 覆写——L-0.2 验收项）
        if detected != locked:
            hint = "" if multilang_probe["evidence"] else (
                f"；embedded 锁定需 steam 台本目录含多语言 json（当前探测：{sdir}，"
                f"files={multilang_probe['files']}）")
            raise E0FailFast(
                f"bilingual_source 锁定值 {locked!r} 与探测结论 {detected!r} 不一致"
                f"（lang={lang_probe['detected_lang']}, "
                f"patch_effective={patch_probe['patch_effective']}, "
                f"multilang={multilang_probe['evidence']}）"
                f"——原料已变更或锁定失效，禁止继续；确认后人工更新 profile{hint}"
            )
        source_final, source_state = locked, "locked"
    else:
        source_final, source_state = detected, "detected"

    if source_final in ("embedded", "patch"):
        alignment = probe_alignment(raw_root, profile, scn_files)
        if source_final == "embedded" and alignment < ALIGNMENT_MIN:
            raise E0FailFast(
                f"alignment_rate={alignment} < {ALIGNMENT_MIN}：禁止以 embedded 模式继续"
                f"（错位对齐会污染 verbal_tics 成对统计——比缺中文更危险）"
            )
    else:
        alignment = None  # translate 路径：对齐率 N/A

    result = {
        "bilingual_source": source_final,
        "source": source_state,
        "alignment_rate": alignment,
        "probes": {"language": lang_probe, "patch": patch_probe,
                   "multilang": multilang_probe},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    LOG.info("input_profile 落盘 source=%s(%s) alignment=%s → %s",
             result["bilingual_source"], source_state, alignment, out_path)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E0 输入验证（三探测项 → input_profile.json）")
    parser.add_argument("--raw", required=True, help="原料区根目录（只读）")
    parser.add_argument("--profile", default="config/game_9nine_ep1.yaml")
    parser.add_argument("--out", default="work/input_profile.json")
    args = parser.parse_args(argv)
    from contracts.logging_util import setup_logging
    setup_logging(Path(args.out).parent)   # 直跑也落 work/logs/pipeline.log

    try:
        result = run(Path(args.raw), Path(args.profile), Path(args.out))
    except E0FailFast as exc:
        LOG.error("fail-fast %s", exc)
        print(f"[E0][fail-fast] {exc}", file=sys.stderr)
        return 2
    print(
        f"[E0] bilingual_source={result['bilingual_source']} ({result['source']})"
        f" alignment_rate={result['alignment_rate']} → {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
