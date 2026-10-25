"""factory_check —— L4.3 出厂检查单（fail-closed：任一项不过不出包，plan §4.1）。

检查项（对齐 pet preflight + life 扩展；协调单 L-3 增两项）：
  1  char_schema 全件校验（card/manifest/relationship/tts_config 四 schema）
  2  9 元情绪闭集（manifest emotion_set 与 contracts 互锁、tone_examples 键 ⊆）
  3  日常态词表复扫零命中（card/relationship/tts_config 全文）
  4  tone_examples 弧线两端（章节回查：head 三分区 ≥1 句 + tail 三分区 ≥1 句）
  5  behavior_rules / anti_patterns / honesty_boundary.notes 三字段非空（协调单 L-3①）
  6  persona 段 estimate_tokens ≤ max_prompt_tokens（协调单 L-3②；硬线 = pet system
     池 0.65×CONTEXT_BUDGET_TOKENS=2662——记忆段初期存活的运行时断点；
     0.30×4096=1229 为舒适线，只记 advisory 不拦包，见 plan v2.8 注记）
  7  verbal_tics 成对概率合计 ≤0.6
  8  voice_tier 一致性（tier≥1 → edge_voice 非空；--voice-less → tier=0）
  9  sprites 直通一致性（包内 manifest.json == E2 验收源）+ version 键在位
 10  决策账本无 stale（decisions.yaml 任一 stale → 不出包——Y5 闭环）
约束 A：本文件不 import 任何工序目录（章节回查为 E3 同口径的轻量内联重实现）。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator

from contracts.emotion_defs import SENTIMENT_CLOSE_SET
from contracts.estimation import estimate_tokens

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = REPO_ROOT / "contracts" / "char_schema"


def _load(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else None


def _schema_check(pkg: Path, name: str, schema_file: str) -> dict:
    f = pkg / name
    if not f.is_file():
        return {"item": f"schema:{name}", "pass": False, "detail": "文件缺失"}
    doc = _load(f)
    schema = json.loads((SCHEMA_DIR / schema_file).read_text(encoding="utf-8"))
    errors = list(Draft202012Validator(schema).iter_errors(doc))
    return {"item": f"schema:{name}", "pass": not errors,
            "detail": "; ".join(e.message[:80] for e in errors[:3]) or "ok"}


def _closed_set_check(pkg: Path) -> dict:
    """闭集互锁锚：sprites manifest（emotion_set/查表键）+ 顶层 axes.expression。"""
    sm_path = pkg / "sprites" / "manifest.json"
    sm = json.loads(sm_path.read_text(encoding="utf-8")) if sm_path.is_file() else {}
    card = _load(pkg / "character_card.yaml") or {}
    m = _load(pkg / "manifest.yaml") or {}
    axes = ((m.get("assets") or {}).get("sprite") or {}).get("axes") or {}
    ok = (sorted(sm.get("emotion_set") or []) == sorted(SENTIMENT_CLOSE_SET)
          and set(sm.get("sprites") or {}) <= set(SENTIMENT_CLOSE_SET)
          and set((card.get("tone_examples") or {})) <= set(SENTIMENT_CLOSE_SET)
          and axes.get("expression") == list(SENTIMENT_CLOSE_SET))
    return {"item": "closed_set", "pass": ok,
            "detail": f"sprites 闭集={len(sm.get('emotion_set') or [])} 元 / 查表键 {len(sm.get('sprites') or {})} / axes.expression {'✓' if axes.get('expression') else '✗'}"}


def _blocklist_rescan(pkg: Path, blocklist: list[str]) -> dict:
    if not blocklist:
        return {"item": "blocklist_rescan", "pass": True,
                "detail": "空表（宁空勿造）——零命中恒成立"}
    blob = "\n".join(
        (pkg / n).read_text(encoding="utf-8") for n in
        ("character_card.yaml", "relationship.yaml", "tts_config.yaml")
        if (pkg / n).is_file())
    hits = [w for w in blocklist if w in blob]
    return {"item": "blocklist_rescan", "pass": not hits,
            "detail": f"命中 {hits}" if hits else "零命中"}


def _arc_ends_check(pkg: Path, work_dir: Path, raw_root: Path,
                    profile: dict) -> dict:
    """tone_examples 章节回查：voice_map 文本 → scn 首现章节；head/tail 各取
    章节三分区，要求各 ≥1 句。文本两侧必须同款归一化（折叠空白）——卡面句
    经 _norm 清洗，vm 原文含折行/全角空格，裸比对会漏配（2026-09-16 实证）。
    约束 A：不 import 工序目录，归一化为内联同口径实现。"""

    def _norm(text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip()

    vm_path = Path(work_dir) / "voice_map.json"
    if not vm_path.is_file():
        return {"item": "arc_ends", "pass": False, "detail": "voice_map 缺失"}
    vm = json.loads(vm_path.read_text(encoding="utf-8"))
    rules = _load(REPO_ROOT / "config" / "extract_rules.yaml") or {}
    core = rules.get("voice", {}).get("id_pattern", r"my\d{4}[a-z]?").strip("^$")
    chap_re = re.compile(rf'"voice"\s*:\s*"({core})"')
    scn_dir = Path(raw_root) / Path(profile["engine"]["scn_pattern"]).parts[0]
    chap: dict[str, str] = {}
    for f in sorted(scn_dir.glob("*.json")):
        if ".resx." in f.name:
            continue
        ch = f.name.split(".")[0]
        for m in chap_re.finditer(f.read_text(encoding="utf-8-sig")):
            chap.setdefault(m.group(1), ch)
    text2chap: dict[str, str] = {}
    for e in vm:
        t = _norm(e.get("text_zh") or "")
        if t and e.get("voice") in chap:
            text2chap.setdefault(t, chap[e["voice"]])
    card = _load(pkg / "character_card.yaml") or {}
    # 章节全集剔除 persona.exclude_chapters（2026-09-19 修复：与 assemble_card 的
    # records 口径对齐——a15 演出章 53 条被池剔除，永不可能出卡面句；留全集会把
    # tail 三分区错位一格（a12 被判 mid），尾段保证替换的 a12 句过不了本检查）
    exclude = set((profile.get("persona", {}) or {}).get("exclude_chapters", []))
    all_ch = sorted({c for c in set(chap.values()) if c not in exclude})
    third = max(1, len(all_ch) // 3)
    head_zone, tail_zone = set(all_ch[:third]), set(all_ch[-third:])
    head_ok = tail_ok = False
    for lines in (card.get("tone_examples") or {}).values():
        for line in lines:
            zh = _norm(line.split("|", 1)[0].split("]", 1)[-1])
            ch = text2chap.get(zh)
            if ch is None:
                continue
            head_ok = head_ok or ch in head_zone
            tail_ok = tail_ok or ch in tail_zone
    return {"item": "arc_ends", "pass": head_ok and tail_ok,
            "detail": f"head 三分区命中={head_ok} / tail 三分区命中={tail_ok}（{all_ch[0] if all_ch else '—'}~{all_ch[-1] if all_ch else '—'}）"}


def _three_fields_check(pkg: Path) -> dict:
    card = _load(pkg / "character_card.yaml") or {}
    ok = (bool(card.get("behavior_rules"))
          and bool(card.get("anti_patterns"))
          and bool((card.get("honesty_boundary") or {}).get("notes")))
    return {"item": "three_fields_nonempty", "pass": ok,
            "detail": f"rules={len(card.get('behavior_rules') or [])} anti={len(card.get('anti_patterns') or [])} honesty={'有' if (card.get('honesty_boundary') or {}).get('notes') else '无'}（P-1 消费前提）"}


def _prompt_budget_check(pkg: Path, profile: dict) -> dict:
    card = _load(pkg / "character_card.yaml") or {}
    est = estimate_tokens(yaml.safe_dump(card, allow_unicode=True, sort_keys=False))
    hard = int(profile.get("pack", {}).get("max_prompt_tokens", 2662))
    comfy = 1229   # 0.30×4096（pet 现行比例舒适线；容器升级后随 profile 走）
    advisory = "；超舒适线（D16 路径：pet 容器升级或 P1 二次密化）" if est > comfy else ""
    return {"item": "prompt_budget", "pass": est <= hard,
            "detail": f"{est} token（硬线 {hard}=system 池；舒适线 {comfy}=0.30×4096{advisory}）"}


def _verbal_tics_check(pkg: Path) -> dict:
    t = _load(pkg / "tts_config.yaml") or {}
    total = sum(p.get("rate_ja", 0) + p.get("rate_zh", 0)
                for p in (t.get("verbal_tics") or []))
    return {"item": "verbal_tics_pair_cap", "pass": total <= 0.6,
            "detail": f"成对合计 {total}（≤0.6）"}


def _tier_check(pkg: Path, voice_less: bool) -> dict:
    m = _load(pkg / "manifest.yaml") or {}
    t = _load(pkg / "tts_config.yaml") or {}
    tier = m.get("voice_tier")
    edge = (t.get("edge_voice") or "").strip()
    if voice_less:
        ok = tier == 0 and not edge
    elif tier == 1:
        ok = bool(edge)
    else:
        ok = tier == 2
    return {"item": "tier_consistency", "pass": ok,
            "detail": f"voice_tier={tier} edge_voice={'非空' if edge else '空'} voice_less={voice_less}"}


def _sprites_passthrough_check(pkg: Path, work_dir: Path) -> dict:
    pkg_m = pkg / "sprites" / "manifest.json"
    src_m = Path(work_dir) / "sprites_manifest.json"
    if not pkg_m.is_file() or not src_m.is_file():
        return {"item": "sprites_passthrough", "pass": False, "detail": "文件缺失"}
    ok = json.loads(pkg_m.read_text(encoding="utf-8")) == \
        json.loads(src_m.read_text(encoding="utf-8"))
    m = json.loads(pkg_m.read_text(encoding="utf-8"))
    return {"item": "sprites_passthrough", "pass": ok and bool(m.get("version")),
            "detail": f"直通一致={ok} / version 键={m.get('version')}（P-3 纪律）"}


def _decisions_check(work_dir: Path) -> dict:
    from pack.report_builder import decisions_status, load_decisions
    decisions = load_decisions(work_dir)
    # 2026-09-19 复查裁定：voice-less 诊断包**不豁免**本项（复查建议驳回）——
    # voice-less 包会替换 dist 生产位（9-16 镜像污染事故即 voice-less 包所致），
    # 不是无害诊断物；"干净 work 无账本"场景本就会在更早的 card 读取处失败，
    # item 13 拦截有明确报错。两形态一致 fail-closed。
    if not decisions:
        # 账本缺失/为空 ≠ "无 stale"——Y5 闭环无账可查，宁严勿松（2026-09-18 修正：
        # 原实现零行全 pass，"账本丢失"与"尚无决策"都静默放行）
        return {"item": "decisions_fresh", "pass": False,
                "detail": "决策账本缺失/为空（work/decisions.yaml）——"
                          "Y5 闭环无账可查，不出包"}
    rows = decisions_status(work_dir, decisions)
    stale = [r["key"] for r in rows if r["stale"]]
    return {"item": "decisions_fresh", "pass": not stale,
            "detail": f"stale={stale}" if stale else f"{len(rows)} 条决策全部新鲜"}


def run_factory_check(pkg: Path, work_dir: Path, raw_root: Path,
                      profile: dict, voice_less: bool = False) -> tuple[bool, list[dict]]:
    pkg = Path(pkg)
    report = [
        _schema_check(pkg, "character_card.yaml", "character_card.schema.json"),
        _schema_check(pkg, "manifest.yaml", "manifest.schema.json"),
        _schema_check(pkg, "relationship.yaml", "relationship.schema.json"),
        _schema_check(pkg, "tts_config.yaml", "tts_config.schema.json"),
        _closed_set_check(pkg),
        _blocklist_rescan(pkg, profile.get("persona", {}).get("daily_corpus_blocklist", [])),
        _arc_ends_check(pkg, work_dir, raw_root, profile),
        _three_fields_check(pkg),
        _prompt_budget_check(pkg, profile),
        _verbal_tics_check(pkg),
        _tier_check(pkg, voice_less),
        _sprites_passthrough_check(pkg, work_dir),
        _decisions_check(work_dir),
    ]
    return all(r["pass"] for r in report), report
