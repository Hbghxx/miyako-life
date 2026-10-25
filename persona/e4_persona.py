"""E4 五层提炼（induce 槽）—— L2.3 实现：抽样 → 样本组提炼 → 合并 → card 组装。

输入纪律（2026-09-11 origin 定案）：机翻(translated)禁入 P1/P2/P4；
官方汉化(localized)准入；daily_corpus_blocklist 命中台词过滤（词表由 E3+人工
产出，E4 不生成自己的输入过滤器——自证隔离）。
断点粒度 = 样本组（弧线段×情绪桶），缓存 work/checkpoints/，重跑只补缺口；
缓存身份 = 组名 + 样本集 sha256（池/配额/分区变更 → 旧缓存自动失效重算）。
refine ≤2（Y9）：输出不达标自动重试，仍败 → 全量扫描兜底档 + 转人工（不静默循环）。
成本：逐调用记账（induce 槽 glm-4.7），并入 work/llm_cost.json。
产出：work/e4_persona.json（五层+证据）+ work/character_card.yaml（过 char_schema）。
"""

from __future__ import annotations

# 直跑脚本兜底（python <subdir>/<file>.py）：sys.path[0] 是本目录而非仓库根，
# contracts 导入先于 __main__ 里的 sys.path.insert 执行——必须在导入前自举。
# （包内导入/-m 方式 __package__ 非空，自动跳过；_sys/_Path 避免遮蔽下方正式导入）
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import hashlib
import json
import re
import time
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator

from contracts.cost import CostMeter
from contracts.estimation import estimate_tokens
from contracts.llm import base_url_from_env, chat_completion, env_key
from contracts.logging_util import get_logger

LOG = get_logger("e4")
from persona.e3_dna_stats import _scn_texts, _voice_chapter_map

REPO_ROOT = Path(__file__).resolve().parents[1]
FACE_MAP_PATH = (REPO_ROOT / "contracts" / "face_map.yaml").resolve()
EXTRACT_RULES_PATH = (REPO_ROOT / "config" / "extract_rules.yaml").resolve()
SCHEMA_PATH = (REPO_ROOT / "contracts" / "char_schema" / "character_card.schema.json").resolve()
PROMPTS_DIR = (REPO_ROOT / "persona" / "prompts").resolve()
ENV_PATH = REPO_ROOT / ".env"
TENSION_SEEDS_DEFAULT = ("殺人", "犯人", "死", "異能", "能力")
# 身份字段已 profile 化（2026-09-13 用户定案：character.name/name_zh/name_en/
# display_name/calls_you）——原 CARD_IDENTITY 硬编码与 calls_you="你" 缺省废除，
# 重跑不再回退用户偏好。
CARD_TABOOS = [  # 产品级护栏（与角色无关，原型同款）
    "不扮演用户的人类伴侣或医生等现实身份",
    "被以恋人或伴侣等现实关系试探时，温和而明确地划清界限，"
    "不给出暧昧、留有余地或期待下次的回应",  # 2026-09-12 收紧（E6 t02 裁决：回应含糊=未达界）
    "不输出现实世界的违法指导",
    "不在引语外使用夸张的动作描写（动作描写交给 voice_postprocess 剥离前的括号）",
]
PREFILL_CONDITIONALS = ("たら", "なら", "れば", "もし", "如果", "要是", "的话")
FALLBACK_SCAN_CAP = 200_000
INDUCE_RATE_DEFAULT = 0.05   # 元/千 tokens 估算占位，glm-4.7 以账单为准


# ═══ L0.5 锁定的助手（过滤/断点/refine——签名与语义不变）═════════════════
CHECKPOINTS_DIR = "checkpoints"


def apply_daily_filter(entries: list[dict], blocklist: list[str]) -> list[dict]:
    """按日常态词表过滤提炼输入：命中词表实体的台词不进提炼。空表 = 恒等（宁空勿造）。"""
    if not blocklist:
        return list(entries)
    return [e for e in entries if not any(term in e["text"] for term in blocklist)]


def _group_path(work_dir: Path, group_id: str) -> Path:
    return Path(work_dir) / CHECKPOINTS_DIR / f"{group_id}.json"


def save_group(work_dir: Path, group_id: str, result: dict) -> Path:
    path = _group_path(work_dir, group_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    return path


def load_group(work_dir: Path, group_id: str) -> dict | None:
    path = _group_path(work_dir, group_id)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def pending_groups(group_ids: list[str], work_dir: Path) -> list[str]:
    """断点续跑的缺口集合：缓存里没有的样本组才需要重烧 API 钱。"""
    return [g for g in group_ids if load_group(work_dir, g) is None]


def refine_loop(quality_fn, max_rounds: int = 2) -> tuple[bool, int]:
    """Y9 refine 上限：quality_fn(round_no) -> bool；返回 (accepted, rounds_used)。"""
    for round_no in range(1, max_rounds + 1):
        if quality_fn(round_no):
            return True, round_no
    return False, max_rounds


# ═══ 语料准备（确定性，L4）══════════════════════════════════════════════
def _load_face_emotion_map() -> dict:
    fm = yaml.safe_load(FACE_MAP_PATH.read_text(encoding="utf-8"))
    return fm["scn_faces"]


def _load_prompts() -> dict[str, str]:
    out = {}
    for name in ("p1", "p2", "p3", "p4", "p5", "merge"):
        out[name] = (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")
    return out


def _fill(template: str, **kw) -> str:
    s = template
    for k, v in kw.items():
        s = s.replace("{" + k + "}", v)
    return s


def _parse_json(content: str) -> dict:
    """容错解析：剥 markdown 围栏，取首个 { 到末个 }。"""
    s = content.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.M).strip()
    lo, hi = s.find("{"), s.rfind("}")
    if lo < 0 or hi <= lo:
        raise ValueError("输出中未找到 JSON 对象")
    return json.loads(s[lo:hi + 1])


def _extract_evidence_ids(samples: list[str]) -> set[str]:
    return set(re.findall(r"\[(evid_\d+)\]", "\n".join(samples)))


def load_records(entries: list[dict], profile: dict, raw_root: Path) -> list[dict]:
    """都语料：机翻过滤（origin==translated 禁入 P 层）+ 词表过滤 + 演出章排除 +
    章节/情绪标签。

    池纪律三连（全部脚本执法、条件人工裁定——自证隔离）：
    ①origin==translated 禁入（D15）；②daily_corpus_blocklist（E3+人工）；③
    persona.exclude_chapters 演出章排除（2026-09-13 用户裁定：a15 无对话句，
    53 条演出行不进归纳——仅限本池，E3/V1 锚点不受影响）。
    """
    from contracts.emotion_defs import DEFAULT_EMOTION, SENTIMENT_CLOSE_SET

    speaker = profile["character"]["speaker_names"][0]
    blocklist = profile.get("persona", {}).get("daily_corpus_blocklist", [])
    exclude_chapters = set(profile.get("persona", {}).get("exclude_chapters", []))
    scn_texts = _scn_texts(raw_root, profile["engine"]["scn_pattern"])
    id_pattern = yaml.safe_load(
        EXTRACT_RULES_PATH.read_text(encoding="utf-8")
    ).get("voice", {}).get("id_pattern", r"^my\d{4}[a-z]?$")
    chapter_of = _voice_chapter_map(scn_texts, id_pattern)
    face2emo = _load_face_emotion_map()

    records = []
    excluded = 0
    for e in entries:
        if e.get("speaker") != speaker or not e.get("voice"):
            continue
        if e.get("origin") == "translated":
            continue  # 机翻禁入 P 层（D15）
        zh = e.get("text_zh")
        if not zh:
            continue  # P 层消费官方汉化简体；无简体的不入抽样
        face = (e.get("self") or {}).get("face")
        emotion = face2emo.get(face) if face else None
        if emotion not in SENTIMENT_CLOSE_SET:
            emotion = DEFAULT_EMOTION if face else None
        rec = {
            "voice": e["voice"], "text": e["text"], "text_zh": zh,
            "origin": e.get("origin", "native"),
            "face": face, "emotion": emotion,
            "chapter": chapter_of.get(e["voice"], "?"),
        }
        if rec["chapter"] in exclude_chapters:
            excluded += 1
            continue  # 演出章禁入 P 层池（2026-09-13 用户裁定）
        if apply_daily_filter([{"text": rec["text"]}], blocklist):
            records.append(rec)
    if excluded:
        print(f"[E4] 池过滤：演出章 {sorted(exclude_chapters)} 剔除 {excluded} 条"
              f"（不入 P 层归纳；E3/V1 不受影响）")
    return records


def _evenly(records: list[dict], n: int) -> list[dict]:
    """等距抽样（确定性）：按 voice 排序后取 n 条均匀铺开。"""
    ordered = sorted(records, key=lambda r: r["voice"])
    if len(ordered) <= n:
        return ordered
    step = len(ordered) / n
    return [ordered[min(int(i * step), len(ordered) - 1)] for i in range(n)]


def build_groups(records: list[dict], chapters: list[str],
                 quota: dict) -> dict[str, list[dict]]:
    """样本组划分（确定性）：P1=情绪×弧线段、P3=弧线三段、P4=情绪、P5=总览。"""
    from contracts.emotion_defs import SENTIMENT_CLOSE_SET

    half = (len(chapters) + 1) // 2
    seg_chapters = {1: set(chapters[:half]), 2: set(chapters[half:])}
    groups: dict[str, list[dict]] = {}

    p1_quota = quota.get("mind_model", {}).get("per_emotion", 11)
    for emo in SENTIMENT_CLOSE_SET:
        for seg, chs in seg_chapters.items():
            pool = [r for r in records if r["emotion"] == emo and r["chapter"] in chs]
            if pool:
                groups[f"e4_p1_{emo}_s{seg}"] = _evenly(pool, p1_quota)

    p3_quota = quota.get("signature", 30)
    third = max(1, len(chapters) // 3)
    zones = [("head", chapters[:third]), ("mid", chapters[third:2 * third]),
             ("tail", chapters[2 * third:])]
    eligible = [r for r in records if r.get("text_zh")]
    per_zone = max(1, p3_quota // len(zones))
    for name, chs in zones:
        pool = [r for r in eligible if r["chapter"] in chs]
        if pool:
            groups[f"e4_p3_{name}"] = _evenly(pool, per_zone)

    p4_quota = quota.get("heuristics", {}).get("per_situation_class", 10)
    for emo in SENTIMENT_CLOSE_SET:
        pool = [r for r in records if r["emotion"] == emo]
        pool.sort(key=lambda r: (not any(m in r["text"] or m in (r.get("text_zh") or "")
                                         for m in PREFILL_CONDITIONALS), r["voice"]))
        if pool:
            groups[f"e4_p4_{emo}"] = pool[:p4_quota]

    # P2 表达风格样本组（覆盖面断言需要 p2 层的 evidence 条目）
    p2_quota = 10
    for emo in SENTIMENT_CLOSE_SET:
        pool = [r for r in records if r["emotion"] == emo and r.get("text_zh")]
        if pool:
            groups[f"e4_p2_{emo}"] = _evenly(pool, p2_quota)

    overview = _evenly([r for r in records if r.get("emotion")],
                       2 * len(SENTIMENT_CLOSE_SET))
    if overview:
        groups["e4_p5_overview"] = overview
    return groups


def _sample_lines(samples: list[dict], with_ja: bool = False) -> list[str]:
    lines = []
    for i, r in enumerate(samples, 1):
        eid = f"evid_{i:03d}"
        emo = r.get("emotion") or "?"
        base = f"[{eid}][情绪:{emo}][章节:{r['chapter']}] {r['text_zh']}"
        if with_ja:
            base += f" ／日文: {r['text']}"
        lines.append(base)
    return lines


# ═══ 组执行（断点 + refine≤2 + 成本）════════════════════════════════════
REQUIRED_KEYS = {
    "p1": ("description", "traits", "values", "core_tensions", "evidence"),
    "p2": ("register", "evidence"),
    "p3": ("tone_examples",),
    "p4": ("behavior_rules",),
    "p5": ("anti_patterns", "evidence"),
}


def samples_signature(samples: list[dict]) -> str:
    """样本组内容指纹（L1 失效传播）：样本内容或池/配额/分区一变 → 指纹变 →
    缓存失效重算。

    2026-09-18 修正：原实现只哈希 voice id 集合——E1 同 id 换内容（官方汉化
    更新、tw2s 修正、清洗口径变更）时旧缓存全部命中，persona 静默沿用旧语料，
    违背 docstring 声明与 Y5"不静默沿用"。改哈希提示词实际消费的全字段
    （voice/emotion/chapter/text_zh/text——p3 的日文行含 text，已覆盖）。
    旧格式缓存按既定语义视为 stale 宁严勿松（plan v2.3 先例；下次 E4 重算
    一次付清）。"""
    fields = ("voice", "emotion", "chapter", "text_zh", "text")
    return hashlib.sha256(
        json.dumps([[r.get(k) for k in fields] for r in samples],
                   ensure_ascii=False).encode("utf-8")).hexdigest()[:12]


def _norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def p3_quotes_verbatim(output: dict, samples: list[dict]) -> tuple[bool, str]:
    """P3 语气示例**逐字引用**校验（统计归脚本纪律，2026-09-19 新增）。

    每条 zh 必须与组内样本 text_zh（归一化后）完全一致——LLM 漂移引用（截断/
    拼接/改写）会使语气示例无法对回语料：arc_ends 章节映射失效（今日实证
    p3_tail 8/8 漂移 → 尾锚断）、pet few-shot 忠实度受损。refine 循环凭本
    校验强制重抄原句。
    """
    corpus = {_norm_ws(r.get("text_zh") or "") for r in samples}
    corpus.discard("")
    for i, ex in enumerate(output.get("tone_examples", []) or [], 1):
        zh = _norm_ws(str(ex.get("zh", "") or "")) if isinstance(ex, dict) else ""
        if not zh:
            return False, f"第{i}条 zh 为空"
        if zh not in corpus:
            return False, (f"第{i}条非逐字引用（必须从样本 text_zh 原句整句复制，"
                           f"不得截断/拼接/改写）：{zh[:24]}…")
    return True, ""


def _run_group(gid: str, layer: str, samples: list[dict], prompt: str,
               model: str, client, meter: CostMeter, work_dir: Path,
               max_rounds: int = 2) -> dict:
    # 样本组身份 = 组名 + 样本集内容 hash（L1 失效传播：池/配额/分区一变，
    # 旧缓存自动失效重算——否则池排除等输入纪律变更不会真正生效，2026-09-13）
    samples_sig = samples_signature(samples)
    cached = load_group(work_dir, gid)
    if (cached and cached.get("samples_sha256") == samples_sig
            and "output" in cached and not cached.get("failed")):
        LOG.info("组缓存命中 %s（指纹 %s）", gid, samples_sig)
        return {"gid": gid, "output": cached["output"], "cached": True,
                "rounds": cached.get("rounds", 1), "failed": False}

    ids = _extract_evidence_ids(_sample_lines(samples))
    sample_text = "\n".join(_sample_lines(samples, with_ja=(layer == "p3")))
    last_err = ""
    result: dict = {}

    def quality(round_no: int) -> bool:
        nonlocal last_err
        feedback = f"\n\n上一次输出不合格（{last_err}），请修正后重新输出严格 JSON。" \
            if last_err else ""
        messages = [{"role": "user",
                     "content": _fill(prompt, samples=sample_text) + feedback}]
        try:
            content, usage = client(model, messages)
        except Exception as exc:  # noqa: BLE001 —— 单组失败走 refine/兜底，不炸全局
            last_err = f"调用失败 {exc.__class__.__name__}"
            return False
        meter.record("induce", model, usage.get("prompt_tokens", 0),
                     usage.get("completion_tokens", 0),
                     (usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0))
                     / 1000 * INDUCE_RATE_DEFAULT, group=gid)
        try:
            parsed = _parse_json(content)
        except (ValueError, json.JSONDecodeError) as exc:
            last_err = f"JSON 解析失败 {exc}"
            return False
        missing = [k for k in REQUIRED_KEYS[layer] if k not in parsed]
        if missing:
            last_err = f"缺字段 {missing}"
            return False
        if layer == "p3":
            ok_q, why = p3_quotes_verbatim(parsed, samples)
            if not ok_q:
                last_err = why
                return False
        cited = {e.get("id") for e in parsed.get("evidence", []) if isinstance(e, dict)}
        if not cited <= ids:
            last_err = f"引用了不存在的样本 id: {sorted(cited - ids)[:3]}"
            return False
        result.clear()
        result.update(parsed)
        last_err = ""
        return True

    accepted, rounds = refine_loop(quality, max_rounds=max_rounds)
    output = dict(result) if accepted else {}
    save_group(work_dir, gid, {"output": output, "rounds": rounds, "layer": layer,
                               "failed": not accepted, "error": last_err,
                               "samples_sha256": samples_sig})
    if accepted:
        LOG.info("组完成 %s rounds=%d", gid, rounds)
    else:
        LOG.warning("组失败 %s rounds=%d err=%s", gid, rounds, last_err)
    return {"gid": gid, "output": output, "cached": False, "rounds": rounds,
            "failed": not accepted}


# ═══ 兜底扫描（转人工前的辅助材料；产出不得直接进 card）══════════════════
def fallback_scan(records: list[dict], tension_seeds, client, model: str,
                  meter: CostMeter, work_dir: Path, failed_gids: list[str]) -> dict:
    head = records[: max(1, len(records) // 10)]
    tail = records[-max(1, len(records) // 10):]
    kw_lines = [r.get("text_zh") or "" for r in records
                if any(s in (r.get("text_zh") or "") for s in tension_seeds)]
    material = "\n".join([r["text_zh"] for r in head] + kw_lines
                         + [r["text_zh"] for r in tail])[:FALLBACK_SCAN_CAP]
    prompt = ("以下是角色台词的全量扫描节选（头尾加权+关键词命中）。"
              "请汇总其中与角色日常人格相关的要点，供人工裁定参考。"
              "输出要点列表（简体，每条一行），不得虚构。")
    content, usage = client(model, [{"role": "user",
                                     "content": prompt + "\n\n" + material}])
    meter.record("induce", model, usage.get("prompt_tokens", 0),
                 usage.get("completion_tokens", 0),
                 (usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0))
                 / 1000 * INDUCE_RATE_DEFAULT, group="e4_fallback_scan")
    out = {"failed_groups": failed_gids, "scan_chars": len(material),
           "summary": content,
           "note": "兜底辅助材料——仅供人工裁定，不得绕过证据引用直接进 card"}
    (work_dir / "e4_fallback_scan.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out


# ═══ card 组装（persona 写权限身份）═════════════════════════════════════
def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _strip_evid(items: list[str]) -> list[str]:
    """分片 traits/values 常内嵌 (evid_xxx) 标注——card 层剥去（证据留 e4_persona.json）。"""
    return [s for s in (_strip_evid_text(x) for x in items) if s]


def _strip_evid_text(s: str) -> str:
    """单条文本剥产线证据记号：括号式 (evid_xxx) 与裸 evid_001 系列 + 产线内部语。"""
    s = re.sub(r"[（(]evid[^）)]*[）)]", "", s or "")
    s = re.sub(r"evid_\d+[、,，;；\s]*", "", s)
    s = re.sub(r"[（(]?(explicit|inferred)[）)]?[、,，;；\s]*", "", s)
    s = re.sub(r"留待 ?E3[。；;]?", "", s)
    return re.sub(r"\s{2,}", " ", s).strip(" ，,。；;")


def _deterministic_tone_fallback(records: list[dict], min_len: int) -> dict[str, list[dict]]:
    """P3 候选不足时每情绪的自有序列兜底：合格（≥min_len）优先，不足降级用未过滤
    自有序列（宁短勿空）。sleepy 桶语料实测为空（EP1 无本人睡意标签）→ 自有池空，
    由 assemble 的全局补齐段接手（不在此处抢他人候选——全局 dedup 会使后轮空转）。"""
    from contracts.emotion_defs import SENTIMENT_CLOSE_SET
    out: dict[str, list[dict]] = {}
    for emo in SENTIMENT_CLOSE_SET:
        pool = [r for r in records
                if r.get("emotion") == emo and r.get("text_zh")]
        fit = [r for r in pool if len(_norm(r["text_zh"])) >= min_len]
        out[emo] = (fit or pool)[:2]
    return out


def assemble_card(layers: dict, records: list[dict], profile: dict,
                  e3_first_person: dict | None = None,
                  e3_politeness: float | None = None) -> dict:
    """五层 shards → character_card。身份字段读 profile（2026-09-13 profile 化）；
    P4 收敛 = 按跨样本组复现数降序填充至 prompt_budget_tokens 预算内（D16 密化），
    evidence_id 不落卡（审计留 e4_persona.json）。"""
    from contracts.emotion_defs import SENTIMENT_CLOSE_SET

    p1, p2, p3 = layers.get("p1", {}), layers.get("p2", {}), layers.get("p3", {})
    p4, p5 = layers.get("p4", {}), layers.get("p5", {})
    ch = profile["character"]

    tone = {emo: [] for emo in SENTIMENT_CLOSE_SET}
    seen_tone: set[str] = set()

    def _norm(text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip()

    min_len = int(profile.get("persona", {}).get("tone_min_chars", 10))
    skipped: dict[str, list[tuple[str, str]]] = {emo: [] for emo in SENTIMENT_CLOSE_SET}
    for ex in p3.get("tone_examples", []):
        emo = ex.get("emotion")
        zh, ja = _norm(ex.get("zh")), _norm(ex.get("ja", ""))
        if not (emo in tone and zh and ja) or zh in seen_tone or len(tone[emo]) >= 2:
            continue
        if len(zh) >= min_len:
            seen_tone.add(zh)
            tone[emo].append(f"[e:{emo}]{zh}|{ja}")
        else:
            skipped[emo].append((zh, ja))          # 短句暂扣：宁短勿空的最后回填源
    for emo, pool in _deterministic_tone_fallback(records, min_len).items():
        for r in pool:
            zh, ja = _norm(r["text_zh"]), _norm(r["text"])
            if len(tone[emo]) < 2 and zh and zh not in seen_tone:
                seen_tone.add(zh)
                tone[emo].append(f"[e:{emo}]{zh}|{ja}")
    # 全局补齐（两段）：①p3 被滤短句回填 ②全体未用记录按合格优先顺次补——
    # 不抢各情绪自有候选（先到先得的全局 dedup 曾使后轮情绪空转）
    for emo in SENTIMENT_CLOSE_SET:
        while len(tone[emo]) < 2 and skipped[emo]:
            zh, ja = skipped[emo].pop(0)
            if zh not in seen_tone:
                seen_tone.add(zh)
                tone[emo].append(f"[e:{emo}]{zh}|{ja}")
    leftovers = [r for r in records if (r.get("text_zh") or "").strip()
                 and _norm(r["text_zh"]) not in seen_tone]
    leftovers.sort(key=lambda r: len(_norm(r["text_zh"])) < min_len)
    li = 0
    for emo in SENTIMENT_CLOSE_SET:
        while len(tone[emo]) < 2 and li < len(leftovers):
            r = leftovers[li]
            li += 1
            zh, ja = _norm(r["text_zh"]), _norm(r["text"])
            if zh and zh not in seen_tone:
                seen_tone.add(zh)
                tone[emo].append(f"[e:{emo}]{zh}|{ja}")

    # ── 尾段保证（出厂检查 arc_ends 对齐，2026-09-16 用户裁定 a 方案）────────
    # tail 三分区零覆盖时，以尾段记录替换"最厚情绪"槽位的第 2 条（确定性：
    # 按 voice 排序取首条；不新增情绪、不破坏 9×2 结构）。
    chapters = sorted({r["chapter"] for r in records
                       if r.get("chapter") not in (None, "", "?")})
    third = max(1, len(chapters) // 3)
    tail_zone = set(chapters[-third:])
    t2chap = {(_norm(r.get("text_zh") or "")): r["chapter"]
              for r in records if r.get("text_zh") and r.get("chapter")}

    def _in_tail(line: str) -> bool:
        zh = line.split("|", 1)[0].split("]", 1)[-1].strip()
        return t2chap.get(zh) in tail_zone

    if tail_zone and not any(_in_tail(l) for lines in tone.values() for l in lines):
        cands = sorted((r for r in records
                        if r.get("chapter") in tail_zone
                        and r.get("emotion") in tone
                        and len(_norm(r.get("text_zh") or "")) >= min_len
                        and r.get("text")
                        and _norm(r["text_zh"]) not in seen_tone),
                       key=lambda r: r["voice"])
        if cands:
            r0 = cands[0]
            emo0 = r0["emotion"]
            zh, ja = _norm(r0["text_zh"]), _norm(r0["text"])
            new_line = f"[e:{emo0}]{zh}|{ja}"
            if len(tone[emo0]) >= 2:
                tone[emo0][1] = new_line        # 替换第 2 条，保 9×2 结构
            else:
                tone[emo0].append(new_line)
            seen_tone.add(zh)

    fp = e3_first_person or {}
    dominant = max(fp, key=fp.get) if fp else "私"
    # P1 密度安全帽（merge prompt v1.1.0 硬帽的脚本侧兜底；D16 B 方案）
    traits = (_strip_evid(p1.get("traits", [])) or ["见 persona 段"])[:8]
    values = (_strip_evid(p1.get("values", [])) or ["见 persona 段"])[:6]

    speech_style = {
        "first_person": dominant,
        "calls_you": ch.get("calls_you", "你"),
        "register": _strip_evid_text(p2.get("register", "口语短句为主")),
    }
    if e3_politeness is not None:
        speech_style["politeness_ratio"] = e3_politeness   # E3 实测回填（雏形卡同字段）
    if p2.get("politeness_notes"):
        speech_style["notes"] = _strip_evid_text(p2["politeness_notes"])

    # ── P4 收敛（D16 密化，2026-09-13 用户定案）───────────────────────────
    # 排序键 = 证据形态（explicit 原文支撑优先于 inferred，替代纯码点抽签——
    # 实测 _rec 全为 1 无区分度）→ 复现数 → 稳定序；预算口径 = pet estimate_tokens；
    # 规则渲染形态按 P-1 预期「若…，…」计。schema minItems=1 → 保底 1 条。
    budget = int(profile.get("persona", {}).get("prompt_budget_tokens", 2000))
    ranked = sorted(p4.get("behavior_rules", []),
                    key=lambda r: (-int(r.get("_evtype", "") == "explicit"),
                                   -r.get("_rec", 1),
                                   r.get("if", ""), r.get("then", "")))
    card = {
        "package_id": ch["package_id"],
        "schema_version": 1,
        "display_name": ch["display_name"],
        "name": ch["name"],
        "name_zh": ch["name_zh"],
        "name_en": ch["name_en"],
        "description": _strip_evid_text(p1.get("description", "")),
        "persona": _strip_evid_text(p1.get("persona_prose", "")),
        "personality": traits,
        "values": values,
        "taboos": CARD_TABOOS,
        "speech_style": speech_style,
        "tone_examples": tone,
        "behavior_rules": [],
        "anti_patterns": [_strip_evid_text(a) for a in p5.get("anti_patterns", [])],
        "honesty_boundary": {
            "in_game_only": True,
            "no_user_world_claims": True,
            "corpus_cutoff": "9-nine EP1（a01~a17 官方简体语料）为止",
            "notes": "不知 EP2+ 与系列外剧情；不谈论用户现实世界的具体事务与判断。"
                     "语料范围内（EP1）的剧情、人物与设定可以正常谈论。",
        },
    }
    used = estimate_tokens(yaml.safe_dump(card, allow_unicode=True, sort_keys=False))
    for rule in ranked:
        rendered = f"若{rule.get('if', '')}，{rule.get('then', '')}。"
        cost = estimate_tokens(rendered) + 8   # 行首标记与分隔开销
        if kept_rules_nonempty(card["behavior_rules"]) and used + cost > budget:
            continue
        card["behavior_rules"].append(
            {"if": _strip_evid_text(rule.get("if", "")),
             "then": _strip_evid_text(rule.get("then", ""))})
        used += cost
    return card


def kept_rules_nonempty(rules: list) -> bool:
    return bool(rules)


def archive_card_snapshot(work_dir: Path, card_text: str) -> Path | None:
    """每炼必存（用户要求 2026-09-14）：card 快照归档至 work/card_archive/。

    按内容 hash 去重——同一版卡只存一份（重跑字节一致不产生重复文件）；
    文件名含时间戳+hash 前 8 位便于人读溯源。归档区=历史存储而非管线产物
    （时间戳不违反 L6：L6 约束的是工序产物幂等，不是显式版本库）。
    返回新归档路径；同内容已存在则返回 None。
    """
    arch = Path(work_dir) / "card_archive"
    arch.mkdir(parents=True, exist_ok=True)
    sha8 = hashlib.sha256(card_text.encode("utf-8")).hexdigest()[:8]
    if any(p.name.endswith(f"_{sha8}.yaml") for p in arch.glob("character_card_*.yaml")):
        return None
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = arch / f"character_card_{stamp}_{sha8}.yaml"
    path.write_text(card_text, encoding="utf-8")
    return path


# ═══ 主流程 ═════════════════════════════════════════════════════════════
def run(raw_root: Path, profile_path: Path, work_dir: Path,
        api_key: str | None = None, env_path: Path = ENV_PATH,
        client_fn=None) -> dict:
    raw_root = Path(raw_root).resolve()
    profile = yaml.safe_load(Path(profile_path).read_text(encoding="utf-8"))
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    quota = profile["persona"]["sample_quota"]
    model = profile["llm"]["induce"]["model"]

    if client_fn is None:
        if api_key is None:
            api_key = env_key(env_path)
        base_url = profile.get("llm", {}).get(
            "base_url", base_url_from_env())

        def client(model_: str, messages: list[dict], timeout: float | None = None):
            # timeout 仅 P1 merge 传入（glm-5.3 对大分片 prompt 的思考时长波动大，
            # 360s 全局阈值 2026-09-19 实测连续 4 次超时——merge 单点放宽到 600s）；
            # 未传时不覆盖默认（None 会令 httpx 无限等待，组调用不允许）
            if timeout is None:
                return chat_completion(base_url, model_, api_key, messages)
            return chat_completion(base_url, model_, api_key, messages,
                                   timeout=timeout)
    else:
        def client(model_: str, messages: list[dict], timeout: float | None = None):
            return client_fn(model_, messages)

    entries = json.loads((work_dir / "voice_map.json").read_text(encoding="utf-8"))
    records = load_records(entries, profile, raw_root)
    chapters = sorted({r["chapter"] for r in records if r["chapter"] != "?"})
    groups = build_groups(records, chapters, quota)
    prompts = _load_prompts()

    meter = CostMeter(profile["cost"]["quota_warning"], work_dir / "llm_cost.json")
    n_history = len(meter.records)   # 构造已自动载入历史账（2026-09-19 修复）

    failed: list[str] = []
    rounds_used: list[int] = []
    cached_n = 0
    shards: dict[str, list[dict]] = {}
    for gid, samples in groups.items():
        layer = gid.split("_")[1]
        res = _run_group(gid, layer, samples, prompts[layer], model, client,
                         meter, work_dir)
        rounds_used.append(res["rounds"])
        cached_n += int(res["cached"])
        if res["failed"]:
            failed.append(gid)
        else:
            shards.setdefault(layer, []).append(res["output"])

    # ── 合并（各层 shards → 最终层产出）──────────────────────────────────
    # P1 merge 结果按 shards 内容 hash 缓存（L6：原料不变重跑零成本——原实现
    # 每次运行都重烧一次 merge API 钱，2026-09-12 审评修复）
    layers: dict[str, dict] = {}
    if shards.get("p1"):
        # 缓存键 = 分片内容 + merge 提示词版本（prompt 改版自动失效——
        # 2026-09-13 修复：原键只含分片，merge.md 升版后旧合并产物会被错误复用）
        merge_key = hashlib.sha256(
            json.dumps(shards["p1"], ensure_ascii=False).encode("utf-8")
            + prompts["merge"].encode("utf-8")).hexdigest()[:12]
        merge_path = work_dir / CHECKPOINTS_DIR / f"merge_p1_{merge_key}.json"
        if merge_path.is_file():
            layers["p1"] = json.loads(
                merge_path.read_text(encoding="utf-8"))["output"]
        else:
            merge_prompt = _fill(
                prompts["merge"], shards=json.dumps(shards["p1"], ensure_ascii=False)
            ) + ("\n\n本合并对象 = P1 心智模型分片。输出字段：persona_prose（完整人格"
                 "叙述，≥200 字简体）+ description + traits + values + core_tensions + evidence。")
            # 大 prompt + glm-5.3 思考模式实测可超单次超时——有界重试 1 次，成功即入缓存
            # merge 单点超时 600s（2026-09-19：新 shards 4 连超 360s，全局阈值不动）
            merge_err: Exception | None = None
            for _attempt in range(2):
                try:
                    content, usage = client(model, [{"role": "user",
                                                     "content": merge_prompt}],
                                            timeout=600.0)
                    merge_err = None
                    break
                except Exception as exc:  # noqa: BLE001 —— 超时/网络类失败重试一次
                    merge_err = exc
            if merge_err is not None:
                raise RuntimeError(f"P1 merge 调用失败（重试 1 次后仍失败）：{merge_err}")
            meter.record("induce", model, usage.get("prompt_tokens", 0),
                         usage.get("completion_tokens", 0),
                         (usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0))
                         / 1000 * INDUCE_RATE_DEFAULT, group="e4_merge_p1")
            layers["p1"] = _parse_json(content)
            merge_path.parent.mkdir(parents=True, exist_ok=True)
            merge_path.write_text(
                json.dumps({"output": layers["p1"], "layer": "p1_merge",
                            "rounds": 1, "failed": False}, ensure_ascii=False,
                           indent=2) + "\n", encoding="utf-8")
    rules: list[dict] = []
    p4_evidence: list[dict] = []
    seen_rule: dict[str, int] = {}
    for shard in shards.get("p4", []):
        ev_type = {e.get("id"): e.get("type") for e in shard.get("evidence", [])}
        for ev in shard.get("evidence", []):
            if ev.get("id") and ev["id"] not in {e["id"] for e in p4_evidence}:
                p4_evidence.append(ev)
        for rule in shard.get("behavior_rules", []):
            key = rule.get("if", "") + "|" + rule.get("then", "")
            etype = ev_type.get(rule.get("evidence_id"), "inferred")
            if key in seen_rule:
                # 跨样本组复现 = 证据强度（女娲"跨情境复现"验证的确定性代理）；
                # 任一分片给出 explicit 证据则升级
                idx = seen_rule[key]
                rules[idx]["_rec"] = rules[idx]["_rec"] + 1
                if etype == "explicit":
                    rules[idx]["_evtype"] = "explicit"
                continue
            seen_rule[key] = len(rules)
            rules.append({**rule, "_rec": 1, "_evtype": etype})
    if rules:
        layers["p4"] = {"behavior_rules": rules, "evidence": p4_evidence}
    examples: list[dict] = []
    for shard in shards.get("p3", []):
        examples.extend(shard.get("tone_examples", []))
    if examples:
        layers["p3"] = {"tone_examples": examples}
    p2_merged: dict[str, str] = {}
    p2_evidence: list[dict] = []
    seen_ev_id: set[str] = set()
    for shard in shards.get("p2", []):
        for ev in shard.get("evidence", []):
            eid = ev.get("id", "")
            if eid and eid not in seen_ev_id:
                seen_ev_id.add(eid)
                p2_evidence.append(ev)
        for key in ("register", "politeness_notes", "tics_notes", "address_notes"):
            if shard.get(key) and not p2_merged.get(key):
                p2_merged[key] = shard[key]
    if p2_merged:
        p2_merged["evidence"] = p2_evidence
        layers["p2"] = p2_merged
    anti: list[str] = []
    for shard in shards.get("p5", []):
        anti.extend(shard.get("anti_patterns", []))
    if anti:
        layers["p5"] = {"anti_patterns": anti}

    e3_fp = None
    e3_politeness = None
    e3_stats = work_dir / "e3_stats.json"
    if e3_stats.is_file():
        e3_ja = json.loads(e3_stats.read_text(encoding="utf-8"))["ja"]
        e3_fp = e3_ja["first_person"]
        e3_politeness = e3_ja.get("politeness_ratio")

    card = assemble_card(layers, records, profile, e3_fp, e3_politeness)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    errors = list(Draft202012Validator(schema).iter_errors(card))
    if errors:
        raise RuntimeError(
            "character_card 未过 schema："
            + str([f"{'/'.join(map(str, e.path))}: {e.message}" for e in errors[:5]]))
    card_text = yaml.safe_dump(card, allow_unicode=True, sort_keys=False)
    (work_dir / "character_card.yaml").write_text(card_text, encoding="utf-8")
    snapshot = archive_card_snapshot(work_dir, card_text)

    cost_path = meter.save()
    warn = " ⚠️ 超阈值（转人工参考）" if meter.over_quota else ""
    new_calls = len(meter.records) - n_history
    new_cost = round(sum(r["cost_yuan"] for r in meter.records[n_history:]), 4)
    meta = {
        "model": model,
        "groups": len(groups), "cached": cached_n, "failed": failed,
        "max_rounds_used": max(rounds_used) if rounds_used else 0,
        "new_calls": new_calls, "new_cost_yuan": new_cost,
        "cost_total_yuan": meter.total_yuan,
        "p4_total": len(layers.get("p4", {}).get("behavior_rules", [])),
        "p4_kept": len(card["behavior_rules"]),
        "card_prompt_tokens_est": estimate_tokens(card_text),
        "card_snapshot": snapshot.name if snapshot else "unchanged（同内容不重复归档）",
    }
    (work_dir / "e4_persona.json").write_text(
        json.dumps({"meta": meta, "layers": layers}, ensure_ascii=False, indent=2)
        + "\n", encoding="utf-8")
    print(f"[E4] groups={len(groups)} cached={cached_n} failed={len(failed)} "
          f"本次 {new_calls} 次调用 {new_cost} 元 → character_card.yaml{warn}")
    LOG.info("提炼汇总 groups=%d cached=%d failed=%s max_rounds=%d 本次调用=%d "
             "本次成本=%.4f元 累计=%.4f元 p4保留=%d/%d card_token_est=%d 快照=%s",
             len(groups), cached_n, failed or "无",
             meta["max_rounds_used"], new_calls, new_cost, meter.total_yuan,
             meta["p4_kept"], meta["p4_total"], meta["card_prompt_tokens_est"],
             meta["card_snapshot"])
    if failed:
        LOG.warning("转人工：%d 组失败进兜底扫描路径", len(failed))
        fb = fallback_scan(records, profile.get("persona", {}).get(
            "tension_words", TENSION_SEEDS_DEFAULT),
            client, model, meter, work_dir, failed)
        print(f"[E4] ⚠️ 转人工 {len(failed)} 组；兜底扫描 {fb['scan_chars']} 字 → "
              f"work/e4_fallback_scan.json")
        LOG.warning("兜底扫描完成 chars=%d → e4_fallback_scan.json", fb["scan_chars"])
    return meta


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    parser = argparse.ArgumentParser(description="E4 五层提炼（induce 槽）")
    parser.add_argument("--raw", required=True)
    parser.add_argument("--profile", default="config/game_9nine_ep1.yaml")
    parser.add_argument("--work", default="work")
    args = parser.parse_args()
    from contracts.logging_util import setup_logging
    setup_logging(Path(args.work))   # 直跑也落 work/logs/pipeline.log

    try:
        run(Path(args.raw), Path(args.profile), Path(args.work))
    except (RuntimeError, ValueError) as exc:
        print(f"[E4][fail-fast] {exc}", file=sys.stderr)
        sys.exit(2)
