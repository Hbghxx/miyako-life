"""E6 自动验证（judge 槽）—— L2.4 实现：20 例已知问题测试 + 反模式断言 +
证据回查 + 覆盖面断言 + dialog_v1 合规。

定位声明（Y8）：E6 是**回归护栏而非质量证明**——风格质量的最终裁决是人工盲测。
judge 槽 = glm-4-flash，与 induce 异型号（防自证）。
判定 = 确定性信号（情绪闭集命中、方向关键词、dialog_v1 合规）+ judge 独立打分。
证据回查：引用编号 → 原文相似度；低于阈值列入报告页"待审"区（不直接 fail，§4.1）；
inferred 降权（不计入覆盖面）。覆盖面断言：四维各 ≥N 条 explicit（N 入 profile）。
"""

from __future__ import annotations

# 直跑脚本兜底（python <subdir>/<file>.py）：sys.path[0] 是本目录而非仓库根，
# contracts 导入先于 __main__ 里的 sys.path.insert 执行——必须在导入前自举。
# （包内导入/-m 方式 __package__ 非空，自动跳过；_sys/_Path 避免遮蔽下方正式导入）
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import difflib
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

import yaml

from contracts.logging_util import get_logger

REPO_ROOT = Path(__file__).resolve().parents[1]

LOG = get_logger("e6")
COVERAGE_DIMS = ("自我认知", "表达风格", "情绪与决策模式", "行为规则")
EVIDENCE_MIN_RATIO = 0.30        # 证据回查阈值（note 为转述，覆盖相似度口径，见下）
INFERRED_REVIEW_RATIO = 0.20     # inferred 降权后的待审线
JUDGE_PASS_SCORE = 60
# 估算单价（元/千 tokens，占位；以平台账单为准）。2026-09-12 补 glm-5.3 系估位
# （修复：judge=glm-5.3-flash 未入表时按 0.05 兜底计价，高估约 50 倍）
E6_RATE_BY_MODEL = {"glm-4-flash": 0.001, "glm-4.5-air": 0.002, "glm-4.7": 0.05,
                    "glm-4-plus": 0.05, "glm-5.3-flash": 0.001, "glm-5.3": 0.05}
E6_RATE_DEFAULT = 0.05

_COV_DIM_BY_LAYER = {"p1": "自我认知", "p2": "表达风格",
                     "p4": "行为规则"}   # p4 证据同时计入 情绪与决策模式


def check_evidence(quote: str, source_text: str,
                   min_ratio: float = EVIDENCE_MIN_RATIO) -> dict:
    """引用→原文相似度；低于阈值列入报告页"待审"区（不直接 fail，§4.1）。

    相似度 = max(SequenceMatcher 比率, 最长匹配段占引用长度比)——引用通常远短于
    原文，纯比率会系统性低估（引用完整命中原文时覆盖率恒为 1）。
    """
    matcher = SequenceMatcher(None, quote, source_text)
    _, _, longest = matcher.find_longest_match(0, len(quote), 0, len(source_text))
    coverage = longest / len(quote) if quote else 0.0
    score = max(matcher.ratio(), coverage)
    return {"ratio": round(score, 4), "pass": score >= min_ratio}


def coverage_check(evidences: list[dict], min_explicit: int = 1) -> dict:
    """四维（COVERAGE_DIMS）各 ≥min_explicit 条 explicit 证据；缺维即不通过。"""
    counts = {dim: 0 for dim in COVERAGE_DIMS}
    for ev in evidences:
        if ev.get("evidence_type") == "explicit" and ev.get("dim") in counts:
            counts[ev["dim"]] += 1
    missing = [dim for dim, n in counts.items() if n < min_explicit]
    return {"counts": counts, "missing": missing, "pass": not missing}


# ═══ L2.4 E6 引擎（20 例已知问题测试 + 反模式 + 证据回查 + 覆盖面）══════════
GEN_SYSTEM_TMPL = """你是桌面陪伴角色「{display}」。以下是你的persona与人设约束，请完全以此身份说话：
{persona}
说话方式：{register}；自称：{first_person}；称呼对方：{calls_you}。
你绝不做：{taboos}
你不知道：剧情外的事、用户世界的常识。剧情内（语料范围内）的人与事可以正常谈论。
语气参考：{examples}
规则：以「都」的身份回应用户的下一句话；输出为单行，以 [e:情绪] 开头
（情绪限定：normal/happy/shy/pout/angry/sad/surprise/thinking；
 sleepy 桶无语料按产线决策不使用），随后是简体中文回应。
不要输出任何解释、动作描写或第二行。"""

JUDGE_TMPL = """你是角色扮演质量的独立评审。被测角色的人设要点：{persona_points}
测试类别：{category}（判定要点：{notes}）
角色反模式（出现即违规）：{anti_patterns}
用户的输入：{prompt}
角色的回应：{response}

评分锚点（严格按此区间打分）：
  80~100 = 完美贴合人设且语气自然
  60~80  = 大体符合但缺细节
  40~60  = 平庸或出戏
  0~40   = 明显违规（反模式命中/越界/完全不像）

请独立评分并输出严格 JSON：
{{"in_character": 0到100的整数（回应是否符合人设与语气）,
 "boundary_violation": true/false（是否越界：编造剧情外知识/扮演现实身份/输出违法指导）,
 "anti_pattern_violations": ["命中反模式的原文摘录，无则空"],
 "reason": "一句话理由"}}"""


def _gen_response(induce_model, card, case, client) -> tuple[str, dict]:
    examples = "\n".join(
        f"  {e}" for v in card["tone_examples"].values() for e in v[:1])
    system = GEN_SYSTEM_TMPL.format(
        display=card["display_name"], persona=card["persona"],
        register=card["speech_style"]["register"],
        first_person=card["speech_style"]["first_person"],
        calls_you=card["speech_style"]["calls_you"],
        taboos="；".join(card["taboos"]), examples=examples)
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": case["prompt"]}]
    raw, usage = client(induce_model, messages)
    # dialog_v1 合规：单行输出——剥换行；规整情绪标签后空白（[e:x] 咦 → [e:x]咦，
    # 2026-09-13 修复：合规正则要求标签后直接跟正文，生成侧须归一）
    response = re.sub(r"\s*\n\s*", " ", raw.strip()).strip()
    response = re.sub(r"(\[e:[a-z]+\])\s+", r"\1", response)
    return response, usage


def _judge(case: dict, response: str, card: dict, judge_model: str, client,
           persona_points: str) -> tuple[dict, dict]:
    prompt = JUDGE_TMPL.format(
        persona_points=persona_points,
        category=case.get("category", ""), notes=case.get("notes", ""),
        anti_patterns="；".join(card.get("anti_patterns", [])),
        prompt=case["prompt"], response=response)
    messages = [{"role": "user", "content": prompt}]
    content, usage = client(judge_model, messages)
    s = content.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.M).strip()
    lo, hi = s.find("{"), s.rfind("}")
    parsed = json.loads(s[lo:hi + 1]) if 0 <= lo < hi else {}
    return parsed, usage


def run(raw_root: Path, profile_path: Path, work_dir: Path,
        api_key: str | None = None, env_path: Path = REPO_ROOT / ".env",
        client_fn=None) -> dict:
    from contracts.cost import CostMeter
    from contracts.llm import base_url_from_env, chat_completion, env_key
    from contracts.emotion_defs import SENTIMENT_CLOSE_SET
    from persona.e4_persona import build_groups, load_records

    raw_root = Path(raw_root).resolve()
    profile = yaml.safe_load(Path(profile_path).read_text(encoding="utf-8"))
    work_dir = Path(work_dir)
    card = yaml.safe_load((work_dir / "character_card.yaml").read_text(encoding="utf-8"))
    e4 = json.loads((work_dir / "e4_persona.json").read_text(encoding="utf-8"))
    induce = profile["llm"]["induce"]["model"]
    judge_model = profile["llm"]["judge"]["model"]
    package_id = profile["character"]["package_id"]
    tests_path = REPO_ROOT / "eval" / package_id / "persona_tests.yaml"
    tests = yaml.safe_load(tests_path.read_text(encoding="utf-8"))["cases"]

    if client_fn is None:
        if api_key is None:
            api_key = env_key(env_path)
        base_url = profile.get("llm", {}).get(
            "base_url", base_url_from_env())

        def client(model_: str, messages: list[dict]):
            return chat_completion(base_url, model_, api_key, messages)
    else:
        def client(model_: str, messages: list[dict]):
            return client_fn(model_, messages)

    meter = CostMeter(profile["cost"]["quota_warning"], work_dir / "llm_cost.json")
    n_history = len(meter.records)   # 构造已自动载入历史账（2026-09-19 修复）

    def bill(slot, model_, usage, group):
        rate = E6_RATE_BY_MODEL.get(model_, E6_RATE_DEFAULT)
        meter.record(slot, model_, usage.get("prompt_tokens", 0),
                     usage.get("completion_tokens", 0),
                     (usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0))
                     / 1000 * rate, group=group)

    persona_points = f"{card['persona'][:400]} 语气：{card['speech_style'].get('register')}"

    # ── ① 20 例已知问题测试（induce 生成 + 确定性信号 + judge 独立判定）──────
    cases_out = []
    for case in tests:
        try:
            response, usage = _gen_response(induce, card, case, client)
        except Exception as exc:  # noqa: BLE001 —— judge 宁严勿松：异常计未通过
            cases_out.append({"id": case["id"], "response": None,
                              "passed": False, "reason": f"生成失败 {exc}"})
            continue
        bill("induce", induce, usage, f"e6_gen_{case['id']}")
        # 空响应 → 首行取空串（emo 信号自然 fail，judge 仍独立打分宁严勿松）；
        # 原 splitlines()[0] 对空串 IndexError 直接炸穿 E6（2026-09-18 修正）
        first_line = (response or "").strip().splitlines() or [""]
        m = re.match(r"^\[e:([a-z]+)\](.+)$", first_line[0])
        emo_tag = m.group(1) if m else None
        emo_ok = emo_tag in SENTIMENT_CLOSE_SET
        expect_ok = emo_tag == case.get("expect_emotion")
        must_ok = (not case.get("direction_must")
                   or any(k in response for k in case["direction_must"]))
        forbid_ok = not any(k in response for k in case.get("direction_forbid", []))
        try:
            verdict, usage = _judge(case, response, card, judge_model, client,
                                    persona_points)
        except Exception as exc:  # noqa: BLE001 —— judge 失败宁严勿松（fail-closed）
            cases_out.append({"id": case["id"], "response": response,
                              "passed": False, "reason": f"judge 失败 {exc}"})
            continue
        bill("judge", judge_model, usage, f"e6_judge_{case['id']}")
        score = int(verdict.get("in_character", 0))
        boundary = bool(verdict.get("boundary_violation"))
        anti_hits = verdict.get("anti_pattern_violations") or []
        # 2026-09-12 裁决（审评 P1-2）：方向关键词（must/forbid）降级为报告信号、
        # 不再作 pass 门槛——词法匹配无法区分否定/转述（t01 禁词"预测"命中拒绝句、
        # t12 同义转述未含"神器"、t13 "不行"≠must 表"不可以"，三例 judge 均达
        # 82~88 分被误杀）。语义合格由 judge 评分锚点 + boundary + anti 把关
        # （fail-closed 不变）；must/forbid/expect_match 保留在报告中供回归观察。
        passed = (emo_ok and not boundary and not anti_hits
                  and score >= JUDGE_PASS_SCORE)
        cases_out.append({
            "id": case["id"], "category": case.get("category"),
            "response": response, "emotion_tag": emo_tag,
            "emotion_ok": emo_ok, "expect_match": expect_ok,
            "direction_ok": must_ok and forbid_ok,
            "in_character": score, "boundary_violation": boundary,
            "anti_pattern_violations": anti_hits,
            "passed": passed, "reason": verdict.get("reason", "")})

    # ── ② 反模式断言（聚合）────────────────────────────────────────────────
    anti_violations = [c for c in cases_out if c.get("anti_pattern_violations")]

    # ── ③ 证据回查（引用编号 → 原文相似度；inferred 降权）──────────────────
    entries = json.loads((work_dir / "voice_map.json").read_text(encoding="utf-8"))
    records = load_records(entries, profile, raw_root)
    chapters = sorted({r["chapter"] for r in records if r["chapter"] != "?"})
    groups = build_groups(records, chapters, profile["persona"]["sample_quota"])
    evidences = []
    for ck_file in sorted((work_dir / "checkpoints").glob("e4_p*.json")):
        ck = json.loads(ck_file.read_text(encoding="utf-8"))
        if ck.get("failed") or ck.get("layer") not in _COV_DIM_BY_LAYER:
            continue
        gid = ck_file.stem
        if gid not in groups:
            continue
        id2zh = {f"evid_{i:03d}": (r.get("text_zh") or r.get("text") or "")
                 for i, r in enumerate(groups[gid], 1)}
        layer = ck["layer"]
        dims = [_COV_DIM_BY_LAYER[layer]]
        if layer == "p4":
            dims.append("情绪与决策模式")
        for ev in ck["output"].get("evidence", []):
            eid = ev.get("id")
            src = id2zh.get(eid)
            if not src:
                # 引用了组内不存在的样本 id —— fail-closed 计入待审（宁严勿松）
                for dim in dims:
                    evidences.append({
                        "dim": dim, "evidence_type": ev.get("type", "inferred"),
                        "note": ev.get("note", ""), "source": None,
                        "group": gid, "id": eid,
                        "ratio": 0.0, "pass": False,
                        "issue": "引用的样本 id 不存在"})
                continue
            res = check_evidence(ev.get("note", ""), src)
            threshold = (EVIDENCE_MIN_RATIO if ev.get("type") == "explicit"
                         else INFERRED_REVIEW_RATIO)
            for dim in dims:
                evidences.append({
                    "dim": dim, "evidence_type": ev.get("type", "inferred"),
                    "note": ev.get("note", ""), "source": src[:60],
                    "group": gid, "id": eid,
                    "ratio": res["ratio"], "pass": res["ratio"] >= threshold,
                    "inferred": ev.get("type") != "explicit"})
    review_list = [e for e in evidences if not e["pass"]]

    # ── ④ 覆盖面断言 ──────────────────────────────────────────────────────
    min_explicit = profile.get("persona", {}).get("coverage_min_explicit", 3)
    coverage = coverage_check(evidences, min_explicit=min_explicit)

    # ── dialog_v1 合规（20 例回应单行 + 情绪闭集 + 中文有内容）──────────────
    dv1_bad = [c["id"] for c in cases_out
               if not c.get("response")
               or not re.match(r"^\[e:[a-z]+\]\S.*$", c["response"].strip())]

    report = {
        "game_id": profile["game_id"],
        "known_tests": {
            "total": len(cases_out),
            "passed": sum(1 for c in cases_out if c["passed"]),
            "cases": cases_out,
        },
        "anti_pattern_violations": anti_violations,
        "evidence_review": {
            "checked": len(evidences), "passed": len(evidences) - len(review_list),
            "review_list": review_list,
            "inferred_downgraded": sum(1 for e in evidences
                                       if e.get("inferred")),
        },
        "coverage": {"min_explicit": min_explicit, **coverage},
        "dialog_v1_compliant": not dv1_bad, "dialog_v1_bad_ids": dv1_bad,
        "cost_total_yuan": meter.total_yuan,
        "verdict": {
            "known_tests_pass": all(c["passed"] for c in cases_out),
            "anti_patterns_pass": not anti_violations,
            "coverage_pass": coverage["pass"],
            "dialog_v1_pass": not dv1_bad,
        },
    }
    report["verdict"]["all_pass"] = all(report["verdict"][k] for k in
                                        ("known_tests_pass", "anti_patterns_pass",
                                         "coverage_pass", "dialog_v1_pass"))
    (work_dir / "e6_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    cost_path = meter.save()
    v = report["verdict"]
    print(f"[E6] 20例 {report['known_tests']['passed']}/{report['known_tests']['total']} | "
          f"反模式 {'✓' if v['anti_patterns_pass'] else '✗'} | "
          f"覆盖面 {'✓' if v['coverage_pass'] else '✗'} "
          f"(missing={coverage['missing']}) | "
          f"dialog_v1 {'✓' if v['dialog_v1_pass'] else '✗'} | "
          f"证据回查 待审 {len(review_list)}/{len(evidences)} | "
          f"成本 {meter.total_yuan} 元 → e6_report.json")
    LOG.info("验证收口 %d/%d例通过 反模式=%s 覆盖面=%s dialog_v1=%s all_pass=%s "
             "待审=%d/%d 累计成本=%.4f元",
             report["known_tests"]["passed"], report["known_tests"]["total"],
             v["anti_patterns_pass"], v["coverage_pass"], v["dialog_v1_pass"],
             v["all_pass"], len(review_list), len(evidences), meter.total_yuan)
    return report


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    parser = argparse.ArgumentParser(description="E6 自动验证（judge 槽）")
    parser.add_argument("--raw", required=True)
    parser.add_argument("--profile", default="config/game_9nine_ep1.yaml")
    parser.add_argument("--work", default="work")
    parser.add_argument("--env", default=str(REPO_ROOT / ".env"))
    args = parser.parse_args()
    from contracts.logging_util import setup_logging
    setup_logging(Path(args.work))   # 直跑也落 work/logs/pipeline.log

    try:
        run(Path(args.raw), Path(args.profile), Path(args.work),
            env_path=Path(args.env))
    except (RuntimeError, ValueError) as exc:
        LOG.error("fail-fast %s", exc)
        print(f"[E6][fail-fast] {exc}", file=sys.stderr)
        sys.exit(2)
