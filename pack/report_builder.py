"""report_builder —— L4.1 报告页 + L4.2 决策账本（decisions.yaml）。

Y5 语义（L0.5 锁定）：每条人工决策记录所审产物 sha256；重跑后产物 hash 变化 →
决策 stale → 报告页重问，**不静默沿用**（安全关键决策不跨版本沿用，plan §4.2）。

decisions.yaml 形态（work/decisions.yaml，人工决策唯一账本，L4.2）：
  <key>:
    decision: <人类可读裁定>
    decided: <YYYY-MM-DD>
    artifact: <所审产物路径（相对 work/）>
    artifact_sha256: <登记时的产物 sha256>
stale 判定 = 现产物 sha256 != 登记值。

报告页（work/report.html，本地 HTML，report/template/report.html 骨架）：
输入档案（常驻）/ 成本面板 / 人格区（词表命中高亮）/ 中文校对区 / 日常态过滤词表 /
语音区（V2 终审材料）/ 决策状态（stale 红标）/ 人工节点清单（≤65min）。
缺文件区块显示占位（报告是视图，不做管线 fail-fast——§4.1 上游 fail-fast 不适用视图层）。
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
import html
import json
from pathlib import Path

import yaml

from contracts.logging_util import get_logger

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = REPO_ROOT / "report" / "template" / "report.html"

LOG = get_logger("report")


# ═══ L4.2 决策账本 ══════════════════════════════════════════════════════
def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_of_path(path: Path) -> str:
    return sha256_of_bytes(Path(path).read_bytes())


def load_decisions(work_dir: Path) -> dict:
    path = Path(work_dir) / "decisions.yaml"
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def save_decisions(work_dir: Path, decisions: dict) -> Path:
    path = Path(work_dir) / "decisions.yaml"
    path.write_text(
        "# work/decisions.yaml —— 人工决策唯一账本（L4.2/Y5）：每条绑定所审产物\n"
        "# sha256；产物重跑即 stale，报告页重问，不静默沿用。\n"
        + yaml.safe_dump(decisions, allow_unicode=True, sort_keys=False),
        encoding="utf-8")
    return path


def record_decision(decisions: dict, key: str, artifact_path: Path,
                    value: dict, decided: str | None = None,
                    artifact_ref: str | None = None) -> None:
    """登记一条人工决策：决策内容 + 所审产物 hash（Y5 绑定）。"""
    decisions[key] = {
        "decision": value,
        "decided": decided or "",
        "artifact": artifact_ref or "",
        "artifact_sha256": sha256_of_path(artifact_path),
    }


def is_stale(decision: dict, artifact_path: Path) -> bool:
    """产物重跑后 hash 变化 → 决策 stale（报告页重问，不静默沿用）。"""
    return decision["artifact_sha256"] != sha256_of_path(artifact_path)


def decisions_status(work_dir: Path, decisions: dict) -> list[dict]:
    """逐条决策的当前状态：stale（产物 hash 变化/产物缺失）或 ok。"""
    rows = []
    for key, dec in decisions.items():
        ref = dec.get("artifact") or ""
        path = Path(work_dir) / ref if ref else None
        if ref and path.is_file():
            stale = is_stale(dec, path)
            current = sha256_of_path(path)[:12]
        else:
            stale = True
            current = "产物缺失"
        rows.append({"key": key, "decision": dec.get("decision", ""),
                     "decided": dec.get("decided", ""), "artifact": ref,
                     "recorded": str(dec.get("artifact_sha256", ""))[:12],
                     "current": current, "stale": stale})
    return rows


# ═══ L4.1 报告页数据收集 ════════════════════════════════════════════════
def _load_json(work_dir: Path, name: str):
    p = Path(work_dir) / name
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def _load_yaml(work_dir: Path, name: str):
    p = Path(work_dir) / name
    return yaml.safe_load(p.read_text(encoding="utf-8")) if p.is_file() else None


def collect_report_data(work_dir: Path, profile_path: Path) -> dict:
    """汇总报告页全部区块数据；缺件 → None（视图层占位渲染）。"""
    work_dir = Path(work_dir)
    profile = yaml.safe_load(Path(profile_path).read_text(encoding="utf-8"))
    d = {
        "profile": profile,
        "input_profile": _load_json(work_dir, "input_profile.json"),
        "llm_cost": _load_json(work_dir, "llm_cost.json"),
        "e3_stats": _load_json(work_dir, "e3_stats.json"),
        "wordlist": _load_json(work_dir, "wordlist_candidates.json"),
        "card": _load_yaml(work_dir, "character_card.yaml"),
        "translate_review": _load_json(work_dir, "translate_review.json"),
        "v1_meta": _load_json(work_dir, "v1_meta.json"),
        "v2_candidates": _load_yaml(work_dir, "reference/candidates.yaml"),
        "sprites_meta": _load_json(work_dir, "sprites_meta.json"),
        "blind_test": _load_json(work_dir, "blind_test_5v5.json"),
    }
    decisions = load_decisions(work_dir)
    d["decision_rows"] = decisions_status(work_dir, decisions)
    d["blocklist"] = profile.get("persona", {}).get("daily_corpus_blocklist", [])
    return d


# ═══ L4.1 HTML 渲染 ═════════════════════════════════════════════════════
def _esc(v) -> str:
    return html.escape(str(v))


def _table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "<p class='dim'>（无）</p>"
    th = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    trs = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
                  for row in rows)
    return f"<table><tr>{th}</tr>{trs}</table>"


def render_sections(d: dict) -> str:
    s: list[str] = []

    # ── 输入档案（常驻）──
    ip = d.get("input_profile") or {}
    probes = ip.get("probes", {})
    lang = probes.get("language", {})
    s.append("<h2>① 输入档案（常驻）</h2>" + _table(
        ["项", "值"],
        [["bilingual_source", f"{_esc(ip.get('bilingual_source'))}（{_esc(ip.get('source'))}）"],
         ["alignment_rate", _esc(ip.get("alignment_rate"))],
         ["语言探测", f"假名 {_esc(lang.get('kana_ratio'))} / 汉字 {_esc(lang.get('han_ratio'))} → {_esc(lang.get('detected_lang'))}"],
         ["resx 汉化", f"文件 {_esc(probes.get('patch', {}).get('resx_files_found'))}，有效条目 {_esc(probes.get('patch', {}).get('resx_effective_entries'))}"],
         ["Steam 多语言", f"{_esc(probes.get('multilang', {}).get('multilang_files'))}/{_esc(probes.get('multilang', {}).get('files'))} 文件含三语节点"]]))

    # ── 成本面板（缺 llm_cost.json 时占位——视图层不 fail-fast）──
    cost = d.get("llm_cost") or {}
    if not cost:
        s.append("<h2>② 成本面板</h2><p class='dim'>"
                 "（缺 llm_cost.json——尚未产生 LLM 调用）</p>")
    else:
        slots = cost.get("by_slot", {})
        s.append("<h2>② 成本面板</h2>" + _table(
            ["槽位", "估算（元）"],
            [[_esc(k), _esc(v)] for k, v in slots.items()]
            + [["合计", f"<b>{_esc(cost.get('total_yuan'))}</b>"
                + (" ⚠️ 超阈值（警示不中断）" if cost.get("over_quota") else "")]]))

    # ── 人格区 ──
    card = d.get("card") or {}
    e3 = d.get("e3_stats") or {}
    ja = e3.get("ja", {})
    anchors = (d.get("wordlist") or {}).get("anchor_hits", {})
    s.append("<h2>③ 人格区</h2>" + _table(
        ["项", "值"],
        [["身份", f"{_esc(card.get('display_name'))} / {_esc(card.get('name'))} / 称呼「{_esc((card.get('speech_style') or {}).get('calls_you'))}」"],
         ["persona", f"{_esc(len(card.get('persona') or ''))} 字"],
         ["性格 / 价值观 / 规则 / 反模式", f"{len(card.get('personality') or [])} / {len(card.get('values') or [])} / {len(card.get('behavior_rules') or [])} / {len(card.get('anti_patterns') or [])}"],
         ["tone_examples", _esc({k: len(v) for k, v in (card.get('tone_examples') or {}).items()})],
         ["第一人称（E3）", _esc(ja.get("first_person"))],
         ["丁寧語率", _esc(ja.get("politeness_ratio"))]]))
    s.append("<h3>日常态过滤词表（勾选 = 加入剔除名单）</h3>" + _table(
        ["词", "全台本命中", "判定"],
        [[_esc(w), _esc(h),
          "已入剔除名单" if w in (d.get("blocklist") or [])
          else ("日常实体/零命中" if h else "零命中")]
         for w, h in anchors.items()])
        + f"<p class='dim'>剔除名单 blocklist = {_esc(d.get('blocklist') or '[]')}——"
          f"命中词的台词整句剔出日常态语料池（Y4 全量候选见 wordlist_candidates.json）</p>")

    # ── 中文校对区 ──
    tr = d.get("translate_review") or []
    s.append("<h2>④ 中文校对区</h2>" + _table(
        ["voice", "原文", "错误"],
        [[_esc(r.get("voice")), _esc(r.get("text")), _esc(r.get("error"))] for r in tr])
        + ("<p class='dim'>embedded 路径：机翻未触发，本区常态为空。</p>" if not tr else ""))

    # ── 语音区 ──
    v1 = d.get("v1_meta") or {}
    cands = (d.get("v2_candidates") or {}).get("candidates", [])
    s.append("<h2>⑤ 语音区（V2 终审材料——人工节点）</h2>" + _table(
        ["项", "值"],
        [["dataset", _esc(v1.get("dataset_selected"))],
         ["audiofile 九桶", _esc(v1.get("audiofile_counts"))],
         ["V2 候选", f"{len(cands)} 条（rank=abs(duration-7.0)，勾选后 pack 代写 tts_config）"]]
        + [[_esc(c.get("voice_id")), f"{_esc(c.get('duration'))}s / {_esc(c.get('face'))} / {_esc(c.get('text'))}"]
           for c in cands]))

    # ── 盲测 ──
    bt = d.get("blind_test") or {}
    s.append("<h2>⑥ 盲测（判定线：首轮 ≥6/10 或累计两轮 ≥70%）</h2>" + _table(
        ["项", "值"],
        [["题面", f"{len(bt.get('items') or [])} 句（5 card + 5 corpus 混排）"],
         ["答案卷", "独立存证于 blind_test_5v5.json answer_key（盲测人勿阅）"]]))

    # ── 决策状态 ──
    rows = [[("⚠️ STALE" if r["stale"] else "ok"), _esc(r["key"]),
             _esc(r["decision"] if isinstance(r["decision"], str) else r["decision"]),
             _esc(r["decided"]), _esc(r["artifact"]),
             f"{_esc(r['recorded'])} → {_esc(r['current'])}"] for r in d["decision_rows"]]
    s.append("<h2>⑦ 决策状态（Y5：产物变更即 stale，重问不沿用）</h2>" + _table(
        ["状态", "key", "裁定", "日期", "所审产物", "hash（登记→当前）"], rows))

    # ── 人工节点清单 ──
    s.append("<h2>⑧ 人工节点清单（≤65min，不含机器时间）</h2>" + _table(
        ["节点", "状态"],
        [["输入档案核对", "报告页 ① 常驻"],
         ["日常态过滤词表勾选", "已决（⑦ wordlist_blocklist）"],
         ["V2 参考音频终审", "待勾选（⑤）"],
         ["盲测判定", "已决（⑦ blind_test）/ 需重测时重问"],
         ["中文校对", "embedded 路径未触发"],
         ["pet 冒烟配合（L4.3 出包后）", "待 L4.3"]]))
    return "\n".join(s)


def build_report(work_dir: Path, profile_path: Path) -> Path:
    """生成 work/report.html（本地 HTML 报告页）。"""
    work_dir = Path(work_dir)
    d = collect_report_data(work_dir, profile_path)
    body = render_sections(d)
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    html_text = template.replace("$body", body)
    out = work_dir / "report.html"
    out.write_text(html_text, encoding="utf-8")
    print(f"[report] 报告页 → {out}")
    stale = [r["key"] for r in d["decision_rows"] if r["stale"]]
    LOG.info("报告页生成 → %s 决策=%d 条（stale=%s）缺件占位=%s", out,
             len(d["decision_rows"]), stale or "无",
             [k for k, v in d.items() if v is None] or "无")
    return out


def run(raw_root: Path, profile_path: Path, work_dir: Path) -> Path:
    """build.py 工序入口（视图层：只读 work/ 产物，无 fail-fast——缺件占位）。"""
    return build_report(work_dir, profile_path)


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    parser = argparse.ArgumentParser(description="L4.1 体检报告页（人工决策收敛单页）")
    parser.add_argument("--raw", default=".")
    parser.add_argument("--profile", default="config/game_9nine_ep1.yaml")
    parser.add_argument("--work", default="work")
    args = parser.parse_args()
    from contracts.logging_util import run_separator, setup_logging
    setup_logging(Path(args.work))
    run_separator("REPORT START")
    build_report(Path(args.work), Path(args.profile))
    run_separator("REPORT END")
