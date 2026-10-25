"""E3 表达 DNA 统计（零 LLM）+ Y4 日常态词表候选（L2.1/L2.2）。

L4 铁律：本文件一切数字由确定性脚本产出，LLM 一个都不许碰。
统计语义与手工雏形（原料区 scripts/build_char_package.py）逐条对齐，
锚点数字已实测复现：第一人称 {私:211, わたし:0, あたし:0, うち:16, 僕:0, 俺:1}、
3496 句切分、句尾后缀 かな1.4%/だね0.8%/…、丁寧語率 0.045（94/2098）、
词表命中口径 = 全台本原文（scn raw text；深沢8/九十九22/其余 0）。

产出（双落盘，write_scopes persona 身份）：
  work/e3_stats.json            全量统计（ja/zh 分列，native/translated 标记）
  work/tts_config.yaml          统计段（replace_map/glossary/verbal_tics；过 char_schema）
  work/wordlist_candidates.json Y4 候选区（分数+章节分布+命中原文例句；勾选=人工，L5 决策）

纪律：E4 消费词表，但**词表不由 E4 生成**（自证隔离）；本文件只产候选与分数，
勾选归报告页人工——空表是合法产出（宁空勿造）。
"""

from __future__ import annotations

# 直跑脚本兜底（python <subdir>/<file>.py）：sys.path[0] 是本目录而非仓库根，
# contracts 导入先于 __main__ 里的 sys.path.insert 执行——必须在导入前自举。
# （包内导入/-m 方式 __package__ 非空，自动跳过；_sys/_Path 避免遮蔽下方正式导入）
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import yaml

from contracts.logging_util import get_logger

REPO_ROOT = Path(__file__).resolve().parents[1]

LOG = get_logger("e3")

# ── 语料常量（雏形同源）─────────────────────────────────────────────────────
FIRST_PERSON_FORMS = ("私", "わたし", "あたし", "うち", "僕", "俺")
# 表记变体 → 规范式（雏形 replace_map.ja：把假名转写拉回转写一致形，TTS 对齐才稳）
FIRST_PERSON_VARIANTS = {"私": ("わたし", "あたし")}
# 句尾后缀与首中优先级（顺序即匹配序，雏形同表同序）
TIC_SUFFIXES = ("なんだよ", "だよね", "じゃん", "でしょ", "だもん", "だね", "だよ",
                "かな", "かも", "なの", "のよ", "わよ", "ですね", "ですよ",
                "ますよ", "なんだ", "だろう", "よね")
TIC_MIN_RATE = 0.02            # 口癖入选阈值（雏形：实测最高 1.4% 未达标 → 空表）
SENT_SPLIT_RE = re.compile(r"[。？！…]")
POLITENESS_MARKS = ("です", "ます")
TRANSITION_SEEDS_JA = ("でも", "しかし", "だけど", "けど", "だが", "それでも")
TRANSITION_SEEDS_ZH = ("但是", "可是", "不过", "然而", "只是")
ZH_TIC_SUFFIXES = ("呢", "吧", "哦", "啦", "呀", "嘛")
GLOSSARY_MIN_COUNT = 3
TTS_EDGE_VOICE_DEFAULT = "ja-JP-NanamiNeural"   # E3 缺省（pack 可代写覆盖）

# 成对口癖的中文等价后缀（校对区可裁定调整；EP1 空表下仅作结构备用）
SUFFIX_EQUIV_ZH = {
    "なんだよ": ("啊",), "だよね": ("呢",), "じゃん": ("嘛",), "でしょ": ("吧",),
    "だもん": ("嘛",), "だね": ("呢",), "だよ": ("哦",), "かな": ("吗", "吧"),
    "かも": ("吧",), "なの": ("呢",), "のよ": ("哦",), "わよ": ("哦",),
    "ですね": ("呢",), "ですよ": ("哦",), "ますよ": ("哦",), "なんだ": ("啊",),
    "だろう": ("吧",), "よね": ("呢",),
}

# ── Y4 常量（契约 §六 Y4）───────────────────────────────────────────────────
NAME_SUFFIX_RE = re.compile(r"([一-龥]{1,4}?)(くん|さん|ちゃん|様|さま|先輩)")
PLACE_SUFFIX_RE = re.compile(r"([一-龥]{1,6}?)(神社|町)")
KATAKANA_LONG_RE_DEFAULT = re.compile(r"[ァ-ヴー]{4,}")
NGRAM_RE = re.compile(r"[一-龥]{2,4}")
NGRAM_MIN_COUNT = 8             # n-gram 候选最低频次（kanji 连串，报告材料）
EXTREME_FACES = ("不安", "悲しみ", "怒り")   # Y4②c face 极端度（契约原文）
# 历史初版词（profile §3.2 注释存证：系手拍未过生成流程，本表仅作对账审计）
Y4_ANCHOR_WORDS = ("深沢", "九十九", "アンチノミック", "伊作", "九九")
DEFAULT_TENSION_SEEDS = ("殺人", "犯人", "死", "異能", "能力")


# ── 文本清洗（雏形同源）─────────────────────────────────────────────────────
def clean_text(raw: str) -> str:
    """台本原文 → 单行转写：剥 %...; 标记、合并折行缩进、去引号壳与全角空格。"""
    t = re.sub(r"%[^;]*;", "", raw)
    t = re.sub(r"\s*\n[\s\u3000]*", "", t)
    t = re.sub(r"^[「『\s\u3000]+", "", t.strip())
    t = re.sub(r"[」』\s\u3000]+$", "", t)
    return t.replace("\u3000", "").strip()


# ── L0.5 已锁定签名的打分骨架（Y4② 三信号，等权暂定，L2.2 校准不换语义）──────
def score_signals(distinct_chapters: int, total_chapters: int,
                  tension_rate: float, extreme_face_rate: float) -> dict:
    """Y4② 剧情态三信号打分（0~1，等权平均）。

    a 章节集中度 = 1 - 命中章节数/总章节数（爆发于少数话 → 高；全程均匀 → 低）
    b 紧张词共现率 / c face 极端度 = 调用侧统计的 0~1 比率
    """
    concentration = 0.0 if total_chapters <= 0 else max(
        0.0, 1.0 - distinct_chapters / total_chapters)
    clamp = lambda v: min(1.0, max(0.0, v))  # noqa: E731
    tension_rate, extreme_face_rate = clamp(tension_rate), clamp(extreme_face_rate)
    score = (concentration + tension_rate + extreme_face_rate) / 3
    return {
        "concentration": round(concentration, 4),
        "tension": round(tension_rate, 4),
        "extreme_face": round(extreme_face_rate, 4),
        "score": round(score, 4),
    }


def extract_katakana_terms(texts: list[str], min_len: int = 4) -> Counter:
    """片假名长串候选频次（Y4① 候选来源之一）。"""
    pat = re.compile(rf"[ァ-ヴー]{{{min_len},}}")
    counts: Counter = Counter()
    for text in texts:
        counts.update(pat.findall(text))
    return counts


# ── 语料装载（雏形 load_corpus 语义；音频过滤用 ogg 存在性，不解码）──────────
def load_corpus(entries: list[dict], speaker: str, ogg_stems: set[str]) -> list[dict]:
    """都语料：去旁白、去重 voice、清清洗文本；附带 face 供 Y4 face 极端度。"""
    seen: set[str] = set()
    records: list[dict] = []
    for e in entries:
        if e.get("speaker") != speaker:
            continue
        vid = e.get("voice")
        if vid in seen or vid not in ogg_stems:
            continue
        seen.add(vid)
        face = (e.get("self") or {}).get("face")
        records.append({"voice": vid, "text": clean_text(e["text"]), "face": face})
    return records


# ── 统计原语（雏形同语义）───────────────────────────────────────────────────
def first_person_counts(records: list[dict]) -> Counter:
    return Counter({p: sum(r["text"].count(p) for r in records)
                    for p in FIRST_PERSON_FORMS})


def corpus_tics(records: list[dict]) -> tuple[list[tuple[str, float]], int]:
    """句尾后缀出现率：按 [。？！…] 切句、strip 后首中即停（雏形同序同 break）。"""
    n_sent, hits = 0, Counter()
    for r in records:
        for sent in SENT_SPLIT_RE.split(r["text"]):
            s = sent.strip()
            if not s:
                continue
            n_sent += 1
            for suf in TIC_SUFFIXES:
                if s.endswith(suf):
                    hits[suf] += 1
                    break
    return [(suf, n / n_sent) for suf, n in hits.most_common()], n_sent


def alnum_tokens(records: list[dict]) -> Counter:
    pat = re.compile(r"[A-Za-zＡ-Ｚａ-ｚ0-9０-９]+")
    c: Counter = Counter()
    for r in records:
        c.update(pat.findall(r["text"]))
    return c


def glossary_entries(records: list[dict]) -> dict:
    """写法归一：高频 token 若用全角，把半角写法拉回语料一致形（雏形同规则）。"""
    tokens = alnum_tokens(records)
    entries: dict[str, str] = {}
    for tok, n in tokens.most_common():
        if n < GLOSSARY_MIN_COUNT:
            continue
        hw = unicodedata.normalize("NFKC", tok)
        if hw != tok:
            entries[hw] = tok
            lw = hw.lower()
            if lw != hw and lw not in entries:
                entries[lw] = tok
    return entries


def _sentences(records: list[dict]) -> list[str]:
    out: list[str] = []
    for r in records:
        out.extend(s.strip() for s in SENT_SPLIT_RE.split(r["text"]) if s.strip())
    return out


def fingerprint(records: list[dict], transition_seeds) -> dict:
    """句式指纹（女娲表达 DNA 量化表·确定性维度）：语气比例/转折/句长/疑问。"""
    sents = _sentences(records)
    n = len(sents)
    polite = sum(1 for r in records if any(m in r["text"] for m in POLITENESS_MARKS))
    return {
        "n_sentences": n,
        "politeness_ratio": round(polite / len(records), 3) if records else 0.0,
        "transition_rate": (round(sum(1 for s in sents
                                      if any(t in s for t in transition_seeds)) / n, 4)
                            if n else 0.0),
        "avg_sentence_len": round(sum(len(s) for s in sents) / n, 2) if n else 0.0,
        "question_rate": (round(sum(1 for s in sents if s.endswith("？")) / n, 4)
                          if n else 0.0),
    }


def zh_tic_rates(records: list[dict]) -> list[tuple[str, float]]:
    """中文侧句尾语气词率（localized/translated 语料；成对口径的 zh 半边）。"""
    sents = _sentences(records)
    n = len(sents)
    if not n:
        return []
    hits = Counter(suf for s in sents for suf in ZH_TIC_SUFFIXES if s.endswith(suf))
    return [(suf, c / n) for suf, c in hits.most_common()]


def build_verbal_tics(ja_tics: list[tuple[str, float]],
                      zh_tics: list[tuple[str, float]]) -> tuple[list[dict], dict]:
    """成对口癖：ja 侧达标（≥TIC_MIN_RATE）配中文等价后缀；成对概率合计 ≤0.6（§0.4）。

    zh 产 string（schema minLength=1；2026-09-18 修正——原实现落 list 且允许空
    等价，EP1 空表侥幸未触发，EP2+ 达标即过不了出厂 schema）。无中文等价映射的
    达标后缀不产条目：造不出成对就不造（宁空勿造，§4.2）。
    """
    zh_rate = dict(zh_tics)
    pairs = []
    for suf, rate in ja_tics:
        if rate < TIC_MIN_RATE:
            continue
        eqs = SUFFIX_EQUIV_ZH.get(suf, ())
        if not eqs:
            continue
        rate_zh = max((zh_rate.get(e, 0.0) for e in eqs), default=0.0)
        pairs.append({"ja": suf, "zh": "/".join(eqs), "rate_ja": round(rate, 4),
                      "rate_zh": round(rate_zh, 4)})
    pair_sum = round(sum(p["rate_ja"] + p["rate_zh"] for p in pairs), 4)
    checks = {"pair_sum": pair_sum, "pair_cap": 0.6, "ok": pair_sum <= 0.6}
    return pairs, checks


# ── Y4① 候选提取 + ②三信号打分 ─────────────────────────────────────────────
def _scn_texts(raw_root: Path, pattern: str) -> dict[str, str]:
    """章节 → 台本原文（Y4 命中计数口径 = 全台本原文，含旁白与他人台词）。"""
    files = sorted(p for p in Path(raw_root).glob(pattern) if ".resx." not in p.name)
    return {p.stem.split(".")[0]: p.read_text(encoding="utf-8-sig") for p in files}


def _voice_chapter_map(scn_texts: dict[str, str], id_pattern: str) -> dict[str, str]:
    """voice_id → 章节（scn json 文本 scraping，免解析；模式取自 extract_rules）。"""
    core = id_pattern.strip("^$")
    pat = re.compile(rf'"voice"\s*:\s*"({core})"')
    out: dict[str, str] = {}
    for chapter, text in scn_texts.items():
        for m in pat.finditer(text):
            out.setdefault(m.group(1), chapter)
    return out


def candidate_terms(scn_texts: dict[str, str]) -> list[str]:
    """Y4① 候选来源：称呼后缀模式 + char 字段名 + 片假名长串 + kanji n-gram（≥频次门）。

    kanji 串按 2~4 全子串切分（非贪心整段）——保证「深沢」这类短实体拿到完整
    出现次数；候选集只是「待打分名词」，命中数在打分阶段按全台本原文口径重算。
    """
    full = "\n".join(scn_texts.values())
    terms: set[str] = set()
    for text in scn_texts.values():
        for m in NAME_SUFFIX_RE.finditer(text):
            terms.add(m.group(1))
        for m in PLACE_SUFFIX_RE.finditer(text):
            terms.add(m.group(1))
        for m in re.finditer(r'"char"\s*:\s*"([^"]{1,6})"', text):
            terms.add(m.group(1))
    terms.update(extract_katakana_terms([full]).keys())
    grams: Counter = Counter()
    for run in re.finditer(r"[一-龥]{2,}", full):
        s = run.group(0)
        length = len(s)
        for i in range(length):
            for span in (2, 3, 4):
                if i + span <= length:
                    grams[s[i:i + span]] += 1
    terms.update(term for term, n in grams.items() if n >= NGRAM_MIN_COUNT)
    return sorted(terms)


def score_candidates(scn_texts: dict[str, str], records: list[dict],
                     tension_seeds, id_pattern: str) -> list[dict]:
    """Y4② 三信号打分 + 报告页候选区数据（分数/章节分布/3~5 句命中原文）。

    命中计数口径 = 全台本原文（含旁白与他人台词，§0.4 锚点：深沢8/九十九22）。
    """
    total_chapters = len(scn_texts)
    chapter_map = _voice_chapter_map(scn_texts, id_pattern)
    lines_by_chapter = {ch: text.splitlines() for ch, text in scn_texts.items()}

    def _score(term: str) -> dict | None:
        hits = sum(text.count(term) for text in scn_texts.values())
        if hits < 2:
            return None  # 零散片段不成候选
        chapters_hit = sorted(ch for ch, text in scn_texts.items() if term in text)
        contain_lines = [line for lines in lines_by_chapter.values()
                         for line in lines if term in line]
        tension_hits = sum(1 for line in contain_lines
                           if any(seed in line for seed in tension_seeds))
        tension_rate = tension_hits / len(contain_lines) if contain_lines else 0.0
        hit_records = [r for r in records if term in r["text"]]
        extreme = sum(1 for r in hit_records if r.get("face") in EXTREME_FACES)
        extreme_rate = extreme / len(hit_records) if hit_records else 0.0
        signals = score_signals(len(chapters_hit), total_chapters,
                                tension_rate, extreme_rate)
        return {
            "term": term, "hits": hits,
            "chapters_hit": chapters_hit,
            "voice_chapters": sorted({chapter_map.get(r["voice"], "?")
                                      for r in hit_records}),
            "signals": signals,
            "score": signals["score"],
            "examples": [r["text"][:80] for r in hit_records[:5]],
        }

    out = [c for term in candidate_terms(scn_texts)
           if (c := _score(term)) is not None]
    out.sort(key=lambda c: (-c["score"], -c["hits"], c["term"]))
    out = out[:50]   # 报告页候选区容量：全量打分后按分数取 top-50
    ranked = {c["term"] for c in out}
    for term in Y4_ANCHOR_WORDS:   # 历史初版词强制入候选区（低分即「空表」的证据链）
        if term not in ranked:
            c = _score(term)
            if c:
                c["anchor"] = True
                out.append(c)
    out.sort(key=lambda c: (-c["score"], -c["hits"], c["term"]))
    return out


# ── 主流程（双落盘）────────────────────────────────────────────────────────
def run(raw_root: Path, profile_path: Path, work_dir: Path) -> dict:
    raw_root = Path(raw_root).resolve()
    profile = yaml.safe_load(Path(profile_path).read_text(encoding="utf-8"))
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    speaker = profile["character"]["speaker_names"][0]
    tension_seeds = tuple(profile.get("persona", {}).get(
        "tension_words", DEFAULT_TENSION_SEEDS))

    entries = json.loads((work_dir / "voice_map.json").read_text(encoding="utf-8"))

    # ogg 存在性过滤（与雏形 load_corpus 同效果；子目录约定取自 extract_rules）
    extract_rules = yaml.safe_load(
        (REPO_ROOT / "config" / "extract_rules.yaml").read_text(encoding="utf-8"))
    voice_sub = extract_rules.get("voice", {}).get("subdir", "my")
    voice_dir = (raw_root / profile["engine"]["voice_dir"]).resolve() / voice_sub
    if not voice_dir.resolve().is_relative_to(raw_root):
        raise ValueError(f"voice 子目录越出原料区：{voice_dir}")
    stems = ({p.stem for p in voice_dir.iterdir() if p.suffix.lower() == ".ogg"}
             if voice_dir.is_dir() else set())
    records = load_corpus(entries, speaker, stems)

    # ── ja 侧（native）统计 ──
    fp = first_person_counts(records)
    ja_tics, n_sent = corpus_tics(records)
    glossary = glossary_entries(records)
    dominant = max(FIRST_PERSON_FORMS, key=lambda f: fp.get(f, 0))
    replace_map_ja = {v: dominant
                      for v in FIRST_PERSON_VARIANTS.get(dominant, ())
                      if v != dominant}
    ja_fp = {
        "first_person": dict(fp),
        "replace_map_ja": replace_map_ja,
        "glossary_ja": glossary,
        "alnum_tokens": dict(alnum_tokens(records).most_common(10)),
        "tics_measured": {suf: round(rate * 100, 1) for suf, rate in ja_tics},
        **fingerprint(records, TRANSITION_SEEDS_JA),
    }

    # ── zh 侧统计（origin 分列：localized 官方汉化 / translated 机翻补缺）──
    # 2026-09-12 修复（审评 P1-1）：原筛选 ==translated 系翻案前残留——embedded
    # 全量 localized 语料下会静默归零；现按 origin∈{localized, translated} 收录。
    zh_src = [e for e in entries
              if e.get("text_zh") and e.get("origin") in ("localized", "translated")]
    zh_records = [{"voice": e["voice"], "text": clean_text(e["text_zh"]),
                   "face": None, "origin": e.get("origin", "translated")}
                  for e in zh_src]
    zh_tics = zh_tic_rates(zh_records)
    zh_fp = {
        "sample_size": len(zh_records),
        "corpus_source": "+".join(sorted({r["origin"] for r in zh_records})) or "none",
        "tics_measured": {suf: round(rate * 100, 1) for suf, rate in zh_tics},
        **fingerprint(zh_records, TRANSITION_SEEDS_ZH),
    }

    verbal_tics, checks = build_verbal_tics(ja_tics, zh_tics)

    # ── Y4 词表候选（勾选=人工，本工序不写 daily_corpus_blocklist）──
    scn_texts = _scn_texts(raw_root, profile["engine"]["scn_pattern"])
    id_pattern = extract_rules.get("voice", {}).get("id_pattern", r"^my\d{4}[a-z]?$")
    candidates = score_candidates(scn_texts, records, tension_seeds, id_pattern)
    anchor_hits = {w: sum(text.count(w) for text in scn_texts.values())
                   for w in Y4_ANCHOR_WORDS}

    stats = {
        "game_id": profile["game_id"],
        "corpus": {"records": len(records), "sentences": n_sent,
                   "speaker": speaker},
        "ja": ja_fp,
        "zh": zh_fp,
        "verbal_tics": verbal_tics,
        "checks": checks,
        "y4": {
            "tension_words": list(tension_seeds),
            "n_candidates": len(candidates),
            "anchor_hits": anchor_hits,
        },
    }
    (work_dir / "e3_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    tts_config = {
        "strip_actions": True,
        "quote_guard": True,
        "edge_voice": TTS_EDGE_VOICE_DEFAULT,   # E3 缺省；pack 代写覆盖（override_rule）
        "replace_map": {"zh": {}, "ja": replace_map_ja},
        "glossary": {"zh": {}, "ja": glossary},
        "verbal_tics": verbal_tics,
    }
    tts_text = (
        "# tts_config.yaml —— E3 统计产出（write_scopes persona 身份；非手拍）\n"
        "# 数据源: work/voice_map.json（sha256 冻结链）× 都语料；edge_voice 由 pack 代写\n"
        + yaml.safe_dump(tts_config, allow_unicode=True, sort_keys=False))
    (work_dir / "tts_config.yaml").write_text(tts_text, encoding="utf-8")

    (work_dir / "wordlist_candidates.json").write_text(
        json.dumps({"candidates": candidates, "anchor_hits": anchor_hits,
                    "tension_words": list(tension_seeds),
                    "note": "勾选=人工（L5 决策节点，绑定 voice_map hash）——本工序不写词表"},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    top = f"{ja_tics[0][0]}({round(ja_tics[0][1] * 100, 1)}%)" if ja_tics else "—"
    print(f"[E3] records={len(records)} sentences={n_sent} tics_top={top} "
          f"politeness={ja_fp['politeness_ratio']} "
          f"verbal_tics={len(verbal_tics)} y4_candidates={len(candidates)}")
    LOG.info("统计完成 records=%d sentences=%d 第一人称=%s 口癖=%d 条 "
             "成对合计=%s y4候选=%d → e3_stats.json + tts_config.yaml",
             len(records), n_sent, ja_fp.get("first_person"), len(verbal_tics),
             checks.get("pair_sum"), len(candidates))
    return stats


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    parser = argparse.ArgumentParser(description="E3 表达 DNA 统计 + Y4 词表候选（零 LLM）")
    parser.add_argument("--raw", required=True)
    parser.add_argument("--profile", default="config/game_9nine_ep1.yaml")
    parser.add_argument("--work", default="work")
    args = parser.parse_args()
    from contracts.logging_util import setup_logging
    setup_logging(Path(args.work))   # 直跑也落 work/logs/pipeline.log

    try:
        run(Path(args.raw), Path(args.profile), Path(args.work))
    except (RuntimeError, ValueError) as exc:
        print(f"[E3][fail-fast] {exc}", file=sys.stderr)
        sys.exit(2)
