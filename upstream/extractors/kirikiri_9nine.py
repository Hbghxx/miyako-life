"""kirikiri_9nine extractor —— 9-nine EP1 台本提取（L1.1；字段映射 = extract_rules.yaml）。

判定语义与手工雏形（原料区 scn/1.py，产出冻结件 sha256 d319f4bd…）逐条对齐：
  * voiced 节点 = len≥4 且 [2] 为台词 str 且 [3] 为非空 dict 列表的 list 节点；
  * voice id 正则 ^my\d{4}[a-z]?$（允许 my1970a 尾缀变体）；
  * 台词 \\n 字面量转真换行；说话人立绘状态取 [5].data 内 char==speaker 的首条；
  * 一节点多 voice → 每 voice 一行（重复 voice 保留，去重是下游 V1 的语义）。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from upstream.extractors.base import collect_character_states, scan_tree

VOICE_RE = re.compile(r"^my\d{4}[a-z]?$")

# voice id 存放子目录（engine 特有；对应原料区 voice/my，extract_rules.voice 落地）
VOICE_SUBDIR = "my"


class Kirikiri9nineExtractor:
    """9-nine（kirikiri/PSB）台本 → voice_map 行集（行键序 = 冻结件键序）。

    支持两种台本形态（自动识别）：
      * 传统单语（已整理区 a??.txt.json）：texts[i][2] 为台词 str；
      * Steam 官方三语内嵌（材料库 scn 经 FreeMote 反编译）：texts[i][2] 为
        语言子表 list——[0]=日文、[1]=英文、[2]=繁体中文（各行首元素为 flag）。
    """

    engine = "kirikiri_9nine"

    # Steam 多语言语言位（extract_rules 的语言位映射落地）
    LANG_JA = 0
    LANG_ZH = 2

    @staticmethod
    def _lang_text(langs: list, idx: int) -> str | None:
        """语言子表取文本：条目 = [flag, 文本, (换行变体...)]，取首个非空文本。"""
        if not isinstance(langs, list) or len(langs) <= idx:
            return None
        entry = langs[idx]
        if not isinstance(entry, list):
            return None
        for el in entry[1:]:
            if isinstance(el, str) and el.strip():
                return el
        return None

    @staticmethod
    def _match_voiced(node: list) -> tuple[list[str], str, str | None] | None:
        """voiced 判定 → (voice ids, 日文台词, 繁中文本或 None)；非 voiced 返回 None。"""
        if not (len(node) >= 4 and node[3]
                and isinstance(node[3], list)
                and all(isinstance(v, dict) for v in node[3])):
            return None
        if isinstance(node[2], str):                       # 传统单语形态
            text_ja, zh_tw = node[2], None
        elif isinstance(node[2], list):                    # Steam 三语形态
            text_ja = Kirikiri9nineExtractor._lang_text(node[2], Kirikiri9nineExtractor.LANG_JA)
            zh_tw = Kirikiri9nineExtractor._lang_text(node[2], Kirikiri9nineExtractor.LANG_ZH)
            if not text_ja:
                return None
        else:
            return None
        voices = [v["voice"] for v in node[3]
                  if VOICE_RE.match(str(v.get("voice", "")))]
        if not voices:
            return None
        return voices, text_ja, zh_tw

    def extract_text(self, text: str) -> list[dict]:
        data = json.loads(text)
        rows: list[dict] = []

        def match(node: list) -> list[dict] | None:
            matched = self._match_voiced(node)
            if matched is None:
                return None
            voices, text_ja, zh_tw = matched
            states: list[dict] = []
            state = node[5] if len(node) > 5 else None
            if isinstance(state, dict):
                states = collect_character_states(state.get("data"))
            speaker_state = next(
                (s for s in states if s["char"] == node[0]), None)
            rows_out = []
            for vid in voices:
                row = {
                    "voice": vid,
                    "speaker": node[0],
                    "costume": node[1],
                    "text": text_ja.replace("\\n", "\n"),
                    "self": speaker_state,
                    "stage": states,
                }
                if zh_tw is not None:
                    row["_zh_tw"] = zh_tw.replace("\\n", "\n")
                rows_out.append(row)
            return rows_out

        scan_tree(data, match, rows.extend)
        return rows

    def extract_file(self, path: Path) -> list[dict]:
        return self.extract_text(Path(path).read_text(encoding="utf-8-sig"))

    def extract_files(self, paths: list[Path]) -> list[dict]:
        rows: list[dict] = []
        for path in paths:
            rows.extend(self.extract_file(path))
        return rows

    @staticmethod
    def voice_stems(voice_dir: Path) -> set[str]:
        """voice 子目录下全部 .ogg 主干名（missing 对账用；子目录=voice/my）。"""
        base = Path(voice_dir).resolve()
        d = (base / VOICE_SUBDIR).resolve()
        if not d.is_relative_to(base):
            raise ValueError(f"voice 子目录越出 voice 根：{d}")
        if not d.is_dir():
            return set()
        return {p.stem for p in d.iterdir() if p.suffix.lower() == ".ogg"}
