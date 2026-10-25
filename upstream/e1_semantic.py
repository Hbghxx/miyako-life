"""E1 语义提取 —— 台本 → voice_map.json（单一事实源 + sha256 冻结链，plan L1.1）。

语义权威 = extract_rules.yaml + 方案契约 §五 E1；实现 = extractors/kirikiri_9nine。
产出：
  work/voice_map.json       冻结事实源（字节级复现手工雏形冻结件 d319f4bd…，
                            含 Windows CRLF 行尾与 indent=1——L1 纪律：hash 永不断）
  work/voice_map.meta.json  {sha256, entries, unique_voices, sprite_coverage, missing}
fail-fast：schema 违例逐条指出（§4.1）；行序即树扫描序，不做任何重排。
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
import os
import tempfile
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator

from upstream.extractors.base import iter_scn_files
from upstream.extractors.kirikiri_9nine import Kirikiri9nineExtractor
from contracts.logging_util import get_logger

LOG = get_logger("e1")

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = (REPO_ROOT / "contracts" / "voice_map.schema.json").resolve()
# 写入目录白名单：仓库工作区（work/）或系统临时目录（测试 tmp_path）
ALLOWED_WORK_BASES = tuple(
    os.path.realpath(p)
    for p in (REPO_ROOT, Path(tempfile.gettempdir()))
)


class E1FailFast(RuntimeError):
    """E1 产线失败（schema 违例等）——上游 fail-fast，禁止 build 继续。"""


_ZH_TW2S = None


def _tw2s(text: str) -> str:
    """繁体（台）→ 简体：OpenCC tw2s，确定性字形转换（2026-09-11 用户批准依赖）。

    专名对照 2026-09-11 用户**全量接受**（对照表 work/review/专名对照_全量.txt，
    3297 最小变化核；无豁免词）。转换仅作用于语料入库层，游戏原文件不受影响。
    """
    global _ZH_TW2S
    if _ZH_TW2S is None:
        from opencc import OpenCC
        _ZH_TW2S = OpenCC("tw2s")
    return _ZH_TW2S.convert(text)


def _resolve(p: str | Path) -> Path:
    """CLI 输入路径规范化：展开用户主目录与相对引用，得到无歧义绝对路径。"""
    return Path(os.path.realpath(Path(p).expanduser()))


def _ensure_writable(work_dir: Path) -> None:
    """写入目录必须落在白名单基目录内（仓库工作区 / 系统临时目录）。"""
    wd = os.path.realpath(work_dir)
    for base in ALLOWED_WORK_BASES:
        if wd == base or wd.startswith(base + os.sep):
            return
    raise E1FailFast(
        f"work 目录越出允许范围（仓库内或系统临时目录）：{work_dir}")


def _validate(entries: list[dict]) -> None:
    """voice_map schema 校验；违例逐条指出后 fail-fast（§4.1）。"""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    errors = sorted(Draft202012Validator(schema).iter_errors(entries),
                    key=lambda e: list(e.path))
    if errors:
        details = "\n".join(
            f"  [{','.join(map(str, e.path)) or '<root>'}] {e.message}"[:200]
            for e in errors[:10])
        raise E1FailFast(
            f"voice_map schema 违例 {len(errors)} 处（前 10）：\n{details}")


def run(raw_root: Path, profile_path: Path, work_dir: Path) -> dict:
    raw_root = _resolve(raw_root)
    profile_path = _resolve(profile_path)
    work_dir = _resolve(work_dir)
    _ensure_writable(work_dir)
    if not raw_root.is_dir():
        raise E1FailFast(f"原料区不存在：{raw_root}")
    if not profile_path.is_file():
        raise E1FailFast(f"profile 不存在：{profile_path}")
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    engine_cfg = profile["engine"]

    extractor = Kirikiri9nineExtractor()
    scn_files = iter_scn_files(raw_root, engine_cfg["scn_pattern"])
    entries = extractor.extract_files(scn_files)

    # ── Steam 三语台本（embedded · zh 来源）：独立于原料区 json 的多语言语料 ──
    from upstream.e0_input_check import steam_json_dir
    steam_dir = steam_json_dir(raw_root, profile)
    steam_files = sorted(steam_dir.glob("a*.steam.json")) if steam_dir.is_dir() else []
    zh_tw_map: dict[str, str] = {}
    steam_ja: dict[str, str] = {}
    for f in steam_files:
        for row in extractor.extract_file(f):
            zh = row.pop("_zh_tw", None)
            if zh:
                zh_tw_map.setdefault(row["voice"], zh)
            steam_ja.setdefault(row["voice"], row["text"])
    ja_mismatch = sum(1 for e in entries
                      if e["voice"] in steam_ja and steam_ja[e["voice"]] != e["text"])

    # ── Steam 官方汉化合并（embedded · 2026-09-11 用户定案）─────────────────
    # 繁中→OpenCC(tw2s) 简体，origin: localized。机翻(translated) 仅作补缺——
    # 官方汉化优先；机翻禁入 P1/P2/P4、官方汉化准入的纪律由 origin 字段承载
    # （E4 消费侧执法）。translate 锁定态 = legacy 锚定模式，不并入官方汉化
    # （保证 ja-only 冻结件 d319f4bd 的回归锚永不断）。
    for row in entries:
        zh_tw = row.pop("_zh_tw", None)
        if zh_tw:
            zh_tw_map.setdefault(row["voice"], zh_tw)
    if zh_tw_map and profile["input"]["bilingual_source"] != "translate":
        localized = 0
        for entry in entries:
            zh_tw = zh_tw_map.get(entry["voice"])
            if zh_tw:
                entry["text_zh"] = _tw2s(zh_tw)
                entry["origin"] = "localized"
                localized += 1
        print(f"[E1] Steam 官方汉化合并：{localized} 条 origin=localized"
              f"（OpenCC tw2s 简体，覆盖 {localized}/{len(entries)}；"
              f"steam 文件 {len(steam_files)}，日文差异行 {ja_mismatch}）")
        LOG.info("官方汉化合并 localized=%d/%d steam文件=%d 日文差异行=%d",
                 localized, len(entries), len(steam_files), ja_mismatch)

    # E1.5 机翻补缺（translated）：仅填官方汉化未覆盖的 voice（localized 优先）
    trans_path = work_dir / "translations.json"
    if trans_path.is_file():
        translations = json.loads(trans_path.read_text(encoding="utf-8"))
        fallback = 0
        for entry in entries:
            if entry.get("text_zh"):
                continue
            zh = translations.get(entry["voice"])
            if zh:
                entry["text_zh"] = zh
                entry["origin"] = "translated"
                fallback += 1
        if fallback:
            print(f"[E1] 机翻补缺合并：{fallback} 条 origin=translated")
            LOG.info("机翻补缺 translated=%d", fallback)

    _validate(entries)

    # 冻结落盘：indent=1 + CRLF 行尾（与手工雏形冻结件逐字节一致；hash 永不断）
    out_path = work_dir / "voice_map.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(entries, ensure_ascii=False, indent=1).encode("utf-8")
    out_path.write_bytes(data.replace(b"\n", b"\r\n"))
    sha256 = hashlib.sha256(out_path.read_bytes()).hexdigest()

    voice_root = _resolve(raw_root / engine_cfg["voice_dir"])
    if not voice_root.is_relative_to(raw_root):
        raise E1FailFast(
            f"voice_dir 越出原料区：{voice_root}（profile engine.voice_dir 非法）")
    stems = extractor.voice_stems(voice_root)
    ref = {e["voice"] for e in entries}
    missing = sorted(ref - stems)
    with_self = sum(1 for e in entries if e["self"])
    meta = {
        "game_id": profile["game_id"],
        "voice_map_sha256": sha256,
        "entries": len(entries),
        "unique_voices": len(ref),
        "sprite_coverage": round(with_self / len(entries), 3) if entries else 0.0,
        "missing": len(missing),
    }
    meta_path = work_dir / "voice_map.meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    print(
        f"[E1] entries={meta['entries']} unique={meta['unique_voices']} "
        f"sprite_coverage={meta['sprite_coverage']} missing={meta['missing']} "
        f"sha256={sha256[:8]}… → {out_path.name}")
    LOG.info("冻结落盘 entries=%d unique=%d sprite_coverage=%s missing=%d "
             "sha256=%s", meta["entries"], meta["unique_voices"],
             meta["sprite_coverage"], meta["missing"], sha256[:12])
    return meta


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 直接调用时的仓库根注入
    parser = argparse.ArgumentParser(description="E1 语义提取（台本 → voice_map 冻结链）")
    parser.add_argument("--raw", required=True)
    parser.add_argument("--profile", default="config/game_9nine_ep1.yaml")
    parser.add_argument("--work", default="work")
    args = parser.parse_args()
    from contracts.logging_util import setup_logging
    setup_logging(Path(args.work))   # 直跑也落 work/logs/pipeline.log

    try:
        run(Path(args.raw), Path(args.profile), Path(args.work))
    except (E1FailFast, RuntimeError) as exc:
        print(f"[E1][fail-fast] {exc}", file=sys.stderr)
        sys.exit(2)
