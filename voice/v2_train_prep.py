"""V2 参考初筛 —— 5~10s 平静清晰句，rank=abs(duration-7.0)（plan L3.3）。

语义与雏形 build_reference_candidates 逐条对齐：
  faces = 通常/微笑/真面目（平静清晰正/中性脸），score ≥4，
  per 选前 REF_CANDIDATES=10 条，rank=|duration-7.0| asc。
产出：work/reference/candidates/ *.wav + candidates.yaml 元数据。

垫尾处理（2026-09-14 用户听审反馈）：游戏语音母带尾部静音极紧（实测 8/8 候选
末 30ms 已衰减到近零），单听"像被截断"。参考音色以自然收尾为佳——写 wav 前
追加 0.4s 静音垫尾（末 60ms 微淡出）；meta duration 仍记语音时长（不含垫尾）。
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
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from voice.v1_audio_prep import (DATASET_MIN_SCORE, EXTRACT_RULES_PATH,
                                 _pad_tail, load_corpus)
from contracts.logging_util import get_logger

REPO_ROOT = Path(__file__).resolve().parents[1]
REF_MIN_SEC, REF_MAX_SEC = 5.0, 10.0
REF_TARGET_SEC = 7.0
REF_CANDIDATES = 10
REF_FACES = ("通常", "微笑", "真面目")

LOG = get_logger("v2")


def run(raw_root: Path, profile_path: Path, work_dir: Path) -> dict:
    raw_root = Path(raw_root).resolve()
    profile = yaml.safe_load(Path(profile_path).read_text(encoding="utf-8"))
    work_dir = Path(work_dir)
    speaker = profile["character"]["speaker_names"][0]
    entries = json.loads((work_dir / "voice_map.json").read_text(encoding="utf-8"))
    # voice 子目录与 E3/E1/V1 同源：extract_rules.voice.subdir（2026-09-18 修复：
    # 原硬编码 "my"——V1 同款 2026-09-12 已修，本文件漏修）
    rules = yaml.safe_load(EXTRACT_RULES_PATH.read_text(encoding="utf-8"))
    voice_sub = rules.get("voice", {}).get("subdir", "my")
    voice_dir = (raw_root / profile["engine"]["voice_dir"]).resolve() / voice_sub
    if not voice_dir.is_relative_to(raw_root.resolve()):
        raise ValueError(f"voice 子目录越出原料区：{voice_dir}")

    records, _ = load_corpus(entries, speaker, voice_dir)
    pool = [r for r in records
            if REF_MIN_SEC <= r["duration"] <= REF_MAX_SEC
            and r["face"] in REF_FACES and r["score"] >= DATASET_MIN_SCORE]
    pool.sort(key=lambda r: (abs(r["duration"] - REF_TARGET_SEC), r["voice"]))
    picked = pool[:REF_CANDIDATES]

    cand_dir = work_dir / "reference" / "candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)
    meta_list = []
    for r in picked:
        data, sr = sf.read(str(voice_dir / f"{r['voice']}.ogg"))
        sf.write(str(cand_dir / f"{r['voice']}.wav"), _pad_tail(data, sr), sr,
                 subtype="PCM_16")
        meta_list.append({"file": f"candidates/{r['voice']}.wav",
                          "voice_id": r["voice"], "text": r["text"],
                          "duration": r["duration"], "face": r["face"]})
    meta = {"candidates": meta_list,
            "criteria": {"duration": [REF_MIN_SEC, REF_MAX_SEC],
                         "faces": list(REF_FACES),
                         "rank": f"abs(duration-{REF_TARGET_SEC}), voice_id",
                         "count": REF_CANDIDATES},
            "note": "GPT-SoVITS 零沉参考音频候选；终审=人工（报告页勾选）。"
                    "wav 附 0.4s 静音垫尾（末 60ms 淡出）——源 ogg 尾静音极紧，"
                    "duration 字段不含垫尾"}
    (work_dir / "reference" / "candidates.yaml").write_text(
        yaml.dump(meta, allow_unicode=True, sort_keys=False, width=100),
        encoding="utf-8")
    print(f"[V2] candidates={len(meta_list)}（rank=abs(duration-7.0)）")
    LOG.info("候选初筛 %d/%d（faces=%s）入选=%s", len(meta_list), len(records),
             "/".join(REF_FACES),
             [c["voice_id"] for c in meta_list] or "—")
    return meta


if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    parser = argparse.ArgumentParser(description="V2 参考初筛")
    parser.add_argument("--raw", required=True)
    parser.add_argument("--profile", default="config/game_9nine_ep1.yaml")
    parser.add_argument("--work", default="work")
    args = parser.parse_args()
    from contracts.logging_util import setup_logging
    setup_logging(Path(args.work))   # 直跑也落 work/logs/pipeline.log

    try:
        run(Path(args.raw), Path(args.profile), Path(args.work))
    except (RuntimeError, ValueError) as exc:
        LOG.error("fail-fast %s", exc)
        print(f"[V2][fail-fast] {exc}", file=sys.stderr)
        sys.exit(2)
