"""V1 音频工程 —— dataset 训练列表 + audiofile 九桶预录（plan L3.1）。

语义与雏形 scripts/build_char_package.py 逐条对齐，锚点已实测复现：
  dataset 1459 / audiofile 159（sleepy=0）
三份 .list：规范（相对路径）/ 绝对路径 / 裸文件名。

纪律：soundfile 读 ogg 时长；WAV 转换 PCM_16；
     audiofile 九桶的 sleepy 空桶为产线决策（不复用 normal 台词）。
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
from pathlib import Path

import soundfile as sf
import yaml

from contracts.logging_util import get_logger

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTRACT_RULES_PATH = REPO_ROOT / "config" / "extract_rules.yaml"

LOG = get_logger("v1")

# ── 阈值（雏形同源）────────────────────────────────────────────────────────
DATASET_MIN_SCORE = 4
DATASET_MIN_SEC = 1.0
CLIP_MIN_SEC, CLIP_MAX_SEC = 1.5, 12.0
CLIP_TARGET_SEC = 4.5
CLIPS_PER_BUCKET = 20
DATASET_SPEAKER = "miyaku"
PAD_SEC, FADE_SEC = 0.4, 0.06   # 垫尾静音 / 末端淡出（2026-09-14 用户裁定"都修"）


def _pad_tail(data, sr: int):
    """末端淡出 + 追加静音垫尾（声道保持）——游戏语音母带尾部静音极紧
    （实测末 30ms 近零），不垫尾则训练/参考/预录都会"像被截断"。
    公共件：v2 参考候选复用（2026-09-14 用户裁定"都修"：dataset/audiofile 同修）。"""
    import numpy as np

    n_fade, n_pad = int(FADE_SEC * sr), int(PAD_SEC * sr)
    data = np.array(data, copy=True)
    if 0 < n_fade < len(data):
        fade = np.linspace(1.0, 0.0, n_fade)
        if data.ndim == 2:
            data[-n_fade:, :] *= fade[:, None]
        else:
            data[-n_fade:] *= fade
    tail = np.zeros((n_pad,) + data.shape[1:], dtype=data.dtype)
    return np.concatenate([data, tail], axis=0)

_KANA_RE = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")
_SMALL_KANA = "ーっゃゅょぁぃぅぇぉァィゥェォャュョッン"


def clean_text(raw: str) -> str:
    t = re.sub(r"%[^;]*;", "", raw)
    t = re.sub(r"\s*\n[\s\u3000]*", "", t)
    t = re.sub(r"^[「『\s\u3000]+", "", t.strip())
    t = re.sub(r"[」』\s\u3000]+$", "", t)
    return t.replace("\u3000", "").strip()


def content_score(text: str) -> int:
    n = 0
    for ch in text:
        if not _KANA_RE.match(ch):
            continue
        if ch in _SMALL_KANA:
            continue
        n += 1
    return n


def entry_face(e: dict) -> str | None:
    return ((e.get("self") or {}).get("face")) or None


def _face_map() -> dict:
    return yaml.safe_load(
        (REPO_ROOT / "contracts" / "face_map.yaml").read_text(encoding="utf-8"))


def entry_emotion(e: dict, face2emo: dict) -> str:
    from contracts.emotion_defs import DEFAULT_EMOTION, SENTIMENT_CLOSE_SET
    face = entry_face(e)
    if not face:
        return DEFAULT_EMOTION
    emo = face2emo.get(face)
    return emo if emo in SENTIMENT_CLOSE_SET else DEFAULT_EMOTION


def load_corpus(entries: list[dict], speaker: str,
                voice_dir: Path) -> tuple[list[dict], dict]:
    seen: set[str] = set()
    records: list[dict] = []
    stats = {"narration_dropped": 0, "dup_dropped": 0, "ogg_missing": 0}
    face2emo = _face_map()["scn_faces"]
    for e in entries:
        if e.get("speaker") != speaker:
            stats["narration_dropped"] += 1
            continue
        vid = e.get("voice")
        if vid in seen:
            stats["dup_dropped"] += 1
            continue
        ogg = Path(voice_dir) / f"{vid}.ogg"
        if not ogg.is_file():
            stats["ogg_missing"] += 1
            continue
        seen.add(vid)
        data, sr = sf.read(str(ogg))
        text = clean_text(e["text"])
        face = entry_face(e)
        records.append({
            "voice": vid, "text": text,
            "duration": round(len(data) / sr, 3),
            "score": content_score(text),
            "face": face,
            "emotion": entry_emotion(e, face2emo),
            "face_known": face is not None and face in face2emo,
        })
    return records, stats


def build_dataset(records: list[dict], voice_dir: Path, ds_out: Path,
                  list_name: str) -> dict:
    pool = [r for r in records
            if r["score"] >= DATASET_MIN_SCORE and r["duration"] >= DATASET_MIN_SEC]
    pool.sort(key=lambda r: r["voice"])
    wav_dir = ds_out / "audio"
    wav_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for r in pool:
        rel = f"audio/{r['voice']}.wav"
        data, sr = sf.read(str(voice_dir / f"{r['voice']}.ogg"))
        sf.write(str(ds_out / rel), _pad_tail(data, sr), sr, subtype="PCM_16")
        text = r["text"].replace("|", "！")
        lines.append(f"{rel}|{DATASET_SPEAKER}|ja|{text}")

    # ── 人工策展增补（dataset_supplement.yaml；2026-09-14 用户听审挑选 myb_sys）──
    # 系统语音 = "对玩家关注感"语域 + 开心 dayo 腔（正篇语料结构性缺失）；
    # config 版本化 → V1 重跑确定性追加不丢失；转写=人工提供（产线不做 ASR）。
    supp_cfg = REPO_ROOT / "config" / "dataset_supplement.yaml"
    manual = 0
    if supp_cfg.is_file():
        supp = yaml.safe_load(supp_cfg.read_text(encoding="utf-8")).get("entries", [])
        supp_dir = (voice_dir.parents[1].parent / "miyako_life材料库"
                    / "vioce" / "my")
        for e in supp:
            src = supp_dir / f"{e['id']}.ogg"
            if not src.is_file():
                raise ValueError(f"人工策展增补源缺失：{src}")
            data, sr = sf.read(str(src))
            rel = f"audio/{e['id']}.wav"
            sf.write(str(ds_out / rel), _pad_tail(data, sr), sr, subtype="PCM_16")
            text = str(e["text"]).replace(" ", "").replace("|", "！")
            lines.append(f"{rel}|{DATASET_SPEAKER}|ja|{text}")
            manual += 1
    (ds_out / list_name).write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    train = [f"{(ds_out / l.split('|', 1)[0]).resolve()}|{DATASET_SPEAKER}|ja|"
             f"{l.split('|', 3)[3]}" for l in lines]
    (ds_out / "gptsovits_train.list").write_text(
        "\n".join(train) + "\n", encoding="utf-8")
    bare = [l.split("|", 1)[0].split("/")[-1] for l in lines]
    (ds_out / "bare.list").write_text("\n".join(bare) + "\n", encoding="utf-8")
    return {"selected": len(pool), "lines": lines, "manual": manual}


def build_audiofile(records: list[dict], voice_dir: Path,
                    audio_out: Path) -> dict:
    from contracts.emotion_defs import DEFAULT_EMOTION, SENTIMENT_CLOSE_SET
    audio_out.mkdir(parents=True, exist_ok=True)
    for emo in SENTIMENT_CLOSE_SET:
        (audio_out / emo).mkdir(exist_ok=True)
    pool = [r for r in records
            if CLIP_MIN_SEC <= r["duration"] <= CLIP_MAX_SEC
            and r["score"] >= DATASET_MIN_SCORE]
    picked: dict[str, list] = {}
    used: set[str] = set()
    for emo in SENTIMENT_CLOSE_SET:
        if emo == "normal":
            tiers = [[r for r in pool if r.get("face") == "通常"],
                     [r for r in pool if not r.get("face_known", False)]]
        else:
            tiers = [[r for r in pool if r.get("emotion") == emo
                      and r.get("face_known", False)]]
        out: list[dict] = []
        for tier in tiers:
            tier = sorted(
                (t for t in tier if t["voice"] not in used),
                key=lambda r: (abs(r["duration"] - CLIP_TARGET_SEC), r["voice"]))
            for r in tier:
                if len(out) >= CLIPS_PER_BUCKET:
                    break
                if r["voice"] in used:
                    continue
                out.append(r)
                used.add(r["voice"])
            if len(out) >= CLIPS_PER_BUCKET:
                break
        picked[emo] = out
        for r in out:
            # 原为 copy2 直拷——2026-09-14"都修"裁定：读源→垫尾→重编码 ogg
            data, sr = sf.read(str(voice_dir / f"{r['voice']}.ogg"))
            sf.write(str(audio_out / emo / f"{r['voice']}.ogg"),
                     _pad_tail(data, sr), sr, format="OGG", subtype="VORBIS")
    return {emo: len(v) for emo, v in picked.items()}


def run(raw_root: Path, profile_path: Path, work_dir: Path) -> dict:
    raw_root = Path(raw_root).resolve()
    profile = yaml.safe_load(Path(profile_path).read_text(encoding="utf-8"))
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    speaker = profile["character"]["speaker_names"][0]
    # voice 子目录与 E3/E1 同源：extract_rules.voice.subdir（2026-09-12 修复——
    # 原为 `if False else {}` 死代码 + 硬编码 "my"，与 extract_rules 脱节）
    rules = yaml.safe_load(EXTRACT_RULES_PATH.read_text(encoding="utf-8"))
    voice_sub = rules.get("voice", {}).get("subdir", "my")
    entries = json.loads((work_dir / "voice_map.json").read_text(encoding="utf-8"))
    voice_dir = (raw_root / profile["engine"]["voice_dir"]).resolve() / voice_sub
    if not voice_dir.is_relative_to(raw_root.resolve()):
        raise ValueError(f"voice 子目录越出原料区：{voice_dir}")

    records, stats = load_corpus(entries, speaker, voice_dir)
    # 主 list 名随 profile（contracts/naming 单一来源；pack 就绪检查同源）——
    # 原与 pack 各自硬编码 "miyako.list"，换角色/改名会静默分叉
    from contracts.naming import dataset_list_name
    ds = build_dataset(records, voice_dir, work_dir / "dataset",
                       dataset_list_name(profile))
    from contracts.emotion_defs import SENTIMENT_CLOSE_SET
    counts = build_audiofile(records, voice_dir, work_dir / "audiofile")
    meta = {"records": len(records), "stats": stats,
            "dataset_selected": ds["selected"],
            "dataset_manual": ds.get("manual", 0),
            "audiofile_counts": counts,
            "audiofile_total": sum(counts.values())}
    (work_dir / "v1_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[V1] records={len(records)} dataset={ds['selected']} "
          f"audiofile={sum(counts.values())}（sleepy={counts.get('sleepy', 0)}）")
    LOG.info("V1 完成 records=%d dataset=%d+%d（人工策展）audiofile=%d "
             "sleepy=%d（空桶=产线决策）→ dataset/ + audiofile/",
             len(records), ds["selected"], ds.get("manual", 0),
             sum(counts.values()), counts.get("sleepy", 0))
    return meta


if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    parser = argparse.ArgumentParser(description="V1 音频工程")
    parser.add_argument("--raw", required=True)
    parser.add_argument("--profile", default="config/game_9nine_ep1.yaml")
    parser.add_argument("--work", default="work")
    args = parser.parse_args()
    from contracts.logging_util import setup_logging
    setup_logging(Path(args.work))   # 直跑也落 work/logs/pipeline.log

    try:
        run(Path(args.raw), Path(args.profile), Path(args.work))
    except (RuntimeError, ValueError) as exc:
        print(f"[V1][fail-fast] {exc}", file=sys.stderr)
        sys.exit(2)
