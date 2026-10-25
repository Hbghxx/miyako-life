"""E2 立绘验收（v2 转型，2026-09-13）—— pet 侧合成产物的契约验收 + 需求模拟 + pack 直通源。

生产链裁决（2026-09-13 用户定案）：图层合成 = **pet 侧 tools/compose_all_sprites.py**
（新合成思路权威实现：全枚举、闭集外差分 expressions_extra、饰品独立层 accessories、
底图无发饰开关化）；归档/生产源 = **材料库**（sprites.compose_dir 可覆盖，缺省
<材料库>/fgimage/sprites_pkg）；**E2 不再自产 manifest**（单一生产者，L1）——只做：
  ①契约验收（fail-fast）：emotion_set 与 9 元闭集互锁、sprites 键 ⊆ 闭集、
    png_expression_map 值 ⊆ 闭集、闭集三元组 PNG 文件存在性、每情绪 default 别名在位、
    dormant 键在位；
  ②voice_map 需求模拟（face_map 日文词 → 闭集 → manifest 查表，fallback 链不变）；
  ③产出 work/sprites_manifest.json（**pet manifest 原样直通**，pack 装配源）+
    work/sprites_meta.json（验收报告：覆盖缺口/需求命中率/计数）。

归档单位约定：compose_dir = **包含 manifest.json 及其引用 PNG 的目录**（= pet 包
miyako.char/sprites/ 子集的复制）——manifest 引用的 expressions_extra/accessories
PNG 只存在于包输出，output_full 归档不含，故归档需连 manifest 一并复制。
fail-fast（Y11 语义更新）：合成产物缺失 → "立绘合成不在 life 范围，先在 pet 侧跑
tools/compose_all_sprites.py 并归档（sprites_pkg）"。
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
import os
import tempfile
from pathlib import Path

import yaml

from contracts.emotion_defs import SENTIMENT_CLOSE_SET
from contracts.logging_util import get_logger
from upstream.extractors.base import UpstreamFailFast

LOG = get_logger("e2")

REPO_ROOT = Path(__file__).resolve().parents[1]
FACE_MAP_PATH = (REPO_ROOT / "contracts" / "face_map.yaml").resolve()
ALLOWED_WORK_BASES = tuple(
    os.path.realpath(p) for p in (REPO_ROOT, Path(tempfile.gettempdir()))
)

POSE_LABELS = {"A": "正面", "B": "右向き", "C": "左向き", "D": "後ろ（背影，无表情轴）"}
POSE_NAME = {"A": "stand", "B": "right", "C": "left", "D": "back"}


class E2FailFast(UpstreamFailFast):
    pass


def _ensure_writable(work_dir: Path) -> None:
    wd = os.path.realpath(work_dir)
    for base in ALLOWED_WORK_BASES:
        if wd == base or wd.startswith(base + os.sep):
            return
    raise E2FailFast(f"work 目录越出允许范围（仓库内或系统临时目录）：{work_dir}")


def load_face_map() -> dict:
    return yaml.safe_load(FACE_MAP_PATH.read_text(encoding="utf-8"))


def compose_dir_of(raw_root: Path, profile: dict) -> Path:
    """合成产物归档位：profile sprites.compose_dir 覆盖，缺省 = 材料库 sprites_pkg。"""
    override = (profile.get("sprites", {}) or {}).get("compose_dir")
    if override:
        return Path(override)
    return Path(raw_root).parent / "miyako_life材料库" / "fgimage" / "sprites_pkg"


def validate_manifest(manifest: dict, compose_dir: Path) -> dict:
    """契约验收（fail-fast）：闭集互锁 + 查表完整性 + PNG 存在性。返回闭集三元组数。"""
    closed = set(SENTIMENT_CLOSE_SET)

    emotion_set = manifest.get("emotion_set")
    if sorted(emotion_set or []) != sorted(closed):
        raise E2FailFast(
            f"manifest emotion_set 与 9 元闭集不一致：{emotion_set}（L7 契约不自造语义；"
            f"权威 = contracts/emotion_defs.py，先在 pet 侧合成工具校正 EMOTION_KEY）")
    sprites = manifest.get("sprites") or {}
    bad_keys = sorted(set(sprites) - closed)
    if bad_keys:
        raise E2FailFast(f"sprites 查表键越出闭集：{bad_keys}")
    if not sprites:
        raise E2FailFast("manifest sprites 查表为空")

    missing_files: list[str] = []
    triples = 0
    for emotion, costumes in sprites.items():
        if not any(ps for ps in costumes.values()):
            raise E2FailFast(f"闭集情绪 {emotion} 无任何 costume/pose 组合")
        if "default" not in costumes:
            raise E2FailFast(f"闭集情绪 {emotion} 缺 default 别名（pet 查表缺 outfit 兜底依赖）")
        for costume, poses in costumes.items():
            for pose, fname in poses.items():
                triples += 1
                if not (compose_dir / fname).is_file():
                    missing_files.append(fname)
    if missing_files:
        raise E2FailFast(
            f"manifest 引用的 PNG 缺失 {len(missing_files)} 个（前 10）："
            f"{missing_files[:10]}——归档不完整（须为包 sprites 子集 + manifest.json）")

    face_map = manifest.get("png_expression_map") or {}
    bad_map = sorted({e for e in face_map.values() if e not in closed})
    if bad_map:
        raise E2FailFast(f"png_expression_map 值越出闭集：{bad_map}")

    dormant = manifest.get("dormant")
    if not dormant or not str(dormant).endswith(".png"):
        raise E2FailFast(f"dormant 键缺失或非法：{dormant!r}")
    if not (compose_dir / dormant).is_file():
        raise E2FailFast(f"dormant 立绘文件缺失：{dormant}")
    return {"closed_triples": triples}


def pose_coverage(sprites: dict) -> dict:
    """per-pose 闭集覆盖（原料层限制的证据链：A=stand 缺 sad、B=right 缺 sleepy、
    C=left 全覆盖、D=back 仅 normal——查表缺项由 sprite_manager fallback normal）。"""
    by_pose = {"stand": "A", "right": "B", "left": "C", "back": "D"}
    out = {}
    for pose, letter in by_pose.items():
        exprs = sorted({e for e, costumes in sprites.items()
                        if any(pose in ps for ps in costumes.values())})
        out[letter] = {"pose": pose, "expressions": exprs,
                       "missing": sorted(set(SENTIMENT_CLOSE_SET) - set(exprs))}
    return out


def simulate_demand(entries: list[dict], sprites: dict, face_map: dict) -> dict:
    """fallback 链模拟：voice_map 各条目的 (服装, 表情, 站位) 需求查 manifest 查表。

    链：exact → 同服装+normal → miss（pet sprite_manager 同语义：缺项逐级 fallback normal）。
    无立绘状态（self 为空）或服装词未收录（如 裸）不计入需求。
    """
    dress_map, pose_map = face_map.get("dress_map", {}), face_map.get("pose_map", {})
    exact = fallback = miss = 0
    unmapped_dresses: set[str] = set()
    for entry in entries:
        state = entry.get("self")
        if not state:
            continue
        costume = dress_map.get(state.get("dress"))
        pose = pose_map.get(state.get("pose"))
        if costume is None or pose is None:
            if state.get("dress") is not None:
                unmapped_dresses.add(state["dress"])
            continue
        emotion = face_map["scn_faces"].get(state.get("face"), "normal")
        if emotion not in SENTIMENT_CLOSE_SET:
            emotion = "normal"
        poses = sprites.get(emotion, {}).get(costume, {})
        if pose in poses:
            exact += 1
        elif pose in sprites.get("normal", {}).get(costume, {}):
            fallback += 1
        else:
            miss += 1
    total = exact + fallback + miss
    return {
        "total": total, "exact": exact, "fallback": fallback, "miss": miss,
        "coverage": round(exact / total, 3) if total else 0.0,
        "unmapped_dresses": sorted(unmapped_dresses),
    }


def run(raw_root: Path, profile_path: Path, work_dir: Path) -> dict:
    raw_root = Path(os.path.realpath(Path(raw_root).expanduser()))
    work_dir = Path(os.path.realpath(Path(work_dir).expanduser()))
    _ensure_writable(work_dir)
    profile = yaml.safe_load(Path(profile_path).read_text(encoding="utf-8"))
    compose_dir = compose_dir_of(raw_root, profile)
    if not compose_dir.is_dir():
        raise E2FailFast(
            f"立绘合成产物归档缺失：{compose_dir}"
            f"（立绘合成不在 life 范围——先在 pet 侧跑 tools/compose_all_sprites.py，"
            f"把包 sprites/ 子集 + manifest.json 归档到材料库该目录，"
            f"或用 profile sprites.compose_dir 指向现有归档位）")
    manifest_path = compose_dir / "manifest.json"
    if not manifest_path.is_file():
        raise E2FailFast(
            f"归档缺 manifest.json：{compose_dir}"
            f"（compose 工具只把 manifest 写进 pet 包——归档时须一并复制）")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("version"):   # 形状变更必须 bump（协调单 P-3 纪律，L-4 余项）
        raise E2FailFast("manifest 缺 version 键（形状变更必须 bump——pet compose 纪律）")
    counts = validate_manifest(manifest, compose_dir)
    coverage = pose_coverage(manifest["sprites"])

    voice_map_path = work_dir / "voice_map.json"
    entries: list[dict] = []
    if voice_map_path.is_file():  # E1 先行时做需求模拟；单独跑 E2 则跳过
        entries = json.loads(voice_map_path.read_text(encoding="utf-8"))
    demand = (simulate_demand(entries, manifest["sprites"], load_face_map())
              if entries else {})

    (work_dir / "sprites_manifest.json").write_bytes(manifest_path.read_bytes())
    meta = {
        "game_id": profile["game_id"],
        "compose_dir": str(compose_dir),
        "manifest_version": manifest.get("version"),
        "counts": {**counts,
                   "extras": sum(len(faces) for faces in
                                 (manifest.get("expressions_extra") or {}).values()),
                   "accessories": sum(len(poses) for poses in
                                      (manifest.get("accessories") or {}).values()),
                   "png_on_disk": sum(1 for _ in compose_dir.glob("*.png"))},
        "coverage": coverage,
        "demand": demand,
        "dormant": manifest.get("dormant"),
        "note": "E2 v2 验收器：pet compose_all_sprites 产物契约验收 + 需求模拟；"
                "sprites_manifest.json 为 pet manifest 原样直通（pack 装配源）",
    }
    (work_dir / "sprites_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    extras = meta["counts"]["extras"]
    print(f"[E2] 验收通过：闭集三元组 {counts['closed_triples']}（extras {extras}/"
          f"accessories {meta['counts']['accessories']}）| dormant={meta['dormant']} | "
          f"demand_coverage={demand.get('coverage', '—')} → sprites_manifest.json（直通）")
    LOG.info("验收通过 compose_dir=%s manifest_version=%s 闭集三元组=%d "
             "extras=%d accessories=%d png_on_disk=%d demand_coverage=%s",
             compose_dir, meta["manifest_version"], counts["closed_triples"],
             extras, meta["counts"]["accessories"], meta["counts"]["png_on_disk"],
             demand.get("coverage", "—"))
    return meta


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    parser = argparse.ArgumentParser(description="E2 立绘验收（pet 合成产物契约验收）")
    parser.add_argument("--raw", required=True)
    parser.add_argument("--profile", default="config/game_9nine_ep1.yaml")
    parser.add_argument("--work", default="work")
    args = parser.parse_args()
    from contracts.logging_util import setup_logging
    setup_logging(Path(args.work))   # 直跑也落 work/logs/pipeline.log

    try:
        run(Path(args.raw), Path(args.profile), Path(args.work))
    except (E2FailFast, RuntimeError) as exc:
        LOG.error("fail-fast %s", exc)
        print(f"[E2][fail-fast] {exc}", file=sys.stderr)
        sys.exit(2)
