"""pack —— L4.3 出包装配：dist/{package_id}.char/（MVP 出口）。

装配清单（write_scopes：包内全部文件由 pack 物理写入，内容所有权归各产出工序）：
  character_card.yaml   ← work（E4 定稿卡，Y5 已绑盲测通过）
  tts_config.yaml       ← work（E3；--voice-less 时删 edge_voice 键 = tier-0 形态）
  relationship.yaml     ← contracts/golden_pair（pet 契约四分支，EP1 直接复用）
  sprites/              ← E2 验收源（compose_dir 全量：manifest.json + PNG）
  audiofile/            ← V1 九桶（--voice-less 跳过）
  reference_audio/      ← V2 候选 wav + candidates.yaml（--voice-less 跳过）
  manifest.yaml         ← pack 计算（schema 全字段 + voice_tier 伪码）
流程：装配到 staging → run_factory_check（fail-closed，任一不过不出包）→
      旧包时间戳备份 → staging 替换 final。
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
import shutil
import time
from pathlib import Path

import yaml

from contracts.emotion_defs import SENTIMENT_CLOSE_SET
from contracts.logging_util import get_logger
from contracts.naming import dataset_list_name
from pack.factory_check import run_factory_check

REPO_ROOT = Path(__file__).resolve().parents[1]
PAD_NOTE = "wav 附 0.4s 静音垫尾（源 ogg 尾静音极紧）"

LOG = get_logger("pack")


def compute_voice_tier(ref_audio_ready: bool, dataset_ready: bool,
                       edge_voice: str | None) -> int:
    """voice_tier 伪码（§0.4，永不变更）——v2.6 起 tier1=数据就绪态：

        ref_audio_ready and dataset_ready → 2
        elif edge_voice 非空              → 1
        else                              → 0

    edge_voice 非空判定 = 非 None 且 strip() 后非空（纯空白视为空）。
    --voice-less 时装配跳过音频产物、tier 按本式自然降档，公式不变。
    """
    if ref_audio_ready and dataset_ready:
        return 2
    if edge_voice and edge_voice.strip():
        return 1
    return 0


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def card_digest_of(card_path: Path) -> str:
    """卡面 digest（**LF 归一化文本口径**）——manifest card_digest 与日志
    card_sha 的唯一来源。

    read_text 的 universal-newlines 把 CRLF→LF 后再哈希：同内容跨平台
    （Windows CRLF / Linux LF）必得同值，是"卡文本"的身份口径——盲测冻结锚
    b006f3a6 即此口径。**与文件原始字节 sha 不是同一个数**（CRLF 下不同）：
    decisions.yaml 的 Y5 绑定用字节口径（sha256_of_path，防篡改），两口径
    各司其职；用本值校验包内 card 文件须先做同样归一化。
    （2026-09-19 复查定案：不改为字节口径——那会与全部文档锚点决裂，
    且字节随平台漂移反而破坏跨平台对账。）"""
    return hashlib.sha256(
        Path(card_path).read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def latest_weight(root: Path, sub: str, glob_pat: str, canonical_re: str) -> Path | None:
    """训练产出目录里的最新正典权重：(epoch, steps) 确定性取最大。

    只认 canonical_re **全名**匹配的训练产出（SoVITS=miyako_e{e}_s{s}.pth /
    GPT=miyako-e{e}.ckpt）；同目录的人工备份/变体名（miyako-e15-fix.ckpt 之类）
    跳过并警告——来路不明的同名文件不静默打进生产包（宁严勿松）。
    2026-09-18 修正：原实现只按 epoch 排序、同 epoch 平局取决于目录枚举序
    （实测盘面 SoVITS 同存原型 e8_s3904 与重训 e8_s3928），且非匹配文件名
    会在 .group(1) 处 AttributeError 炸穿装配。
    """
    ranked: list[tuple[tuple[int, int, str], Path]] = []
    for p in (Path(root) / sub).glob(glob_pat):
        m = re.fullmatch(canonical_re, p.name)
        if not m:
            print(f"[pack] ⚠ 跳过非常典权重文件（不匹配 {canonical_re}）："
                  f"{sub}/{p.name}")
            LOG.warning("跳过非常典权重文件（不匹配 %s）：%s/%s",
                        canonical_re, sub, p.name)
            continue
        ranked.append(((int(m.group(1)), int(m.group(2) or 0), p.name), p))
    return max(ranked)[1] if ranked else None


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _load(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else None


def _build_manifest(work: Path, profile: dict, pkg_id: str, voice_tier: int,
                    sprites_manifest: dict, sprites_png: int, cand_count: int,
                    face_map: dict, e3: dict, card_sha: str,
                    tts_models: list[str]) -> dict:
    vm_meta = _load_json(work / "voice_map.meta.json") or {}
    v1_meta = _load_json(work / "v1_meta.json") or {}
    compose_dir = (profile.get("sprites", {}) or {}).get("compose_dir") or "材料库 sprites_pkg"
    pack_cfg = profile.get("pack", {}) or {}
    return {
        "package_id": pkg_id,
        "schema_version": 1,
        "package_version": pack_cfg.get("package_version", "1.0.0"),
        "spec_version": pack_cfg.get("spec_version", "miyako_v1.9"),
        "generated_by": "miyako_life pack（E0~E6 + V1~V2 产线；sprites=E2 验收直通）",
        "character_package_dir": f"data/characters/{pkg_id}.char",
        "voice_tier": voice_tier,
        "source": {
            "voice_map": {"path": "work/voice_map.json",
                          "sha256": vm_meta.get("voice_map_sha256", ""),
                          "entries": vm_meta.get("entries", 0)},
            "sprites_dir": {"path": compose_dir, "png_count": sprites_png},
        },
        "assets": {
            "sprite": {"manifest": "sprites/manifest.json", "format": "png/rgba",
                       "axes": {"expression": list(SENTIMENT_CLOSE_SET),
                                "costume": ["casual", "uniform", "waitress",
                                            "parka", "naked"],
                                # pose 从 sprites manifest 权威源推导（2026-09-18：
                                # 原三处硬编码各说各话——profile sprites.axes=3、
                                # E2 POSE_LABELS、此处四值；pet manifest pose_labels
                                # 为实际合成枚举，查表缺项由 sprite_manager fallback）
                                "pose": list((sprites_manifest.get("pose_labels")
                                              or {}).keys()) or ["A", "B", "C", "D"]},
                       "coverage_note": "查表缺项由 sprite_manager 逐级 fallback normal"},
            "live2d": {"enabled": False, "note": "预留资产节（pet 设计 7.9）"},
            "reference_audio": {"candidates_meta": "reference_audio/candidates.yaml",
                                "candidates_count": cand_count,
                                "note": PAD_NOTE},
            "audiofile": {"root": "audiofile",
                          "counts": v1_meta.get("audiofile_counts", {}),
                          "note": "预录层停用——纯数据留档（恢复时直接复用）"},
            "tts_model": {"files": tts_models,
                          "note": "重训权重随卡（2026-09-16 用户裁定）；部署时放入"
                                  " GPT-SoVITS 对应目录即可加载"} if tts_models
                   else {"note": "gptsovits_root 无 miyako 权重"},
        },
        "emotion_map": {"note": "游戏 face 标签 → 9 元闭集；未知/缺失 fallback normal（硬约束）",
                        "face_to_emotion": face_map.get("scn_faces", {})},
        "first_person_stat": (e3 or {}).get("ja", {}).get("first_person", {}),
        "fallbacks": {"unknown_emotion": "normal", "empty_audiofile_bucket": "normal"},
        "game_assets_note": "sprites/*.png、audiofile/**、reference_audio/** 为游戏版权物料："
                            "不进 git；dataset 训练集留产线（1466 = 1459+7 人工策展 myb_sys）",
        "card_digest": card_sha[:12],
    }


def assemble(work: Path, raw_root: Path, profile_path: Path,
             out_root: Path, voice_less: bool = False) -> tuple[Path, bool, list[dict]]:
    profile = yaml.safe_load(Path(profile_path).read_text(encoding="utf-8"))
    pkg_id = profile["character"]["package_id"]
    LOG.info("装配开始 pkg_id=%s voice_less=%s work=%s", pkg_id, voice_less, work)
    dist = Path(out_root)
    dist.mkdir(parents=True, exist_ok=True)
    final = dist / f"{pkg_id}.char"
    staging = dist / f".staging_{pkg_id}"

    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    # ── 内容件（write_scopes：pack 物理写入，内容所有权归产出工序）────────
    card_text = (work / "character_card.yaml").read_text(encoding="utf-8")
    (staging / "character_card.yaml").write_text(card_text, encoding="utf-8")
    tts = yaml.safe_load((work / "tts_config.yaml").read_text(encoding="utf-8"))
    if voice_less:
        tts.pop("edge_voice", None)   # tier-0 形态：键缺失（schema 2026-09-16 修正）
    (staging / "tts_config.yaml").write_text(
        yaml.safe_dump(tts, allow_unicode=True, sort_keys=False), encoding="utf-8")
    shutil.copy2(REPO_ROOT / "contracts" / "golden_pair" / "miyako_sample"
                 / "relationship.yaml", staging / "relationship.yaml")

    compose_dir = (profile.get("sprites", {}) or {}).get("compose_dir")
    if not compose_dir:
        # 缺省 = 材料库 sprites_pkg（plan v2.4 定案；与 E2 compose_dir_of 同口径——
        # 约束 A 禁止 import upstream，内联同款规则。Y11：缺失 fail-fast）
        compose_dir = Path(raw_root).parent / "miyako_life材料库" / "fgimage" / "sprites_pkg"
    compose_dir = Path(compose_dir)
    if not compose_dir.is_dir():
        raise FileNotFoundError(
            f"sprites 合成产物源缺失：{compose_dir}（E2 验收直通需要材料库归档位，"
            f"或 profile sprites.compose_dir 指向现有目录）")
    _copytree = shutil.copytree
    _copytree(compose_dir, staging / "sprites")

    ref_ready = ds_ready = False
    if not voice_less:
        cand_src = work / "reference" / "candidates"
        cand_meta = work / "reference" / "candidates.yaml"
        ref_ready = cand_src.is_dir() and cand_meta.is_file()
        if ref_ready:
            _copytree(cand_src, staging / "reference_audio" / "candidates")
            shutil.copy2(cand_meta, staging / "reference_audio" / "candidates.yaml")
        aud_src = work / "audiofile"
        if aud_src.is_dir():
            _copytree(aud_src, staging / "audiofile")
    ds_ready = (work / "dataset" / dataset_list_name(profile)).is_file()

    # ── TTS 权重随卡（用户裁定 2026-09-16：炼好的 tts 跟着 card 走）──────────
    # 从 gptsovits_root 取最新 miyako 正典权重入包 tts/（选取语义见 latest_weight）。
    # 权重缺失不拦包（纯随卡资产，与 tier 伪码无关）。
    gptsovits_root = Path((profile.get("pack", {}) or {}).get("gptsovits_root", ""))
    tts_models: list[str] = []
    sov = gpt = None
    if gptsovits_root and gptsovits_root.is_dir():
        sov = latest_weight(gptsovits_root, "SoVITS_weights_v2",
                            "miyako_*.pth", r"miyako_e(\d+)(?:_s(\d+))?\.pth")
        gpt = latest_weight(gptsovits_root, "GPT_weights_v2",
                            "miyako-*.ckpt", r"miyako-e(\d+)(?:-(\d+))?\.ckpt")
    for sub, f in ([("SoVITS_weights_v2", sov)] if sov else []) + \
            ([("GPT_weights_v2", gpt)] if gpt else []):
        (staging / "tts" / sub).mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, staging / "tts" / sub / f.name)
        tts_models.append(f"tts/{sub}/{f.name}")

    voice_tier = 0 if voice_less else compute_voice_tier(
        ref_ready, ds_ready, tts.get("edge_voice"))
    face_map = yaml.safe_load((REPO_ROOT / "contracts" / "face_map.yaml")
                              .read_text(encoding="utf-8"))
    e3 = _load_json(work / "e3_stats.json")
    sprites_png = len(list((staging / "sprites").glob("*.png")))
    sprites_manifest = json.loads((staging / "sprites" / "manifest.json")
                                  .read_text(encoding="utf-8"))
    cand_meta_path = staging / "reference_audio" / "candidates.yaml"
    cand_count = len(_load(cand_meta_path).get("candidates", [])) \
        if cand_meta_path.is_file() else 0
    card_sha = card_digest_of(work / "character_card.yaml")
    manifest = _build_manifest(work, profile, pkg_id, voice_tier,
                               sprites_manifest, sprites_png, cand_count, face_map,
                               e3, card_sha, tts_models)
    (staging / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False), encoding="utf-8")

    # ── 出厂检查（fail-closed，§4.1：任一不过不出包）────────────────────
    ok, report = run_factory_check(staging, work, raw_root, profile, voice_less)
    for r in report:
        print(f"  [{'✓' if r['pass'] else '✗'}] {r['item']}: {r['detail']}")
        (LOG.info if r["pass"] else LOG.error)(
            "出厂检查 %s [%s] %s", r["item"], "✓" if r["pass"] else "✗",
            r["detail"])
    LOG.info("出厂检查汇总 %d/%d voice_less=%s", sum(r["pass"] for r in report),
             len(report), voice_less)
    if not ok:
        failed_dir = dist / f"{pkg_id}.char.failed.{time.strftime('%Y%m%d_%H%M%S')}"
        shutil.move(str(staging), str(failed_dir))
        print(f"[pack] ✗ 出厂检查未过——不出包（残件留 {failed_dir.name} 供排查）")
        LOG.error("不出包：出厂检查未过，残件=%s", failed_dir)
        return failed_dir, False, report

    # ── 旧包时间戳备份 → 替换 ──
    if final.exists():
        bak = dist / f"{pkg_id}.char.bak.{time.strftime('%Y%m%d_%H%M%S')}"
        shutil.move(str(final), str(bak))
        print(f"[pack] 旧包备份 → {bak.name}")
        LOG.info("旧包备份 → %s", bak)
    shutil.move(str(staging), str(final))
    print(f"[pack] ✓ 出包 → {final}（voice_tier={voice_tier}）")
    LOG.info("出包完成 %s voice_tier=%d 权重=%s card_sha=%s", final,
             voice_tier, tts_models or "无", card_sha[:12])

    # ── 完整 card 产出镜像（用户裁定 2026-09-16：生产时同步落 card/{character_name}）──
    # 仅完整包镜像（voice-less 诊断包不覆盖 card/ 库）。
    if not voice_less:
        card_root = REPO_ROOT / "card" / pkg_id
        if card_root.exists():
            shutil.rmtree(card_root)
        shutil.copytree(final, card_root)
        print(f"[pack] 完整 card 镜像 → {card_root}")
        LOG.info("完整 card 镜像 → %s（voice-less 诊断包不写镜像）", card_root)
    return final, True, report


def run(raw_root: Path, profile_path: Path, work_dir: Path,
        out_root: Path | None = None, voice_less: bool = False) -> Path:
    out_root = Path(out_root) if out_root else REPO_ROOT / "dist"
    final, ok, _ = assemble(work_dir, raw_root, profile_path, out_root, voice_less)
    if not ok:
        raise SystemExit(2)
    return final


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    parser = argparse.ArgumentParser(description="L4.3 pack 出包装配（fail-closed）")
    parser.add_argument("--raw", required=True)
    parser.add_argument("--profile", default="config/game_9nine_ep1.yaml")
    parser.add_argument("--work", default="work")
    parser.add_argument("--out", default=str(REPO_ROOT / "dist"))
    parser.add_argument("--voice-less", action="store_true")
    args = parser.parse_args()
    from contracts.logging_util import run_separator, setup_logging
    setup_logging(Path(args.work))
    run_separator("PACK START", f"raw={args.raw}")
    run(Path(args.raw), Path(args.profile), Path(args.work),
        Path(args.out), args.voice_less)
    run_separator("PACK END")
