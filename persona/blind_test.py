"""L2.5 盲测材料 5v5 —— card 产出句 vs 真实语料句混排（plan L2.5）。

判定线（Y7）：首轮辨认率 ≥60%（6/10）或累计两轮 ≥70% 为 pass；
未达线 → P3/P5 加固后重测（受 refine_loop_max 约束）。
产出：work/blind_test_5v5.json {items: [{label, text}], answer_key: {label: source}}
  items 给盲测人（不泄露来源），answer_key 独立存证（L4.2 决策绑定用）。

清洗对称纪律（2026-09-14 用户报障修复）：语料句与生成句**同一清洗器**（E1/E3 同款
clean_text——剥「」引号壳、合并 \n 折行与全角空格）。否则格式差异（引号壳/折行）
成为盲测人可辨的来源信号，测试鉴别力被泄漏。
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
from pathlib import Path

import yaml

from contracts.llm import base_url_from_env, chat_completion, env_key
from persona.e3_dna_stats import clean_text

REPO_ROOT = Path(__file__).resolve().parents[1]
N_PER_SIDE = 5
MIN_SENT_LEN = 10  # 盲测素材最短字数——短于此无鉴别信号量

BLIND_PROMPT_TMPL = """你是桌面陪伴角色「都」。以下是你的persona：
{persona}
说话方式：{register}；自称：{first_person}。
请以「都」的身份用简体中文回应以下场景，输出单行台词，不要任何解释或格式标记。
场景：{scene}"""

SCENES = [
    "被夸奖做料理好吃",
    "深夜用户还没睡，提醒休息",
    "聊到喜欢的动画话题",
    "被请求帮忙整理房间",
    "下雨天聊天气",
]


def run(work_dir: Path, profile_path: Path, raw_root: Path,
        api_key: str | None = None, env_path: Path = Path(".env"),
        client_fn=None) -> dict:
    profile = yaml.safe_load(Path(profile_path).read_text(encoding="utf-8"))
    work_dir = Path(work_dir)
    card = yaml.safe_load((work_dir / "character_card.yaml").read_text(encoding="utf-8"))
    entries = json.loads((work_dir / "voice_map.json").read_text(encoding="utf-8"))

    if client_fn is None:
        if api_key is None:
            api_key = env_key(Path(env_path))
        base_url = base_url_from_env()

        def client(model_: str, messages: list[dict]):
            return chat_completion(base_url, model_, api_key, messages)
    else:
        def client(model_: str, messages: list[dict]):
            return client_fn(model_, messages)

    induce = profile["llm"]["induce"]["model"]
    speaker = profile["character"]["speaker_names"][0]

    # ── 真实语料句：官方汉化简体，5 情绪各取 1 条（多样性）──────────────────
    from contracts.emotion_defs import SENTIMENT_CLOSE_SET
    face2emo = yaml.safe_load(
        (REPO_ROOT / "contracts" / "face_map.yaml").read_text(encoding="utf-8")
    )["scn_faces"]
    by_emo: dict[str, list[str]] = {}
    for e in entries:
        if e.get("speaker") != speaker or e.get("origin") == "translated":
            continue
        zh = clean_text(e.get("text_zh") or "")
        if len(zh) < MIN_SENT_LEN:
            continue
        face = (e.get("self") or {}).get("face")
        emo = face2emo.get(face, "normal")
        if emo not in by_emo:
            by_emo[emo] = []
        by_emo[emo].append(zh)
    # 均匀取 5 种不同情绪的首句
    corpus_sents: list[str] = []
    used_emos: set[str] = set()
    for emo in sorted(by_emo):
        if len(corpus_sents) >= N_PER_SIDE:
            break
        if emo not in used_emos and by_emo[emo]:
            corpus_sents.append(by_emo[emo][0])
            used_emos.add(emo)

    # ── card 产出句：5 个场景 × induce 生成 ────────────────────────────────
    persona = card.get("persona", "")
    register = card.get("speech_style", {}).get("register", "口语短句")
    first_person = card.get("speech_style", {}).get("first_person", "私")
    card_sents: list[str] = []
    for scene in SCENES[:N_PER_SIDE]:
        prompt = BLIND_PROMPT_TMPL.format(
            persona=persona[:600], register=register,
            first_person=first_person, scene=scene)
        messages = [{"role": "user", "content": prompt}]
        content, _ = client(induce, messages)
        text = clean_text(content.strip().splitlines()[0])
        card_sents.append(text)

    # ── 确定性混排（seed = voice_map 内容 hash 前 8 位）───────────────────
    vm_hash = hashlib.sha256(
        (work_dir / "voice_map.json").read_bytes()).hexdigest()[:8]
    items = [("card", s) for s in card_sents] + [("corpus", s) for s in corpus_sents]
    rng = vm_hash
    for i in range(len(items) - 1, 0, -1):
        rng = int(hashlib.sha256((str(rng) + str(i)).encode()).hexdigest()[:8], 16)
        j = rng % (i + 1)
        items[i], items[j] = items[j], items[i]

    labels = "ABCDEFGHIJ"
    blind_items = [{"label": labels[i], "text": s} for i, (_, s) in enumerate(items)]
    answer_key = {labels[i]: src for i, (src, _) in enumerate(items)}

    out = {"items": blind_items, "answer_key": answer_key,
           "judgment_line": "首轮辨认率 ≥60%（≥6/10 正确）或累计两轮 ≥70%",
           "note": "请逐句判断哪句更像「都」的真实台词。答案卷勿盲测人翻阅。"}
    out_path = work_dir / "blind_test_5v5.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(f"[盲测] 5v5 材料生成完毕 → {out_path}")
    return out


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    parser = argparse.ArgumentParser(description="L2.5 盲测材料 5v5 生成")
    parser.add_argument("--work", default="work")
    parser.add_argument("--profile", default="config/game_9nine_ep1.yaml")
    parser.add_argument("--raw", default="")
    args = parser.parse_args()
    run(Path(args.work), Path(args.profile), Path(args.raw) if args.raw else Path("."))
