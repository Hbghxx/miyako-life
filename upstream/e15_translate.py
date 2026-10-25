"""E1.5 翻译工序（translate 路径 · plan L1.2；语义源 = 方案契约 §五 E1.5）。

输入：kirikiri 台本提取的日文 voice_map 行集（与 E1 同一提取器，确定性零成本）；
工序：translate 槽（glm-4-plus）批量粗译，上下文窗口 = 前后各 2 句；
纪律：保持语气/第一人称/口癖；**专名/作品名一律保留原文写法**；
落盘：work/translations.json {voice_id: 中文}（由 E1 合成双语冻结件，origin: translated）；
失败：单条失败进 work/translate_review.json 待校对清单，**不中断批次**（§4.1）；
成本：逐调用记账 → work/llm_cost.json（槽位 translate，配额见 profile）。

key 纪律：GLM_API_KEY 写在仓库根 .env（BYOK，gitignore）；缺 key = fail-fast，
**禁止假 key 伪装验收**（plan L1.2 人工介入点）。

复跑语义（L6）：translations.json 已存在的条目不重译——重跑只补缺口，零重复花费。
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
import re
import tempfile
from pathlib import Path

import yaml

from contracts.cost import CostMeter
from contracts.llm import base_url_from_env, chat_completion, env_key
from contracts.logging_util import get_logger
from upstream.extractors.base import iter_scn_files
from upstream.extractors.kirikiri_9nine import Kirikiri9nineExtractor

LOG = get_logger("e15")

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = REPO_ROOT / ".env"
# 写入白名单（与 E1 同纪律）：仓库工作区 / 系统临时目录
ALLOWED_WORK_BASES = tuple(
    os.path.realpath(p) for p in (REPO_ROOT, Path(tempfile.gettempdir()))
)

# 估算单价（元/千 tokens，含输入与输出）；实际以平台账单为准，成本面板为估算值。
# 2026-09-12 补 glm-5.3 系（估位：flash 档 0.001 / 全量档 0.05，与 E6 估算表同口径）
GLM_PRICING_YUAN_PER_1K = {"glm-4-plus": 0.05, "glm-4-flash": 0.001,
                           "glm-4.5-air": 0.002, "glm-5.3-flash": 0.001,
                           "glm-5.3": 0.05}
CONTEXT_WINDOW = 2            # 上下文窗口 = 前后各 2 句（契约 §五 E1.5）
HTTP_TIMEOUT = 60.0
MAX_ATTEMPTS = 2              # 单条最多尝试 2 次，仍败进待校对清单

# 抽查选句的确定性信号（L4：选择归脚本；契约 §五：高频口癖/第一人称密集/语气词句）
TIC_SUFFIXES = ("かな", "だね", "よね", "かも", "だよね", "なの", "だよ", "だろう")
FIRST_PERSON_RE = re.compile(r"私|わたし|あたし|うち|僕|俺")
INTERJECTION_RE = re.compile(r"[ねよわさ]$")


class E15FailFast(RuntimeError):
    """E1.5 工序失败（缺 key 等）——fail-fast，不伪装验收。"""


def load_api_key(env_path: Path = ENV_PATH) -> str:
    """从 .env 读 GLM_API_KEY（兼容 ZHIPU_API_KEY）；缺失即 fail-fast。

    实现收敛到 contracts/llm.env_key（2026-09-12，审评 P1-4）；E15FailFast 包装
    保持本工序 fail-fast 语义与 CLI 退出码不变。
    """
    try:
        return env_key(Path(env_path))
    except RuntimeError as exc:
        raise E15FailFast(str(exc)) from exc


def build_context(entries: list[dict], index: int,
                  window: int = CONTEXT_WINDOW) -> list[str]:
    """上下文窗口：前后各 window 句的台词原文（不含自身）。"""
    lo = max(0, index - window)
    hi = min(len(entries), index + window + 1)
    return [e["text"] for j, e in enumerate(entries) if lo <= j < hi and j != index]


def translation_score(text: str) -> int:
    """抽查选句确定性打分：口癖后缀×3 + 第一人称密度×2 + 句尾语气词×1。"""
    fp = len(FIRST_PERSON_RE.findall(text))
    suffix = any(text.rstrip("」。？！… ").endswith(s) for s in TIC_SUFFIXES)
    interject = bool(INTERJECTION_RE.search(text.rstrip("」。？！… ")))
    return 3 * int(suffix) + 2 * min(fp, 3) + int(interject)


def select_review_quota(entries: list[dict], quota: int) -> list[dict]:
    """报告页中文校对区材料（≈40 条）：确定性打分取 top-N，平分以 voice_id 定序。"""
    ranked = sorted(entries,
                    key=lambda e: (-translation_score(e["text"]), e["voice"]))
    return ranked[:quota]


def build_messages(text: str, context: list[str]) -> list[dict]:
    ctx = "\n".join(f"・{c}" for c in context) or "（无）"
    return [
        {"role": "system",
         "content": "你是 galgame 台本翻译。把日文台词翻译成简体中文，要求："
                    "保持角色的语气与第一人称；句尾口癖等语言习惯在中文里保留等价痕迹；"
                    "专名、作品名、地名、人名一律保留日文原文写法，不翻译不改写；"
                    "只输出译文，不要任何解释。"},
        {"role": "user",
         "content": f"上下文（前后各{CONTEXT_WINDOW}句，仅供理解，不要翻译）：\n{ctx}\n\n"
                    f"待翻译台词：\n{text}"},
    ]


def chat_translate(base_url: str, model: str, api_key: str,
                   text: str, context: list[str]) -> tuple[str, dict]:
    """单条翻译：返回 (译文, usage)。

    HTTP 层统一走 contracts/llm.chat_completion（429 退避重试同款；2026-09-12
    审评 P1-4 收敛——原自带 httpx 客户端 + 非 coding 端点兜底已漂移）。
    """
    content, usage = chat_completion(
        base_url, model, api_key, build_messages(text, context),
        temperature=0.2, timeout=HTTP_TIMEOUT)
    return content, usage


def _ensure_writable(work_dir: Path) -> None:
    wd = os.path.realpath(work_dir)
    for base in ALLOWED_WORK_BASES:
        if wd == base or wd.startswith(base + os.sep):
            return
    raise E15FailFast(f"work 目录越出允许范围（仓库内或系统临时目录）：{work_dir}")


def run(raw_root: Path, profile_path: Path, work_dir: Path,
        quota: int | None = None, api_key: str | None = None,
        env_path: Path = ENV_PATH, client_fn=chat_translate) -> dict:
    work_dir = Path(os.path.realpath(Path(work_dir).expanduser()))
    _ensure_writable(work_dir)
    profile = yaml.safe_load(Path(profile_path).read_text(encoding="utf-8"))
    llm_cfg = profile["llm"]["translate"]
    model = llm_cfg["model"]
    base_url = (llm_cfg.get("base_url")
                or profile.get("llm", {}).get("base_url")
                or base_url_from_env())   # 与 E4/E6 同序：profile 优先（coding 端点）
    rate = GLM_PRICING_YUAN_PER_1K.get(model, 0.05)
    if api_key is None:
        api_key = load_api_key(env_path)

    # 与 E1 同一提取器同一顺序（确定性；extract 结果即 E1 冻结件内容）
    extractor = Kirikiri9nineExtractor()
    scn_files = iter_scn_files(Path(os.path.realpath(raw_root)),
                               profile["engine"]["scn_pattern"])
    entries = extractor.extract_files(scn_files)

    # 补缺口集（E1.5 降级补缺 · 2026-09-11 embedded 定案）：
    # 当前 voice_map 已有 localized 覆盖时，只翻"仍缺 text_zh"的 voice；
    # embedded 态且无基线 voice_map → 不宜全量机翻，跳过并提示先跑 E1。
    vm_path = work_dir / "voice_map.json"
    embedded = profile["input"]["bilingual_source"] == "embedded"
    lacking: set[str] | None = None
    if vm_path.is_file():
        vm = json.loads(vm_path.read_text(encoding="utf-8"))
        lacking = {e["voice"] for e in vm if not e.get("text_zh")}
    elif embedded:
        print("[E1.5] embedded 锁定但尚无 E1 基线——机翻全量被禁止（降级补缺角色），跳过")
        return {"translated": 0, "review": 0, "cost_yuan": 0.0}

    trans_path = work_dir / "translations.json"
    done: dict[str, str] = {}
    if trans_path.is_file():  # L6：已译条目不重烧 API 钱
        done = json.loads(trans_path.read_text(encoding="utf-8"))
    todo = [e for e in entries
            if e["text"] and e["voice"] not in done
            and (lacking is None or e["voice"] in lacking)]
    if quota is not None:
        picked = {e["voice"] for e in select_review_quota(todo, quota)}
        todo = [e for e in todo if e["voice"] in picked]
    print(f"[E1.5] 待译 {len(todo)} 条（模型 {model}，已缓存 {len(done)}）")
    LOG.info("批次开始 待译=%d 已缓存=%d 模型=%s 缺口集=%s",
             len(todo), len(done), model, "限定" if lacking is not None else "全量")

    meter = CostMeter(profile["cost"]["quota_warning"],
                      work_dir / "llm_cost.json")
    review: list[dict] = []
    # 上下文窗口必须取全量 entries 中的位置——todo 是过滤/抽样子集，enumerate(todo)
    # 的下标切片出来的"前后句"是无关台词（2026-09-18 修正；按对象身份映射，对
    # 提取层的重复 voice id 也精确定位）
    idx_of = {id(e): k for k, e in enumerate(entries)}
    for i, entry in enumerate(todo):
        context = build_context(entries, idx_of[id(entry)])
        last_err = ""
        for _ in range(MAX_ATTEMPTS):
            try:
                zh, usage = client_fn(base_url, model, api_key,
                                      entry["text"], context)
                done[entry["voice"]] = zh
                cost = (usage.get("prompt_tokens", 0)
                        + usage.get("completion_tokens", 0)) / 1000 * rate
                meter.record("translate", model,
                             usage.get("prompt_tokens", 0),
                             usage.get("completion_tokens", 0), cost)
                last_err = ""
                break
            except Exception as exc:  # noqa: BLE001 —— 单条失败不中断批次（§4.1）
                last_err = f"{exc.__class__.__name__}: {exc}"
        if last_err:
            review.append({"voice": entry["voice"], "text": entry["text"],
                           "error": last_err})
            LOG.warning("译文失败进待校对 voice=%s err=%s", entry["voice"], last_err)
        if (i + 1) % 50 == 0:
            print(f"[E1.5] 进度 {i + 1}/{len(todo)}（失败 {len(review)}）")
            LOG.info("进度 %d/%d 失败=%d", i + 1, len(todo), len(review))

    trans_path.write_text(json.dumps(done, ensure_ascii=False, indent=1) + "\n",
                          encoding="utf-8")
    review_path = work_dir / "translate_review.json"
    review_path.write_text(json.dumps(review, ensure_ascii=False, indent=1) + "\n",
                           encoding="utf-8")
    cost_path = meter.save()
    warn = " ⚠️ 超阈值（报告页警示）" if meter.over_quota else ""
    print(f"[E1.5] 完成：译得 {len(todo) - len(review)} 条，待校对 {len(review)} 条 → "
          f"{review_path.name}；本次成本 {meter.total_yuan} 元 → {cost_path.name}{warn}")
    LOG.info("批次完成 译得=%d 待校对=%d 本次成本=%.4f元",
             len(todo) - len(review), len(review), meter.total_yuan)
    return {"translated": len(todo) - len(review), "review": len(review),
            "cost_yuan": meter.total_yuan}


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    parser = argparse.ArgumentParser(description="E1.5 翻译工序（translate 槽批量粗译）")
    parser.add_argument("--raw", required=True)
    parser.add_argument("--profile", default="config/game_9nine_ep1.yaml")
    parser.add_argument("--work", default="work")
    parser.add_argument("--quota", type=int, default=None,
                        help="抽查配额（报告页中文校对区 ≈40 条）；缺省=全量未译条目")
    parser.add_argument("--env", default=str(ENV_PATH))
    args = parser.parse_args()
    from contracts.logging_util import setup_logging
    setup_logging(Path(args.work))   # 直跑也落 work/logs/pipeline.log

    try:
        run(Path(args.raw), Path(args.profile), Path(args.work),
            quota=args.quota, env_path=Path(args.env))
    except E15FailFast as exc:
        LOG.error("fail-fast %s", exc)
        print(f"[E1.5][fail-fast] {exc}", file=sys.stderr)
        sys.exit(2)
