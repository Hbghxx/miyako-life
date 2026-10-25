"""LLM 调用客户端（三方共享 · contracts，约束 A 允许的唯一共享层）。

induce / judge / translate 三槽位统一经此发起 chat 调用；模型路由来自
profile llm 节，key 来自 .env（BYOK）。纯 httpx，无第三方 SDK。
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx

DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/coding/paas/v4"
HTTP_TIMEOUT = 360.0   # glm-5.3 思考模式 + 大样本/合并 prompt 实测可超 4 分钟（240s 曾超时，2026-09-12 上调）


def env_key(env_path: Path, names: tuple[str, ...] = ("GLM_API_KEY", "ZHIPU_API_KEY")) -> str:
    """从 .env 读取 API key（首个命中的变量）；缺失抛 RuntimeError。"""
    path = Path(env_path)
    if not path.is_file():
        raise RuntimeError(f".env 不存在：{path}（GLM_API_KEY=<你的智谱 key>，BYOK 不入库）")
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        for name in names:
            if s.startswith(name + "="):
                key = s.split("=", 1)[1].strip().strip('"').strip("'")
                if key:
                    return key
    raise RuntimeError(f".env 中没有 {'/'.join(names)}")


def chat_completion(base_url: str, model: str, api_key: str, messages: list[dict],
                    temperature: float = 0.3, timeout: float = HTTP_TIMEOUT,
                    retries: int = 3) -> tuple[str, dict]:
    """单次对话补全：返回 (回复文本, usage)。

    仅 429 限频按指数退避重试（尊重 Retry-After 头）；其余 HTTP 异常立即向上抛
    （fail-fast——重试语义与文档一致，2026-09-12 修正：原实现会把非 429 也无退避重试）。
    """
    import time

    for attempt in range(retries):
        try:
            resp = httpx.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": model, "messages": messages, "temperature": temperature},
                timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip(), data.get("usage", {})
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 429 or attempt >= retries - 1:
                raise
            retry_after = exc.response.headers.get("Retry-After")
            delay = float(retry_after) if (retry_after or "").replace(".", "", 1).isdigit() \
                else 20.0 * (attempt + 1)
            time.sleep(delay)
    raise RuntimeError("chat_completion: 不可达（retries 循环守卫）")


def base_url_from_env() -> str:
    return os.environ.get("GLM_BASE_URL", DEFAULT_BASE_URL)
