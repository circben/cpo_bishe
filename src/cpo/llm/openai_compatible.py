from __future__ import annotations

import json
import socket
import time
from dataclasses import dataclass
from urllib import error, request


@dataclass
class OpenAICompatibleConfig:
    api_key: str
    base_url: str = "https://api.siliconflow.cn/v1"
    model: str = "Qwen/Qwen3-8B"
    timeout: int = 120
    enable_thinking: bool | None = None
    max_retries: int = 3
    retry_backoff_sec: float = 1.0


def chat_completion(
    *,
    config: OpenAICompatibleConfig,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> str:
    payload = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if config.enable_thinking is not None:
        payload["enable_thinking"] = config.enable_thinking
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    url = config.base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }
    req = request.Request(url=url, data=body, headers=headers, method="POST")

    last_exc: Exception | None = None
    for attempt in range(max(1, config.max_retries)):
        try:
            with request.urlopen(req, timeout=config.timeout) as resp:
                raw = resp.read().decode("utf-8")
            break
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            # Do not retry auth/permission/client errors.
            if exc.code in {400, 401, 403, 404}:
                raise RuntimeError(f"API HTTPError {exc.code}: {detail}") from exc
            last_exc = RuntimeError(f"API HTTPError {exc.code}: {detail}")
        except error.URLError as exc:
            last_exc = RuntimeError(f"API URLError: {exc}")
        except (TimeoutError, socket.timeout) as exc:
            last_exc = RuntimeError(f"API TimeoutError: {exc}")

        if attempt + 1 < max(1, config.max_retries):
            time.sleep(config.retry_backoff_sec * (2**attempt))

    else:
        raise RuntimeError("API request failed with unknown error")

    if 'raw' not in locals():
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("API request failed without response")

    data = json.loads(raw)
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"API returned no choices: {data}")

    msg = choices[0].get("message") or {}
    content = msg.get("content", "")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"API returned empty content: {data}")
    return content.strip()
