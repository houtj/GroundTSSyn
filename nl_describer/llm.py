"""Azure OpenAI client wrapper with structured-output helper.

Reads keys + deployment from environment variables (loaded via python-dotenv).
Exposes one public function:

    structured_call(messages, response_model, *, agent_name, temperature)

which calls `client.chat.completions.parse(response_format=response_model)`
with retries and a timeout, returning the parsed Pydantic instance.

Vision messages are constructed via `make_vision_message(text, images)`,
where `images` is a list of `(label, base64_png_bytes)` tuples.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Type, TypeVar

from dotenv import load_dotenv
from openai import APIError, APITimeoutError, AzureOpenAI, RateLimitError
from pydantic import BaseModel

from .config import AzureConfig, Config

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger("nl_describer.llm")


def _load_env() -> None:
    """Load .env from repo root if present."""
    repo_root = Path(__file__).resolve().parents[1]
    env_path = repo_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def make_client(azure_cfg: AzureConfig) -> tuple[AzureOpenAI, str]:
    """Build an AzureOpenAI client and return (client, deployment_name)."""
    _load_env()
    endpoint = os.environ.get(azure_cfg.endpoint_env)
    api_key = os.environ.get(azure_cfg.api_key_env)
    deployment = os.environ.get(azure_cfg.deployment_env)
    api_version = os.environ.get(azure_cfg.api_version_env)
    missing = [
        n
        for n, v in [
            (azure_cfg.endpoint_env, endpoint),
            (azure_cfg.api_key_env, api_key),
            (azure_cfg.deployment_env, deployment),
            (azure_cfg.api_version_env, api_version),
        ]
        if not v
    ]
    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {missing}. "
            "Add them to .env or export them."
        )
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )
    return client, deployment  # type: ignore[return-value]


def make_vision_message(
    text: str,
    images: list[tuple[str, bytes]] | None = None,
) -> dict[str, Any]:
    """Build a `user` message with one or more inline base64 PNGs.

    `images` is a list of (label, png_bytes); the label is rendered as text
    immediately before the corresponding image so the model can refer to it.
    """
    if not images:
        return {"role": "user", "content": text}
    import base64

    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for label, png_bytes in images:
        content.append({"type": "text", "text": f"\n[{label}]"})
        b64 = base64.b64encode(png_bytes).decode("ascii")
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{b64}",
                },
            }
        )
    return {"role": "user", "content": content}


def structured_call(
    client: AzureOpenAI,
    deployment: str,
    messages: list[dict[str, Any]],
    response_model: Type[T],
    *,
    cfg: Config,
    agent_name: str,
    temperature: float | None = None,
) -> T:
    """Call `chat.completions.parse` with retries.

    `agent_name` is used purely for logging.
    """
    retries = cfg.llm.retries
    last_error: Exception | None = None
    kwargs: dict[str, Any] = {
        "model": deployment,
        "messages": messages,
        "response_format": response_model,
        "timeout": cfg.llm.request_timeout_s,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    for attempt in range(retries + 1):
        try:
            t0 = time.perf_counter()
            completion = client.chat.completions.parse(**kwargs)
            dt = time.perf_counter() - t0
            parsed = completion.choices[0].message.parsed
            if parsed is None:
                raise RuntimeError(
                    f"[{agent_name}] response.parsed is None — likely a refusal "
                    f"or model failed schema. raw={completion.choices[0].message}"
                )
            logger.info("[%s] ok (%.1fs, attempt %d)", agent_name, dt, attempt + 1)
            return parsed  # type: ignore[return-value]
        except (RateLimitError, APITimeoutError, APIError) as e:
            last_error = e
            wait = 2.0 * (2**attempt)
            logger.warning(
                "[%s] attempt %d failed (%s); retrying in %.1fs",
                agent_name,
                attempt + 1,
                type(e).__name__,
                wait,
            )
            time.sleep(wait)
        except Exception as e:
            # Non-retryable error
            raise RuntimeError(f"[{agent_name}] unrecoverable: {e}") from e
    raise RuntimeError(
        f"[{agent_name}] exhausted {retries + 1} attempts; last error: {last_error}"
    )
