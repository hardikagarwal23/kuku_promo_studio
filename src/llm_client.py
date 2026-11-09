"""Configurable OpenAI-compatible LLM client."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from src.config import LLMEnvConfig, load_llm_env, load_yaml_config
from src.reliability import redact_secrets

logger = logging.getLogger(__name__)
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

# Rate limiter for 15 RPM (1 request every 4.0 seconds, padded to 4.2)
_RATE_LIMIT_LOCK = asyncio.Lock()
_LAST_CALL_TIME = 0.0
_MIN_REQUEST_INTERVAL = 4.5

class LLMClient:
    """Small provider wrapper. Never hardcodes keys or models in callers."""

    def __init__(
        self,
        *,
        env: LLMEnvConfig | None = None,
        max_concurrency: int | None = None,
    ) -> None:
        settings = load_yaml_config()
        self.env = env or load_llm_env(settings)
        self.temperature = self.env.temperature
        self.timeout_seconds = self.env.timeout_seconds
        self.max_concurrency = max_concurrency or self.env.max_concurrency
        self.semaphore = asyncio.Semaphore(self.max_concurrency)
        self.model = self.env.model
        self.call_count = 0
        self.fail_count = 0

        if not self.env.api_key:
            raise RuntimeError("LLM_API_KEY is not set. Add it to .env.")
        from openai import AsyncOpenAI

        kwargs: dict[str, Any] = {"api_key": self.env.api_key}
        if self.env.base_url:
            kwargs["base_url"] = self.env.base_url
        self._client = AsyncOpenAI(**kwargs)

    def load_prompt(self, name: str) -> str:
        path = PROMPTS_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"Prompt file missing: {path}")
        return path.read_text(encoding="utf-8")

    def render_prompt(self, name: str, payload: str) -> str:
        template = self.load_prompt(name)
        return template.replace("{{INPUT}}", payload)

    async def complete(self, prompt: str, *, stage: str = "generic", model_override: str = None) -> str:
        async with self.semaphore:
            target_model = model_override or self.model
            
            # 1. Enforce strict 4.2s pacing to stay under 15 RPM
            global _LAST_CALL_TIME
            async with _RATE_LIMIT_LOCK:
                now = time.monotonic()
                elapsed = now - _LAST_CALL_TIME
                if elapsed < _MIN_REQUEST_INTERVAL:
                    await asyncio.sleep(_MIN_REQUEST_INTERVAL - elapsed)
                _LAST_CALL_TIME = time.monotonic()

            for attempt in range(4):
                try:
                    self.call_count += 1
                    
                    resp = await asyncio.wait_for(
                        self._client.chat.completions.create(
                            model=target_model,
                            temperature=self.temperature,
                            max_completion_tokens=2500,
                            messages=[
                                {
                                    "role": "system",
                                    "content": "Return strict JSON only. No markdown.",
                                },
                                {"role": "user", "content": prompt},
                            ],
                            response_format={"type": "json_object"},
                        ),
                        timeout=self.timeout_seconds,
                    )
                    content = resp.choices[0].message.content or ""
                    return content
                except Exception as exc:
                    if hasattr(exc, "response") and exc.response.status_code == 429:
                        retry_after = float(exc.response.headers.get("retry-after", 5.0))
                        logger.warning(f"429 Rate Limit on {target_model}. Sleeping {retry_after}s...")
                        await asyncio.sleep(retry_after)
                        continue 
                        
                    self.fail_count += 1
                    logger.error("LLM call failed (%s): %s", stage, redact_secrets(str(exc)))
                    raise
            
            raise RuntimeError(f"Max retries exceeded for 429 Rate Limit on {target_model}.")