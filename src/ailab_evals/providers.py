"""Model providers. Stdlib-only HTTP so the package has zero runtime dependencies.

A provider turns ``(ModelSpec, system, prompt)`` into a :class:`Completion`. Any
callable with that shape works, which is how a host application plugs in its own
LLM client (with its own keys, retries, and spend controls) without this package
ever seeing a credential.

``ReplayProvider`` wraps another provider with a JSONL cassette: record once
against real models, then replay deterministically in CI with no network and no key.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from .types import Completion, ModelSpec


class Provider(Protocol):
    def __call__(self, spec: ModelSpec, system: str, prompt: str, *, json_mode: bool = False) -> Completion: ...


def _post_json(url: str, payload: dict, headers: dict[str, str], timeout: float) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", **headers}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - caller-supplied base URL
        return json.loads(resp.read())


class OllamaProvider:
    """Local models via the Ollama ``/api/chat`` endpoint. Cost is whatever the spec says (usually $0)."""

    def __init__(self, base_url: str | None = None, timeout: float = 300.0):
        self.base_url = (base_url or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        if not self.base_url.startswith("http"):
            self.base_url = "http://" + self.base_url
        self.timeout = timeout

    def __call__(self, spec: ModelSpec, system: str, prompt: str, *, json_mode: bool = False) -> Completion:
        payload = {
            "model": spec.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": spec.params.get("temperature", 0)},
        }
        if json_mode:
            payload["format"] = "json"
        if "think" in spec.params:
            payload["think"] = spec.params["think"]
        t0 = time.perf_counter()
        try:
            data = _post_json(f"{self.base_url}/api/chat", payload, {}, self.timeout)
        except (urllib.error.URLError, TimeoutError) as exc:
            return Completion(text="", error=f"ollama: {exc}", latency_ms=(time.perf_counter() - t0) * 1000)
        return Completion(
            text=data.get("message", {}).get("content", ""),
            input_tokens=int(data.get("prompt_eval_count", 0)),
            output_tokens=int(data.get("eval_count", 0)),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )


class AnthropicProvider:
    """Anthropic Messages API. Reads ``ANTHROPIC_API_KEY`` from the environment only."""

    def __init__(self, api_key: str | None = None, timeout: float = 120.0):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.timeout = timeout

    def __call__(self, spec: ModelSpec, system: str, prompt: str, *, json_mode: bool = False) -> Completion:
        if not self.api_key:
            return Completion(text="", error="anthropic: ANTHROPIC_API_KEY not set")
        payload = {
            "model": spec.model,
            "max_tokens": spec.params.get("max_tokens", 512),
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}
        t0 = time.perf_counter()
        try:
            data = _post_json("https://api.anthropic.com/v1/messages", payload, headers, self.timeout)
        except urllib.error.HTTPError as exc:
            return Completion(text="", error=f"anthropic: HTTP {exc.code}", latency_ms=(time.perf_counter() - t0) * 1000)
        except (urllib.error.URLError, TimeoutError) as exc:
            return Completion(text="", error=f"anthropic: {exc}", latency_ms=(time.perf_counter() - t0) * 1000)
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        usage = data.get("usage", {})
        return Completion(
            text=text,
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )


class ProviderRegistry:
    """Maps ``ModelSpec.provider`` to a provider callable."""

    def __init__(self, providers: dict[str, Provider] | None = None):
        self._providers: dict[str, Provider] = dict(providers or {})

    @classmethod
    def default(cls) -> ProviderRegistry:
        return cls({"ollama": OllamaProvider(), "anthropic": AnthropicProvider()})

    def register(self, name: str, provider: Provider | Callable[..., Completion]) -> None:
        self._providers[name] = provider

    def __call__(self, spec: ModelSpec, system: str, prompt: str, *, json_mode: bool = False) -> Completion:
        try:
            provider = self._providers[spec.provider]
        except KeyError:
            return Completion(text="", error=f"no provider registered for {spec.provider!r}")
        return provider(spec, system, prompt, json_mode=json_mode)


def cassette_key(spec: ModelSpec, system: str, prompt: str, json_mode: bool) -> str:
    raw = json.dumps([spec.provider, spec.model, spec.params, system, prompt, json_mode], sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


class ReplayProvider:
    """Record/replay cassette over another provider.

    mode="replay": serve from the cassette; a miss is an error (CI must be hermetic).
    mode="record": serve hits from the cassette, call through on a miss and append it.
    """

    def __init__(self, path: str | Path, inner: Provider | None = None, mode: str = "replay"):
        if mode not in {"replay", "record"}:
            raise ValueError("mode must be 'replay' or 'record'")
        if mode == "record" and inner is None:
            raise ValueError("record mode needs an inner provider")
        self.path = Path(path)
        self.inner = inner
        self.mode = mode
        self._lock = threading.Lock()
        self._cache: dict[str, dict] = {}
        if self.path.exists():
            with self.path.open(encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        row = json.loads(line)
                        self._cache[row["hash"]] = row

    def __call__(self, spec: ModelSpec, system: str, prompt: str, *, json_mode: bool = False) -> Completion:
        key = cassette_key(spec, system, prompt, json_mode)
        hit = self._cache.get(key)
        if hit is not None:
            return Completion(
                text=hit["text"],
                input_tokens=hit["input_tokens"],
                output_tokens=hit["output_tokens"],
                latency_ms=hit["latency_ms"],
            )
        if self.mode == "replay":
            return Completion(text="", error=f"cassette miss for {spec.name} ({key[:12]})")
        completion = self.inner(spec, system, prompt, json_mode=json_mode)
        if completion.error is None:
            row = {
                "hash": key,
                "model": spec.model,
                "text": completion.text,
                "input_tokens": completion.input_tokens,
                "output_tokens": completion.output_tokens,
                "latency_ms": round(completion.latency_ms, 1),
            }
            with self._lock:
                self._cache[key] = row
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return completion
