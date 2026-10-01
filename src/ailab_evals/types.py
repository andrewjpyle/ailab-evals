"""Core value types shared by every runner, judge, and gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Example:
    """One dataset row. ``expected`` is the gold label / answer; ``meta`` is free-form."""

    id: str
    input: Any
    expected: Any = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelSpec:
    """A candidate model and its list price (USD per million tokens).

    Price lives on the spec, not the provider, so the cost column is reproducible
    from the results file alone and a local model can be priced at $0 honestly.
    """

    name: str
    provider: str
    model: str
    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0
    params: dict[str, Any] = field(default_factory=dict)

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok) / 1_000_000


@dataclass
class Completion:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    error: str | None = None


@dataclass
class Prediction:
    example_id: str
    output: Any
    raw: str = ""
    confidence: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
