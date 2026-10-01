"""JSONL datasets with a content hash, so a results file names exactly what it measured."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from .types import Example


@dataclass(frozen=True)
class Dataset:
    name: str
    examples: tuple[Example, ...]

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self) -> Iterator[Example]:
        return iter(self.examples)

    @property
    def sha256(self) -> str:
        """Stable hash of the examples (order-sensitive). Two runs on the same hash are comparable."""
        h = hashlib.sha256()
        for ex in self.examples:
            row = {"id": ex.id, "input": ex.input, "expected": ex.expected}
            h.update(json.dumps(row, sort_keys=True, ensure_ascii=False).encode())
            h.update(b"\n")
        return h.hexdigest()

    @classmethod
    def from_records(cls, name: str, records: Iterable[dict]) -> Dataset:
        examples = []
        seen: set[str] = set()
        for i, rec in enumerate(records):
            ex_id = str(rec.get("id", i))
            if ex_id in seen:
                raise ValueError(f"{name}: duplicate example id {ex_id!r}")
            seen.add(ex_id)
            examples.append(
                Example(id=ex_id, input=rec["input"], expected=rec.get("expected"), meta=rec.get("meta", {}))
            )
        return cls(name=name, examples=tuple(examples))

    @classmethod
    def load_jsonl(cls, path: str | Path, name: str | None = None) -> Dataset:
        path = Path(path)
        with path.open(encoding="utf-8") as fh:
            records = [json.loads(line) for line in fh if line.strip()]
        return cls.from_records(name or path.stem, records)
