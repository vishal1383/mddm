from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ProbeExample:
    """Prompt/completion pair used by the anchor probes."""

    example_id: str
    dataset: str
    prompt: str
    completion: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EncodedExample:
    """Tokenized prompt and gold completion for one probe example."""

    example_id: str
    prompt_token_ids: list[int]
    completion_token_ids: list[int]
    completion_token_texts: list[str]
    prompt_text: str
    completion_text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    anchor_positions: list[int] | None = None

    @property
    def completion_length(self) -> int:
        return len(self.completion_token_ids)

    @property
    def prompt_length(self) -> int:
        return len(self.prompt_token_ids)


@dataclass(frozen=True)
class ConfidenceResult:
    """Model confidence summaries for completion positions."""

    anchor_positions: tuple[int, ...]
    p_gt: list[float]
    entropy: list[float]
