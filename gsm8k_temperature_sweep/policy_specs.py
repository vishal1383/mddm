"""Architecture contracts shared by training and evaluation."""
from __future__ import annotations

from typing import Any


APPLE_POLICY_ARCHITECTURE: dict[str, Any] = {
    "policy_type": "dit_confidence",
    "hidden_dim": 128,
    "feedforward_dim": 512,
    "num_heads": 2,
    "dropout": 0.0,
    "time_embed_dim": 128,
    "confidences_top_p": 1,
    "smart_init": -2.0,
    "num_blocks": 1,
    "time_period": 1,
    "full_context": True,
}
