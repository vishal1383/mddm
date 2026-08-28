#!/usr/bin/env python3
"""Resolve mutable model references once before the chained submission."""
from __future__ import annotations

import os

from huggingface_hub import HfApi

from evaluate import BASE_MODEL_ID, DPARALLEL_MODEL_ID


def main() -> None:
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    base = api.model_info(BASE_MODEL_ID, revision=os.environ.get("BASE_MODEL_REVISION", "main")).sha
    dparallel = api.model_info(
        DPARALLEL_MODEL_ID, revision=os.environ.get("DPARALLEL_MODEL_REVISION", "main")
    ).sha
    if not base or not dparallel or any(character.isspace() for character in f"{base}{dparallel}"):
        raise RuntimeError("failed to seal immutable model revisions")
    print(base, dparallel)


if __name__ == "__main__":
    main()
