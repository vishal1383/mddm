#!/usr/bin/env python3
"""Resolve mutable model references once before the chained submission."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi

from evaluate import BASE_MODEL_ID, DPARALLEL_MODEL_ID


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve() if args.output else None
    if output and output.is_file():
        sealed = json.loads(output.read_text(encoding="utf-8"))
        print(str(sealed["base_model_revision"]), str(sealed["dparallel_model_revision"]))
        return
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    base = api.model_info(BASE_MODEL_ID, revision=os.environ.get("BASE_MODEL_REVISION", "main")).sha
    dparallel = api.model_info(
        DPARALLEL_MODEL_ID, revision=os.environ.get("DPARALLEL_MODEL_REVISION", "main")
    ).sha
    if not base or not dparallel or any(character.isspace() for character in f"{base}{dparallel}"):
        raise RuntimeError("failed to seal immutable model revisions")
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
        temporary.write_text(
            json.dumps(
                {"base_model_revision": str(base), "dparallel_model_revision": str(dparallel)},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    print(base, dparallel)


if __name__ == "__main__":
    main()
