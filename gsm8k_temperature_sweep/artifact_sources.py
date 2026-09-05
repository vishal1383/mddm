"""Canonical artifact sources for the standalone sweep."""
from __future__ import annotations

from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parent

BASE_MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
DPARALLEL_MODEL_ID = "Zigeng/dParallel-LLaDA-8B-instruct"
JUSTGRPO_MODEL_ID = "nzl-thu/LLaDA-Instruct-JustGRPO-GSM8K"

# Retained only for the historical 16-example diagnostic script.  The
# production chain neither downloads nor evaluates this unofficial artifact.
PAPER_POLICY_REPO_ID = "orkunkinay/ml-rl-dllm-gs8"
PAPER_POLICY_FILENAME = "checkpoint-best/model.safetensors"

# The exact standard GSM8K LoRA used by the existing mddm full256 baseline.
# It is committed with this standalone experiment so a clean clone needs no
# shared-filesystem checkpoint path or environment variable.
SFT_ADAPTER_PATH = EXPERIMENT_ROOT / "artifacts" / "gsm8k_lora_sft"
